"""Document ingestion must publish an index only after every write succeeds."""

import asyncio
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import chromadb
from chromadb.errors import NotFoundError

import services.vectorstore as vectorstore


def _embedding_response(texts):
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=[float(i), 1.0]) for i, _ in enumerate(texts)]
    )


class TestDocumentIngestAtomicity(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cache_snapshot = dict(vectorstore._bm25_cache)
        vectorstore._bm25_cache.clear()
        vectorstore._active_staging_names.clear()

    def tearDown(self):
        vectorstore._bm25_cache.clear()
        vectorstore._bm25_cache.update(self.cache_snapshot)
        vectorstore._active_staging_names.clear()

    async def test_embedding_failure_never_creates_collection(self):
        client = MagicMock()
        client.get_collection.side_effect = NotFoundError("missing")
        provider_error = RuntimeError("embedding unavailable")

        with patch.object(vectorstore, "chromadb_client", client), \
             patch.object(vectorstore, "_embed", AsyncMock(side_effect=provider_error)):
            with self.assertRaises(RuntimeError) as raised:
                await vectorstore.deal_document("notes.md", "notes.md", ["content"])

        self.assertIs(raised.exception, provider_error)
        client.create_collection.assert_not_called()
        client.delete_collection.assert_not_called()

    async def test_all_embedding_batches_finish_before_staging_write(self):
        events = []
        client = MagicMock()

        def get_collection(*, name):
            events.append(("lookup", name))
            raise NotFoundError("missing")

        staging = MagicMock()
        staging.name = "studyloop-staging-test"
        staging.add.side_effect = lambda **_: events.append(("add", staging.name))
        staging.modify.side_effect = lambda **kwargs: events.append(("publish", kwargs["name"]))
        client.get_collection.side_effect = get_collection
        client.create_collection.side_effect = lambda **kwargs: (
            events.append(("create", kwargs["name"])) or staging
        )

        async def embed(batch):
            events.append(("embed", len(batch)))
            return _embedding_response(batch)

        chunks = [f"chunk {i}" for i in range(vectorstore.EMBED_BATCH_SIZE + 1)]
        with patch.object(vectorstore, "chromadb_client", client), \
             patch.object(vectorstore, "_embed", embed):
            count = await vectorstore.deal_document("notes.md", "notes.md", chunks)

        self.assertEqual(count, len(chunks))
        writes = [event[0] for event in events if event[0] != "lookup"]
        self.assertEqual(writes, ["embed", "embed", "create", "add", "publish"])
        self.assertEqual(staging.modify.call_args.kwargs["name"], "notes.md")
        self.assertEqual(
            staging.modify.call_args.kwargs["metadata"]["ingest_status"], "indexed"
        )

    async def test_write_failure_rolls_back_only_staging_collection(self):
        client = MagicMock()
        client.get_collection.side_effect = NotFoundError("missing")
        staging = MagicMock()
        staging.name = "studyloop-staging-owned"
        write_error = RuntimeError("index write failed")
        staging.add.side_effect = write_error
        client.create_collection.return_value = staging
        vectorstore._bm25_cache["notes.md"] = {"existing": True}

        with patch.object(vectorstore, "chromadb_client", client), \
             patch.object(vectorstore, "_embed", AsyncMock(
                 return_value=_embedding_response(["content"])
             )):
            with self.assertRaises(RuntimeError) as raised:
                await vectorstore.deal_document("notes.md", "notes.md", ["content"])

        self.assertIs(raised.exception, write_error)
        client.delete_collection.assert_called_once_with(name="studyloop-staging-owned")
        staging.modify.assert_not_called()
        self.assertIn("notes.md", vectorstore._bm25_cache)

    async def test_cleanup_failure_does_not_mask_index_failure(self):
        client = MagicMock()
        client.get_collection.side_effect = NotFoundError("missing")
        staging = MagicMock()
        staging.name = "studyloop-staging-owned"
        write_error = RuntimeError("primary write failure")
        staging.add.side_effect = write_error
        client.create_collection.return_value = staging
        client.delete_collection.side_effect = RuntimeError("cleanup failure")

        with patch.object(vectorstore, "chromadb_client", client), \
             patch.object(vectorstore, "_embed", AsyncMock(
                 return_value=_embedding_response(["content"])
             )), self.assertLogs(vectorstore.logger, level="ERROR") as logs:
            with self.assertRaises(RuntimeError) as raised:
                await vectorstore.deal_document("notes.md", "notes.md", ["content"])

        self.assertIs(raised.exception, write_error)
        self.assertTrue(any("cleanup failure" in line for line in logs.output))

    async def test_cancellation_waits_for_write_then_removes_staging(self):
        client = MagicMock()
        client.get_collection.side_effect = NotFoundError("missing")
        staging = MagicMock()
        staging.name = "studyloop-staging-owned"
        started = threading.Event()
        release = threading.Event()

        def slow_add(**_):
            started.set()
            release.wait(timeout=2)

        staging.add.side_effect = slow_add
        client.create_collection.return_value = staging

        with patch.object(vectorstore, "chromadb_client", client), \
             patch.object(vectorstore, "_embed", AsyncMock(
                 return_value=_embedding_response(["content"])
             )):
            task = asyncio.create_task(
                vectorstore.deal_document("notes.md", "notes.md", ["content"])
            )
            await asyncio.to_thread(started.wait, 2)
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        client.delete_collection.assert_called_once_with(name="studyloop-staging-owned")

    async def test_existing_document_is_rejected_without_mutation(self):
        client = MagicMock()
        client.get_collection.return_value = MagicMock(name="existing")
        embed = AsyncMock()

        with patch.object(vectorstore, "chromadb_client", client), \
             patch.object(vectorstore, "_embed", embed):
            with self.assertRaises(vectorstore.DocumentAlreadyExistsError):
                await vectorstore.deal_document("notes.md", "notes.md", ["new content"])

        embed.assert_not_awaited()
        client.create_collection.assert_not_called()
        client.delete_collection.assert_not_called()

    async def test_concurrent_publish_conflict_is_reported_as_duplicate(self):
        client = MagicMock()
        existing = MagicMock(name="published")
        client.get_collection.side_effect = [
            NotFoundError("missing before embedding"),
            NotFoundError("missing before publish"),
            existing,
        ]
        staging = MagicMock()
        staging.name = "studyloop-staging-owned"
        staging.modify.side_effect = RuntimeError("rename conflict")
        client.create_collection.return_value = staging

        with patch.object(vectorstore, "chromadb_client", client), \
             patch.object(vectorstore, "_embed", AsyncMock(
                 return_value=_embedding_response(["content"])
             )):
            with self.assertRaises(vectorstore.DocumentAlreadyExistsError):
                await vectorstore.deal_document("notes.md", "notes.md", ["content"])

        client.delete_collection.assert_called_once_with(name="studyloop-staging-owned")

    async def test_document_list_hides_unpublished_staging_collections(self):
        client = MagicMock()
        client.list_collections.return_value = [
            SimpleNamespace(
                name="studyloop-staging-orphan",
                metadata={"ingest_status": "indexing", "created_at": 0},
            ),
            SimpleNamespace(
                name="partial.md",
                metadata={"ingest_status": "indexing", "created_at": 0},
            ),
            SimpleNamespace(name="ready.md", metadata={"ingest_status": "indexed"}),
            SimpleNamespace(
                name="studyloop-staging-legitimate.md",
                metadata={"ingest_status": "indexed"},
            ),
            SimpleNamespace(name="legacy-empty.md", metadata=None, count=lambda: 0),
            SimpleNamespace(name="legacy.md", metadata=None, count=lambda: 1),
        ]

        with patch.object(vectorstore, "chromadb_client", client):
            collections = await vectorstore.get_all_document()

        self.assertEqual(
            [item.name for item in collections],
            ["ready.md", "studyloop-staging-legitimate.md", "legacy.md"],
        )

    async def test_janitor_never_deletes_fresh_or_active_staging(self):
        now = vectorstore.time.time()
        client = MagicMock()
        client.list_collections.return_value = [
            SimpleNamespace(
                name="active-staging",
                metadata={"ingest_status": "indexing", "created_at": 0},
            ),
            SimpleNamespace(
                name="fresh-staging",
                metadata={"ingest_status": "indexing", "created_at": int(now)},
            ),
            SimpleNamespace(
                name="stale-staging",
                metadata={"ingest_status": "indexing", "created_at": 0},
            ),
        ]
        vectorstore._active_staging_names.add("active-staging")

        with patch.object(vectorstore, "chromadb_client", client):
            collections = await vectorstore.get_all_document()

        self.assertEqual(collections, [])
        client.delete_collection.assert_called_once_with(name="stale-staging")

    async def test_real_chroma_migrates_legacy_ghost_and_publishes_staging(self):
        with tempfile.TemporaryDirectory(prefix="study-loop-real-chroma-") as tmp:
            client = chromadb.PersistentClient(tmp)
            client.create_collection("legacy-empty.md")
            legacy_full = client.create_collection("legacy-full.md")
            legacy_full.add(
                documents=["old valid content"],
                embeddings=[[1.0, 0.0]],
                ids=["legacy-full_chunk_0"],
            )

            with patch.object(vectorstore, "chromadb_client", client):
                visible = await vectorstore.get_all_document()
            self.assertEqual([item.name for item in visible], ["legacy-full.md"])
            with self.assertRaises(NotFoundError):
                client.get_collection("legacy-empty.md")

            client.create_collection("direct-empty.md")
            with patch.object(vectorstore, "chromadb_client", client), \
                 patch.object(vectorstore, "_embed", AsyncMock(
                     return_value=_embedding_response(["new valid content"])
                 )):
                count = await vectorstore.deal_document(
                    "direct-empty.md", "direct-empty.md", ["new valid content"]
                )
                visible = await vectorstore.get_all_document()

            self.assertEqual(count, 1)
            self.assertEqual(
                sorted(item.name for item in visible),
                ["direct-empty.md", "legacy-full.md"],
            )
            published = client.get_collection("direct-empty.md")
            self.assertEqual(published.count(), 1)
            self.assertEqual(published.metadata["ingest_status"], "indexed")
            self.assertFalse(any(
                (item.metadata or {}).get("ingest_status") == "indexing"
                for item in client.list_collections()
            ))

    async def test_delete_invalidates_cache_only_after_storage_success(self):
        client = MagicMock()
        vectorstore._bm25_cache["notes.md"] = {"cached": True}

        with patch.object(vectorstore, "chromadb_client", client):
            client.delete_collection.side_effect = RuntimeError("storage down")
            with self.assertRaises(RuntimeError):
                await vectorstore.delete_document("notes.md")
            self.assertIn("notes.md", vectorstore._bm25_cache)

            client.delete_collection.side_effect = None
            await vectorstore.delete_document("notes.md")
            self.assertNotIn("notes.md", vectorstore._bm25_cache)


if __name__ == "__main__":
    unittest.main()
