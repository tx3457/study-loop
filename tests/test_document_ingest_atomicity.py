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
        staging = MagicMock()
        staging.name = "studyloop-staging-owned"
        staging.metadata = {
            "source_document_id": "notes.md",
            "ingest_status": "indexing",
        }
        client.get_collection.side_effect = [
            NotFoundError("missing"),
            NotFoundError("missing"),
            staging,
        ]
        write_error = RuntimeError("index write failed")
        staging.add.side_effect = write_error
        client.create_collection.return_value = staging
        vectorstore._bm25_cache["notes.md"] = {"existing": True, "cached_chars": 0}

        with patch.object(vectorstore, "chromadb_client", client), \
             patch.object(vectorstore, "_embed", AsyncMock(
                 return_value=_embedding_response(["content"])
             )):
            with self.assertRaises(RuntimeError) as raised:
                await vectorstore.deal_document("notes.md", "notes.md", ["content"])

        self.assertIs(raised.exception, write_error)
        client.delete_collection.assert_called_once()
        self.assertNotEqual(client.delete_collection.call_args.kwargs["name"], "notes.md")
        staging.modify.assert_not_called()
        self.assertIn("notes.md", vectorstore._bm25_cache)

    async def test_cleanup_failure_does_not_mask_index_failure(self):
        client = MagicMock()
        staging = MagicMock()
        staging.name = "studyloop-staging-owned"
        staging.metadata = {
            "source_document_id": "notes.md",
            "ingest_status": "indexing",
        }
        client.get_collection.side_effect = [
            NotFoundError("missing"),
            NotFoundError("missing"),
            staging,
        ]
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
        staging = MagicMock()
        staging.name = "studyloop-staging-owned"
        staging.metadata = {
            "source_document_id": "notes.md",
            "ingest_status": "indexing",
        }
        client.get_collection.side_effect = [
            NotFoundError("missing"),
            NotFoundError("missing"),
            staging,
        ]
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

        client.delete_collection.assert_called_once()
        self.assertNotEqual(client.delete_collection.call_args.kwargs["name"], "notes.md")

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
        staging = MagicMock()
        staging.name = "studyloop-staging-owned"
        staging.metadata = {
            "source_document_id": "notes.md",
            "ingest_status": "indexing",
        }
        client.get_collection.side_effect = [
            NotFoundError("missing before embedding"),
            NotFoundError("missing before publish"),
            existing,
            staging,
        ]
        staging.modify.side_effect = RuntimeError("rename conflict")
        client.create_collection.return_value = staging

        with patch.object(vectorstore, "chromadb_client", client), \
             patch.object(vectorstore, "_embed", AsyncMock(
                 return_value=_embedding_response(["content"])
             )):
            with self.assertRaises(vectorstore.DocumentAlreadyExistsError):
                await vectorstore.deal_document("notes.md", "notes.md", ["content"])

        client.delete_collection.assert_called_once()
        self.assertNotEqual(client.delete_collection.call_args.kwargs["name"], "notes.md")

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

    async def test_delete_is_retryable_from_a_hidden_tombstone(self):
        client = MagicMock()
        collection = MagicMock()
        collection.name = "notes.md"
        collection.metadata = {
            "source_document_id": "notes.md",
            "source_filename": "notes.md",
            "ingest_status": "indexed",
        }
        collection.get.return_value = {"ids": ["notes.md_chunk_0"]}
        collection.count.return_value = 0

        def update_metadata(*, metadata):
            collection.metadata = metadata

        collection.modify.side_effect = update_metadata
        client.get_collection.return_value = collection
        vectorstore._bm25_cache["notes.md"] = {"cached": True, "cached_chars": 0}

        with patch.object(vectorstore, "chromadb_client", client):
            collection.delete.side_effect = RuntimeError("storage down")
            with self.assertRaises(RuntimeError):
                await vectorstore.delete_document("notes.md")
            self.assertNotIn("notes.md", vectorstore._bm25_cache)
            self.assertEqual(collection.metadata["ingest_status"], "deleting")

            collection.delete.side_effect = None
            client.list_collections.return_value = [collection]
            visible = await vectorstore.get_all_document()
            replay = await vectorstore.delete_document("notes.md")

        self.assertEqual(visible, [])
        self.assertEqual(replay, "material_deleted")
        self.assertEqual(collection.metadata["ingest_status"], "deleted")
        collection.delete.assert_called_with(ids=["notes.md_chunk_0"])

    async def test_delete_missing_document_creates_a_permanent_tombstone(self):
        client = MagicMock()
        tombstone = MagicMock()
        tombstone.name = "legacy.md"
        tombstone.metadata = {
            "source_document_id": "legacy.md",
            "source_filename": "legacy.md",
            "ingest_status": "deleted",
        }
        tombstone.count.return_value = 0
        lookups = 0

        def get_collection(*, name):
            nonlocal lookups
            lookups += 1
            if lookups == 1:
                raise NotFoundError("missing")
            return tombstone

        client.get_collection.side_effect = get_collection

        with patch.object(vectorstore, "chromadb_client", client):
            status = await vectorstore.delete_document("legacy.md")
            replay = await vectorstore.delete_document("legacy.md")
            with self.assertRaises(vectorstore.DocumentAlreadyExistsError):
                await vectorstore.deal_document(
                    "legacy.md",
                    "legacy.md",
                    ["不能继承旧学习历史"],
                )

        self.assertEqual(status, "material_deleted")
        self.assertEqual(replay, "material_deleted")
        client.create_collection.assert_called_once()
        metadata = client.create_collection.call_args.kwargs["metadata"]
        self.assertEqual(metadata["ingest_status"], "deleted")
        self.assertEqual(metadata["source_document_id"], "legacy.md")

    async def test_concurrent_cold_bm25_misses_build_once_on_single_worker(self):
        client = MagicMock()
        collection = MagicMock()
        collection.name = "notes.md"
        collection.metadata = {
            "source_document_id": "notes.md",
            "source_filename": "notes.md",
            "ingest_status": "indexed",
        }
        loop = asyncio.get_running_loop()
        build_started = asyncio.Event()
        release_build = threading.Event()

        def get_chunks(*, include):
            if include == ["documents"]:
                return {"ids": ["notes.md_chunk_0"], "documents": ["secret text"]}
            return {"ids": ["notes.md_chunk_0"]}

        original_build = vectorstore.build_bm25_index

        def gate_bm25_build(documents):
            loop.call_soon_threadsafe(build_started.set)
            if not release_build.wait(timeout=5):
                raise RuntimeError("test BM25 build was not released")
            return original_build(documents)

        collection.get.side_effect = get_chunks
        client.get_collection.return_value = collection

        with patch.object(vectorstore, "chromadb_client", client), patch.object(
            vectorstore,
            "build_bm25_index",
            gate_bm25_build,
        ):
            first_task = asyncio.create_task(
                vectorstore._get_bm25_index(collection, "notes.md")
            )
            await asyncio.wait_for(build_started.wait(), timeout=2)
            second_task = asyncio.create_task(
                vectorstore._get_bm25_index(collection, "notes.md")
            )
            try:
                await asyncio.sleep(0)
            finally:
                release_build.set()
            first, second = await asyncio.gather(first_task, second_task)

        self.assertIs(first, second)
        self.assertEqual(collection.get.call_count, 1)

    async def test_cancelled_late_bm25_build_never_publishes_cache(self):
        client = MagicMock()
        collection = MagicMock()
        collection.name = "notes.md"
        collection.metadata = {
            "source_document_id": "notes.md",
            "source_filename": "notes.md",
            "ingest_status": "indexed",
        }

        def get_chunks(*, include):
            if include == ["documents"]:
                return {"ids": ["notes.md_chunk_0"], "documents": ["secret text"]}
            return {"ids": ["notes.md_chunk_0"]}

        collection.get.side_effect = get_chunks
        client.get_collection.return_value = collection

        loop = asyncio.get_running_loop()
        build_started = asyncio.Event()
        release_build = threading.Event()
        original_build = vectorstore.build_bm25_index

        def gate_bm25_build(documents):
            loop.call_soon_threadsafe(build_started.set)
            if not release_build.wait(timeout=5):
                raise RuntimeError("test BM25 build was not released")
            return original_build(documents)

        with patch.object(vectorstore, "chromadb_client", client), patch.object(
            vectorstore,
            "build_bm25_index",
            gate_bm25_build,
        ):
            build_task = asyncio.create_task(
                vectorstore._get_bm25_index(collection, "notes.md")
            )
            await asyncio.wait_for(build_started.wait(), timeout=2)
            try:
                build_task.cancel()
            finally:
                release_build.set()
            with self.assertRaises(asyncio.CancelledError):
                await build_task
            await asyncio.wait_for(vectorstore.get_all_document(), timeout=2)

        self.assertNotIn("notes.md", vectorstore._bm25_cache)

    async def test_timed_out_bm25_build_never_publishes_cache(self):
        client = MagicMock()
        collection = MagicMock()
        collection.name = "notes.md"
        collection.metadata = {
            "source_document_id": "notes.md",
            "source_filename": "notes.md",
            "ingest_status": "indexed",
        }
        loop = asyncio.get_running_loop()
        build_started = asyncio.Event()
        build_finished = asyncio.Event()
        release_build = threading.Event()
        original_build = vectorstore.build_bm25_index

        def get_chunks(*, include):
            if include == ["documents"]:
                return {"ids": ["notes.md_chunk_0"], "documents": ["secret text"]}
            return {"ids": ["notes.md_chunk_0"]}

        def gate_bm25_build(documents):
            loop.call_soon_threadsafe(build_started.set)
            if not release_build.wait(timeout=5):
                raise RuntimeError("test BM25 build was not released")
            loop.call_soon_threadsafe(build_finished.set)
            return original_build(documents)

        collection.get.side_effect = get_chunks
        client.get_collection.return_value = collection

        with patch.object(vectorstore, "chromadb_client", client), patch.object(
            vectorstore,
            "build_bm25_index",
            gate_bm25_build,
        ), patch.object(vectorstore, "CHROMA_IO_OPERATION_TIMEOUT_SECONDS", 0.05):
            build_task = asyncio.create_task(
                vectorstore._get_bm25_index(collection, "notes.md")
            )
            await asyncio.wait_for(build_started.wait(), timeout=2)
            with self.assertRaises(vectorstore.ChromaIOOperationTimeoutError):
                await asyncio.wait_for(build_task, timeout=1)
            release_build.set()
            await asyncio.wait_for(build_finished.wait(), timeout=2)
            await asyncio.wait_for(vectorstore.get_all_document(), timeout=2)

        self.assertNotIn("notes.md", vectorstore._bm25_cache)

    async def test_delete_queued_after_bm25_build_invalidates_completed_cache(self):
        client = MagicMock()
        collection = MagicMock()
        collection.name = "notes.md"
        collection.metadata = {
            "source_document_id": "notes.md",
            "source_filename": "notes.md",
            "ingest_status": "indexed",
        }
        loop = asyncio.get_running_loop()
        build_started = asyncio.Event()
        release_build = threading.Event()
        original_build = vectorstore.build_bm25_index

        def get_chunks(*, include):
            if include == ["documents"]:
                return {"ids": ["notes.md_chunk_0"], "documents": ["secret text"]}
            return {"ids": ["notes.md_chunk_0"]}

        def gate_bm25_build(documents):
            loop.call_soon_threadsafe(build_started.set)
            if not release_build.wait(timeout=5):
                raise RuntimeError("test BM25 build was not released")
            return original_build(documents)

        def update_metadata(*, metadata):
            collection.metadata = metadata

        collection.get.side_effect = get_chunks
        collection.modify.side_effect = update_metadata
        collection.count.return_value = 0
        client.get_collection.return_value = collection

        with patch.object(vectorstore, "chromadb_client", client), patch.object(
            vectorstore,
            "build_bm25_index",
            gate_bm25_build,
        ):
            build_task = asyncio.create_task(
                vectorstore._get_bm25_index(collection, "notes.md")
            )
            await asyncio.wait_for(build_started.wait(), timeout=2)
            delete_task = asyncio.create_task(vectorstore.delete_document("notes.md"))
            release_build.set()
            await build_task
            self.assertEqual(await delete_task, "material_deleted")

        self.assertNotIn("notes.md", vectorstore._bm25_cache)


if __name__ == "__main__":
    unittest.main()
