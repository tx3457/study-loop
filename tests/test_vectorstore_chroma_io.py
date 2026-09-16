"""Concurrency contracts around the embedded Chroma worker boundary."""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from chromadb.errors import NotFoundError

import services.vectorstore as vectorstore
from services.vectorstore import DEFAULT_DOCUMENT_OWNER


class _BlockingCatalogClient:
    def __init__(self, loop, *, late_error=None):
        self.loop = loop
        self.started = asyncio.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.late_error = late_error
        self.calls = 0
        self.active = 0
        self.peak_active = 0
        self._active_lock = threading.Lock()

    def list_collections(self):
        self.calls += 1
        self.loop.call_soon_threadsafe(self.started.set)
        with self._active_lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            if not self.release.wait(timeout=5):
                raise RuntimeError("test worker was not released")
            if self.late_error is not None:
                error, self.late_error = self.late_error, None
                raise error
            return []
        finally:
            with self._active_lock:
                self.active -= 1
            self.finished.set()


class _AckLossCollection:
    def __init__(self, owner, name, metadata):
        self._owner = owner
        self.name = name
        self.metadata = metadata

    def add(self, **_):
        return None

    def modify(self, *, name, metadata):
        # The server accepted the rename, but its acknowledgement was lost.
        # A competing owner is now authoritative at the canonical name.
        takeover = _AckLossCollection(self._owner, name, metadata)
        self._owner.collections.pop(self.name, None)
        self._owner.collections[name] = takeover
        self.name = name
        raise RuntimeError("rename acknowledgement lost")


class _AckLossClient:
    def __init__(self):
        self.collections = {}
        self.deleted_names = []

    def get_collection(self, *, name):
        try:
            return self.collections[name]
        except KeyError as error:
            raise NotFoundError(name) from error

    def create_collection(self, *, name, metadata):
        collection = _AckLossCollection(self, name, metadata)
        self.collections[name] = collection
        return collection

    def delete_collection(self, *, name):
        self.deleted_names.append(name)
        self.collections.pop(name, None)


class TestChromaIOBoundary(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Other application-lifespan tests share this module singleton.  Make
        # each boundary assertion start from an accepting, lazy worker state.
        vectorstore.start_vectorstore_io()

    async def asyncTearDown(self):
        # Every lifecycle test leaves the process-global worker ready for the
        # next isolated loop, even if an assertion fails midway through drain.
        await vectorstore.shutdown_vectorstore_io()

    async def test_blocked_catalog_read_keeps_loop_alive_without_default_executor(self):
        loop = asyncio.get_running_loop()
        client = _BlockingCatalogClient(loop)
        heartbeat = asyncio.Event()

        async def tick_after_worker_starts():
            await client.started.wait()
            await asyncio.sleep(0)
            heartbeat.set()

        with patch.object(vectorstore, "chromadb_client", client), patch.object(
            vectorstore.asyncio,
            "to_thread",
            side_effect=AssertionError("Chroma must not use the default executor"),
        ):
            task = asyncio.create_task(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER))
            ticker = asyncio.create_task(tick_after_worker_starts())
            await asyncio.wait_for(heartbeat.wait(), timeout=2)
            client.release.set()
            self.assertEqual(await asyncio.wait_for(task, timeout=2), [])
            await ticker

    async def test_queue_rejects_excess_pending_work_without_default_executor_growth(self):
        loop = asyncio.get_running_loop()
        client = _BlockingCatalogClient(loop)
        task_count = vectorstore.CHROMA_IO_MAX_PENDING + 1

        with patch.object(vectorstore, "chromadb_client", client), patch.object(
            vectorstore,
            "CHROMA_IO_QUEUE_WAIT_SECONDS",
            0.05,
        ), patch.object(
            vectorstore.asyncio,
            "to_thread",
            side_effect=AssertionError("Chroma must not use the default executor"),
        ):
            tasks = [
                asyncio.create_task(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER))
                for _ in range(task_count)
            ]
            try:
                await asyncio.wait_for(client.started.wait(), timeout=2)
                done, _ = await asyncio.wait_for(
                    asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION),
                    timeout=2,
                )
                failures = [
                    task.exception()
                    for task in done
                    if not task.cancelled() and task.exception() is not None
                ]
                self.assertTrue(
                    any(
                        isinstance(error, vectorstore.ChromaIOWaitTimeoutError)
                        for error in failures
                    )
                )
            finally:
                client.release.set()
                await asyncio.gather(*tasks, return_exceptions=True)

        # One worker may run, but saturation must not cause the normal pool to
        # accumulate hidden Chroma work.
        self.assertGreaterEqual(client.calls, 1)
        self.assertLessEqual(client.calls, vectorstore.CHROMA_IO_MAX_PENDING)
        self.assertEqual(client.peak_active, 1)

    async def test_repeated_cancellation_has_one_deadline_and_late_error_is_consumed(self):
        loop = asyncio.get_running_loop()
        client = _BlockingCatalogClient(loop, late_error=RuntimeError("late catalog failure"))
        unhandled = []
        old_handler = loop.get_exception_handler()

        def record_unhandled(current_loop, context):
            unhandled.append(context)
            if old_handler is not None:
                old_handler(current_loop, context)

        loop.set_exception_handler(record_unhandled)
        try:
            with patch.object(vectorstore, "chromadb_client", client), patch.object(
                vectorstore,
                "CHROMA_IO_CANCEL_DRAIN_SECONDS",
                0.05,
            ), patch.object(
                vectorstore.asyncio,
                "to_thread",
                side_effect=AssertionError("Chroma must not use the default executor"),
            ):
                task = asyncio.create_task(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER))
                await asyncio.wait_for(client.started.wait(), timeout=2)
                task.cancel()
                task.cancel()  # A second client disconnect must not extend the drain deadline.
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)

                # The late synchronous failure occurs after the caller has gone
                # away.  It must be consumed by the worker hand-off, not leak as
                # an unhandled task exception during loop shutdown.
                client.release.set()
                follow_up = await asyncio.wait_for(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER), timeout=2)
                self.assertEqual(follow_up, [])
        finally:
            loop.set_exception_handler(old_handler)

        self.assertFalse(unhandled)

    async def test_publish_ack_loss_never_deletes_canonical_takeover(self):
        client = _AckLossClient()
        embeddings = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])

        with patch.object(vectorstore, "chromadb_client", client), patch.object(
            vectorstore,
            "_embed",
            AsyncMock(return_value=embeddings),
        ):
            with self.assertRaises(vectorstore.DocumentAlreadyExistsError):
                await vectorstore.deal_document("notes.md", "notes.md", ["content"], owner_id=DEFAULT_DOCUMENT_OWNER)

        self.assertIn("notes.md", client.collections)
        self.assertNotIn("notes.md", client.deleted_names)

    async def test_cancelled_waiter_never_runs_after_the_active_operation_settles(self):
        loop = asyncio.get_running_loop()
        client = _BlockingCatalogClient(loop)

        with patch.object(vectorstore, "chromadb_client", client), patch.object(
            vectorstore,
            "CHROMA_IO_CANCEL_DRAIN_SECONDS",
            0.05,
        ):
            active = asyncio.create_task(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER))
            await asyncio.wait_for(client.started.wait(), timeout=2)
            waiting = asyncio.create_task(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER))
            await asyncio.sleep(0)
            waiting.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(waiting, timeout=1)

            client.release.set()
            self.assertEqual(await asyncio.wait_for(active, timeout=2), [])
            # A later public request creates the only second catalog call.  If
            # the cancelled waiter had been submitted early, this would be 3.
            self.assertEqual(await asyncio.wait_for(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER), timeout=2), [])

        self.assertEqual(client.calls, 2)

    async def test_hung_read_has_a_public_deadline_then_later_settles(self):
        loop = asyncio.get_running_loop()
        client = _BlockingCatalogClient(loop)

        with patch.object(vectorstore, "chromadb_client", client), patch.object(
            vectorstore,
            "CHROMA_IO_OPERATION_TIMEOUT_SECONDS",
            0.05,
        ):
            task = asyncio.create_task(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER))
            await asyncio.wait_for(client.started.wait(), timeout=2)
            with self.assertRaises(vectorstore.ChromaIOOperationTimeoutError):
                await asyncio.wait_for(task, timeout=1)
            client.release.set()
            self.assertEqual(await asyncio.wait_for(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER), timeout=2), [])

    async def test_readiness_probe_uses_the_same_single_worker_channel(self):
        loop = asyncio.get_running_loop()
        client = _BlockingCatalogClient(loop)
        probe_finished = asyncio.Event()
        probe_error = []

        def count_collections():
            return 0

        client.count_collections = count_collections

        def probe_in_thread():
            try:
                vectorstore.probe_vectorstore_readiness()
            except BaseException as error:
                probe_error.append(error)
            finally:
                loop.call_soon_threadsafe(probe_finished.set)

        with patch.object(vectorstore, "chromadb_client", client):
            active = asyncio.create_task(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER))
            await asyncio.wait_for(client.started.wait(), timeout=2)
            threading.Thread(target=probe_in_thread, daemon=True).start()
            client.release.set()
            self.assertEqual(await asyncio.wait_for(active, timeout=2), [])
            await asyncio.wait_for(probe_finished.wait(), timeout=2)

        self.assertEqual(probe_error, [])

    async def test_shutdown_rejects_new_work_drains_short_write_and_reopens_lazily(self):
        loop = asyncio.get_running_loop()
        client = MagicMock()
        staging = MagicMock()
        staging.name = "owned-staging"
        staging.metadata = {
            "source_document_id": "notes.md",
            "ingest_status": "indexing",
        }
        add_started = asyncio.Event()
        release_add = threading.Event()

        def slow_add(**_):
            loop.call_soon_threadsafe(add_started.set)
            if not release_add.wait(timeout=5):
                raise RuntimeError("test upload was not released")

        staging.add.side_effect = slow_add
        client.create_collection.return_value = staging
        client.get_collection.side_effect = [
            NotFoundError("missing"),
            NotFoundError("missing"),
        ]
        embeddings = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])

        with patch.object(vectorstore, "chromadb_client", client), patch.object(
            vectorstore,
            "_embed",
            AsyncMock(return_value=embeddings),
        ):
            write = asyncio.create_task(
                vectorstore.deal_document("notes.md", "notes.md", ["content"], owner_id=DEFAULT_DOCUMENT_OWNER)
            )
            await asyncio.wait_for(add_started.wait(), timeout=2)
            shutdown = asyncio.create_task(vectorstore.shutdown_vectorstore_io())
            await asyncio.sleep(0)
            with self.assertRaises(vectorstore.ChromaIOShuttingDownError):
                await asyncio.wait_for(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER), timeout=1)

            release_add.set()
            self.assertEqual(await asyncio.wait_for(write, timeout=2), 1)
            await asyncio.wait_for(shutdown, timeout=2)

        staging.modify.assert_called_once()
        vectorstore.start_vectorstore_io()
        next_client = _BlockingCatalogClient(loop)
        next_client.release.set()
        with patch.object(vectorstore, "chromadb_client", next_client):
            self.assertEqual(await asyncio.wait_for(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER), timeout=2), [])

    async def test_shutdown_of_hung_worker_returns_bounded_and_stays_fail_closed(self):
        loop = asyncio.get_running_loop()
        client = _BlockingCatalogClient(loop)

        with patch.object(vectorstore, "chromadb_client", client), patch.object(
            vectorstore,
            "CHROMA_IO_SHUTDOWN_DRAIN_SECONDS",
            0.05,
        ):
            active = asyncio.create_task(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER))
            await asyncio.wait_for(client.started.wait(), timeout=2)
            await asyncio.wait_for(vectorstore.shutdown_vectorstore_io(), timeout=1)
            with self.assertRaises(vectorstore.ChromaIOShuttingDownError):
                await asyncio.wait_for(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER), timeout=1)

            client.release.set()
            self.assertEqual(await asyncio.wait_for(active, timeout=2), [])
            # A later shutdown owns the already-finished worker's sentinel and
            # returns the channel to lazy-reopenable state.
            await asyncio.wait_for(vectorstore.shutdown_vectorstore_io(), timeout=1)

    async def test_pre_shutdown_waiters_cannot_cross_lifecycle_generation(self):
        """A waiter captured before shutdown must never submit in the next run."""

        async def assert_stale_waiter_is_rejected(waiting_for: str):
            loop = asyncio.get_running_loop()
            client = _BlockingCatalogClient(loop)
            waiter_reached = asyncio.Event()
            resume_waiter = asyncio.Event()
            original_slot = vectorstore._acquire_chroma_io_slot
            original_active = vectorstore._acquire_chroma_io_active
            slot_calls = 0
            active_calls = 0
            slots = threading.BoundedSemaphore(1 if waiting_for == "pending_slot" else 2)

            async def gate_slot(deadline):
                nonlocal slot_calls
                slot_calls += 1
                await original_slot(deadline)
                if waiting_for == "pending_slot" and slot_calls == 2:
                    waiter_reached.set()
                    await resume_waiter.wait()

            async def gate_active(deadline):
                nonlocal active_calls
                active_calls += 1
                await original_active(deadline)
                if waiting_for == "active_worker" and active_calls == 2:
                    # The old request really acquired the active gate after
                    # waiting behind the first call.  Give shutdown a clean
                    # channel, then reacquire only in the new generation.
                    vectorstore._chroma_io_active.release()
                    waiter_reached.set()
                    await resume_waiter.wait()
                    await original_active(deadline)

            with patch.object(vectorstore, "chromadb_client", client), patch.object(
                vectorstore,
                "_chroma_io_slots",
                slots,
            ), patch.object(
                vectorstore,
                "_acquire_chroma_io_slot",
                gate_slot,
            ), patch.object(
                vectorstore,
                "_acquire_chroma_io_active",
                gate_active,
            ):
                active = asyncio.create_task(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER))
                await asyncio.wait_for(client.started.wait(), timeout=2)
                old_waiter = asyncio.create_task(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER))
                shutdown = asyncio.create_task(vectorstore.shutdown_vectorstore_io())
                await asyncio.sleep(0)
                client.release.set()
                self.assertEqual(await asyncio.wait_for(active, timeout=2), [])
                await asyncio.wait_for(waiter_reached.wait(), timeout=2)
                await asyncio.wait_for(shutdown, timeout=2)
                vectorstore.start_vectorstore_io()

                resume_waiter.set()
                with self.assertRaises(vectorstore.ChromaIOShuttingDownError):
                    await asyncio.wait_for(old_waiter, timeout=2)
                self.assertEqual(await asyncio.wait_for(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER), timeout=2), [])

            self.assertEqual(client.calls, 2)

        for waiting_for in ("active_worker", "pending_slot"):
            with self.subTest(waiting_for=waiting_for):
                await assert_stale_waiter_is_rejected(waiting_for)

    def test_late_completion_after_request_loop_closes_releases_the_channel(self):
        client_holder = {}

        async def cancel_in_first_loop():
            loop = asyncio.get_running_loop()
            client = _BlockingCatalogClient(loop)
            client_holder["client"] = client
            with patch.object(vectorstore, "chromadb_client", client), patch.object(
                vectorstore,
                "CHROMA_IO_CANCEL_DRAIN_SECONDS",
                0.05,
            ):
                task = asyncio.create_task(vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER))
                await asyncio.wait_for(client.started.wait(), timeout=2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)

        asyncio.run(cancel_in_first_loop())
        client = client_holder["client"]
        client.release.set()
        self.assertTrue(client.finished.wait(timeout=2))

        async def read_in_second_loop():
            loop = asyncio.get_running_loop()
            next_client = _BlockingCatalogClient(loop)
            next_client.release.set()
            with patch.object(vectorstore, "chromadb_client", next_client):
                return await vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER)

        self.assertEqual(asyncio.run(read_in_second_loop()), [])

    def test_completed_work_can_be_awaited_from_two_fresh_event_loops(self):
        """The process-wide worker must not retain a closed request loop."""

        async def read_once():
            loop = asyncio.get_running_loop()
            client = _BlockingCatalogClient(loop)
            client.release.set()
            with patch.object(vectorstore, "chromadb_client", client):
                return await vectorstore.get_all_document(owner_id=DEFAULT_DOCUMENT_OWNER)

        self.assertEqual(asyncio.run(read_once()), [])
        self.assertEqual(asyncio.run(read_once()), [])
