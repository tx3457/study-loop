"""PostgreSQL runtime boundaries for learner-memory store I/O.

These tests use controlled blocking fakes rather than unreachable network
addresses.  That keeps the suite deterministic while exercising the same
event-loop and cancellation boundary as a stalled PostgreSQL operation.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch


async def _wait_for_thread_event(event: threading.Event, timeout: float = 1) -> None:
    async def poll() -> None:
        while not event.is_set():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=timeout)


async def _wait_until(predicate, timeout: float = 1) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=timeout)


class _BlockingReadStore:
    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.failure = failure
        self.worker_thread_id: int | None = None

    def get(self, namespace, key):
        self.worker_thread_id = threading.get_ident()
        self.started.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("test did not release the fake PostgreSQL read")
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(value={"source": "postgres"})


class _CountingBlockingStore:
    def __init__(self) -> None:
        self.first_started = threading.Event()
        self.release = threading.Event()
        self._call_lock = threading.Lock()
        self.calls = 0

    def get(self, namespace, key):
        with self._call_lock:
            self.calls += 1
            if self.calls == 1:
                self.first_started.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("test did not release the fake PostgreSQL reads")
        return SimpleNamespace(value={"key": key})


class TestPostgresMemoryRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_at_gate_acquire_delivery_releases_late_lock(self):
        import services.memory as memory

        gate = asyncio.Lock()
        original_acquire = gate.acquire
        original_release = gate.release
        release_calls = 0
        waiter: asyncio.Task | None = None

        async def acquire_then_cancel_waiter() -> bool:
            acquired = await original_acquire()
            # Reproduce the same-tick race: Lock.acquire has set _locked and
            # this child is about to publish True, but its parent has not yet
            # observed that result when cancellation arrives.
            waiter.cancel("client disconnected")
            return acquired

        def counted_release() -> None:
            nonlocal release_calls
            release_calls += 1
            original_release()

        with (
            patch.object(gate, "acquire", new=acquire_then_cancel_waiter),
            patch.object(gate, "release", new=counted_release),
        ):
            waiter = asyncio.create_task(memory._acquire_store_io_gate(gate, timeout_seconds=1))
            with self.assertRaises(asyncio.CancelledError) as raised:
                await waiter

        self.assertEqual(raised.exception.args, ("client disconnected",))
        await asyncio.sleep(0)
        self.assertFalse(gate.locked())
        self.assertEqual(release_calls, 1)

        self.assertTrue(await asyncio.wait_for(gate.acquire(), timeout=0.1))
        gate.release()

    async def test_gate_timeout_never_releases_the_current_owner(self):
        import services.memory as memory

        gate = asyncio.Lock()
        await gate.acquire()

        with self.assertRaises(asyncio.TimeoutError):
            await memory._acquire_store_io_gate(gate, timeout_seconds=0.01)

        # Let cancellation and the late-result callback both settle. The gate
        # still belongs to the original owner and must be released exactly by it.
        await asyncio.sleep(0)
        self.assertTrue(gate.locked())
        gate.release()
        await asyncio.sleep(0)
        self.assertFalse(gate.locked())

        self.assertTrue(await asyncio.wait_for(gate.acquire(), timeout=0.1))
        gate.release()

    async def test_slow_postgres_store_read_does_not_block_event_loop(self):
        import services.memory as memory

        fake_store = _BlockingReadStore()
        loop_thread_id = threading.get_ident()
        watchdog = threading.Timer(1, fake_store.release.set)
        watchdog.start()
        try:
            with (
                patch.object(memory, "DATABASE_URL", "postgresql://memory-test"),
                patch.object(memory, "store", fake_store),
            ):
                read = asyncio.create_task(memory.read_bank_state("runtime-user", "mastery"))
                await _wait_for_thread_event(fake_store.started, timeout=0.25)

                # A stalled synchronous Store.get must not prevent unrelated
                # timers/callbacks from running on the request event loop.
                heartbeat_started = time.monotonic()
                await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.1)
                self.assertLess(time.monotonic() - heartbeat_started, 0.1)
                self.assertNotEqual(fake_store.worker_thread_id, loop_thread_id)

                fake_store.release.set()
                self.assertEqual(
                    await asyncio.wait_for(read, timeout=0.5),
                    {"source": "postgres"},
                )
        finally:
            fake_store.release.set()
            watchdog.cancel()

    async def test_concurrent_store_reads_queue_before_entering_executor(self):
        import services.memory as memory

        fake_store = _CountingBlockingStore()
        watchdog = threading.Timer(1, fake_store.release.set)
        watchdog.start()
        reads: list[asyncio.Task] = []
        try:
            with (
                patch.object(memory, "DATABASE_URL", "postgresql://memory-test"),
                patch.object(memory, "store", fake_store),
            ):
                reads = [
                    asyncio.create_task(
                        memory.read_bank_state(f"concurrent-user-{index}", "mastery")
                    )
                    for index in range(8)
                ]
                await _wait_for_thread_event(fake_store.first_started, timeout=0.25)
                await asyncio.sleep(0.03)

                # The gate belongs before asyncio.to_thread.  Queuing inside
                # PostgresStore would consume all default-executor threads.
                self.assertEqual(fake_store.calls, 1)

                fake_store.release.set()
                results = await asyncio.wait_for(
                    asyncio.gather(*reads),
                    timeout=1,
                )
                self.assertEqual(len(results), 8)
                self.assertEqual(fake_store.calls, 8)
        finally:
            fake_store.release.set()
            watchdog.cancel()
            for read in reads:
                if not read.done():
                    read.cancel()
            if reads:
                await asyncio.gather(*reads, return_exceptions=True)

    async def test_cancelled_postgres_io_returns_within_budget_and_is_reaped(self):
        import services.memory as memory

        fake_store = _BlockingReadStore()
        original_background = set(memory._BACKGROUND_STORE_IO)
        watchdog = threading.Timer(1, fake_store.release.set)
        watchdog.start()
        try:
            with (
                patch.object(memory, "DATABASE_URL", "postgresql://memory-test"),
                patch.object(memory, "store", fake_store),
                patch.dict(
                    os.environ,
                    {"MEMORY_STORE_PG_CANCEL_DRAIN_TIMEOUT_SECONDS": "0.01"},
                ),
            ):
                read = asyncio.create_task(memory.read_bank_state("cancelled-user", "mastery"))
                await _wait_for_thread_event(fake_store.started, timeout=0.25)

                started = time.monotonic()
                read.cancel("client disconnected")
                with self.assertRaises(asyncio.CancelledError) as raised:
                    await asyncio.wait_for(read, timeout=0.25)

                self.assertLess(time.monotonic() - started, 0.25)
                self.assertEqual(raised.exception.args, ("client disconnected",))
                self.assertEqual(
                    len(set(memory._BACKGROUND_STORE_IO) - original_background),
                    1,
                )

                fake_store.release.set()
                await _wait_until(
                    lambda: not (set(memory._BACKGROUND_STORE_IO) - original_background),
                    timeout=0.5,
                )
        finally:
            fake_store.release.set()
            watchdog.cancel()

    async def test_cancelled_running_worker_keeps_gate_until_background_completion(self):
        import services.memory as memory

        fake_store = _CountingBlockingStore()
        original_background = set(memory._BACKGROUND_STORE_IO)
        watchdog = threading.Timer(1, fake_store.release.set)
        watchdog.start()
        first: asyncio.Task | None = None
        second: asyncio.Task | None = None
        try:
            with (
                patch.object(memory, "DATABASE_URL", "postgresql://memory-test"),
                patch.object(memory, "store", fake_store),
                patch.dict(
                    os.environ,
                    {"MEMORY_STORE_PG_CANCEL_DRAIN_TIMEOUT_SECONDS": "0.01"},
                ),
            ):
                first = asyncio.create_task(memory.read_bank_state("first-user", "mastery"))
                await _wait_for_thread_event(fake_store.first_started, timeout=0.25)
                first.cancel("client disconnected")
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(first, timeout=0.25)

                second = asyncio.create_task(memory.read_bank_state("second-user", "mastery"))
                await asyncio.sleep(0.03)
                self.assertEqual(
                    fake_store.calls,
                    1,
                    "a cancelled request released the gate while its worker was alive",
                )

                fake_store.release.set()
                self.assertEqual(
                    await asyncio.wait_for(second, timeout=0.5),
                    {"key": "current"},
                )
                await _wait_until(
                    lambda: not (set(memory._BACKGROUND_STORE_IO) - original_background),
                    timeout=0.5,
                )
                self.assertEqual(fake_store.calls, 2)
        finally:
            fake_store.release.set()
            watchdog.cancel()
            for task in (first, second):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (first, second) if task is not None),
                return_exceptions=True,
            )

    async def test_store_gate_wait_has_a_finite_budget(self):
        import services.memory as memory

        fake_store = _CountingBlockingStore()
        watchdog = threading.Timer(1, fake_store.release.set)
        watchdog.start()
        first: asyncio.Task | None = None
        try:
            with (
                patch.object(memory, "DATABASE_URL", "postgresql://memory-test"),
                patch.object(memory, "store", fake_store),
                patch.dict(
                    os.environ,
                    {"MEMORY_STORE_PG_IO_WAIT_TIMEOUT_SECONDS": "0.01"},
                ),
            ):
                first = asyncio.create_task(memory.read_bank_state("first-user", "mastery"))
                await _wait_for_thread_event(fake_store.first_started, timeout=0.25)

                with self.assertRaises(memory.PostgresStoreIOWaitTimeoutError):
                    await asyncio.wait_for(
                        memory.read_bank_state("queued-user", "mastery"),
                        timeout=0.25,
                    )
                self.assertEqual(fake_store.calls, 1)

                fake_store.release.set()
                await asyncio.wait_for(first, timeout=0.5)
        finally:
            fake_store.release.set()
            watchdog.cancel()
            if first is not None and not first.done():
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)

    async def test_repeated_cancellation_does_not_reset_the_drain_deadline(self):
        import services.memory as memory

        fake_store = _BlockingReadStore()
        original_background = set(memory._BACKGROUND_STORE_IO)
        watchdog = threading.Timer(1, fake_store.release.set)
        watchdog.start()
        read: asyncio.Task | None = None
        repeated_cancels: asyncio.Task | None = None
        try:
            with (
                patch.object(memory, "DATABASE_URL", "postgresql://memory-test"),
                patch.object(memory, "store", fake_store),
                patch.dict(
                    os.environ,
                    {"MEMORY_STORE_PG_CANCEL_DRAIN_TIMEOUT_SECONDS": "0.03"},
                ),
            ):
                read = asyncio.create_task(
                    memory.read_bank_state("repeated-cancel-user", "mastery")
                )
                await _wait_for_thread_event(fake_store.started, timeout=0.25)

                async def keep_cancelling() -> None:
                    for _ in range(100):
                        await asyncio.sleep(0.002)
                        read.cancel("later cancellation")

                started = time.monotonic()
                read.cancel("original cancellation")
                repeated_cancels = asyncio.create_task(keep_cancelling())
                with self.assertRaises(asyncio.CancelledError) as raised:
                    await asyncio.wait_for(read, timeout=0.5)

                self.assertLess(time.monotonic() - started, 0.15)
                self.assertEqual(raised.exception.args, ("original cancellation",))
                self.assertEqual(
                    len(set(memory._BACKGROUND_STORE_IO) - original_background),
                    1,
                )

                fake_store.release.set()
                await _wait_until(
                    lambda: not (set(memory._BACKGROUND_STORE_IO) - original_background),
                    timeout=0.5,
                )
        finally:
            fake_store.release.set()
            watchdog.cancel()
            if repeated_cancels is not None:
                repeated_cancels.cancel()
            await asyncio.gather(
                *(task for task in (read, repeated_cancels) if task is not None),
                return_exceptions=True,
            )

    async def test_late_store_failure_does_not_replace_original_cancellation(self):
        import services.memory as memory

        fake_store = _BlockingReadStore(failure=RuntimeError("database disconnected"))
        original_background = set(memory._BACKGROUND_STORE_IO)
        watchdog = threading.Timer(1, fake_store.release.set)
        watchdog.start()
        try:
            with (
                patch.object(memory, "DATABASE_URL", "postgresql://memory-test"),
                patch.object(memory, "store", fake_store),
                patch.dict(
                    os.environ,
                    {"MEMORY_STORE_PG_CANCEL_DRAIN_TIMEOUT_SECONDS": "0.01"},
                ),
            ):
                read = asyncio.create_task(memory.read_bank_state("cancelled-user", "mastery"))
                await _wait_for_thread_event(fake_store.started, timeout=0.25)
                read.cancel("client disconnected")

                with self.assertRaises(asyncio.CancelledError) as raised:
                    await asyncio.wait_for(read, timeout=0.25)
                self.assertEqual(raised.exception.args, ("client disconnected",))

                with (
                    self.assertLogs("services.memory", level="ERROR") as store_logs,
                    self.assertNoLogs("asyncio", level="ERROR"),
                ):
                    fake_store.release.set()
                    await _wait_until(
                        lambda: not (set(memory._BACKGROUND_STORE_IO) - original_background),
                        timeout=0.5,
                    )
                    await asyncio.sleep(0)

                rendered_logs = "\n".join(store_logs.output)
                self.assertIn("error_type=RuntimeError", rendered_logs)
                self.assertNotIn("database disconnected", rendered_logs)
        finally:
            fake_store.release.set()
            watchdog.cancel()


if __name__ == "__main__":
    unittest.main(verbosity=2)
