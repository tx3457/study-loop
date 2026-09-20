from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager


class KeyedRWLocks:
    """Process-local writer preference lock; PostgreSQL epoch checks cover restarts."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def _lock(self, key: str) -> asyncio.Lock:
        async with self._guard:
            return self._locks.setdefault(key, asyncio.Lock())

    @asynccontextmanager
    async def read(self, key: str):
        lock = await self._lock(key)
        async with lock:
            yield

    @asynccontextmanager
    async def write(self, key: str):
        lock = await self._lock(key)
        async with lock:
            yield
