from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path


class MaterialStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, knowledge_base_id: str, version_id: str, content: bytes) -> str:
        # Canonical, so every spelling of one id shares the directory the purge removes.
        directory = self.root / str(uuid.UUID(knowledge_base_id))
        directory.mkdir(parents=True, exist_ok=True)
        final_path = directory / f"{version_id}.bin"
        temporary = directory / f".{version_id}.{os.getpid()}.tmp"
        temporary.write_bytes(content)
        temporary.replace(final_path)
        return str(final_path.relative_to(self.root))

    def read(self, relative_path: str) -> bytes:
        return (self.root / relative_path).read_bytes()

    def delete(self, relative_path: str) -> None:
        (self.root / relative_path).unlink(missing_ok=True)

    def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        # The UUID round-trip keeps a stored id from ever naming a path outside root.
        directory = self.root / str(uuid.UUID(knowledge_base_id))
        if directory.exists():
            shutil.rmtree(directory)
