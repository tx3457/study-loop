from __future__ import annotations

import os
from pathlib import Path


class MaterialStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, knowledge_base_id: str, version_id: str, content: bytes) -> str:
        directory = self.root / knowledge_base_id
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
