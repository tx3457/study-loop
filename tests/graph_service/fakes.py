from __future__ import annotations

from collections import defaultdict

from graph_service.engine import DeletionRefused


class FakeEngine:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, str]] = defaultdict(dict)
        self.corrections: list[tuple[str, str, dict]] = []
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self.fail_next = False
        self.rebuild_clear_flags: list[bool] = []
        self.rebuilds: list[str] = []
        self.forgotten: list[str] = []
        self.busy_workspaces: set[str] = set()
        self.refuse_deletion: set[str] = set()

    async def insert(self, workspace: str, version_id: str, text: str, source_token: str) -> None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("synthetic indexing failure")
        self.documents[workspace][version_id] = text

    async def delete_document(self, workspace: str, version_id: str) -> None:
        if version_id in self.refuse_deletion:
            raise DeletionRefused(version_id)
        self.documents[workspace].pop(version_id, None)

    async def forget(self, workspace: str) -> bool:
        if workspace in self.busy_workspaces:
            return False
        self.documents.pop(workspace, None)
        self.forgotten.append(workspace)
        return True

    async def apply_correction(self, workspace: str, kind: str, payload: dict) -> None:
        self.corrections.append((workspace, kind, payload))

    async def query(self, workspace: str, query: str) -> dict:
        chunks = []
        for version_id, text in self.documents[workspace].items():
            if query.lower() in text.lower():
                chunks.append(
                    {
                        "chunk_id": f"chunk-{version_id}",
                        "content": text,
                        "full_doc_id": version_id,
                    }
                )
        return {"chunks": chunks, "entities": [], "relationships": []}

    async def graph(self, workspace: str, search: str | None, max_nodes: int) -> dict:
        return {"nodes": self.nodes, "edges": self.edges}

    async def entity_exists(self, workspace: str, label: str) -> bool:
        return any(node["id"] == label for node in self.nodes)

    async def rebuild(
        self,
        workspace: str,
        documents: list[dict],
        corrections: list[dict],
        *,
        clear_llm_cache: bool = False,
    ) -> None:
        self.rebuild_clear_flags.append(clear_llm_cache)
        self.rebuilds.append(workspace)
        self.documents[workspace].clear()
        for document in documents:
            if not document.get("active", True):
                continue
            await self.insert(
                workspace,
                document["version_id"],
                document["parsed_text"],
                document["source_token"],
            )
        for correction in corrections:
            await self.apply_correction(workspace, correction["kind"], correction["payload"])
