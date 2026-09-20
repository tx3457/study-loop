from __future__ import annotations

from pathlib import Path

import pytest

from .support import (
    chunk_doc_ids,
    immutable_doc_id,
    make_rag,
    opaque_source_token,
    unique_workspace,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.lightrag_live]


async def _open_rag(tmp_path: Path, label: str):
    rag = make_rag(
        workspace=unique_workspace(label),
        working_dir=tmp_path / label,
    )
    await rag.initialize_storages()
    return rag


async def _insert(rag, text: str, doc_id: str, source_token: str) -> None:
    await rag.ainsert(text, ids=doc_id, file_paths=source_token)
    status = await rag.aget_docs_by_ids(doc_id)
    assert doc_id in status, f"ainsert lost caller-assigned document ID {doc_id}"
    status_value = status[doc_id]["status"]
    status_value = getattr(status_value, "value", status_value)
    assert status_value == "processed", (
        f"document {doc_id} did not reach processed status: {status[doc_id]}"
    )


async def _query_chunks(rag, query: str) -> list[dict]:
    from lightrag import QueryParam

    result = await rag.aquery_data(
        query,
        QueryParam(mode="naive", top_k=20, chunk_top_k=20, enable_rerank=False),
    )
    assert result["status"] == "success", f"aquery_data failed: {result}"
    return result["data"]["chunks"]


async def test_workspaces_isolate_same_ids_names_files_text_and_unique_data(
    tmp_path: Path,
    contract_database_url: str,
) -> None:
    first = await _open_rag(tmp_path, "isolation_a")
    second = await _open_rag(tmp_path, "isolation_b")
    shared_id = immutable_doc_id()
    shared_token = opaque_source_token()
    shared_text = "ENTITY[Shared] ENTITY[Anchor] REL[Shared|Anchor] shared passage"
    first_only_id = immutable_doc_id()
    second_only_id = immutable_doc_id()
    try:
        await _insert(first, shared_text, shared_id, shared_token)
        await _insert(second, shared_text, shared_id, shared_token)
        await _insert(
            first,
            "ENTITY[FirstOnly] ENTITY[Anchor] REL[FirstOnly|Anchor] first_only_marker",
            first_only_id,
            opaque_source_token(),
        )
        await _insert(
            second,
            "ENTITY[SecondOnly] ENTITY[Anchor] REL[SecondOnly|Anchor] second_only_marker",
            second_only_id,
            opaque_source_token(),
        )

        first_chunks = await _query_chunks(first, "first_only_marker")
        second_chunks = await _query_chunks(second, "second_only_marker")
        assert all("second_only_marker" not in row["content"] for row in first_chunks), (
            "workspace A retrieved workspace B content"
        )
        assert all("first_only_marker" not in row["content"] for row in second_chunks), (
            "workspace B retrieved workspace A content"
        )
        assert (await first.get_entity_info("SecondOnly"))["graph_data"] is None, (
            "workspace A graph contains workspace B entity"
        )
        assert (await second.get_entity_info("FirstOnly"))["graph_data"] is None, (
            "workspace B graph contains workspace A entity"
        )
        assert first_only_id not in await second.aget_docs_by_ids(first_only_id), (
            "workspace B status store contains workspace A document"
        )
        assert second_only_id not in await first.aget_docs_by_ids(second_only_id), (
            "workspace A status store contains workspace B document"
        )
    finally:
        await first.finalize_storages()
        await second.finalize_storages()


async def test_query_chunk_maps_to_caller_assigned_document_id(
    tmp_path: Path,
    contract_database_url: str,
) -> None:
    rag = await _open_rag(tmp_path, "provenance")
    doc_id = immutable_doc_id()
    source_token = opaque_source_token()
    try:
        await _insert(
            rag,
            "ENTITY[Provenance] ENTITY[Anchor] REL[Provenance|Anchor] provenance_marker",
            doc_id,
            source_token,
        )
        chunks = await _query_chunks(rag, "provenance_marker")
        matching = [row for row in chunks if "provenance_marker" in row["content"]]
        assert matching, "aquery_data did not return the inserted provenance chunk"
        assert {row["file_path"] for row in matching} == {source_token}, (
            "aquery_data changed or exposed more than the opaque source token"
        )
        resolved = await chunk_doc_ids(rag, matching)
        assert set(resolved.values()) == {doc_id}, (
            "text_chunks.full_doc_id did not map every retrieved chunk to the "
            f"caller-assigned document ID: {resolved}"
        )
    finally:
        await rag.finalize_storages()


async def test_same_workspace_duplicate_content_has_consistent_provenance(
    tmp_path: Path,
    contract_database_url: str,
) -> None:
    rag = await _open_rag(tmp_path, "duplicate")
    doc_id = immutable_doc_id()
    content = "ENTITY[Duplicate] ENTITY[Anchor] REL[Duplicate|Anchor] duplicate_marker"
    try:
        source_token = opaque_source_token()
        await _insert(rag, content, doc_id, source_token)
        before = await _query_chunks(rag, "duplicate_marker")
        before_ids = {row["chunk_id"] for row in before}

        await _insert(rag, content, doc_id, source_token)

        statuses = await rag.aget_docs_by_ids(doc_id)
        after = await _query_chunks(rag, "duplicate_marker")
        after_ids = {row["chunk_id"] for row in after}
        assert set(statuses) == {doc_id}, (
            "idempotent same-ID insertion created or lost document status"
        )
        assert after_ids == before_ids, (
            "same-KB duplicate insertion created duplicate chunks instead of reusing "
            f"the existing document: before={before_ids}, after={after_ids}"
        )
    finally:
        await rag.finalize_storages()


async def test_deleting_one_shared_source_preserves_other_graph_evidence(
    tmp_path: Path,
    contract_database_url: str,
) -> None:
    rag = await _open_rag(tmp_path, "shared_delete")
    first_id, second_id = immutable_doc_id(), immutable_doc_id()
    try:
        await _insert(
            rag,
            "ENTITY[SharedEvidence] ENTITY[Anchor] REL[SharedEvidence|Anchor] "
            "shared_evidence source_a_marker",
            first_id,
            opaque_source_token(),
        )
        await _insert(
            rag,
            "ENTITY[SharedEvidence] ENTITY[Anchor] REL[SharedEvidence|Anchor] "
            "shared_evidence source_b_marker",
            second_id,
            opaque_source_token(),
        )
        await rag.adelete_by_doc_id(first_id)

        assert (await rag.get_entity_info("SharedEvidence"))["graph_data"] is not None, (
            "deleting one source removed an entity still supported by another document"
        )
        relation = await rag.get_relation_info("SharedEvidence", "Anchor")
        assert relation["graph_data"] is not None, (
            "deleting one source removed a relation still supported by another document"
        )
        chunks = await _query_chunks(rag, "shared_evidence")
        resolved_ids = set((await chunk_doc_ids(rag, chunks)).values())
        assert first_id not in resolved_ids, "deleted document remains retrievable"
        assert second_id in resolved_ids, "surviving document evidence was removed"
    finally:
        await rag.finalize_storages()


async def _reopen(workspace: str, working_dir: Path):
    reopened = make_rag(workspace=workspace, working_dir=working_dir)
    await reopened.initialize_storages()
    return reopened


async def test_entity_rename_survives_reopen(
    tmp_path: Path,
    contract_database_url: str,
) -> None:
    workspace = unique_workspace("rename")
    working_dir = tmp_path / "rename"
    rag = make_rag(workspace=workspace, working_dir=working_dir)
    await rag.initialize_storages()
    try:
        await _insert(
            rag,
            "ENTITY[Alpha] ENTITY[Beta] REL[Alpha|Beta] rename_marker",
            immutable_doc_id(),
            opaque_source_token(),
        )
        await rag.aedit_entity(
            "Alpha", {"entity_name": "RenamedAlpha"}, allow_rename=True
        )
        assert (await rag.get_entity_info("Alpha"))["graph_data"] is None, (
            "rename left old entity live"
        )
        assert (await rag.get_entity_info("RenamedAlpha"))["graph_data"] is not None, (
            "rename did not create the requested entity identity"
        )
    finally:
        await rag.finalize_storages()

    reopened = await _reopen(workspace, working_dir)
    try:
        assert (await reopened.get_entity_info("RenamedAlpha"))["graph_data"] is not None, (
            "renamed entity was lost after storage reopen"
        )
        assert (await reopened.get_entity_info("Alpha"))["graph_data"] is None
    finally:
        await reopened.finalize_storages()


async def test_entity_merge_survives_reopen(
    tmp_path: Path,
    contract_database_url: str,
) -> None:
    workspace = unique_workspace("merge")
    working_dir = tmp_path / "merge"
    rag = make_rag(workspace=workspace, working_dir=working_dir)
    await rag.initialize_storages()
    try:
        await _insert(
            rag,
            "ENTITY[Alpha] ENTITY[Beta] ENTITY[Gamma] "
            "REL[Alpha|Beta] REL[Gamma|Beta] merge_marker",
            immutable_doc_id(),
            opaque_source_token(),
        )
        await rag.amerge_entities(["Gamma"], "Alpha")
        assert (await rag.get_entity_info("Gamma"))["graph_data"] is None, (
            "merge left source entity live"
        )
        assert (await rag.get_entity_info("Alpha"))["graph_data"] is not None, (
            "merge removed its target entity"
        )
    finally:
        await rag.finalize_storages()

    reopened = await _reopen(workspace, working_dir)
    try:
        assert (await reopened.get_entity_info("Gamma"))["graph_data"] is None, (
            "merged source entity returned after storage reopen"
        )
        assert (await reopened.get_entity_info("Alpha"))["graph_data"] is not None, (
            "merged target entity was lost after storage reopen"
        )
    finally:
        await reopened.finalize_storages()


async def test_relation_delete_survives_reopen(
    tmp_path: Path,
    contract_database_url: str,
) -> None:
    workspace = unique_workspace("relation_delete")
    working_dir = tmp_path / "relation_delete"
    rag = make_rag(workspace=workspace, working_dir=working_dir)
    await rag.initialize_storages()
    try:
        await _insert(
            rag,
            "ENTITY[Alpha] ENTITY[Beta] REL[Alpha|Beta] relation_delete_marker",
            immutable_doc_id(),
            opaque_source_token(),
        )
        await rag.adelete_by_relation("Alpha", "Beta")
        assert (await rag.get_relation_info("Alpha", "Beta"))["graph_data"] is None, (
            "relation delete did not remove the real SDK edge"
        )
    finally:
        await rag.finalize_storages()

    reopened = await _reopen(workspace, working_dir)
    try:
        assert (
            await reopened.get_relation_info("Alpha", "Beta")
        )["graph_data"] is None, (
            "deleted relation returned after storage reopen"
        )
    finally:
        await reopened.finalize_storages()
