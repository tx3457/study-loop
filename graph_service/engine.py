from __future__ import annotations

import asyncio
import hashlib
import json
import os
import unicodedata
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from .config import Settings
from .errors import TooLarge


class GraphEngine(Protocol):
    async def insert(
        self, workspace: str, version_id: str, text: str, source_token: str
    ) -> None: ...
    async def delete_document(self, workspace: str, version_id: str) -> None: ...
    async def apply_correction(self, workspace: str, kind: str, payload: dict) -> None: ...
    async def query(self, workspace: str, query: str) -> dict: ...
    async def graph(self, workspace: str, search: str | None, max_nodes: int) -> dict: ...
    async def entity_exists(self, workspace: str, label: str) -> bool: ...
    async def rebuild(
        self,
        workspace: str,
        documents: list[dict],
        corrections: list[dict],
        *,
        clear_llm_cache: bool = False,
    ) -> None: ...


def index_config_hash(settings: Settings) -> str:
    def origin(value: str | None) -> str:
        if not value:
            return ""
        parsed = urlparse(value)
        host = (parsed.hostname or "").lower()
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme.lower()}://{host}{port}{parsed.path.rstrip('/')}"

    public = {
        "version": settings.index_config_version,
        "provider": settings.provider,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "embedding_max_input_tokens": settings.embedding_max_input_tokens,
        "embedding_token_budget": settings.embedding_token_budget,
        "embedding_max_utf8_bytes": settings.embedding_max_utf8_bytes,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "llm_max_output_tokens": settings.llm_max_output_tokens,
        "llm_enable_thinking": settings.llm_enable_thinking,
        "llm_temperature": settings.llm_temperature,
        "llm_top_p": settings.llm_top_p,
        "llm_top_k": settings.llm_top_k,
        "llm_min_p": settings.llm_min_p,
        "llm_presence_penalty": settings.llm_presence_penalty,
        "extraction_max_records": settings.extraction_max_records,
        "extraction_max_entities": settings.extraction_max_entities,
        "extraction_max_gleaning": settings.extraction_max_gleaning,
        "tokenizer_model": settings.tokenizer_model,
        "llm_origin": origin(settings.llm_base_url),
        "embedding_origin": origin(settings.embedding_base_url),
        "storages": ["PGKVStorage", "PGDocStatusStorage", "PGTableGraphStorage", "PGVectorStorage"],
    }
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


class LightRAGEngine:
    """Lazy single-event-loop adapter for the pinned LightRAG SDK."""

    def __init__(self, settings: Settings, *, llm_func=None, embedding_func=None) -> None:
        if settings.max_cached_instances < 1:
            raise ValueError("KNOWLEDGE_MAX_CACHED_INSTANCES must be positive")
        self.settings = settings
        self._llm_func = llm_func
        self._embedding_func = embedding_func
        self._instances: OrderedDict[str, Any] = OrderedDict()
        self._active: dict[str, int] = {}
        self._guard = asyncio.Lock()
        self._configure_postgres()

    def _configure_postgres(self) -> None:
        parsed = urlparse(self.settings.database_url)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
            raise ValueError("KNOWLEDGE_DATABASE_URL must be a PostgreSQL DSN")
        password = (
            unquote(parsed.password)
            if parsed.password is not None
            else os.environ.get("PGPASSWORD", "")
        )
        os.environ.update(
            POSTGRES_HOST=parsed.hostname,
            POSTGRES_PORT=str(parsed.port or 5432),
            POSTGRES_USER=unquote(parsed.username or ""),
            POSTGRES_PASSWORD=password,
            POSTGRES_DATABASE=unquote(parsed.path.lstrip("/")),
            POSTGRES_SSL_MODE="disable",
        )
        # Empty is intentional: it blocks both an inherited environment override and
        # LightRAG's config-file fallback while leaving each SDK instance's workspace active.
        os.environ["POSTGRES_WORKSPACE"] = ""

    @staticmethod
    def _record(value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "__dict__"):
            return dict(value.__dict__)
        if isinstance(value, dict):
            return value
        raise TypeError(f"unsupported LightRAG graph record: {type(value).__name__}")

    def _provider_callbacks(self):
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc, is_truncated_response

        if self.settings.provider != "openai":
            raise RuntimeError(f"unsupported KNOWLEDGE_PROVIDER: {self.settings.provider}")
        if not self.settings.llm_model or not self.settings.embedding_model:
            raise RuntimeError("LLM_MODEL and LLM_EMBEDDING_MODEL are required")

        async def llm(prompt: str, **kwargs):
            requested_max_tokens = kwargs.pop("max_tokens", None)
            effective_max_tokens = self.settings.llm_max_output_tokens
            if requested_max_tokens is not None:
                try:
                    requested_limit = int(requested_max_tokens)
                except (TypeError, ValueError) as error:
                    raise ValueError("max_tokens must be a positive integer") from error
                if requested_limit <= 0:
                    raise ValueError("max_tokens must be a positive integer")
                effective_max_tokens = min(effective_max_tokens, requested_limit)
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
            kwargs.pop("top_k", None)
            kwargs.pop("min_p", None)
            kwargs.pop("presence_penalty", None)
            supplied_client_config = kwargs.pop("openai_client_configs", None) or {}
            client_config = {**supplied_client_config, "max_retries": 0}
            extra_body = dict(kwargs.pop("extra_body", None) or {})
            if self.settings.llm_enable_thinking is not None:
                extra_body["enable_thinking"] = self.settings.llm_enable_thinking
            if self.settings.llm_top_k is not None:
                extra_body["top_k"] = self.settings.llm_top_k
            if self.settings.llm_min_p is not None:
                extra_body["min_p"] = self.settings.llm_min_p
            if extra_body:
                kwargs["extra_body"] = extra_body
            if self.settings.llm_top_p is not None:
                kwargs["top_p"] = self.settings.llm_top_p
            if self.settings.llm_presence_penalty is not None:
                kwargs["presence_penalty"] = self.settings.llm_presence_penalty
            result = await openai_complete_if_cache(
                self.settings.llm_model,
                prompt,
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url,
                timeout=self.settings.llm_timeout_seconds,
                max_tokens=effective_max_tokens,
                temperature=self.settings.llm_temperature,
                openai_client_configs=client_config,
                **kwargs,
            )
            if is_truncated_response(result):
                raise RuntimeError(
                    "LLM response was truncated at the configured output token limit"
                )
            return result

        async def embed(texts: list[str], max_token_size: int | None = None, **kwargs):
            del max_token_size
            normalized = self._validated_embedding_texts(texts, context="embedding input")
            supplied_client_config = kwargs.pop("client_configs", None) or {}
            client_config = {
                **supplied_client_config,
                "timeout": self.settings.embedding_timeout_seconds,
                "max_retries": 0,
            }
            return await openai_embed.func(
                normalized,
                model=self.settings.embedding_model,
                api_key=self.settings.embedding_api_key,
                base_url=self.settings.embedding_base_url,
                max_token_size=0,
                client_configs=client_config,
                **kwargs,
            )

        embedding = EmbeddingFunc(
            embedding_dim=self.settings.embedding_dim,
            max_token_size=self.settings.embedding_token_budget,
            model_name=self.settings.embedding_model,
            func=embed,
            supports_asymmetric=True,
        )
        return llm, embedding

    def _validated_embedding_texts(self, texts: list[str], *, context: str) -> list[str]:
        normalized: list[str] = []
        for index, text in enumerate(texts):
            value = unicodedata.normalize("NFKC", text)
            byte_count = len(value.encode("utf-8"))
            if byte_count > self.settings.embedding_max_utf8_bytes:
                raise TooLarge(
                    f"{context} {index + 1} exceeds the configured "
                    f"{self.settings.embedding_max_utf8_bytes}-byte embedding safety limit"
                )
            normalized.append(value)
        return normalized

    async def _acquire_rag(self, workspace: str):
        async with self._guard:
            if workspace in self._instances:
                self._instances.move_to_end(workspace)
                self._active[workspace] += 1
                return self._instances[workspace]
            from lightrag import LightRAG

            llm = self._llm_func
            embedding = self._embedding_func
            if llm is None or embedding is None:
                llm, embedding = self._provider_callbacks()
            working = self.settings.working_dir / workspace
            working.mkdir(parents=True, exist_ok=True)
            rag = LightRAG(
                working_dir=str(working),
                workspace=workspace,
                kv_storage="PGKVStorage",
                doc_status_storage="PGDocStatusStorage",
                graph_storage="PGTableGraphStorage",
                vector_storage="PGVectorStorage",
                llm_model_func=llm,
                llm_model_name=self.settings.llm_model,
                embedding_func=embedding,
                chunk_token_size=self.settings.chunk_tokens,
                chunk_overlap_token_size=self.settings.chunk_overlap_tokens,
                embedding_chunk_overlap_token_size=self.settings.chunk_overlap_tokens,
                tiktoken_model_name=self.settings.tokenizer_model,
                auto_manage_storages_states=False,
                llm_model_max_async=self.settings.llm_max_async,
                entity_extract_max_records=self.settings.extraction_max_records,
                entity_extract_max_entities=self.settings.extraction_max_entities,
                entity_extract_max_gleaning=self.settings.extraction_max_gleaning,
                default_llm_timeout=self.settings.llm_sdk_timeout_seconds,
                default_embedding_timeout=self.settings.embedding_sdk_timeout_seconds,
            )
            await rag.initialize_storages()
            self._instances[workspace] = rag
            self._active[workspace] = 1
            return rag

    @asynccontextmanager
    async def _use(self, workspace: str):
        rag = await self._acquire_rag(workspace)
        try:
            yield rag
        finally:
            evicted: list[Any] = []
            async with self._guard:
                self._active[workspace] -= 1
                while len(self._instances) > self.settings.max_cached_instances:
                    idle_workspace = next(
                        (name for name in self._instances if self._active[name] == 0), None
                    )
                    if idle_workspace is None:
                        break
                    evicted.append(self._instances.pop(idle_workspace))
                    self._active.pop(idle_workspace, None)
            for idle_rag in evicted:
                await idle_rag.finalize_storages()

    async def close(self) -> None:
        for rag in list(self._instances.values()):
            await rag.finalize_storages()
        self._instances.clear()
        self._active.clear()

    async def insert(self, workspace: str, version_id: str, text: str, source_token: str) -> None:
        async with self._use(workspace) as rag:
            await rag.ainsert(text, ids=version_id, file_paths=source_token)
            status = await rag.aget_docs_by_ids(version_id)
            value = status.get(version_id, {}).get("status") if status else None
            value = getattr(value, "value", value)
            if value != "processed":
                raise RuntimeError("LightRAG did not flush the canonical source version")

    async def delete_document(self, workspace: str, version_id: str) -> None:
        async with self._use(workspace) as rag:
            known = await rag.aget_docs_by_ids(version_id)
            if version_id in known:
                await rag.adelete_by_doc_id(version_id)

    async def apply_correction(self, workspace: str, kind: str, payload: dict) -> None:
        async with self._use(workspace) as rag:
            if kind == "rename_entity":
                if payload["entity_label"] == payload["label"]:
                    return
                source = (await rag.get_entity_info(payload["entity_label"]))["graph_data"]
                target = (await rag.get_entity_info(payload["label"]))["graph_data"]
                if source is None:
                    return
                if target is not None:
                    if payload.get("target_is_owned_alias"):
                        await rag.amerge_entities([payload["entity_label"]], payload["label"])
                        return
                    raise RuntimeError("rename target exists; explicit merge required")
                await rag.aedit_entity(
                    payload["entity_label"],
                    {"entity_name": payload["label"]},
                    allow_rename=True,
                    allow_merge=False,
                )
            elif kind == "merge_entities":
                if (await rag.get_entity_info(payload["target_label"]))["graph_data"] is None:
                    return
                live_sources = [
                    label
                    for label in payload["source_labels"]
                    if (await rag.get_entity_info(label))["graph_data"] is not None
                ]
                if live_sources:
                    await rag.amerge_entities(live_sources, payload["target_label"])
            elif kind == "delete_entity":
                if (await rag.get_entity_info(payload["entity_label"]))["graph_data"] is not None:
                    await rag.adelete_by_entity(payload["entity_label"])
            elif kind == "delete_relation":
                relation = await rag.get_relation_info(
                    payload["source_label"], payload["target_label"]
                )
                if relation["graph_data"] is not None:
                    await rag.adelete_by_relation(payload["source_label"], payload["target_label"])
            else:
                raise ValueError(f"unknown correction kind: {kind}")

    async def query(self, workspace: str, query: str) -> dict:
        from lightrag import QueryParam

        self._validated_embedding_texts([query], context="query")
        async with self._use(workspace) as rag:
            result = await rag.aquery_data(
                query,
                QueryParam(mode="mix", top_k=40, chunk_top_k=20, enable_rerank=False),
            )
            if result.get("status") != "success":
                raise RuntimeError("LightRAG query failed")
            data = result["data"]
            chunks = []
            for chunk in data.get("chunks", []):
                stored = await rag.text_chunks.get_by_id(chunk["chunk_id"])
                if stored and stored.get("full_doc_id"):
                    chunks.append({**chunk, "full_doc_id": stored["full_doc_id"]})
            return {
                "chunks": chunks,
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", []),
            }

    async def graph(self, workspace: str, search: str | None, max_nodes: int) -> dict:
        async with self._use(workspace) as rag:
            graph = await rag.get_knowledge_graph(search or "*", max_depth=3, max_nodes=max_nodes)
            chunk_sources: dict[str, str | None] = {}

            async def with_sources(value: Any) -> dict[str, Any]:
                record = self._record(value)
                properties = record.get("properties") or {}
                source_versions: list[str] = []
                for chunk_id in str(properties.get("source_id", "")).split("<SEP>"):
                    if not chunk_id:
                        continue
                    if chunk_id not in chunk_sources:
                        chunk = await rag.text_chunks.get_by_id(chunk_id)
                        chunk_sources[chunk_id] = (
                            str(chunk["full_doc_id"])
                            if chunk and chunk.get("full_doc_id")
                            else None
                        )
                    if chunk_sources[chunk_id]:
                        source_versions.append(chunk_sources[chunk_id])
                record["source_version_ids"] = list(dict.fromkeys(source_versions))
                return record

            return {
                "nodes": [await with_sources(node) for node in graph.nodes],
                "edges": [await with_sources(edge) for edge in graph.edges],
                "truncated": graph.is_truncated,
            }

    async def entity_exists(self, workspace: str, label: str) -> bool:
        async with self._use(workspace) as rag:
            return (await rag.get_entity_info(label))["graph_data"] is not None

    async def rebuild(
        self,
        workspace: str,
        documents: list[dict],
        corrections: list[dict],
        *,
        clear_llm_cache: bool = False,
    ) -> None:
        async with self._use(workspace) as rag:
            # Reconstruct from an empty workspace. Document-wise SDK deletion cannot
            # discover application-renamed/merged graph identities from the original
            # per-document extraction anchors, so it can leave graph/vector source IDs
            # pointing at chunks it just removed. The SDK storage drop contract clears
            # this workspace only; the LLM cache is intentionally retained so unchanged
            # canonical documents normally reuse their extraction responses.
            storages = [
                rag.chunk_entity_relation_graph,
                rag.entities_vdb,
                rag.relationships_vdb,
                rag.chunks_vdb,
                rag.entity_chunks,
                rag.relation_chunks,
                rag.full_entities,
                rag.full_relations,
                rag.text_chunks,
                rag.full_docs,
                rag.doc_status,
            ]
            if clear_llm_cache:
                storages.append(rag.llm_response_cache)
            for storage in storages:
                result = await storage.drop()
                if result.get("status") != "success":
                    raise RuntimeError(
                        f"LightRAG workspace reconstruction failed for {storage.namespace}"
                    )
            for document in documents:
                if not document.get("active", True):
                    continue
                await rag.ainsert(
                    document["parsed_text"],
                    ids=document["version_id"],
                    file_paths=document["source_token"],
                )
                status = await rag.aget_docs_by_ids(document["version_id"])
                value = status.get(document["version_id"], {}).get("status") if status else None
                value = getattr(value, "value", value)
                if value != "processed":
                    raise RuntimeError("LightRAG rebuild did not flush a source version")
        for correction in corrections:
            await self.apply_correction(workspace, correction["kind"], correction["payload"])
