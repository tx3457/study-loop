# V1 application boundary

Implemented V1 boundary; execution evidence and limitations are recorded in LIGHTRAG_STATUS.md.

## Transport and identity

The public StudyLoop API resolves the existing authenticated subject. The graph service
accepts only the internal service token plus `X-StudyLoop-Subject`; browsers cannot address
it directly. Body-supplied identities and workspace names never grant authority.

Public capabilities: `GET /knowledge-bases/capabilities` returns
`{enabled, available, web_search_available}`. Disabled or unavailable graph features do not
make the legacy application unready. The public client must retain request IDs on errors.

Graph-service internal paths mirror the `/knowledge-bases` and `/knowledge-jobs` groups;
the main backend authenticates, validates uploads, and forwards authorized operations.

## Resource views

- Knowledge base: `id, name, description, status, revision, epoch, document_count,
  created_at`. Status is `ready | updating | dirty | deleting`.
- Document: `id, knowledge_base_id, name, kind, version_id, status, content_hash,
  created_at`, with optional source URL/time and legacy origin ID. Kind is
  `file | web | legacy_copy`. Display names are not identities.
- Job: `id, knowledge_base_id, status, stage, revision, error_code`. Status is
  `queued | running | succeeded | failed`. Error codes are public-safe, not exceptions.
- Scope: `knowledge_base_id, revision, epoch`; resolved by the server, not trusted from
  a model tool call. Source evidence additionally pins `source_version_id`.
- Graph: `nodes, edges, truncated, revision, epoch`. Nodes expose stable application IDs,
  labels and aliases; edges expose stable IDs/endpoints and source references. Detailed
  graph results are bounded to 200 nodes / 400 edges. Source details are separately fetched.

List envelopes use `knowledge_bases`, `documents`, or `jobs`, respectively.
Document pages return `{documents, total, limit, offset}`, defaulting to 50 rows per
page (maximum 100). `total` counts visible documents in the authorized KB even when
the requested offset is beyond the last page. Ordering is stable by creation time and ID.
Creation returns the created resource. Mutations with indexing work return `202` with
`{job_id, knowledge_base_id}`. Keep that `job_id` while polling the job resource,
which uses `id`; only `succeeded` or `failed` ends the wait.
Idempotency keys are owner-global. Reusing a key with a different operation, target
KB/document/snapshot, or body returns 409. Exact same-target retries replay the original job.
Pre-fix material/correction/delete-document receipt hashes lack full target identity;
they are not accepted as an exact replay by the corrected implementation (409). Inspect
the original job and KB state before issuing a new intent; do not automatically retry
an ambiguous old mutation with a fresh key.

## Operations

| Method | Public path | Request |
| --- | --- | --- |
| GET/POST | /knowledge-bases | POST: name, description |
| GET/PATCH/DELETE | /knowledge-bases/{kb_id} | writes: expected_revision; PATCH metadata |
| GET | /knowledge-bases/{kb_id}/documents | pagination |
| POST | /knowledge-bases/{kb_id}/documents/upload | multipart file, expected_revision |
| PUT/DELETE | /knowledge-bases/{kb_id}/documents/{document_id} | replacement file or deletion, expected_revision |
| POST | /knowledge-bases/{kb_id}/documents/import-legacy | legacy_document_id, expected_revision |
| GET | /knowledge-bases/{kb_id}/graph | search/focus ID and bounded limits |
| GET | /knowledge-bases/{kb_id}/sources/{source_version_id} | bounded source excerpt/locator |
| POST | /knowledge-bases/{kb_id}/corrections | kind, target fields, expected_revision |
| POST | /knowledge-bases/{kb_id}/web-import | snapshot_id, expected_revision |
| GET | /knowledge-jobs/{job_id} | — |
| POST | /knowledge-jobs/{job_id}/retry | expected_revision |

Correction kinds are `rename_entity`, `merge_entities`, `delete_entity`, `delete_relation`.
Rename takes `entity_id, label`; merge takes `entity_ids, target_id`; entity deletion takes
`entity_id`; relation deletion takes `edge_id`. The actor and KB are server-bound.
Merge accepts one or more distinct source IDs into an existing, distinct target ID;
the target must not also appear in `entity_ids`. A two-entity merge uses one source ID.

Graph service also exposes internal-only retrieval/scope checks for the authorized Agent:
`POST /knowledge-bases/{kb_id}/query` with `query, expected_revision, expected_epoch`, and
`GET /knowledge-bases/{kb_id}/scope`. Retrieval returns `scope, evidence, entities,
relationships`; only evidence with a resolved immutable source version is citeable.

## Agent and citations

Autonomous requests add `knowledge_base_id: string | null` and `web_enabled: boolean`.
`document_id` and `knowledge_base_id` are mutually exclusive. Knowledge-base mode enforces
grounding and uses a read-only research tool set. Legacy document behavior remains unchanged.

The additive `source_citations` array contains:

- KB chunk: `kind: "kb_chunk", evidence_id, knowledge_base_id, document_id,
  source_version_id, chunk_id, title, snippet`, plus verified locator/URL when available.
- Web: `kind: "web_snapshot", evidence_id, snapshot_id, title, url, fetched_at,
  content_hash, snippet`.

All fields are server-resolved from evidence actually observed in the current run. New
source kinds must not be coerced into legacy citations. Unknown kinds fail validation.
Graph summaries without underlying text evidence cannot satisfy grounding.

Raw webpage bodies are stored outside the bounded Autonomous snapshot. Import is an
explicit authenticated UI action using a previously captured snapshot, never an Agent write.

## Error and lifetime rules

- 404: resource absent or outside the caller's scope.
- 409: revision/epoch changed, conflicting idempotency payload, or ambiguous correction.
- 410: required web snapshot expired; do not silently refetch.
- 413: upload/body exceeds the configured bounded limit.
- 503: feature unavailable or index updating/dirty; no stale fallback as current evidence.

Use versioned snapshots, scoped tool fingerprints and the original request's idempotency key
for recovery. Finalized historical answers may show retained excerpts marked as historical;
live queries and paused sessions must revalidate current material availability.
