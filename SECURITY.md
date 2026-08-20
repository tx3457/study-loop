# Security and data handling

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting feature instead of opening
a public issue containing exploit details or secrets.

## Local data boundary

StudyLoop can store uploaded documents, embeddings, learning profiles,
checkpoints, and traces. The following paths are intentionally ignored and must
not be committed:

- `.env`
- `chroma_db/`
- `.checkpoints/`
- `.memory_snapshot.json`
- `.idempotency.sqlite3`
- `.deepeval/`, `artifacts/`, `traces/`
- `.omc/`, `.omx/`, `.playwright-cli/`

Cloud model providers and optional tracing/MCP integrations may receive user
content when enabled. Review the provider's data policy before uploading
sensitive material. Tracing and live MCP integrations are disabled in the
example configuration.

Public HTTP and SSE failures use fixed error codes and do not serialize raw
provider, parser, MCP, or storage exception text. Every HTTP response carries a
strictly validated `X-Request-ID` for correlation; runtime exception logs retain
the request or component context and exception type, but not the exception
message or traceback. Application logs can still contain ordinary operational
metadata and must remain access-controlled.
The bundled Uvicorn process and Nginx `/api/` proxy disable their raw-URL access
logs; the application instead records a route-template, status, and request ID.

Although individual PostgreSQL-backed stores implement cross-worker fencing,
the supported bundled deployment remains one backend worker and one replica.
Embedded Chroma plus in-process BM25/ingest coordination are not a supported
multi-process data plane; `DATABASE_URL` alone does not remove that boundary.

## Known boundaries

- Tool schemas constrain what the model is asked to emit. Required arguments,
  undeclared top-level keys in a closed schema (`additionalProperties: false`),
  and values incompatible with an explicit handler signature are rejected before
  a handler starts. Open schemas may pass extension keys, and not every value has
  an independent Pydantic type-validation layer.
- `ToolMetadata.permission` is descriptive metadata; it is not an authorization
  system.
- Read-only and idempotent tools may retry automatically. Unknown and
  non-idempotent tools do not. Autonomous, quiz-answer, adaptive-submit, and
  tool-chat clients may send an `Idempotency-Key`; completed responses are
  replayed from a persistent receipt, while a crash after a write starts remains
  non-retryable unless an exact durable session outcome can be independently
  validated. Quiz question indexes and adaptive turn numbers
  reject stale submissions. The receipt provides at-most-once replay
  protection, not a transaction spanning the receipt and the affected state
  store.
- New receipts use a bounded owner lease. A clean `pending_v2` receipt may be
  taken over after its lease expires, but the operation and payload fingerprint
  remain immutable and every mutation is fenced by the owner token. Once a
  non-replayable handler starts, the receipt moves to `effect_started_v2` and is
  never taken over to rerun that handler. An exact, separately validated
  Autonomous session outcome may reconcile the receipt without rerunning the
  handler; otherwise a crash remains an ambiguous, fail-closed conflict. Legacy
  unleased `pending` rows are also never taken over during a rolling upgrade.
- Receipt owner leases are execution fencing, not record retention. Completed,
  ambiguous, and legacy receipt rows currently have no automatic retention TTL;
  production deployments should monitor table growth and apply an audited
  retention policy.
- Autonomous HITL pause snapshots use versioned JSON in PostgreSQL when
  `DATABASE_URL` is set, or SQLite for local development. Resume uses an atomic
  leased fencing-token claim, enforces expiry at claim time, and records
  progress before a business tool can dispatch. A clean abandoned claim may be
  taken over after lease expiry; an abandoned claim that crossed the progress
  barrier becomes an `ambiguous` tombstone and is never resumed. Pause handoff
  and terminal completion store the exact public response as a durable outcome,
  allowing a retry to repair an uncompleted outer receipt without rerunning the
  Agent.
- Pause snapshots may contain user messages, tool arguments and results, and
  retrieved evidence text. The default TTL is one hour and the default encoded
  payload limit is 2 MiB. Database files, backups, and logs must be protected
  according to the sensitivity of the uploaded learning material. Autonomous
  queries and replies are capped at 8,000 characters; user, document, and
  conversation identifiers also have bounded lengths before execution starts.
- Before an Autonomous start or continue request is sent, the browser stores the
  exact request body and idempotency key in `sessionStorage`. Ambiguous network,
  server, rate-limit, and in-progress responses keep that request read-only for
  exact replay; only a confirmed pre-execution rejection unlocks editing. HITL
  recovery also stores the bearer `conversation_id`, question, and unsent draft.
  A confirmed cancel deletes a still-paused server snapshot; canceling an
  in-flight operation is rejected. This data remains available to scripts
  running in the same origin and tab.
- The standalone API currently has no trusted authentication subject. A
  `conversation_id` is therefore a high-entropy bearer capability, not an
  authorization boundary, and deployments must be treated as single-user or
  placed behind authentication. The default Compose file binds the Web and API
  ports to loopback and does not publish PostgreSQL; preserve an equivalent
  boundary when adapting it. The ID is no longer logged in full.
- Audit responses are redacted by default. Full tool arguments, output previews,
  and retrieved text remain unavailable unless a trusted deployment explicitly
  sets `AUDIT_PAYLOAD_ENABLED=true`; this switch is not a substitute for
  authentication.
- Session transitions and idempotency receipts share the configured database by
  default; `QUIZ_SESSION_DB_PATH`, `ADAPTIVE_SESSION_DB_PATH`, and
  `AUTONOMOUS_SESSION_DB_PATH` may override their local SQLite files.
  They are still separate transactions, so this is not a general exactly-once
  protocol. Autonomous pause, handoff, and finish transitions mitigate the
  cross-store crash window with a canonical session outcome that repairs the
  receipt on retry. A crash after an external non-idempotent handler starts but
  before its effect can be proven remains intentionally ambiguous and requires
  investigation. `Idempotency-Key` also remains optional outside the Web flow.
- Web Quiz and Adaptive sessions use versioned private aggregates in PostgreSQL
  when `DATABASE_URL` is set, or SQLite for local development. Lease claims,
  fencing tokens, revision checks, TTL checks, and payload bounds prevent stale
  workers from overwriting newer state. Browser recovery data in
  `sessionStorage` includes stable idempotency keys and, while an Adaptive turn
  is pending, the user's submitted answers. It remains accessible to scripts in
  the same origin and tab. Tutor Lab workflow state has separate experimental
  boundaries and must not be inferred to have these guarantees.
- A resumed LangGraph node may replay earlier code before an `interrupt`.
  The interrupt-capable assistant therefore exposes and dispatches only tools
  declared read-only or idempotent; unknown MCP tools and profile writes are
  excluded. Future side-effectful tools need durable handler-level idempotency
  before they can enter that path.
