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
  a non-retryable conflict. Quiz question indexes and adaptive turn numbers
  reject stale submissions. The receipt provides at-most-once replay
  protection, not a transaction spanning the receipt and the affected state
  store.
- Receipts currently have no automatic expiry. This preserves fail-closed retry
  behavior, but operators must monitor storage and manually investigate stale
  `pending` receipts; adding a simple TTL would weaken at-most-once protection.
- Autonomous HITL pause snapshots use versioned JSON in PostgreSQL when
  `DATABASE_URL` is set, or SQLite for local development. Resume uses an atomic
  fencing-token claim, enforces expiry at claim time, and records progress
  before any returned tool call can mutate messages or dispatch a handler.
  An abandoned `in_flight` claim is deliberately never unlocked by timeout;
  automatic takeover could run concurrently with the old worker and duplicate
  a side effect. Operators must investigate stale claims and restart the user
  flow.
- Pause snapshots may contain user messages, tool arguments and results, and
  retrieved evidence text. The default TTL is one hour and the default encoded
  payload limit is 2 MiB. Database files, backups, and logs must be protected
  according to the sensitivity of the uploaded learning material. Autonomous
  queries and replies are capped at 8,000 characters; user, document, and
  conversation identifiers also have bounded lengths before execution starts.
- While an Autonomous HITL question is pending, the browser stores the minimal
  recovery state in `sessionStorage`, including the bearer `conversation_id`,
  the question, and the unsent draft reply. It is cleared on completion, reset,
  or terminal failure, but remains available to scripts running in the same
  origin and tab.
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
  They are still separate transactions. This is fail-closed at-most-once
  protection, not exactly-once execution: a process crash between transitions
  can leave a pending receipt or an abandoned claim that requires operator
  cleanup. `Idempotency-Key` also remains optional.
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
