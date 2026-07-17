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
- The standalone autonomous endpoint keeps paused conversations in process
  memory. Only the tutor interrupt/resume path uses a LangGraph checkpointer.
- Read-only and idempotent tools may retry automatically. Unknown and
  non-idempotent tools do not. Autonomous and tool-chat clients may send an
  `Idempotency-Key`; completed responses are replayed from a persistent receipt,
  while a crash after a write starts remains a non-retryable conflict. The
  receipt provides at-most-once replay protection, not a transaction spanning
  the receipt and LangGraph memory store.
- Receipts currently have no automatic expiry. This preserves fail-closed retry
  behavior, but operators must monitor storage and manually investigate stale
  `pending` receipts; adding a simple TTL would weaken at-most-once protection.
- The paused Autonomous HITL session store is still process-local. A restart or
  request routed to another worker cannot resume that paused conversation.
- A resumed LangGraph node may replay earlier code before an `interrupt`.
  The interrupt-capable assistant therefore exposes and dispatches only tools
  declared read-only or idempotent; unknown MCP tools and profile writes are
  excluded. Future side-effectful tools need durable handler-level idempotency
  before they can enter that path.
