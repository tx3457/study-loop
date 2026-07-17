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
- `.deepeval/`, `artifacts/`, `traces/`
- `.omc/`, `.omx/`, `.playwright-cli/`

Cloud model providers and optional tracing/MCP integrations may receive user
content when enabled. Review the provider's data policy before uploading
sensitive material. Tracing and live MCP integrations are disabled in the
example configuration.

## Known boundaries

- Tool schemas constrain what the model is asked to emit, but not every tool
  argument currently has an independent Pydantic validation layer.
- `ToolMetadata.permission` is descriptive metadata; it is not an authorization
  system.
- The standalone autonomous endpoint keeps paused conversations in process
  memory. Only the tutor interrupt/resume path uses a LangGraph checkpointer.
- Read-only and idempotent tools may retry automatically. Unknown and
  non-idempotent tools do not; an ambiguous result returns a non-retryable
  conflict, and an in-process autonomous continuation is consumed instead of
  replayed. The audit marker and paused-session store are still process-local,
  so they are not an exactly-once guarantee across crashes or multiple workers.
- A resumed LangGraph node may replay earlier code before an `interrupt`.
  The interrupt-capable assistant therefore exposes and dispatches only tools
  declared read-only or idempotent; unknown MCP tools and profile writes are
  excluded. Future side-effectful tools need durable handler-level idempotency
  before they can enter that path.
