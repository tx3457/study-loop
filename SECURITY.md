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
- A resumed LangGraph node may replay code before an `interrupt`; tools with
  side effects need idempotency safeguards before this is used with untrusted
  workloads.
