# Configuration and operations

StudyLoop is configured through a repository-local `.env` copied from
`.env.example`. The file may contain credentials and is intentionally ignored by
Git.

## Model providers

The complete product flow requires OpenAI-compatible Chat, Structured Output,
and Embeddings APIs.

| Capability | Environment variables |
| --- | --- |
| Chat and Tool Calling | `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` |
| Structured Output | `STRUCTURED_API_KEY`, `STRUCTURED_BASE_URL`, `STRUCTURED_MODEL` |
| Embeddings | `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `LLM_EMBEDDING_MODEL` |

The Structured Output and Embeddings key/address pairs must be configured as a
complete group. Only when a complete group is left empty does it fall back to
the corresponding `LLM_*` provider.

When connecting a provider for the first time:

1. Copy `.env.example` to `.env`.
2. Replace the `LLM_*` values with the real Chat endpoint, key, and model.
3. Configure `STRUCTURED_*` separately if the Chat provider does not support the
   required structured-output behavior.
4. Configure `EMBEDDING_*` and `LLM_EMBEDDING_MODEL` separately if embeddings
   are not available from the Chat provider.

Run the explicit capability probe after configuration:

```bash
python scripts/check_provider_capabilities.py
```

This command sends a small number of real requests and may incur a small cost.
It checks normal Chat, JSON mode, production tool choice, Structured Output, and
Embeddings. It never prints the API key, provider address, response body, or raw
exception. It is not a public HTTP endpoint or a default CI step.

## Optional integrations

- Local Cross-Encoder reranking requires `requirements-reranker.txt` and
  `RERANKER_ENABLED=true`.
- Query rewriting, HyDE, multi-query retrieval, and reranking are feature
  flagged and disabled in `.env.example` for a lightweight first run.
- LangSmith, Langfuse, tracing, and live MCP integrations are opt-in and disabled
  by default.

## Durable Web workflows

The Web client attaches idempotency keys to Autonomous starts/resumes, Learning
Path creation, Quiz starts/answers, and Adaptive starts/submissions. While an
operation is uncertain, the browser keeps the exact request and key read-only so
that a refresh or response loss replays the same intent instead of creating a
second mutation.

Server-side state includes:

- immutable Learning Path resources and stage progress;
- complete private Quiz and Adaptive session aggregates;
- Autonomous HITL pause snapshots and canonical terminal outcomes;
- learner profiles, mastery, weak points, and learning events;
- persistent idempotency receipts.

With `DATABASE_URL`, these stores use PostgreSQL. Local development falls back
to the SQLite paths documented in `.env.example`; learner memory additionally
has a local JSON snapshot fallback. Default retention is one hour for
Autonomous and Adaptive sessions and 24 hours for Web Quiz sessions.

Quiz question indexes, Adaptive turn numbers, and stored revisions reject stale
updates. A clean abandoned lease may be taken over safely. Once execution has
crossed an unprovable side-effect boundary, StudyLoop fails closed instead of
blindly rerunning the operation. The exact failure and recovery boundaries are
documented in [`SECURITY.md`](../SECURITY.md).

## Supported storage topology

The shipped Docker/Compose topology supports one backend worker and one backend
replica. It uses an embedded Chroma `PersistentClient`; multiple processes must
not write the same Chroma directory.

`DATABASE_URL` coordinates receipts, learning sessions, paths, and learner
memory across processes, but it does not make embedded Chroma multi-process
safe. A future multi-worker deployment must move Chroma behind a service using
`HttpClient`, retain PostgreSQL for application state, and replace the
in-process BM25 cache invalidation and ingest coordination with cross-process
protocols. See [`ARCHITECTURE.md`](ARCHITECTURE.md).

All Chroma calls in the supported process pass through one serialized daemon
channel. The relevant budgets are:

| Setting | Purpose |
| --- | --- |
| `CHROMA_IO_MAX_PENDING` | admission limit |
| `CHROMA_IO_QUEUE_WAIT_SECONDS` | queue wait budget |
| `CHROMA_IO_OPERATION_TIMEOUT_SECONDS` | public operation wait budget |
| `CHROMA_IO_CANCEL_DRAIN_SECONDS` | cancellation cleanup budget |
| `CHROMA_IO_SHUTDOWN_DRAIN_SECONDS` | graceful shutdown drain budget |

A request timeout or disconnect does not abandon an already-started staging or
tombstone transaction. The work continues toward a bounded cleanup. These
settings are not multi-worker support and do not turn non-idempotent writes into
automatic retries. Forced process termination can still interrupt an unfinished
local Chroma write.

PostgreSQL-backed state components expose additional bounded connection, lock,
statement, queue, and cancellation-cleanup settings in `.env.example`, including
`MEMORY_STORE_SETUP_LOCK_TIMEOUT_SECONDS` and
`QUIZ_SESSION_PG_LOCK_TIMEOUT_MS`. They protect individual components; they do
not change the supported application topology.

## Health and capability endpoints

After starting the backend:

```bash
curl -i http://localhost:8001/health/live
curl -i http://localhost:8001/health/ready
curl -i http://localhost:8001/health/providers
```

- `/health/live` checks only that the process is alive.
- `/health/ready` checks the Chroma collection catalog and the state stores used
  by the current process. With PostgreSQL it performs a bounded query and reads
  the application's learner-memory Store; without PostgreSQL it checks the
  selected SQLite files and local memory snapshot directory.
- `/health/providers` calls the provider's model-list endpoint and caches the
  result for 30 seconds. It does not run Chat, Structured Output, Tool Calling,
  or Embeddings requests.

A readiness success proves that the required storage is currently accessible;
it is not a complete upload or generation test. Restart the backend after a
PostgreSQL restart so that long-running learner-memory connections are rebuilt.
From the Web container, use
`http://localhost:4001/api/health/ready`; `/health/ready` without `/api` is a
frontend route.

## Provider deadlines and public errors

The shared Chat, Structured Output, and Embeddings retry chains each have a
60-second end-to-end budget by default, including backoff. Configure it with
`PROVIDER_REQUEST_DEADLINE_SECONDS`.

The HTTP boundary returns stable provider errors:

| Condition | Code | HTTP status |
| --- | --- | ---: |
| Rate limited | `provider_rate_limited` | 429 |
| Deadline exceeded | `provider_timeout` | 504 |
| Other upstream failure | `provider_unavailable` | 503 |
| Provider not configured | `provider_not_configured` | 503 |

Every response carries a generated or strictly validated `X-Request-ID` so the
Web UI can correlate a safe public message with protected server logs. Raw
provider responses, storage paths, and original exception messages are not
returned to the browser.

## Browser and network boundary

The default browser-origin allowlist covers the local Web, Vite development
server, and API documentation origins on ports 4001, 5173, and 8001. A custom
domain or HTTPS reverse proxy must add its exact scheme, host, and port to the
shared trusted-origin source before rebuilding.

This origin check is not authentication. Any shared or remote deployment must
add authentication, TLS, and trusted-proxy/Host enforcement before exposing the
application. Full details are in [`SECURITY.md`](../SECURITY.md).

