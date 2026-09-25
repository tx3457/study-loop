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

Uploaded DOCX files are validated before any parser expands them. The archive
guard enforces independent budgets on entry count, central-directory size,
per-entry and total decompressed bytes, XML payload size, and a 200:1
compression-ratio ceiling, and rejects ZIP64 and Unicode Path extra fields
outright. A hostile or malformed archive is refused before it reaches the
document loader.

Cloud model providers and optional tracing/MCP integrations may receive user
content when enabled. Review the provider's data policy before uploading
sensitive material. Tracing and live MCP integrations are disabled in the
example configuration.

Deleting a document uses **material-only deletion**. It removes the source
material and retrieval index from StudyLoop, but intentionally retains learning
history, learner profiles, wrong questions, quizzes, and saved Agent-session
artifacts. Previously saved questions or cited snippets may therefore remain
visible inside their original session. A tombstone also reserves the deleted
document name so newly uploaded content cannot be confused with old learning
records; rename the file before uploading it again. This action is not a full
privacy erasure workflow.

Deleting from an optional knowledge base removes the source content, not only
its index. A deleted document loses every version's original file, parsed text,
content hash and source URL, the body of the web snapshot it was imported from
(unless another live document still uses it), and the LightRAG extraction-cache
entries no remaining chunk uses; only a titled tombstone remains so old citations
resolve as deleted instead of failing. A deleted knowledge base keeps only a
nameless tombstone and its delete job: its graph, vectors, caches, working
directory, files, documents, corrections and graph-identity rows are removed, and
every later read or write reports it as absent. A crash after the delete job
commits is finished by the knowledge worker's periodic sweep; a delete job that
fails leaves the documents readable (its index and caches may already be gone)
until the deletion is retried. Deleting a
single document does not remove LightRAG's merged entity-description cache or
entity names registered by corrections. Existing backups, saved StudyLoop
sessions, and physical erasure below the database (dead PostgreSQL tuples before
VACUUM, filesystem blocks) are outside this guarantee; see
[`docs/LIGHTRAG.md`](docs/LIGHTRAG.md).

Public HTTP and SSE failures use fixed error codes and do not serialize raw
provider, parser, MCP, or storage exception text. Every HTTP response carries a
strictly validated `X-Request-ID` for correlation; runtime exception logs retain
the request or component context and exception type, but not the exception
message or traceback. Application logs can still contain ordinary operational
metadata and must remain access-controlled.
The bundled Uvicorn process and Nginx `/api/` proxy disable their raw-URL access
logs; the application instead records a route-template, status, and request ID.
Browser requests that can change state require one exact trusted `Origin`. The
bundled allowlist covers only the `localhost` and `127.0.0.1` origins on ports
4001, 5173, and 8001; an absent Origin remains available to command-line clients.
Custom domains and HTTPS reverse proxies are not trusted automatically. Such a
deployment must add its exact scheme, host, and port to the shared
`ALLOWED_BROWSER_ORIGINS` source constant and rebuild the application; changing
only a forwarded `Host` does not extend the trust boundary.
This is a browser CSRF/origin control, not authentication or general Host
validation: a non-browser client can omit or forge `Origin`. Any deployment
beyond the bundled loopback topology still needs authentication, TLS, and
trusted-proxy/Host enforcement at its network boundary.

Although individual PostgreSQL-backed stores implement cross-worker fencing,
the supported bundled deployment remains one backend worker and one replica.
Embedded Chroma plus in-process BM25/ingest coordination are not a supported
multi-process data plane; `DATABASE_URL` alone does not remove that boundary.

Dependabot reports four ChromaDB advisories with no patched release:
GHSA-36p7-vc44-83pf and GHSA-f4j7-r4q5-qw2c (code injection through the
`/api/v2/.../collections` endpoints when a request supplies a model
repository with `trust_remote_code`), GHSA-2wm9-hf6c-p5cr and
GHSA-xph7-9rjv-w5fr (missing tenant, database and collection checks in
`SimpleRBACAuthorizationProvider`). All four describe the ChromaDB HTTP
server. This project embeds Chroma through `PersistentClient` against a
local directory: it serves no Chroma endpoint, accepts no Chroma HTTP
request, and configures no tenant or RBAC provider. Uploads cannot supply
Chroma collection configuration, embedding-function configuration, model
repository names, or `trust_remote_code`; the application supplies its own
embedding vectors. These boundaries exclude the reported remote attack paths.

Embedded storage alone is not a defense against poisoned persisted embedding
configuration: the Python client may load such configuration from a collection.
`CHROMA_DIR` and its contents must therefore remain operator-controlled; do not
point the application at an untrusted Chroma database or import an untrusted
Chroma volume. Remote `HttpClient`/Chroma-server deployments or externally
supplied databases are outside this assessment and require a new review.

The scheduled/manual CI dependency audit excludes exactly those four advisory
IDs via `pip-audit --ignore-vuln`, following the deployment assessment above.
This is a deployment-specific exception for the reviewed `chromadb==1.5.9`
pin, not an upstream patch or a claim that the package has no vulnerabilities.
Dependency resolution and auditing of all other findings remain enabled; the
job does not use `continue-on-error`. Regression tests bind the exception list
to the reviewed pin, the application's persistent embedded Rust client, the
absence of Chroma collection HTTP endpoints, and uploads remaining document
text rather than executable embedding configuration. Any Chroma upgrade, remote
client/server deployment, or changed advisory must trigger a new review of
these exceptions. Remove an exception when a patched version is adopted.

## Known boundaries

- Tool schemas constrain what the model is asked to emit. Required arguments,
  undeclared top-level keys in a closed schema (`additionalProperties: false`),
  and values incompatible with an explicit handler signature are rejected before
  a handler starts. Open schemas may pass extension keys, and not every value has
  an independent Pydantic type-validation layer.
- `ToolMetadata.permission` is descriptive metadata; it is not an authorization
  system.
- Request identity comes from the server, not from the caller. `user_id` used to
  be an ordinary body, query or path field, so any caller could name any user and
  read that user's documents, profile and wrong-question bank.
  `services/auth.py` is now the single source: with `STUDYLOOP_AUTH_TOKEN` set,
  every business endpoint requires `Authorization: Bearer <token>` and the subject
  is derived from it; without it the deployment runs in anonymous single-user
  mode, which `/health/live` reports as `auth: "anonymous"` so the mode is visible
  without guessing a token first. The gate is attached at `include_router`, so a
  newly added endpoint sits inside it by default, and
  `tests/test_auth_subject.py` pins the exemption list to `/`, `/health/live` and
  `/health/ready`. `/health/providers` is gated because it makes real provider
  calls. Documents are additionally scoped by owner in storage, so identity and
  data separation do not depend on the same check.
  Boundaries: a shared token authenticates the deployment, not a person. It does
  not separate two humans using the same instance, anyone holding it is the single
  user, and there is no rotation, revocation or rate limiting on it. A token short
  enough to guess makes the process refuse to start rather than pretend to be
  protected.
- Retrieved document text is untrusted input. `search_document` returns it inside
  an envelope marked `content_trust: untrusted_document_text`, and a layer-1
  pattern scan sets `injection_flagged` when the passage contains
  instruction-shaped text. Detection deliberately does not block retrieval: the
  corpus is the user's own study material, and a document *about* prompt
  injection would match. Enforcement happens in the registry instead — once a run
  observes flagged content, every non-idempotent or unknown-effect tool in that
  run is rejected with `untrusted_content_taint`, while read-only and idempotent
  tools keep working. The ReAct system prompt also states that retrieved passages
  are data, never instructions.
  Boundaries: the scan is regex-only and can be evaded by rephrasing, so it
  reduces blast radius rather than preventing injection; the taint is scoped to a
  single continuous dispatch run and is not carried across a human-in-the-loop
  pause, so content retrieved before a pause does not re-taint the resumed run
  unless it is retrieved again. The durable protection against cross-user writes
  remains `ToolMetadata.owner_argument`, which fails closed when no trusted user
  context is present.
- Outbound web access is a separate capability with its own boundary, available
  only inside a knowledge-base run and only when the deployment enables both
  `KNOWLEDGE_BASES_ENABLED` and `MCP_LIVE_ENABLED`. The remote DuckDuckGo MCP
  tools are registered in an isolated registry (`services/mcp_client.py`), not
  the global one, so they never appear in `get_tool_definitions`,
  `allowed_tool_names`, `replay_safe_tool_names` or any other list the model
  chooses from. Reaching the public web goes through
  `services/knowledge_web.py`, which owns the security contract the remote
  server does not provide: scheme and port validation, rejection of
  non-public addresses, a fresh resolution pinned to the connection so a
  rebind cannot follow the check, per-hop revalidation across at most three
  redirects, a content-type allowlist, bounded decompression, and a total
  deadline. `search_web` takes no arguments -- the query is fixed to the user's
  own words -- and `fetch_web` accepts only a URL the user supplied or a search
  already returned, so retrieved private text cannot become an outbound
  destination. Each run is further bounded by
  `MAX_WEB_SEARCHES_PER_RUN`/`MAX_WEB_FETCHES_PER_RUN`.
  Boundaries: address filtering relies on `ipaddress.is_global`, which admits
  NAT64-mapped and site-local forms, so a DNS64 deployment should add an
  explicit deny list; the fetch path has no process-wide concurrency limit; and
  this environment's DNS resolves public names into a reserved range, so live
  public retrieval has never been exercised end to end here.
- Search results are untrusted text and are scanned per row. A result whose
  title or snippet carries instruction-shaped prose is dropped and counted in
  `filtered_result_count`; the surviving rows stay usable and only they are
  authorized for `fetch_web`. This deliberately does not withdraw web access
  for the whole run, matching the rule above that read-only tools keep working:
  a learner asking what prompt injection is would otherwise disable their own
  web access, and `fetch_web`'s authorization list already denies any target a
  poisoned row might name. URLs are excluded from the scan because ordinary
  paths (`/docs/system-prompt-basics`, `/wiki/Jailbreak_(film)`) match the
  patterns without being an injection. A fetched page body that trips the scan
  still withdraws the outbound tools for that run, and because `ask_user`
  remains available as a control tool, a question asked after that point is
  prefixed with a visible notice that the material may have influenced it.
- Citations carry a provenance tier (`origin`): material the user uploaded,
  a web page this deployment ingested into the knowledge base, or a snapshot
  fetched during the current run. It records where a passage came from and how
  accountable that origin is; it is not a claim that the passage is true, and
  it is a separate axis from `source_status`, which says whether the source is
  still live. Persisted sessions written before the tier existed deserialize
  with it derived from the record's kind.
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
- In anonymous mode (no `STUDYLOOP_AUTH_TOKEN` configured) every caller shares
  the same default subject. A `conversation_id` is therefore a high-entropy
  bearer capability, not an authorization boundary, and deployments must be
  treated as single-user or placed behind authentication. The default Compose
  file binds the Web and API ports to loopback and does not publish PostgreSQL;
  preserve an equivalent boundary when adapting it. The ID is no longer logged
  in full.
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
