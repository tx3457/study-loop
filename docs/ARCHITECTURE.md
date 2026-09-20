# Architecture and execution boundaries

StudyLoop has several execution styles. They share services, but they are not
all Agents.

The simplified showcase diagram is maintained as
[`diagrams/architecture.mmd`](diagrams/architecture.mmd) and exported to
[`assets/architecture.svg`](assets/architecture.svg). The sections below are the
source of truth for the detailed execution and deployment boundaries.

## Main learning loop

```text
upload material
      |
      v
parse -> chunk -> Chroma vector index + BM25 corpus
      |
      v
retrieve -> generate exercise -> answer -> grade
                                      |
                                      v
                             update learner memory
                                      |
                                      v
                         choose the next learning action
```

The learner's grade and profile are observations for the next adaptive
decision. The action executor is still code-defined, so this is an agentic
workflow rather than an unconstrained tool Agent.

## Execution-style matrix

| Surface | Actual control flow | State and stop condition | Accurate label |
| --- | --- | --- | --- |
| `/agent/autonomous` | The model chooses a registered business tool, `ask_user`, or `finalize`; tool output returns as an observation before the next decision | PostgreSQL/SQLite pause snapshots with guarded resume; at most 8 rounds | Tool-using Agent |
| `/chat/tools` | The same tool loop without a session: the model chooses a registered business tool and answers from the observations | No server session; at most 3 rounds | Bounded tool-using endpoint |
| `/agent/tutor/assist` (Lab) | The embedded assistant uses the same tool loop; `ask_user` pauses through LangGraph `interrupt` | SQLite checkpointer + `thread_id`; at most 8 assistant rounds | Experimental tool-using Agent with HITL |
| `/agent/adaptive/*` | The model selects a structured teaching action; application code executes a known branch | Adaptive session state, mastery/round stop rules | Agentic workflow |
| `/agent/tutor/start` and `/agent/tutor/submit` | A supervisor node reasons over the full observation and dispatches the next worker with `Command(goto=...)`; workers flow back to it for the next decision | SQLite checkpointer + `thread_id`; `interrupt` per answer turn, at most 8 handoffs | Supervisor-based Multi-Agent with HITL |
| `/agent/run`, `/agent/stream` and quiz/critic/reviser graphs | Node order and retry routes are encoded by the developer | Graph state and bounded revision counts | Predefined LangGraph workflow |
| `/chat/stream` | No decision loop: a single model call streamed back to the caller | No server session; ends when the stream closes | Streaming chat completion |
| `/chat/history` | No decision loop; earlier turns are replayed as context and the model summarises the older ones once they pass a threshold | In-process bounded LRU: lost on restart and not shared across replicas | Multi-turn chat completion |
| `/generate/quiz/native` | No decision loop: one structured generation outside the learning path | No server session; ends with the returned quiz | Structured generation |
| document retrieval and learner memory | No autonomous decision loop | Chroma/BM25 and Store-compatible memory | RAG / application memory |

The optional supervisor graph is experimental: it has no Web surface and is not
registered in FastAPI unless `MAS_SUPERVISOR_ENABLED=true` at process startup.
The `/agent/tutor/*` routes are a Lab surface for architecture experiments, not
part of the product surface.

`/chat/*`, `/generate/quiz/native`, `/agent/run` and `/agent/stream` are
registered by default and sit behind the authentication gate, but none of them
has a Web surface; they are called directly. `/eval/ab` is the same kind of
operator entry point and spends real model credits on every call, so it is
documented separately in [Evaluation](EVALUATION.md).

## Tool loop

```text
user task + current state
          |
          v
    model tool choice
          |
          v
JSON parse -> allowlist -> timeout/retry -> real handler
          |                                  |
          +---------- tool observation <-----+
          |
          +---- next model decision / ask_user / finalize
```

`services/tool_registry.py` owns registered handlers, JSON schemas exposed to
the model, per-tool timeout/retry settings, and local audit records.
`services/tool_loop.py` parses tool calls, executes allowed tools, and appends
their results as tool messages. The current schema is not a complete server-side
authorization or Pydantic-validation boundary; see `SECURITY.md`.

`/chat/tools` is the smallest entry point into this loop: one request, no server
session, at most three rounds. `/agent/autonomous` runs the same
`run_tool_round` and adds the control tools, the pause snapshot, and the
eight-round budget on top of it.

## Persistence boundaries

- `services/checkpoint.py` persists per-thread graph execution state for
  interrupt/resume.
- `services/memory.py` stores learner profiles, mastery, weak points, preferences,
  and session events across conversations.
- `services/memory_persist.py` is a local JSON snapshot fallback when PostgreSQL
  is not configured. This fallback is intentionally single-worker; multi-worker
  memory storage must configure PostgreSQL rather than share one snapshot file.
  That removes only the learner-memory fallback constraint, not the application
  topology limits described below.
- `services/quiz_sessions.py` persists stable Web Quiz and wrong-question
  practice sessions. SQLite supports local restart recovery; PostgreSQL adds
  cross-worker claims and fencing.
- `services/adaptive_sessions.py` persists the complete private Adaptive
  aggregate, including partial grades, memory-write markers, decisions, and
  replay artifacts. Its browser projection excludes answers, explanations, and
  source chunks. A terminal `switch_to_plan` artifact acts as a durable outbox:
  only after that terminal aggregate is committed is its immutable plan
  idempotently published to the Learning Path store, and start/submit replay or
  snapshot reads repair a lost publication acknowledgement. SQLite supports
  local restart recovery and PostgreSQL adds cross-worker claims, revision CAS,
  and fencing.
- Tutor remains an opt-in experimental Lab with its own narrower workflow-state
  boundary; it must not inherit Quiz or Adaptive durability claims by analogy.

Checkpoint state and learner memory solve different problems and should not be
described as one generic “memory” feature.

## Deployment topology

The bundled Docker/Compose topology intentionally runs one backend worker and
one backend replica. Document retrieval uses an embedded Chroma
`PersistentClient` over a local volume; that data plane must not be shared by
multiple backend processes. The image pins `--workers 1`, and Compose overrides
`WEB_CONCURRENCY` to `1` even if a local `.env` contains another value.

PostgreSQL makes receipts, durable sessions, Learning Paths, and learner memory
capable of cross-process coordination. It does **not** make the embedded Chroma
client multi-process safe. A future multi-worker/replica deployment must first
move Chroma behind a separate server and use `HttpClient`, then keep PostgreSQL
for the application state stores. Those changes are necessary but not
sufficient: the in-process BM25 cache invalidation and document-ingest staging
coordination would also need cross-process protocols. That externalized
topology is not shipped by this repository today.

## Optional integrations

The optional knowledge-base service adds a separate LightRAG data plane, described in
[LIGHTRAG.md](LIGHTRAG.md). It owns source versions, canonical entity/edge identities,
correction replay and a PostgreSQL-backed write queue. The existing document-centric
learning path, Quiz and Adaptive routes retain their original scope.

When Autonomous selects `knowledge_base_id`, it uses a request-local read-only tool registry
and source citations. Its snapshot pins the knowledge-base revision/epoch and permitted web
policy. Legacy request serialization and global tool fingerprints remain compatible. Both
citation systems validate observed source identity and scope; neither proves semantic
entailment for every generated statement.

- Query rewriting, HyDE, multi-query retrieval, and cross-encoder reranking are
  feature-flagged and disabled in `.env.example` for a lightweight first run.
- LangSmith, Langfuse, and live MCP integrations are opt-in and disabled by
  default.
- The Docker backend includes Poppler and Tesseract OCR dependencies. OCR
  quality still depends on input quality and installed language packs.

## Framework references

The terminology and persistence boundaries in this document follow the
official LangGraph documentation:

- [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [Interrupts and resume](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Persistence and checkpointers](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Long-term memory stores](https://docs.langchain.com/oss/python/langgraph/add-memory)
