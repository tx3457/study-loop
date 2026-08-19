# Architecture and execution boundaries

StudyLoop has several execution styles. They share services, but they are not
all Agents.

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
| `/agent/tutor/assist` (Lab) | The embedded assistant uses the same tool loop; `ask_user` pauses through LangGraph `interrupt` | SQLite checkpointer + `thread_id`; at most 8 assistant rounds | Experimental tool-using Agent with HITL |
| `/agent/adaptive/*` | The model selects a structured teaching action; application code executes a known branch | Adaptive session state, mastery/round stop rules | Agentic workflow |
| `/agent/run` and quiz/critic/reviser graphs | Node order and retry routes are encoded by the developer | Graph state and bounded revision counts | Predefined LangGraph workflow |
| document retrieval and learner memory | No autonomous decision loop | Chroma/BM25 and Store-compatible memory | RAG / application memory |

The optional supervisor graph is experimental: it has no Web surface, is not
registered in FastAPI unless `MAS_SUPERVISOR_ENABLED=true` at process startup,
and contains an unfinished pure-tutor worker. The `/agent/tutor/*` routes are a
Lab surface for architecture experiments, not a headline product or resume
claim.

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

## Persistence boundaries

- `services/checkpoint.py` persists per-thread graph execution state for
  interrupt/resume.
- `services/memory.py` stores learner profiles, mastery, weak points, preferences,
  and session events across conversations.
- `services/memory_persist.py` is a local JSON snapshot fallback when PostgreSQL
  is not configured. This fallback is intentionally single-worker; multi-worker
  deployments must configure PostgreSQL rather than share one snapshot file.

Checkpoint state and learner memory solve different problems and should not be
described as one generic “memory” feature.

## Optional integrations

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
