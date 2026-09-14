# Evaluation policy

## What the public repository verifies

The default test suite uses mocks, fake policies, and local state. It verifies
control-flow contracts such as:

- tool allowlisting, execution, observation reinjection, timeout, retry, and
  audit recording;
- bounded multi-round Agent loops and explicit finalization;
- LangGraph interrupt/resume and checkpointer behavior;
- adaptive-action routing and termination rules;
- retrieval fusion/query-rewrite helpers without paid model calls;
- prompt-injection and output-leak detectors;
- FastAPI routing and frontend build/lint integrity.

`scripts/demo_react_tutor_agent.py` is deterministic. It exercises the real
assistant and ToolRegistry path with a scripted model policy and fake handlers;
it is not evidence of real-model task success.

## Reproducible retrieval regression

The business BM25 path now uses the same versioned, language-aware heuristic
tokenizer for documents and queries (`BM25_TOKENIZER_ID` in
`services/tokenization.py`). It normalizes text with NFKC and case folding,
keeps ASCII words/numbers as tokens, and emits CJK unigrams plus adjacent
bigrams. This is intentionally described as a heuristic tokenizer rather than
a general Chinese word segmenter.

The checked-in evidence under [`evaluation/scifact_bm25/`](../evaluation/scifact_bm25/)
contains a sanitized 300-query metric artifact, a manifest with dataset/source
bindings, and a standard-library verifier.  The evaluated ranking path
(`services/tokenization.py`, `services/bm25.py`) is hash-pinned;
`services/vectorstore.py` is bound by behavioural invariants instead, because
the runner never imports it and pinning its digest only coupled a frozen
experiment to unrelated feature work.  The verifier requires neither the
dataset nor a model and recomputes the aggregate metrics and paired bootstrap
interval from the published query-level rows:

```bash
python evaluation/scifact_bm25/verify.py
```

The original offline regression ran the exact business tokenizer on the public
BEIR SciFact test split (5,183 documents, 300 queries). With all other BM25
settings held fixed, the pre-fix character baseline versus the current
tokenizer produced:

| Metric | Character baseline | Current tokenizer |
| --- | ---: | ---: |
| Recall@10 | 0.1286 | 0.7757 |
| MRR@10 | 0.0734 | 0.6184 |
| nDCG@10 | 0.0852 | 0.6523 |

The public artifact records per-query metrics, source hashes, corpus hashes, a
fixed bootstrap seed, and a paired 95% confidence interval for the nDCG@10
delta ([0.5197, 0.6141]).  It excludes query/document text, local paths, model
artifacts, credentials, and the SciFact dataset itself.

A full recomputation is also provided, but it deliberately does not download
or redistribute SciFact.  The user must obtain the dataset from the upstream
BEIR source, review its current license/terms, and provide the extracted files:

```bash
python evaluation/scifact_bm25/run.py \
  --data-dir /path/to/scifact \
  --zip-path /path/to/scifact.zip \
  --output /tmp/studyloop-scifact-results.jsonl
python evaluation/scifact_bm25/verify.py \
  --results /tmp/studyloop-scifact-results.jsonl
```

A fresh clone can therefore verify the published artifact immediately, but
cannot honestly claim a fresh full-dataset rerun until the upstream dataset is
supplied. These numbers measure English BM25 retrieval only. They do not
establish Chinese retrieval quality, hybrid/RRF uplift, answer quality, Agent
task success, production latency, or user impact.

## Citation contract and remaining measurement boundary

The autonomous endpoint now registers retrieved chunk IDs server-side, binds
document-scoped tool calls before dispatch, expands snippets only from that
registry, and fails closed to an explicit abstention when grounding is required
and citations are missing or any submitted ID is invalid. Unit tests cover
mixed valid/forged IDs, no-citation abstention, document-scope violations, and
evidence continuity across human-in-the-loop resume.

This contract is implementation evidence, not a citation-accuracy result. A
publishable percentage still requires a versioned answerable/unanswerable set,
claim-support labels, multiple real-model runs, and a deterministic scoring
artifact.

## What is deliberately not claimed

This public snapshot does not publish a real-model task completion rate,
citation-accuracy percentage, real-provider latency, or token cost. Earlier
local experiments depended on private vector collections and
interview-question data whose redistribution provenance was unclear, so their
datasets, outputs, logs, and aggregate reports were excluded rather than
presented as reproducible evidence.

Unit-test pass counts must not be described as Agent success rates.

## Requirements for a future end-to-end benchmark

A publishable evaluation should use a redistributable corpus and versioned task
set, pin provider/model settings, store per-case trajectories without secrets,
and report at least:

- retrieval Recall@K/MRR for each retrieval arm and citation support rate;
- end-to-end task completion and first-tool accuracy;
- average/P95 steps and latency, tool failure rate, and truncation rate;
- token usage/cost when the provider exposes usage;
- baseline and ablation definitions plus failure examples.

Results should be generated by a documented command in CI or a reproducible
offline job, not copied from an unavailable local knowledge base.
