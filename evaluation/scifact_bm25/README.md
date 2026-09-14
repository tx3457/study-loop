# SciFact BM25 regression evidence

This directory publishes the evidence behind StudyLoop's BM25 tokenizer
regression.  It is intentionally narrower than an end-to-end RAG or Agent
evaluation.

## What is checked in

- `per_query_metrics.jsonl`: 300 SciFact test-query rows containing only query
  IDs and per-arm metrics.  Query text, document text, rankings, local paths,
  model output, and credentials are excluded.
- `manifest.json`: dataset/source hashes, aggregate results, paired-bootstrap
  settings, and claim boundaries.
- `verify.py`: standard-library verification of artifact hashes, source
  bindings, aggregates, and 10,000-sample paired-bootstrap intervals.
- `run.py`: optional full recomputation using user-supplied SciFact files and
  StudyLoop's checked-in `services/tokenization.py` / `services/bm25.py`.

## Verify the published artifact offline

From the repository root:

```bash
python evaluation/scifact_bm25/verify.py
```

This command does **not** download a dataset, call a model, read `.env`, or
recompute BM25 rankings.  It verifies that the published query-level metrics
match the manifest and that the manifest is bound to the current business
source files.

Source binding has two tiers, described by `studyloop_source.binding_policy`:

- **Hash-pinned** (`files`): `services/tokenization.py` and `services/bm25.py`.
  `run.py` reproduces the evaluated ranking from these bytes, so any change to
  them invalidates the published numbers and verification fails.
- **Invariant-bound** (`invariant_bound_files`): `services/vectorstore.py`.
  `run.py` never imports this module, so its contents cannot move a metric. It
  is recorded to evidence that production retrieval uses the same pure BM25
  helpers, and that property is asserted behaviourally rather than by a digest.
  Pinning its digest previously bound a frozen experiment to a file under active
  feature development, so unrelated changes broke verification without saying
  anything about the measurement.

The manifest declares which invariants each invariant-bound file must satisfy
and `verify.py` implements them. A declared invariant with no implementation, an
empty invariant list, or a file listed in both tiers all fail verification, so a
file can never be recorded as bound while going unchecked.

## Full recomputation with user-supplied data

The repository does not redistribute the SciFact archive or extracted corpus.
Obtain SciFact from the upstream BEIR source recorded in `manifest.json`, review
the upstream license/terms, and extract this layout:

```text
scifact/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    └── test.tsv
```

Then run:

```bash
python evaluation/scifact_bm25/run.py \
  --data-dir /path/to/scifact \
  --zip-path /path/to/scifact.zip \
  --output /tmp/studyloop-scifact-results.jsonl

python evaluation/scifact_bm25/verify.py \
  --results /tmp/studyloop-scifact-results.jsonl
```

`run.py` performs no network access.  It verifies the supplied file hashes
against the manifest before evaluation and refuses to overwrite the checked-in
artifact.  Omitting `--zip-path` is allowed when only the extracted files are
retained; the three extracted-file SHA-256 values are still mandatory.

## Exact claim boundary

The comparison uses the same 5,183 English SciFact title+abstract documents,
300 test queries, 339 qrel pairs, BM25 implementation, top-k, and tie-break.
Only tokenization changes:

- baseline: the former `list(text)` character behavior;
- current: `nfkc_casefold_ascii_words_cjk_unigram_bigram_v1` from the business
  source.

The result supports an English **BM25 retrieval-component** regression claim.
It does not measure Chinese retrieval, Dense/RRF/CrossEncoder uplift, generated
answer quality, citation accuracy, Agent task success, latency, cost, or user
impact.
