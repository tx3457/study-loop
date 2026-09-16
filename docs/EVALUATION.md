# Evaluation

`POST /eval/ab` runs one A/B experiment over a document already in the
retrieval index and scores both arms with an LLM judge. It is an operator
tool: it sits behind the authentication gate, has no Web surface, and
**spends real model credits on every call**.

## What the experiment compares

`experiment` selects the dimension under test:

| Value | Baseline | Treatment |
| --- | --- | --- |
| `ce` | generation without the learner context | generation with `difficulty_score` and `weak_points` |
| `rag` | dense retrieval only | hybrid retrieval (BM25 + vector, fused with RRF) |

One run retrieves the material once, generates both question sets
concurrently, judges every question concurrently, and aggregates the two
sides into a single `ABResult`.

## What the judge scores

Each question is scored on `relevance` (1-5), `clarity` (1-5),
`difficulty_feel`, `covers_weak_point` with the concrete `matched_point`,
and `faithfulness` — whether the question and its answer stay within the
source text instead of inventing content. The judge also returns a
one-sentence `reasoning` for its verdict.

## Failed judge calls do not become zeros

A judge call that fails is recorded with `status: "error"` and its
`error_type`, and is then **excluded from the quality metrics** rather
than scored as zero. Silently counting failures as zero is the usual way
an evaluation harness reports a regression that never happened.

The counts stay visible, so a run whose judge was mostly failing is
recognisable instead of merely looking bad:

- `total_count` — questions submitted to the judge
- `valid_count` — verdicts that came back usable
- `failed_count` — judge calls that errored
- `judge_success_rate` — `valid_count / total_count`

Aggregate quality is reported as `weak_point_coverage`, `avg_relevance`,
`avg_clarity`, `faithfulness_rate`, and a `difficulty_dist` histogram.

## Running one

The document must already be ingested, and the request needs the same
authentication as any other business route.

```bash
curl -X POST http://localhost:8001/eval/ab \
  -H 'Content-Type: application/json' \
  -d '{
    "document_id": "sample_document.md",
    "query": "反向传播",
    "experiment": "ce",
    "weak_points": ["链式法则", "梯度消失"],
    "difficulty_score": 0.65
  }'
```

`count` (default 5) sets the questions per arm, so a single call issues
roughly `2 × count` generation requests plus `2 × count` judge requests.
Start small.

Judge calls are wrapped with `@traceable`, so a configured LangSmith or
Langfuse endpoint records each verdict individually. Tracing is disabled
in the example configuration; see [`CONFIGURATION.md`](CONFIGURATION.md).
