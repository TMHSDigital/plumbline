# Plan

The build's memory, not a spec. What is left, what was already decided, and what
is still unknown. Kept short on purpose: a cleared context should be able to read
this page and pick up where the work stopped.

## Remaining phases

- **Phase 6 — adapters: `local_logits` and `generative`.** Landed. The restricted
  softmax over a pinned checkpoint, and the text-generating control arm.
- **Phase 7 — datasets.** Landed. The JSONL loader that refuses unscoreable
  rows, the JevBench public fixture and its translation, and the end-to-end
  smoke run in `examples/smoke_public_dataset.py`.
- **Phase 8 — report and CLI.** Landed. The markdown report groups arms by
  `probability_semantics`, states every figure's row count and null, and demotes
  MCE to diagnostics. `src/plumbline/cli.py` exists, so the `plumbline` console
  script pyproject declares now works: `run`, `report`, `adapters`, `version`.
- **Phase 9 — what the report still does not show.** Recalibration (fit a
  temperature on a held-out split and report what it did not fix) and the
  cascade (threshold sweep, cost at the chosen threshold) are implemented in
  `metrics/` and have no section yet. The METHODOLOGY sections still marked
  pending land with them.

## v0.2

- Ordinal score questions. The six score rows in the public fixture are loaded,
  marked and excluded; scoring them needs rank-aware metrics, because every
  metric here treats wrong-by-one and wrong-by-three identically.
- Adapters do not receive `label_descriptions`, so a dataset's per-option
  criteria never reach the wire. A Choice takes them directly and a Noul takes
  true/false descriptions.

## Decisions

Settled during the build. Reopen one only with a reason, not from scratch.

- ECE and MCE are always reported against their calibrated-null floor.
- MCE is demoted to a diagnostics block, never beside ECE, because it cannot
  detect gross overconfidence at 500 rows.
- Recalibration has three verdicts — recommended, partial, refused — and emits no
  temperature on refusal.
- Latency percentiles use nearest rank, not interpolation.
- Adapters are organized by transport; a new vendor is config, not code.
- plumbline is not a leaderboard. JevBench is.
- Every pricing entry carries its source and the date it was read; an entry older
  than `DEFAULT_PRICING_MAX_AGE_DAYS` is used but flagged rather than presented as
  current.
- A blank cost column names its reason (`CostBasis`): an adapter that reports no
  tokens at all is a different finding from an API that reported none on this run.
- `local_logits` is fixed at `restricted_softmax` and `generative` at `none`;
  neither is configurable, because the report groups on that field.
- A row whose gold label is not one of its options is refused by the loader,
  never scored: it would mark every system wrong and read as a model failure.
- Every calibration figure carries its row count, and the artifact records
  `dataset_rows` beside `dataset_hash`, because the floor depends on n.
- The JevBench public rows are vendored as a fixture under MIT with attribution.
  They are asked as one-of-n choices through plumbline's own harness, so results
  from them are never comparable with JevBench's published numbers.
- A yes/no row is asked as a Noul where the transport has one, and every record
  carries both what the row asks and how it was asked. A noul figure is never
  compared with a two-option-choice figure without that line between them.
- Ordinal score rows are loaded, marked and excluded from every figure in v0.1.
- Artifacts never overwrite each other: the timestamp is only accurate to the
  second, so a repeated name gets a suffix rather than replacing user records.
- `generative` configures no fallback model. This is deliberate and deviates
  from the SDK's default advice for this model family: server-side fallback
  would re-run a declined case on another model inside the same call, so one
  dataset would be answered partly by two models and the run would no longer be
  about the system under test. A decline is recorded as a refusal with its
  category and counted. The tradeoff is stated at the call site in
  `adapters/generative.py`; it is not an oversight.

## Open questions

- Does `response.usage` actually populate token counts on a live Jev call? The
  wire schema marks both counts required, the SDK widens them to `int | None`.
  There is a TODO in `adapters/typesafe_wire.py` for whoever makes the first live
  call: record the answer, the model, and the date here.
- Are Jev output tokens billed at zero? The only source is the wire schema's
  field description, "Output tokens are currently stated at https://docs.typesafe.ai/models", read
  2026-09-20. No input price is published anywhere, so the shipped entry prices
  nothing and cost is reported as not available.
