# Plan

The build's memory, not a spec. What is left, what was already decided, and what
is still unknown. Kept short on purpose: a cleared context should be able to read
this page and pick up where the work stopped.

## Remaining phases

- **Phase 6 — adapters: `local_logits` and `generative`.** Landed. The restricted
  softmax over a pinned checkpoint, and the text-generating control arm.
- **Phase 7 — datasets.** The JevBench public loader (`datasets/public/`), the
  private JSONL loader, and the dataset hash that ties a result to its rows.
- **Phase 8 — CLI.** `pyproject.toml` declares `plumbline = "plumbline.cli:app"`
  against a module that does not exist, so a fresh install ships a broken entry
  point: `plumbline` on the path fails at import. Phase 8 either writes
  `src/plumbline/cli.py` or removes the `[project.scripts]` entry. Until then,
  the library imports fine and only the console script is broken.
- **Phase 9 — the report.** Grouping by `probability_semantics`, the refusal to
  compare across groups, the cost and staleness lines, and the METHODOLOGY
  sections currently marked pending.

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
