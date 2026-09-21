# plumbline

Choose and configure a decision model on your own labeled data.

plumbline is not a leaderboard. It answers three questions no public ranking
can: how each candidate behaves on your labeled data, what its reported
probability becomes after recalibration fitted on that data, and what a cascade
at the resulting threshold actually costs you.

## Try it without spending anything

```
python examples/smoke_public_dataset.py
```

Loads the public fixture, runs the deterministic mock over it, computes the
metrics, and writes a results artifact. No network call, no key, no cost.

The fixture is the 111 public rows of the JevBench hard tier, vendored under MIT
and attributed in [datasets/public/README.md](datasets/public/README.md). It is a
smoke test, not a reproduction: plumbline's harness, prompts and scoring all
differ from JevBench's, so a number produced here must never be compared with a
number JevBench publishes. Your own data goes in `datasets/private/`, which is
gitignored, and that is the only path on which the recalibration numbers mean
anything.

## The command line

```
plumbline run datasets/public/jevbench-hard.jsonl --format jevbench --results results
plumbline report results/<artifact>.json --out report.md \
    --escalation-cost 0.02 --error-cost 1.00
plumbline adapters
```

`run` loads a dataset, runs one adapter over it, writes the artifact, and renders
the report. `report` renders a document from runs that already happened, so a
finished run is never repeated to get a write-up out of it.

The two cost flags are the two numbers no benchmark can know: what one escalation
to a more expensive system costs you, and what one wrong answer costs you.
Supply both and the report ends in the sentence the tool exists to produce -- at
this threshold, this much traffic stays on the cheap arm, at this expected cost,
versus this much if everything escalated. Leave them out and that section says
so rather than inventing them.

## Development

- [docs/PLAN.md](docs/PLAN.md) — remaining phases, decisions already made, and
  open questions.
- [METHODOLOGY.md](METHODOLOGY.md) — how each number is computed and what it does
  not mean.

```
uv run pytest        # tests
uv run ruff check .  # lint
uv run mypy          # types
```
