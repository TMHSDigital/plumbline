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
