# Public datasets

A smoke-test fixture: something real to run on day one, and nothing more.

Your own data goes in `datasets/private/`, which is gitignored. That is the
intended path for real use, and it is the only path on which plumbline's
recalibration numbers mean anything.

## jevbench-hard.jsonl

The 111 public rows of the JevBench hard tier, vendored unchanged.

- Source: <https://github.com/fstandhartinger/jevbench>, `datasets/public/hard.jsonl`
- License: MIT, Copyright (c) 2026 Florian Standhartinger and contributors. The
  full license text is in [LICENSE-jevbench](LICENSE-jevbench) beside the file.
- JevBench is Benchmark Heaven's benchmark. It is not affiliated with plumbline,
  and plumbline is not affiliated with it.

### These numbers are not JevBench's numbers

plumbline is not reproducing this benchmark, and a figure produced here must
never be compared with a figure JevBench publishes. The harness differs, the
prompts differ, and the scoring differs:

- JevBench asks each row through its own harness. plumbline composes a case text
  from the row's `question.instructions` above its `state` and sends that through
  whichever adapter is under test, with that adapter's own prompt shape.
- JevBench's rows come in three question types — `choice`, `noul`, and `score`.
  plumbline asks each row as the type it declares, where the transport has one:
  a `noul` row is asked as a Noul by `typesafe_wire`, and as a two-option choice
  by a transport with no Noul. Those are different questions, so every record
  carries both what the row asks and how it was asked, and the report keeps them
  apart rather than averaging across the difference. The six `score` rows are
  loaded, marked, and excluded from every figure: v0.1 has no ordinal support,
  and flattening ordered levels into unordered options discards the ordering.
- JevBench's score combines intelligence, calibration, speed, and cost into one
  number. plumbline computes its own metrics, against its own calibrated-null
  floor, and deliberately publishes no combined score.
- Six rows write the gold label as a JSON number against string options; the
  loader matches `1` to `"1"` and reports how many rows needed it.
- Thirty-eight rows carry criteria that describe the statement rather than each
  option, so their option descriptions are dropped rather than guessed at.

`plumbline.datasets.load_jevbench` states the same thing in its docstring, and
every load report repeats it in its notes, so the caveat travels with the data
rather than living only here.

### Running it

```
python examples/smoke_public_dataset.py
```

Loads this file, runs the mock adapter over it, computes the metrics, and writes
an artifact. It makes no network call and spends nothing. The numbers are
meaningless — a seeded mock is answering — and the point is to prove the loader,
the runner, the metrics, and the artifact compose on a real file before a live
run turns a mistake into money.
