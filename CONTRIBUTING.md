# Contributing

## New vendors are config, not code

This is the one thing to read before opening a pull request.

plumbline has three transports, and almost every "please support X" is already
one of them. Adding a vendor should not add a file.

- **Anything serving the Jev wire format is a `base_url` for `typesafe_wire`.**
  A self-hosted endpoint, a compatible server in front of an open model, a
  provider that implemented the same shape. It is a config entry. That includes
  the open decision models that serve this wire format: the unchanged adapter
  runs them.
- **Any open-weights checkpoint is a config entry for `local_logits`**, as a
  HuggingFace model id and a pinned revision. Pin the revision. A moving
  checkpoint makes every stored result unreproducible.
- **Any chat model you want as a control arm is a model string for
  `generative`.**

A new adapter module is for a genuinely new *transport*, which is rare. If you
are writing one, say in the issue what wire shape it speaks and why none of the
three fit. There is an issue template for exactly this, and it exists to route
the conversation toward config before anyone writes code.

If a vendor is nearly compatible and something small blocks it, that is a bug in
the adapter worth fixing, not a reason for a fourth one.

## The rule about figures

**No figure may be rendered without its n and its null.**

This is not a style preference. A calibration number without its sample size and
its floor is not a weaker version of a finding, it is a number that reads like a
finding and is not one, and the entire point of this tool is to not do that. A
pull request that prints a bare metric will be asked to change, however correct
the arithmetic is.

The same applies to refusals. When there is not enough data, print the verdict
and the reason, not a value with a caveat attached.

## Before you open a pull request

All three must pass:

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict
```

CI runs these on Python 3.12 and 3.13, on Ubuntu and Windows.

**Tests never need a network connection or an API key.** Every test must pass
without either. If a change needs a live call to be tested, the live call is
mocked and the real one goes in a manual check, recorded in `docs/PLAN.md`. A
test that reaches the network is a test that fails in CI for reasons unrelated
to the change.

## Commits

One concern per commit, conventional messages.

```
fix(adapters): request a model the API actually serves
feat(report): print the recalibration verdict and the cascade sentence
docs(methodology): write down what this build measured
```

The body should say why, not what. The diff already says what. If a commit is
fixing something that was wrong, say what the wrong behavior was, because that
is the part nobody can reconstruct later.

## Pricing entries

Every pricing entry carries the source it was read from and the date it was
read. Both are required, in shipped config and in a user-supplied table alike.
An undated price with no source reads in a report exactly like one checked this
morning.

Do not add a figure for a vendor whose terms treat pricing as confidential.
Those ship unpriced, naming the page to read, and the operator supplies the
numbers with `--pricing`. See `docs/pricing.example.json`.

Use `null`, never `0`, where a vendor publishes no price. `null` is the absence
of a claim; `0` is the claim that those tokens are free.

## Datasets

Never commit anything to `datasets/private/`. It is gitignored and it is where
real labeled data goes.

Run artifacts in `results/` are gitignored too. They contain per-case records
from your own data, and on a live run, from a vendor call. Do not paste one into
an issue without reading it first.
