# plumbline

[![CI](https://github.com/TMHSDigital/plumbline/actions/workflows/ci.yml/badge.svg)](https://github.com/TMHSDigital/plumbline/actions/workflows/ci.yml)

Measure whether a decision model's probabilities are trustworthy on your own
labeled data, and decide what to do about it.

Check where your own ECE sits against its floor, in the browser, with nothing
installed: **[tmhsdigital.github.io/plumbline](https://tmhsdigital.github.io/plumbline/)**.

> **v0.1.0, one maintainer.** The measurement behaviour is settled; the Python
> API and the CLI flags are not, and will change in v0.2. Pin a version if you
> build on it.

## What it is

A **decision model** is one you ask a closed question and get back a label plus
a number: "which of these five categories is this ticket" together with "0.83".
That number is the point. If it is honest, your code can act on 0.95 and route
0.6 to a human, and you have a system. If it is not, you have a confident
guess in a trench coat.

plumbline is a command line tool that takes **your** labeled rows, runs one or
more models over them, and tells you three things:

1. **Is the number honest?** A model is *calibrated* when the things it calls
   70% likely happen about 70% of the time. plumbline measures that, and
   crucially measures it against what the same figure would look like if the
   model were perfect, because on a few hundred rows those are closer than
   anyone expects.
2. **Where do I set my threshold?** Act automatically above it, escalate below
   it.
3. **What does that save me?** A **cascade** runs a cheap model first and sends
   only the uncertain cases to an expensive one. Given what an escalation costs
   you and what a mistake costs you, plumbline says where to cut and what you
   get for it.

It is a measuring instrument. It has no opinion about which model you should
pick, and it will refuse to answer a question your data cannot support.

## What this is not

plumbline is not a leaderboard. It does not rank vendors, it publishes no
combined score, and it will not tell you which model is best.

If you want a cross-vendor ranking of decision models, go to
[JevBench](https://github.com/fstandhartinger/jevbench) and
[Benchmark Heaven](https://benchmarkheaven.com/jev-models). That is their job and
they do it properly, across many models, on a shared dataset, with a published
methodology. plumbline deliberately does not compete with it, and a number
produced here must never be compared with a number they publish: different
harness, different prompts, different scoring.

plumbline answers a different question, and it is the question a ranking
structurally cannot answer:

- Does this model work on *your* labeled data, with your label set and your
  distribution?
- Where should you set your confidence threshold?
- What does a cascade at that threshold actually save you?

A leaderboard tells you how a model did on someone else's rows. Only your rows
can tell you whether its probabilities mean anything where you intend to use
them.

## The argument

The standard way to score calibration is **Expected Calibration Error**: sort
the predictions into bins by confidence, and in each bin compare the claimed
confidence against how often the model was actually right. Average the gaps.
Zero would be perfect.

Zero is not achievable, and that is the problem. With a finite number of rows,
each bin holds a handful of cases, and a handful of coin flips does not land
exactly on its own probability. That scatter, **binning noise**, puts a floor
under ECE that has nothing to do with the model. The floor rises as your row
count falls. **A perfectly calibrated model on a few hundred rows does not
score 0, and if you do not know what it would score, you cannot read your
own number.**

This is not a rounding concern. On a few hundred rows, a calibration claim is
frequently not measurable at all.

So plumbline computes that floor by simulation and prints every inferential
figure against it. Here is a real line from the example report:

> ECE 0.0740 over 105 rows (10 equal width bins), against a calibrated-model
> floor of 0.0707 (95th percentile 0.1109): INCONCLUSIVE at this sample size. A
> perfectly calibrated model would often score this badly on this many rows, so
> this dataset cannot tell the two apart. This is not a clean bill of health:
> nothing was established either way. Collect more rows to make the question
> answerable.

**Read "inconclusive" as an absence of a result, not a pass.** It means your
rows cannot tell your model apart from a perfect one, so nothing was
established in either direction. A genuinely well calibrated model and a badly
calibrated one both land there on too few rows, and the figure does not say
which you have. Taking it as a clean bill of health inverts the conclusion, and
it is the easiest mistake to make with this tool.

On that same 105-row run, three of the four headline figures came back
inconclusive and only accuracy cleared its null. A tool that printed the other
three alone would be handing you numbers that look like findings and are not.
That refusal is the product.

## Quickstart

**Prerequisites:** Python 3.12 or later, and [uv](https://docs.astral.sh/uv/).
An older Python gives a resolver error rather than a clear message, so check
with `python --version` first.

**plumbline is not on PyPI.** `pip install plumbline` will not work. Clone the
repository; that is the intended install path for v0.1.

Nothing in this first section needs an API key or spends anything.

<details open>
<summary><b>PowerShell</b></summary>

```powershell
git clone https://github.com/TMHSDigital/plumbline
cd plumbline
uv sync

uv run plumbline run datasets/public/jevbench-hard.jsonl `
    --adapter mock --format jevbench `
    --results results --report results/report.md
```

</details>

<details>
<summary><b>bash or zsh</b></summary>

```bash
git clone https://github.com/TMHSDigital/plumbline
cd plumbline
uv sync

uv run plumbline run datasets/public/jevbench-hard.jsonl \
    --adapter mock --format jevbench \
    --results results --report results/report.md
```

</details>

That loads the vendored public fixture, runs a deterministic seeded mock over it,
computes every metric against its null, and writes both a results artifact and a
report. No network call. Both land in `results/`, which is gitignored, so
following this leaves your clone clean.

Expected output shape:

```
111 rows read from datasets/public/jevbench-hard.jsonl, 111 loaded, 0 refused. ...
artifact: results/20260921T222053+0000-mock-c18e9496.json
report: results/report.md
```

Those lines go to stderr. Without `--report`, the report itself is the only
thing on stdout, so `plumbline run ... > report.md` captures just the report.
The exit code is 0 when the run produced figures and 1 when it could not start
(a bad option, a missing key, an unreadable file) or when every case failed; in
that last case the artifact and the report are still written, and the reason is
printed when all the cases share one.

```
uv run plumbline adapters   # what this install can run
uv run plumbline version
uv run plumbline run --help
```

### Running a local model, still without a key

`local_logits` reads option-token probabilities out of a checkpoint on your own
machine, so it needs no API key. It does need the optional `local` extra, which
a plain `uv sync` does not install:

```
uv sync --extra local
```

Without it every case fails with a message telling you this, so if a local run
reports no figures at all, that is the first thing to check.

### Running a hosted vendor

This one spends money. Set a key, name an adapter, and cap the run.

Cost needs a pricing table **you** supply, because plumbline ships no figures
for vendors whose terms treat pricing as confidential. Copy the template and
fill in the rates from the vendor's own page, and the date you read them.
plumbline refuses the copy until you have, so an unedited template can never
price a run at nothing:

```
cp docs/pricing.example.json my-pricing.json
```

<details open>
<summary><b>PowerShell</b></summary>

```powershell
$env:TYPESAFE_API_KEY = "your-key-here"

uv run plumbline run datasets/public/jevbench-hard.jsonl `
    --adapter typesafe_wire --model jev-latest --format jevbench `
    --results results --report results/report.md `
    --max-cases 40 --pricing my-pricing.json
```

</details>

<details>
<summary><b>bash or zsh</b></summary>

```bash
export TYPESAFE_API_KEY="your-key-here"

uv run plumbline run datasets/public/jevbench-hard.jsonl \
    --adapter typesafe_wire --model jev-latest --format jevbench \
    --results results --report results/report.md \
    --max-cases 40 --pricing my-pricing.json
```

</details>

Without a pricing table the run still works; cost reports as unpriced, and
`--max-cost-usd` refuses rather than bounding a run it cannot cost. See
[Limitations](#limitations).

Your own data goes in `datasets/private/`, which is gitignored, and that is the
only path on which the recalibration numbers mean anything.

## Example report

[docs/example-report.md](docs/example-report.md) is the output of one seeded mock
run, and CI holds it to that: on every change and before every deploy it reruns
the report's recorded command and refuses if any line differs, other than the
date. The ECE line is quoted in [The argument](#the-argument) above. Three more,
each showing the tool declining to do something:

> Not reported. recalibration needs at least 200 held-out evaluation rows and
> this split has 53. Fitting a temperature on fewer rows produces a number whose
> uncertainty is larger than the correction it claims to make, and it arrives
> looking like a measurement.

> Not reported. A threshold is decided by two numbers no benchmark can know:
> what one escalation to the expensive arm costs, and what one wrong answer
> costs.

> 38 noul rows were asked as choice questions, which is a different question from
> the one the dataset states. Not comparable with an arm that asked them as noul.

A **Noul** is a yes/no question that returns one probability directly, rather
than a distribution over options. Asking a yes/no row as a two-option choice is
a different question, so the report keeps the two apart instead of averaging
across the difference.

The arm in that report is the seeded mock, labeled as such at the top of the
page, so anyone can reproduce it with one command and no key. Its numbers are
properties of plumbline's harness, not a measurement of any vendor.

## Adapters and probability semantics

Three real transports, plus a mock for smoke tests. **The intent is that adding
a vendor is config rather than code**, and that intent has so far been verified
against one endpoint; demonstrating it against a second is
[issue #11](https://github.com/TMHSDigital/plumbline/issues/11).

The "Run for real" column is deliberate. Two of these have only ever run
against test fakes, which is [issue #3](https://github.com/TMHSDigital/plumbline/issues/3)
and the highest-value item in the backlog.

The **Jev wire format** below is one vendor's HTTP shape for typed questions:
you send some state plus a question with declared options, and you get back a
selected option, a probability for each, and a confidence. Several open models
and self-hosted servers speak it, which is why one adapter covers all of them.

| Adapter | Transport | Semantics | Run for real | Adding one |
|---|---|---|---|---|
| `typesafe_wire` | Jev wire format over HTTP | `calibrated_claim` | Yes, 40 rows against a hosted vendor | A `base_url`. Anything serving the same wire format is a config entry, including self-hosted endpoints and open models behind a compatible server. |
| `local_logits` | Option-token logits from a local checkpoint | `restricted_softmax` | **No, tests only** | A HuggingFace model id and a pinned revision. Needs the optional `local` extra. |
| `generative` | Chat completion, parsed | `none` | **No, tests only** | A model string. |
| `mock` | None, seeded | configurable | Yes, it is the example report | Built in. A deterministic stand-in, not a system under test. |

The semantics classes are the whole reason the report refuses some comparisons:

- **`calibrated_claim`.** The vendor asserts these probabilities are calibrated.
  plumbline records the claim and never asserts it is true. Testing it is the
  point.
- **`restricted_softmax`.** A softmax over the declared options only, with no
  calibration claim. Add an option and every number moves without anything about
  the case having changed, so it is a statement about your option set rather than
  a probability of correctness in the world. Whether it behaves like a calibrated
  probability on your workload is measurable, so plumbline measures it.
- **`none`.** No probability at all. Excluded from calibration entirely, never
  imputed, never defaulted to zero.

The report groups arms by this field and will not place figures from different
groups side by side.

## Limitations

Specific, and none of them are going to surprise you later.

- **Choice and Noul only.** Ordinal Score rows load, are marked, and are excluded
  from every figure. Flattening ordered levels into unordered options discards
  the ordering that makes them a score, so v0.1 declines rather than
  approximating.
- **One request per case, no batching.** Cost and latency figures are therefore
  conservative relative to batched use, where a single call carrying many
  questions against one shared state is materially cheaper and faster.
- **Temperature scaling only.** Per-label and vector scaling are not fitted. When
  the residual says temperature is the wrong correction, the tool refuses and
  emits no temperature rather than returning one that does not fit.
- **Recalibration needs 200 held-out rows.** Below that it refuses. Most datasets
  people try first will not reach it.
- **Cost requires a pricing table you supply.** plumbline ships no figures for
  vendors whose terms treat pricing as confidential. The shipped entry names the
  page to read, and you pass your own table with `--pricing`. Without one, cost
  reports as `model_not_priced` and `--max-cost-usd` refuses the run outright,
  because a guard cannot bound a run it cannot cost.
- **The METHODOLOGY numbers derived from the seeded mock are properties of the
  harness's resolution, not measurements of any vendor,** and are labeled as
  such wherever they appear.
- **Adapters reporting no distribution are excluded from multiclass Brier** and
  recalibrate materially worse, because a single scalar carries less to correct
  with.
- **Probabilities from a hosted API may arrive quantized.** That bounds the
  resolution of any threshold or bin computed from them. METHODOLOGY says what
  the bound is and where it bites.
- **Verified on Windows and Ubuntu, Python 3.12 and 3.13.** macOS is untested.
- **Two of the three transports have never run outside the test suite.** See
  the adapters table above and
  [issue #3](https://github.com/TMHSDigital/plumbline/issues/3).

## Related work

- **[JevBench](https://github.com/fstandhartinger/jevbench)** and
  [Benchmark Heaven](https://benchmarkheaven.com/jev-models). Cross-vendor
  ranking on a shared dataset, combining intelligence, calibration, speed and
  cost into one score. plumbline computes its own metrics against its own nulls
  on your data and publishes no combined score. Use theirs to shortlist, this to
  decide.
- **SemIf** (formerly OpenJev). Reads typed option probabilities straight out of
  an open model. Its README states the constraint plumbline's
  `restricted_softmax` class exists for: "Returned probabilities are conditional
  on the supplied options. Calibrate and validate them on the workload where they
  will make decisions." plumbline is a tool for doing exactly that. SemIf is an
  independent project and states it is "not affiliated with or endorsed by
  TypeSafe".
- **fastjev**, cited separately because it is a separate thing. An independently
  maintained fork of SemIf that preserves its history and MIT license, follows
  its own roadmap, and states plainly that it "does not serve Jev or reproduce
  Jev calibration".
- **[Mapika/decider](https://github.com/Mapika/decider)**. One-pass typed
  decisions with calibrated probabilities, fine-tuned from Qwen3.5-2B, in several
  sizes. Its own guidance is to check calibration on your own labels before
  routing on confidence. It serves the same typed question shape, so it is a
  `base_url` config entry here rather than new code.
- **[Bespoke Nimble](https://github.com/bespokelabsai/nimble)**. An open recipe
  for typed decision models, a LoRA fine-tune on Qwen3.5-9B, trained with
  contrastive data curation. It serves the Jev wire format, so the unchanged
  `typesafe_wire` adapter runs it.

The pattern across the last three: they are models, and plumbline is the
instrument you point at them. It has no opinion about which one you should pick.

## Documentation

- [METHODOLOGY.md](METHODOLOGY.md). How each number is computed, what it does not
  mean, and why each refusal is a refusal.
- [docs/PLAN.md](docs/PLAN.md). Decisions already made, what a live run measured,
  and what v0.2 is for.
- [docs/example-report.md](docs/example-report.md). Real output.
- [datasets/public/README.md](datasets/public/README.md). What the vendored
  fixture is, and why its numbers are not JevBench's numbers.
- [TypeSafe documentation](https://docs.typesafe.ai/). The wire format the
  `typesafe_wire` adapter speaks.

## Contributing

New vendors are config entries, not new adapter modules. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).

The vendored JevBench fixture in `datasets/public/` is MIT, Copyright (c) 2026
Florian Standhartinger and contributors, and is attributed in
[datasets/public/README.md](datasets/public/README.md) with the full license text
beside it in [LICENSE-jevbench](datasets/public/LICENSE-jevbench).
