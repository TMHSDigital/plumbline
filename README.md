# plumbline

[![CI](https://github.com/TMHSDigital/plumbline/actions/workflows/ci.yml/badge.svg)](https://github.com/TMHSDigital/plumbline/actions/workflows/ci.yml)

Measure whether a decision model's probabilities are trustworthy on your own
labeled data, and decide what to do about it.

## What this is not

plumbline is not a leaderboard. It does not rank vendors, it publishes no
combined score, and it will not tell you which model is best.

If you want a cross-vendor ranking of Jev-class decision models, go to
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

Expected Calibration Error has a floor that is not zero, and the floor depends on
how many rows you have. A perfectly calibrated model, measured on a few hundred
rows, does not score 0. It scores some positive number determined by binning
noise and sample size. If you do not know that number, you cannot read your own.

This is not a rounding concern. On a few hundred rows a calibration claim is
frequently not measurable at all.

So plumbline reports every figure against its own null, and says plainly when a
value is indistinguishable from a calibrated model. From the example report, 105
rows:

- ECE 0.0740, against a calibrated-model floor of 0.0707 and a 95th percentile of
  0.1109. Not distinguishable.
- Brier 0.1711, against a floor of 0.1489. Not distinguishable.
- Confidence AUROC 0.6034, against a permutation null of 0.4997 and a 95th
  percentile of 0.6116. Not distinguishable.
- Accuracy 0.7714, against a chance null of 0.3416. Better than chance.

Four figures, one of which supports a conclusion. A tool that printed the first
three on their own would be handing you numbers that look like findings and are
not. That refusal is the product.

## Quickstart

PowerShell. Nothing below needs an API key or spends anything.

```powershell
git clone https://github.com/TMHSDigital/plumbline
cd plumbline
uv sync

uv run plumbline run datasets/public/jevbench-hard.jsonl `
    --adapter mock --format jevbench `
    --results results --report results/report.md
```

That loads the vendored public fixture, runs a deterministic seeded mock over it,
computes every metric against its null, and writes both a results artifact and a
report. No network call. Both land in `results/`, which is gitignored, so
following this leaves your clone clean.

Expected output shape:

```
111 rows read from datasets\public\jevbench-hard.jsonl, 111 loaded, 0 refused. ...
artifact: results\20260921T222053+0000-mock-c18e9496.json
report: results\report.md
```

To run a real vendor instead, set a key and name an adapter. Cost needs a pricing
table you supply; see [Limitations](#limitations) and
[docs/pricing.example.json](docs/pricing.example.json).

```powershell
$env:TYPESAFE_API_KEY = "your-key-here"

uv run plumbline run datasets/public/jevbench-hard.jsonl `
    --adapter typesafe_wire --model jev-latest --format jevbench `
    --results results --report results/report.md `
    --max-cases 40 --pricing my-pricing.json
```

Your own data goes in `datasets/private/`, which is gitignored, and that is the
only path on which the recalibration numbers mean anything.

```powershell
uv run plumbline adapters   # what this install can run
uv run plumbline version
```

## Example report

[docs/example-report.md](docs/example-report.md) is real, unedited output. Four
lines from it:

> ECE 0.0740 over 105 rows (10 equal width bins), against a calibrated-model
> floor of 0.0707 (95th percentile 0.1109): not distinguishable from a perfectly
> calibrated model at this sample size. Collect more rows before reading anything
> into it.

> No cost available. None of the 105 cases could be priced, so cost is not
> reported rather than being shown as zero. 105 rows: tokens were reported, but
> the model that answered is not priced.

> Not reported. recalibration needs at least 200 held-out evaluation rows and
> this split has 53. Fitting a temperature on fewer rows produces a number whose
> uncertainty is larger than the correction it claims to make, and it arrives
> looking like a measurement.

> 38 noul rows were asked as choice questions, which is a different question from
> the one the dataset states. Not comparable with an arm that asked them as noul.

The arm in that report is the seeded mock, labeled as such at the top of the
page, so anyone can reproduce it with one command and no key.

## Adapters and probability semantics

Three transports. Adding a vendor is config, not code.

| Adapter | Transport | Semantics | Adding one |
|---|---|---|---|
| `typesafe_wire` | Jev wire format over HTTP | `calibrated_claim` | A `base_url`. Anything serving the Jev wire format is a config entry, including self-hosted endpoints and open models behind a compatible server. |
| `local_logits` | Option-token logits from a local checkpoint | `restricted_softmax` | A HuggingFace model id and a pinned revision. Requires the optional `local` extra. |
| `generative` | Chat completion, parsed | `none` | A model string. |
| `mock` | None, seeded | configurable | Built in. A deterministic stand-in for smoke tests. |

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
- **Windows is the only verified platform** until CI says otherwise.

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
