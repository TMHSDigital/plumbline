# Plan

The build's memory, not a spec. What is left, what was already decided, and what
is still unknown. Kept short on purpose: a cleared context should be able to read
this page and pick up where the work stopped.

**v0.1 is complete.** Code, docs, CI, licence and the live validation all
landed on 2026-09-21. CI is green on Ubuntu and Windows across Python 3.12 and
3.13. `v0.1.0` is tagged locally and has not been pushed.

What remains is a short list of manual steps for a human, in "Pre-public
checklist" at the bottom of this page. The repository is still private, and
flipping it is a deliberate act that nothing here does for you.

## Phases

- **Phase 6 — adapters: `local_logits` and `generative`.** Landed. The restricted
  softmax over a pinned checkpoint, and the text-generating control arm.
- **Phase 7 — datasets.** Landed. The JSONL loader that refuses unscoreable
  rows, the JevBench public fixture and its translation, and the end-to-end
  smoke run in `examples/smoke_public_dataset.py`.
- **Phase 8 — report and CLI.** Landed. The markdown report groups arms by
  `probability_semantics`, states every figure's row count and null, and demotes
  MCE to diagnostics. `src/plumbline/cli.py` exists, so the `plumbline` console
  script pyproject declares now works: `run`, `report`, `adapters`, `version`.
- **Phase 9 — recalibration, the cascade, and the methodology.** Landed. The
  report fits a temperature on a held-out half and prints the verdict rather
  than the number when the verdict is a refusal; the cascade section ends in one
  sentence naming the threshold, the coverage, the expected cost, and the cost
  of escalating everything. METHODOLOGY has no pending sections left.

## Before this goes public

All three items are done. Kept here because the answers matter, not the list.

1. **Terms read**, 2026-09-21. No clause mentions benchmarking or restricts
   publishing results. Three others constrain what a public README may carry,
   and all three are handled: 14.1 (pricing confidential) is why no figure
   ships, 16.4 (publicity) is why the example report uses the mock, and 2.3(c)
   (reverse engineering) turned out not to apply because the vendor publishes
   the confidence formula themselves.
2. **Live call made**, 2026-09-21. See "Live validation" below. Both open
   questions are settled, and the call found two bugs that had never been
   exercised.
3. **Decision on going public** is the human's, and the checklist at the bottom
   of this page is what is left to do.

## v0.2

- **Ordinal score questions.** The six score rows in the public fixture are
  loaded, marked and excluded; scoring them needs rank-aware metrics, because
  every metric here treats wrong-by-one and wrong-by-three identically.
- **Batching.** One request per case today, so cost and latency are both
  conservative relative to batched use. The vendor's own documentation describes
  packing many questions against one shared state in a single call, which is a
  materially different cost and latency profile and is the single largest
  measurement gap in v0.1.
- **Per-label and vector scaling.** Only temperature is fitted. When the
  residual says temperature is the wrong correction the tool refuses, which is
  right but leaves the user with nothing to apply.
- **Adapters do not receive `label_descriptions`**, so a dataset's per-option
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
- A refused recalibration prints no number: the verdict, the split sizes, and
  nothing that could be lifted into production code.
- The cascade threshold is chosen on held-out rows, on the recalibrated scale
  when a temperature was recommended, because a threshold set against a raw
  overconfident probability sits in the wrong place.
- Escalation cost and error cost are supplied by the caller and never defaulted.
  No benchmark can know them, and a made-up default would decide the threshold.
- Below 200 held-out rows no threshold is printed at all.
- `generative` configures no fallback model. This is deliberate and deviates
  from the SDK's default advice for this model family: server-side fallback
  would re-run a declined case on another model inside the same call, so one
  dataset would be answered partly by two models and the run would no longer be
  about the system under test. A decline is recorded as a refusal with its
  category and counted. The tradeoff is stated at the call site in
  `adapters/generative.py`; it is not an oversight.

## Live validation

Run on **2026-09-21**. Model requested `jev-latest`; model that answered
`jev-1.13.0`. 40 choice rows from the vendored JevBench public fixture
(`datasets/public/jevbench-hard.jsonl`, choice rows only, first 40 of 67), one
request per case, 4 concurrent workers, no cache. Total billed input 66,458
tokens, output 2,555 tokens.

The artifact is not committed: it holds per-case records from a vendor call and
lives in gitignored `results/`.

No dollar figure from this run appears in this file or in any other committed
file. The vendor's MCA makes its pricing information confidential and overrides
the public-knowledge carve-out, and a spend total stated next to a token count
lets a reader divide one out. The figures exist in the local artifact under
`results/`, which is gitignored, and that is where they stay.

### Two bugs this run found before it could answer anything

- **`jev-1` is not a model.** The adapter's default `model_requested` was
  `jev-1`, which the API rejects with `400 Unknown model: jev-1`. The SDK's own
  default is `jev-latest` (`typesafe_sdk/constants.py`, `DEFAULT_MODEL`). Fixed
  in `adapters/typesafe_wire.py`. The first five-case probe spent nothing
  because every request 400'd.
- **The pricing table was keyed on aliases only.** `pricing_for` prefers the
  model the API *reported*, and a call to `jev-latest` reports `jev-1.13.0`.
  With only alias keys, every live row priced as `model_not_priced`. The table
  is now keyed on the reported version as well as the aliases, with no prefix
  matching, so a future `jev-1.14.0` will not silently inherit a superseded
  rate.

### Answers

1. **Does `response.usage` populate the token counts?** Yes. 40 of 40 rows
   returned non-None `input_tokens` and `output_tokens`. The wire schema is
   right and the SDK's `int | None` is a widening, not a description of
   behaviour. The None-handling path stays: it costs nothing and it is the
   difference between a blank cost and a false zero. But the report no longer
   needs to hedge about how often `tokens_not_reported` is taken on this
   vendor. Not observed once in 40 calls.

2. **What is `response.model`, and does it match what was requested?** It is
   `jev-1.13.0` on every row, and it does **not** match the requested
   `jev-latest`. This is documented behaviour, not a fault: TypeSafe's models
   page states that `jev-latest` and `jev-preview` are aliases that both
   currently resolve to `jev-1.13.0`, and that "the response's `model` field
   reports the versioned ID that answered". Recording both strings in the
   artifact is what makes a result readable after an alias moves.

3. **Do `probabilities` sum to 1?** Yes, to floating-point exactness. Maximum
   observed deviation across 40 rows is **1.11e-16**, mean 2.78e-18, and 39 of
   40 rows sum to exactly 1.0. Every one of the 165 probability values returned
   lies exactly on a two-decimal grid, so the distribution is quantized to 0.01
   on the wire. Worth knowing for calibration work: with 10 equal-width ECE
   bins, a 0.01 grid is finer than the binning and does not bias it, but it
   does put a floor on how finely a threshold can be tuned.

4. **Is `probabilities[choice]` always the maximum?** The *value* always is:
   zero rows of 40 had `prob_selected` differing from
   `max(distribution.values())`. But the reported choice is not always what a
   naive argmax picks, because **ties happen**. One row
   (`hard-opus-c-temporal_numeric-04`) returned two options tied at 0.23; the
   API selected `usd_2303_01` and the gold label was the other member of the
   tie, `usd_2335_00`. That row scored wrong on a tie-break. At a 0.01
   quantization over 3 to 6 options, ties are not rare events and code that
   re-derives the choice by argmax instead of reading `answer.choice` will
   disagree with the vendor. plumbline reads `answer.choice`, which is correct.

5. **Does `confidence` equal `(n * max_prob - 1) / (n - 1)`?** Effectively yes,
   to within wire rounding. Maximum absolute deviation over 40 rows is
   **0.0167**, mean 0.0053, median 0.0050; 10 rows match exactly and all 40 are
   within 0.02. Both the probabilities and the confidences are returned on a
   two-decimal grid, and a +/-0.005 rounding of `max_prob` propagates to
   +/-0.005 * n/(n-1) in the confidence, which accounts for the spread. The
   residual is consistent with the vendor computing confidence from
   full-precision internal probabilities and rounding both for the wire.

   This is **not** a reverse-engineering finding. TypeSafe publishes the
   formula themselves on <https://docs.typesafe.ai/confidence>, where the
   interactive explainer computes `(count * peak - 1) / (count - 1)` clamped to
   [0, 1], and the accompanying text describes it as an approximation of the
   production definition. The docs also confirm that Noul answers carry no
   confidence, which is what the adapter already assumes.

   Consequence for METHODOLOGY: confidence is a deterministic function of
   `max_prob` **at fixed n**, so AUROC parity between the probability column and
   the confidence column on fixed-width rows is structural rather than
   empirical, and the divergence on mixed-width rows follows from n varying.
   That is exactly what METHODOLOGY already claims from the seeded mock, so the
   existing hedged wording stands and now has a mechanism behind it. Per the
   decision recorded below, the relationship is not published.

6. **Latency percentiles, nearest rank, over 40 live calls** (4 concurrent
   workers, so these include some self-inflicted queueing):
   min 135ms, p50 **178ms**, p90 235ms, p95 **367ms**, p99 469ms, max 469ms.

7. **Did anything fail, rate limit, or refuse?** No. 40 of 40 rows produced a
   prediction, zero refusals, zero rate limits. One row needed a second attempt
   and succeeded on it. No `429` was seen; the published limits are 250,000
   tokens per second and 1,200 requests per minute, far above this run.

### Pricing

TypeSafe publishes a tariff for `jev-1.13.0` on
<https://docs.typesafe.ai/models.md>, covering both the input and the output
side. The figures are deliberately not reproduced here; read the page.

That settles the open question that had blocked cost entirely. The previous
shipped entry quoted an SDK field description for the output side and carried no
input price at all, so `is_priced` was False and **the cost guard refused any run
with `--max-cost-usd` set** — it cannot bound a run it cannot cost.

**What plumbline ships is a separate decision from what the tariff says.** MCA
14.1 makes the vendor's pricing information confidential and explicitly
overrides the public-knowledge carve-out in 14.3, so a figure written into the
default table would be published to everyone who clones this repository. The
shipped entries therefore price nothing and name the page instead.

Of the two options considered, **the number moves out of the committed table
into operator-supplied config**, rather than the table keeping the number while
the source string omits it. The second does not work: a populated
`input_usd_per_million` field in a committed file *is* the per-token price,
whatever the adjacent string says. Only removing the number removes the
disclosure.

So:

- `config.py` ships `jev-1.13.0`, `jev-latest` and `jev-preview` with both prices
  None and a source naming the tariff page. None rather than 0.0 on the output
  side, because a zero there would be plumbline asserting those tokens are free.
- `load_pricing_file` and `merged_pricing_table` read an operator's own JSON
  table and lay it over the shipped one, wired to `plumbline run --pricing`.
  Every supplied entry must carry its own `source` and `as_of`, on the same
  reasoning that applies to a shipped one.
- `docs/pricing.example.json` is a template with placeholder figures.
- A test asserts no shipped entry for this vendor carries a numeric price, so a
  figure cannot drift back in unnoticed.

Verified end to end with a local supplied table: estimate before the run, actual
after, `cost_basis` `priced` on all 40 rows, and the guard admitted the run on
its own arithmetic rather than being bypassed. With no supplied table the same
run reports `model_not_priced` and `--max-cost-usd` refuses, which is the honest
default.

### Already-committed material that states a price

Flagged and fixed in this pass:

- `src/plumbline/config.py` quoted the SDK's "Output tokens are currently stated at https://docs.typesafe.ai/models" as `JEV_OUTPUT_SOURCE`. Removed.
- `METHODOLOGY.md` repeated that quotation in "Prices are dated, and so is every
  result". Rewritten without it.
- `docs/PLAN.md` carried it in the old open-questions section. Replaced by this
  section.

Those three strings are still in **git history**, in commits `5324bc8`,
`865dc85` and `0202d0c`. The repository is not public yet, so this is the moment
to decide: either rewrite history before the first push, or accept that a
quotation of a vendor field description sits in the log. A decision for the
manual pre-public pass, recorded in the checklist below. (Matches on "free of
charge" in `datasets/public/LICENSE-jevbench` and the fixture are MIT licence
boilerplate and are unrelated.)

### What the run says about the tool

Accuracy 0.6500 over 40 rows against a chance null of 0.2571, better than
chance. ECE 0.1070 against a calibrated-model floor of 0.1400, **not
distinguishable from a perfectly calibrated model at this sample size**. Brier
0.1737 against a floor of 0.1512, also not distinguishable. Confidence AUROC
0.7981 against a permutation null of 0.5013, separates correct from incorrect.
Recalibration refused: it needs 200 held-out rows and this split has 20.

That is the intended behaviour at n = 40 and it is the argument the README
makes.

## Open questions

- **Ask TypeSafe, in writing, for permission to name them in a public
  repository's example output.** **MCA 16.4 is the clause to ask about**: it
  bars either party from publicly announcing that they entered the agreement,
  and a committed report from a keyed run arguably does that. Note the clause is
  one-sided in the other direction, since it expressly lets TypeSafe name its
  customers.

  This is why `docs/example-report.md` currently shows the seeded mock arm
  rather than the live run, and says so at the top. Publicity clauses of this
  kind are boilerplate and routinely waived, and a vendor is usually glad to be
  named by a tool that measures it fairly and refuses to rank it. A one-line
  written yes swaps the mock report for a real vendor one, which is a strictly
  better page: a reader could then see the tool decline to draw a conclusion
  from an actual vendor at 40 rows.

  Worth asking about 14.1 in the same message, since a written yes on restating
  the published tariff would let the shipped pricing table carry figures again
  and remove the `--pricing` step for every user of this vendor.

  Follow-up, not a release blocker. Nothing about v0.1 depends on the answer.
- Whether the residual in the confidence relationship (max 0.0167) is purely
  wire rounding or a slightly different production formula. Not worth another
  spend to settle, and nothing in plumbline depends on the answer.
- `datasets/private/.gitkeep` exists on disk but is not tracked, because the
  `datasets/private/` ignore rule matches it. A fresh clone therefore has no
  such directory even though the README names it as where your own data goes.
  Harmless, and fixing it means `datasets/private/*` plus a negation, which
  changes the ignore semantics of a data directory. Left alone deliberately
  rather than changed on the way out the door.

## Repo description and topics

For the GitHub About box. Paste as is.

**Description** (109 characters):

```
Measure whether a decision model's probabilities hold up on your own labeled data. Not a leaderboard.
```

**Topics:**

```
calibration
evaluation
llm
classification
machine-learning
benchmarking
uncertainty-quantification
model-evaluation
confidence-calibration
expected-calibration-error
python
cli
```

No vendor name is included. The tool is not about one vendor, the README says
so in its first section, and a vendor topic would file it as a fan project.

## Pre-public checklist

Everything below is a manual step. Nothing in this repository does any of it
for you.

**Verified already, listed so you can re-check rather than re-derive:**

- [x] Full secret scan across all history, not just the working tree. The live
      key appears in no commit and no blob; `.env` has never been committed;
      `apik_`, `sk-ant-` and `Bearer ` match nothing anywhere in history.
- [x] `results/`, `cache/`, `models/` and `datasets/private/` are all ignored
      and none has ever been committed. No run artifact is in history.
- [x] `uv.lock` is tracked and current (`uv lock --check` passes).
- [x] CI green on `ubuntu-latest` and `windows-latest`, Python 3.12 and 3.13.
      No cross-platform failure appeared; the suspected path handling was fine.
- [x] Fresh install from the built wheel: entry point resolves, `--help`,
      `adapters` and `version` all work, the package imports with core
      dependencies only, and nothing writes into the working directory.
- [x] `LICENSE` present, complete, Apache-2.0, dated 2026, holder TMHSDigital.

**Yours to do:**

1. **Decide about the three strings in git history.** Commits `5324bc8`,
   `865dc85` and `0202d0c` quote a vendor field description that states a price
   ("Output tokens are currently stated at https://docs.typesafe.ai/models"). They are gone from the
   working tree. The repository is still private, so this is the last moment
   where a history rewrite is cheap. Either rewrite before the first public
   push, or accept them. This is a judgement about 14.1, not a technical
   problem.
2. **Read the README yourself, once, as a stranger.** It is the whole public
   interface and it was written by someone who already knew the answer.
3. **Push the tag** if you are happy with it: `git push origin v0.1.0`. It is
   tagged locally and deliberately not pushed.
4. **Flip the repository public.** Not done here, by instruction.
5. **Set the About box** from the description and topics above.
6. **Email TypeSafe** about 16.4 and 14.1 (see "Open questions"). A written yes
   converts the mock example report into a real vendor one and would let the
   shipped pricing table carry figures again. Not a blocker for anything.
7. **Check the CI badge renders** once the repository is public. A badge
   pointing at a private repository's workflow shows as unknown to logged-out
   readers.
