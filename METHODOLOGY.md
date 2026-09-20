# Methodology

How plumbline measures what it measures, and what each number does not mean.

This document is written as the implementation lands. Sections marked as pending
arrive with the phase that produces them.

## Adapters that report no distribution

Some systems return a probability for the selected label and nothing else. A
TypeSafe Noul answer is the clearest case: it returns `noul`, a bare probability
that the answer is yes, with no distribution by construction. A generative
adapter returns no probability at all and is excluded from calibration entirely.
Between those sits any adapter that reports a top-line probability without the
distribution behind it.

That middle case is worse off than it first appears, in two separate ways.

### It is excluded from the multiclass Brier score

The binary Brier score plumbline reports is `mean((prob_selected - correct) ** 2)`,
the two-class form applied to the top label. It is not the multiclass Brier
score and the two are not comparable. Where a full distribution exists plumbline
also computes the multiclass form, `mean over cases of sum_k (p_k - y_k) ** 2`,
in the unnormalized convention, so its range is `[0, 2]` and a perfect model
scores 0. An adapter with no distribution gets the binary figure only, and the
report renders the multiclass column as not reported rather than as zero.

### It is materially harder to recalibrate, even when its miscalibration is simple

This is the less obvious cost, and it is the one worth stating plainly.

Temperature scaling on a full distribution is `softmax(log p / T)`. Temperature
scaling on a lone top-label probability is `sigmoid(logit(p) / T)`. These are not
the same correction. The second is a one-parameter approximation of the first,
and it is misspecified against any distortion that actually lives in the full
distribution.

Measured on the mock, whose injected skew is a pure temperature on the log
probabilities and therefore exactly the kind of miscalibration temperature
scaling is built for:

| injected skew | form | fitted T | ECE before | ECE after | vs floor | outcome |
|---|---|---|---|---|---|---|
| T = 0.5, overconfident | multiclass | 1.940 | 0.1614 | 0.0124 | 0.46x | recommended |
| T = 0.5, overconfident | binary | 2.415 | 0.1614 | 0.0260 | 0.99x | recommended |
| T = 2.0, underconfident | multiclass | 0.485 | 0.2133 | 0.0124 | 0.46x | recommended |
| T = 2.0, underconfident | binary | 0.408 | 0.2133 | 0.1739 | 5.07x | partial |

Identical rows, identical outcomes, identical underlying distortion. On the
underconfident case the multiclass form returns ECE to the noise floor and the
binary form leaves it at roughly five times the floor, removing less than a
fifth of the miscalibration present.

So an adapter that reports a probability without a distribution is penalized
twice: it loses a metric, and the correction plumbline can fit for it is weaker
even in the easy case. This is a consequence of the API shape rather than of the
model behind it. A vendor returning the full distribution is handing back
something a caller can act on; a vendor returning only the top line is not.

The report states this in one line wherever it applies, on any adapter whose
`probability_semantics` is not `"none"` that supplied no distribution. A Noul
answer meets that condition by construction.

## What a blank cost column means

Cost is derived from reported tokens against a pricing table held in config, and
a case that cannot be priced is reported as not priced rather than as zero. Free
and unknown are different claims, and a zero in a cost column reads as free.

"Not priced" has more than one cause, and each row records which one applied.

| basis | what it says |
|---|---|
| `adapter_reports_no_tokens` | This adapter cannot report cost at all. A local checkpoint is the clear case: its cost is hardware and wall-clock, and there is no token count to price. Every row is blank and that says nothing about the run. |
| `tokens_not_reported` | This adapter does report tokens and the API returned none on this call. A fact about the run, worth counting. |
| `model_not_priced` | Tokens arrived, but the model that answered has no usable entry in the pricing table. |
| `cache_hit` | No call went out, so there is nothing to charge. |

The report keeps these apart. A tool that showed one blank column for all four
would let "this vendor stopped reporting usage halfway through the run" hide
behind "local models do not report tokens".

## Prices are dated, and so is every result

A price is a current-state claim, not a property of a model. Jev's output tokens
are the clearest case: the only published statement is a field description in the
wire schema saying they are "currently stated at https://docs.typesafe.ai/models", which was true on the day
it was read and says nothing about the day a report is printed.

So every pricing entry carries the source it was read from and the date it was
read, and every run artifact records which entry priced it along with that date.
An entry older than 90 days is still used, and the report says plainly that it may
be out of date instead of presenting its numbers as current. A result from last
year is never silently re-scored against this year's prices.

Where a vendor publishes no price at all, the entry carries None rather than a
guess, and cost is reported as not available. plumbline ships a Jev entry with
output at 0.0 and no input price for exactly this reason.

## A row that cannot be scored is refused, not scored

A dataset row whose gold label is not one of its own options marks every system
wrong on that row, by construction. It arrives in a report looking like a model
failure and it is a typo in a file. The loader refuses such a row: it never
reaches the runner, the refusal carries the line number and the id, and the load
report states how many rows were read, loaded and refused. A caller that needs
the whole dataset can demand it and find out before the run rather than after.

Nothing is repaired. The one narrow exception is representation rather than
content: a gold label written as the JSON number `1` against the string option
`"1"` is the same option, so it is matched, and the number of rows that needed
that is reported rather than absorbed.

## Every calibration figure carries its row count

ECE has no fixed meaning without the sample size beside it. The calibrated-null
floor that ECE has to clear is a function of the row count, the bin count and the
shape of the predicted probabilities, so 0.03 over 5,000 rows and 0.03 over 80
rows are different findings and the second one is usually no finding at all.

So the row count travels with the figure: `CalibrationFigure` holds the value,
the count, the bin count and the floor together, and its one-line statement
prints all four. The run artifact records `dataset_rows` next to `dataset_hash`,
so a stored result carries the sample size that its numbers were computed on.
The hash says which rows; the count says how many, and a reader needs both.

## Probabilities versus confidence

Pending. Lands with Phase 9.

## The three probability_semantics classes

Pending. Lands with Phase 9.

## The mandatory fit and eval split

Pending. Lands with Phase 9.

## The binary Brier formulation

Partially covered above. The full statement lands with Phase 9.

## One request per case

Pending. Lands with Phase 9.

## Binning scheme and the ECE floor

Pending. Lands with Phase 9.

## Requested and reported model strings

Pending. Lands with Phase 9.
