# Methodology

How plumbline measures what it measures, and what each number does not mean.

Written as the implementation landed. Every claim here is either a rule the code
enforces or a number this build measured, and the measured ones say what they
were measured on.

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

#### More rows do not close the gap

That table is one sample size, so it could have been a small-sample artifact.
It is not. `scripts/distribution_penalty.py` repeats the comparison from 500 to
20,000 rows, averaged over five seeds of the same mock (accuracy 0.75, four
options). Each cell is post-scaling ECE on the held-out half, with its ratio to
that half's floor:

| rows | injected skew | floor | multiclass | binary |
|---|---|---|---|---|
| 500 | T = 0.5 | 0.0489 | 0.0515 (1.05x) | 0.0438 (0.98x) |
| 2,000 | T = 0.5 | 0.0251 | 0.0306 (1.22x) | 0.0266 (1.18x) |
| 5,000 | T = 0.5 | 0.0158 | 0.0181 (1.14x) | 0.0238 (1.66x) |
| 20,000 | T = 0.5 | 0.0079 | 0.0113 (1.43x) | 0.0210 (2.92x) |
| 500 | T = 2.0 | 0.0489 | 0.0515 (1.05x) | 0.1815 (2.88x) |
| 2,000 | T = 2.0 | 0.0251 | 0.0306 (1.22x) | 0.1836 (5.78x) |
| 5,000 | T = 2.0 | 0.0158 | 0.0181 (1.14x) | 0.1756 (8.80x) |
| 20,000 | T = 2.0 | 0.0079 | 0.0113 (1.43x) | 0.1756 (17.23x) |

The binary form's leftover miscalibration does not shrink with rows: about
0.18 on the underconfident case and 0.02 on the overconfident one, at every
size. Only its ratio to the floor grows, because the floor falls as rows are
added. So the penalty is a bias of the answer shape, not noise, and the more
rows a dataset has, the more plainly a top-line-only transport shows it. On the
overconfident case it is hidden at a few hundred rows and plain by a few
thousand.

The multiclass form lands on identical figures for both skews, and on the same
figures again with no skew injected at all. The fit inverts the injected
temperature exactly, and what remains, including the drift to 1.43x at 20,000
rows, is the mock's own base calibration rather than anything the correction
left behind.

So an adapter that reports a probability without a distribution is penalized
twice: it loses a metric, and the correction plumbline can fit for it is weaker
even in the easy case. This is a consequence of the API shape rather than of the
model behind it. A vendor returning the full distribution is handing back
something a caller can act on; a vendor returning only the top line is not.

The report states this in one line wherever it applies, on any adapter whose
`probability_semantics` is not `"none"` that supplied no distribution. A Noul
answer meets that condition by construction.

That is no longer hypothetical. plumbline now asks a yes/no row as a Noul, so
every such row is measured as one probability with nothing behind it. Those rows
are outside the multiclass Brier column (the column renders as not reported for
them, never as zero), and the temperature that can be fitted for them is the
one-parameter approximation in the table above, the form that left ECE at five
times the floor on the underconfident case. The penalty is a property of the
answer shape, not of the model that produced it, and it is the price of a wire
format that hands back a top line instead of a distribution.

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

A price is a current-state claim, not a property of a model. A vendor can change
one between the day a run happens and the day its report is read, and a tariff
page carries no promise that it will not.

So every pricing entry carries the source it was read from and the date it was
read, and every run artifact records which entry priced it along with that date.
An entry older than 90 days is still used, and the report says plainly that it may
be out of date instead of presenting its numbers as current. A result from last
year is never silently re-scored against this year's prices.

Where a vendor publishes no price at all, the entry carries None rather than a
guess, and cost is reported as not available. None is not zero: None is the
absence of a claim, and zero is the claim that those tokens are free.

There is a second reason an entry can ship unpriced, and it is not about what the
vendor publishes. Some vendors' terms make their pricing information
confidential, and a figure written into a file that plumbline publishes is a
disclosure by plumbline no matter how public the page it was copied from. For
those vendors the shipped entry names the page to read and prices nothing, and a
cost column exists only when the operator reads the tariff and supplies it with
`--pricing` (see `docs/pricing.example.json`). Reading a published page and
writing the number down is the operator's act, in their own working copy. The
TypeSafe entries ship this way.

The consequence is visible in the report rather than hidden: with no supplied
table, a run against such a vendor reports cost as `model_not_priced`, and
`--max-cost-usd` refuses the run outright rather than bounding it, because a
guard cannot bound a run it cannot cost.

## A yes/no row is asked as a yes/no question

A Noul returns one number: the probability that the answer is yes. There is no
distribution behind it and therefore no confidence statistic computed from one.
That makes it the cleanest calibration target in the API: nothing is
renormalized, nothing is derived, and `prob_selected` is exactly what the vendor
reported.

Asking the same row as a two-option choice is a different question. It asks for
a distribution over the two strings "yes" and "no" rather than for the
probability that a statement is true, and the number that comes back is a
softmax entry rather than a stated probability. plumbline therefore carries each
row's question type from the dataset and lets the adapter dispatch on it:
`typesafe_wire` asks a yes/no row as a Noul, and reports `noul` for a yes answer
and `1 - noul` for a no answer, because the calibration column holds the
probability of the answer that was actually given.

Which option is the yes is a fact about the dataset. An adapter handed two
options it cannot read as a yes/no pair refuses the case rather than picking
one, because picking wrong inverts every probability on that row.

`local_logits` and `generative` have no equivalent primitive, so they ask these
rows as two-option choices, which is all either can do. Every record carries both
what the row asks and how the adapter asked it, and the report states the
difference wherever it occurs. A noul measurement and a two-option-choice
measurement are never compared without that line between them.

Confidence on a noul is not reported, and the report says so in those words. A
blank cell would suggest the vendor failed to send something; the statistic does
not exist for an answer with no distribution.

## Ordinal score questions are scored by rank

Some datasets ask for a level rather than a label: 0, 1, 2, or 3 daily-rest
violations. The levels are ordered, and every choice metric here is rank-blind:
being wrong by one level and wrong by three score identically, and a
distribution piled on the levels beside the answer is treated no differently
from one spread to both ends. So a score row is never scored as a choice. It is
asked as a score, answered as a distribution over its levels, and read by three
figures of its own, in a block of its own, never beside a choice figure.

### What an answer is

A score answer is a probability for each level, from 0 to K minus 1. Its point
answer is the expected score, the probability-weighted mean of the levels,
which can fall between two of them. That is the answer a score vendor returns
and the one a caller would act on, so it is read as given, never replaced by the
most probable level. An arm that returns a single level and no distribution has
that level as its expected score, and only the first figure below applies to it.

### The three figures

**Mean absolute error of the expected score, in levels.** The rank-aware
accuracy: an answer one level off costs 1, three levels off costs 3. Lower is
better. It is read against a permutation null: the arm's own expected scores
shuffled across the rows, which keeps how the arm answers and breaks the link to
what each row's level was. An arm clears the null only when its answers carry
information about the gold level. A uniform guess would be the wrong null here,
since an arm that always answered the middle level would beat it knowing
nothing.

**Ranked probability score.** For each threshold between two adjacent levels,
the squared gap between the predicted probability that the answer is at or below
it and whether it was, summed over the thresholds and divided by their number,
then averaged over rows. It is the ordinal counterpart of the Brier score and a
proper scoring rule: it is best in expectation when the probabilities are the
true ones, and it charges mass by its distance from the answer, so a near miss
costs less than a far one. Zero is perfect.

**Cumulative calibration error.** What calibration means for an ordered
distribution. A distribution is calibrated when, at every threshold, the
predicted probability that the level is at or below it matches how often it is.
Each row contributes one event per threshold, the predicted cumulative
probability against whether the level was at or below the threshold, and the
events are pooled and read exactly as ECE reads a choice column: ten equal-width
bins and the count-weighted mean gap. The probability of the single most likely
level is not used, because it ignores order: it cannot tell an answer spread
over neighbouring levels from one split between the two ends.

### The floors

The ranked probability score and the cumulative calibration error are read
against a calibrated-model floor built the way every floor here is built. The
predicted distributions are held fixed and the gold level of each row is redrawn
from its own distribution, two thousand times, so every resample is calibrated
by construction. The spread of each figure across the resamples is the noise
floor for this exact set of distributions at this row count. A figure above the
floor's 95th percentile is distinguishable from sampling noise; one inside it is
INCONCLUSIVE, with the same wording as every other figure, because nothing was
established either way.

### What is not done for scores

Recalibration and the cascade are choice-question tools and are not applied to
score rows. A temperature fitted to a distribution over ordered levels is a
different correction, and a threshold on an expected score is a different
decision; each would need its own design. The score block says so. Score rows
are also left out of every choice figure, as they always were.

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

## Every figure carries its row count and its null

ECE has no fixed meaning without the sample size beside it. The floor it has to
clear is a function of the row count, the bin count, and the shape of the
predicted probabilities, so 0.03 over 5,000 rows and 0.03 over 80 rows are
different findings and the second is usually no finding at all. The same is true
of everything else the report prints, so every figure carries both.

ECE, MCE, Brier and multiclass Brier are read against a calibrated-null floor
from the parametric bootstrap. Accuracy is read against chance on this dataset's
own mix of option widths, which is not 1/n for any single n once the widths
differ. AUROC is read against a permutation null that keeps the observed ties and
class balance.

Both of those nulls are read in both directions. Above the 95th percentile is a
result (better than chance, or a score that separates right from wrong). Below
the 5th percentile is a result too, of the opposite kind: accuracy worse than
guessing means the answers are systematically wrong, usually a misalignment of
labels and options, and AUROC below its null means the score is inverted. Only a
value between the two is INCONCLUSIVE, and only there is "collect more rows" the
right advice. The calibration floors are one-sided, because calibration error has
no "better than perfect" to fall into.

Nothing prints as a bare number. A figure whose null cannot be built (MCE when
no bin holds enough rows, AUROC when every case is correct or every case is wrong)
is reported as not reported with the reason, rather than as a number standing
on its own.

The row count travels with the figure in the types as well as on the page:
`CalibrationFigure` and `Figure` each hold the value, the count and the null
together, and their one-line statements print all of it. The run artifact records
`dataset_rows` next to `dataset_hash`, so a stored result carries the sample size
its numbers were computed on. The hash says which rows; the count says how many,
and a reader needs both.

## Probabilities versus confidence

These are two different numbers and plumbline never lets one stand in for the
other.

`prob_selected` is P(the selected label is correct), as the system reported it.
It is the only quantity ECE, MCE, and Brier are ever computed against.

`confidence` is a vendor summary statistic derived from the shape of the
distribution: the TypeSafe docs describe it as a statistic computed from the
distribution the answer already gives you, collapsed "into a single number from 0
to 1, so you can threshold on it". It is a gating and ranking signal. It is not a
probability of correctness, and no calibration metric is computed against it.

The separation is enforced by the type system rather than by a naming
convention. `ProbabilitySeries` and `ConfidenceSeries` are different types, the
calibration functions accept the first and nothing else, and the discrimination
functions accept either. There is no call site at which a confidence column can
reach ECE by mistake, which matters because the mistake is invisible once made:
a reliability diagram drawn from confidence looks exactly like a reliability
diagram.

What confidence is measured with instead is AUROC and a threshold sweep: can it
separate the cases the system got right from the ones it got wrong, and what
does accuracy and coverage look like at each cut.

### Confidence and prob_selected rank together, until the option counts vary

The docs' confidence statistic divides out the option count. Within a fixed
label set it is a monotone transform of the top probability, so the two columns
rank identically and their AUROCs are the same number. Across cases with
differing option counts they come apart, and that divergence is the only place
confidence carries information `prob_selected` does not.

Measured on the public fixture with the mock adapter (seed 7, accuracy 0.8):

| rows | option widths | confidence AUROC | prob_selected AUROC |
|---|---|---|---|
| 32 | 4 only | 0.6667 | 0.6667 |
| 105 | 2 through 6 | 0.6034 | 0.6188 |

Identical on the fixed-width subset, to the digit, which is the Phase 2 parity
result showing up in a real file. On the full fixture they differ, and the
variable option widths are the whole reason. When a report shows the two
diverging, that is the first thing to check, and it is not evidence that the
vendor statistic is adding judgment.

## Read the vendor's selected label. Never recompute it.

An answer carries both a selected label and a distribution over the labels. It
is tempting to treat the distribution as the source of truth and recover the
selection with an argmax, because it is one line and it looks equivalent.

It is not equivalent, and the two disagree silently.

On a live run of 40 choice rows against hosted Jev on 2026-09-21, one row
returned two options tied for the maximum at identical probability. The vendor
selected one of them. An argmax over the distribution picked the other, because
which one an argmax returns is decided by dictionary order rather than by
anything about the answer. On that row the gold label was the option the vendor
did not select, so the two implementations disagreed about whether the system
was right, and neither would have logged anything unusual.

This is guidance for anyone writing an adapter, not a note about one vendor. The
rule is that the selected label is whatever the vendor said it was. A
distribution is evidence about the selection; it is not the selection. plumbline
reads `answer.choice` and looks up `probabilities[answer.choice]` for the
calibratable column, which is correct even on a tie, and `adapters/base.py`
expects the same of any adapter added later.

The related check is worth stating too: the adapter verifies that the selected
label is one of the labels that were asked about, and raises rather than scoring
the row if it is not. A service answering a different question than the one
posed is not a prediction to score.

A tie is recorded, not only survived. Each record in the artifact carries
`tied_for_top`, the options that shared the highest probability when two or more
did, so a reader auditing a close result can see which rows were decided by the
vendor's tie-break rather than by a margin. When any row tied, the report says
how many beside the accuracy figure, and on how many of them the gold label was
a tied option the vendor did not choose: those rows count as wrong by a
tie-break, which is a fact about the measurement's precision rather than about
the model's preference.

Whether a vendor's tie-break is deterministic is not known. If it is not, the
same tied row can resolve differently between calls, and a cached answer and a
live one can disagree on it for a reason that is nobody's fault. The flag is
what lets that be checked.

## Probabilities arrive quantized, which bounds the resolution of any figure here

Hosted Jev returns probabilities on a two-decimal grid. All 165 probability
values across the 40-row live run landed exactly on a multiple of 0.01, as did
all 40 confidence values. Nothing here treats that as a defect. A vendor is
entitled to round what it puts on the wire, and two decimals is a reasonable
place to round for a decision API whose output is meant to be thresholded.

It does, however, bound what can be measured *through* that API, and that bound
belongs on the measurement rather than on the vendor:

- **Binning.** The default ECE scheme is ten equal-width bins, which are wider
  than the 0.01 grid, so the quantization does not bias the binning. A scheme
  with more than 100 bins would be measuring the grid rather than the model.
- **Thresholds.** A cascade threshold cannot be tuned more finely than 0.01.
  A sweep that reports a cut to four decimals is reporting three digits of
  noise.
- **Ties.** At 0.01 resolution over three to six options, exact ties for the
  maximum are ordinary rather than freak events. One in 40 rows on the run
  above. This is the mechanism behind the previous section.
- **Recalibration.** A fitted temperature is applied to quantized inputs, so the
  corrected probabilities are smooth but the information they carry is not finer
  than what arrived.

None of this makes the vendor's numbers worse than an unrounded column would be
for the purpose they exist for. It means a report should not claim resolution
its inputs do not have, and it is a fact about the measurement that a reader of
these figures needs in order to know what a small difference is worth.

So the report says it at the point of use. When every probability an arm
returned sits on a grid of 0.001 or coarser, over at least 20 values, the arm's
section carries a **Resolution** line naming the grid, and adds a warning when
the bins are narrower than it.

### The grid does not raise the floor

The open question was whether rounding costs calibration the floor does not
model, which would read every figure through this transport against a floor
slightly too low. It does not, at any size this tool reports on.

`scripts/quantization_floor.py` simulates a perfectly calibrated model whose
true probabilities are continuous, sends them rounded to the grid, and reads
ECE on the rounded values against `calibration_floor` of those same values, as
the report does. If rounding cost calibration, the calibrated model would clear
the floor's 95th percentile more often than 5 percent of the time. Over 200
trials per row, so a rate within about three points of 5 percent is noise:

| rows | bins | grid | mean ECE | floor mean | above p95 |
|---|---|---|---|---|---|
| 40 | 10 | 0.01 | 0.1141 | 0.1149 | 5.5% |
| 105 | 10 | 0.01 | 0.0743 | 0.0728 | 6.0% |
| 500 | 10 | 0.01 | 0.0345 | 0.0339 | 5.5% |
| 2,000 | 10 | 0.01 | 0.0164 | 0.0171 | 5.5% |
| 10,000 | 10 | 0.01 | 0.0074 | 0.0076 | 2.5% |
| 105 | 10 | 0.05 | 0.0727 | 0.0714 | 6.0% |
| 10,000 | 10 | 0.05 | 0.0080 | 0.0075 | 6.5% |
| 105 | 20 | 0.01 | 0.1002 | 0.0994 | 7.0% |
| 10,000 | 20 | 0.01 | 0.0106 | 0.0107 | 4.5% |

The measured ECE and the floor agree to the third decimal throughout, and the
rate stays at the nominal 5 percent, even on a grid five times coarser than the
vendor's. The reason is that the floor is computed from the rounded values
themselves: rounding moves each probability by at most half a step, in both
directions, and within a bin those errors cancel rather than accumulate. What
the grid bounds is resolution, above; it does not bias the verdict.

## The three probability_semantics classes

Every adapter declares what kind of number it reports, and the report groups on
that field and refuses to compare across groups.

**`calibrated_claim`.** The vendor asserts the probabilities are calibrated.
Calibration metrics are meaningful on their own terms here, and whether the claim
survives contact with a dataset is what plumbline exists to answer. The label
records the claim; it never asserts the claim is true.

**`restricted_softmax`.** A softmax over the declared option tokens only, with no
calibration claim attached. The upstream project is explicit about it. SemIf
(formerly OpenJev), which reads typed option probabilities straight out of an
open model, states the constraint in its README:

> Returned probabilities are conditional on the supplied options. Calibrate and
> validate them on the workload where they will make decisions.

That sentence is the whole argument for this class existing. A number normalized
across the options you happened to supply is a statement about that option set,
not a probability of correctness in the world: add an option and every number
moves, without anything about the case having changed. It can behave like a
calibrated probability on a given workload, and whether it does is measurable,
which is why plumbline measures it rather than assuming either way.

The local arm reads that softmax in one of two ways, and the artifact and the
report say which. By default it reads each option's own token, which means every
option must be a single token for the checkpoint: an option that tokenizes into
several pieces is refused by name, because a softmax over first tokens answers a
different question than the dataset asks. Options written as identifiers, such
as `pay_subject_to_10000_sublimit`, are almost never one token, so on a dataset
of them this reading scores little beyond the yes/no rows.

`--option-style letter` asks the same options by letter instead: the prompt lists
them as A, B, C, and the softmax is over the letter tokens, each of which is one
token. That scores options of any length and is the usual way multiple-choice
questions are put to an open model. It is a different question, though, and is
labelled as one. The letters bring position into the answer, which is why the
cache keys the option order for this arm, and the numbers are conditional on the
lettered option set exactly as the plain reading is on the words. Scoring each
option by the probability of its whole token sequence was the other candidate,
and was not taken: longer options would lose probability for being long, and
every way of correcting for length is a choice the result would silently depend
on.

Citations, kept separate on purpose. SemIf is an independent project and says so:
"not affiliated with or endorsed by TypeSafe". fastjev is an independently
maintained fork of SemIf that preserves its history and MIT license, follows its
own roadmap, and carries the same sentence about conditional probabilities; it
serves TypeSafe's documented wire shape while stating plainly that it "does not
serve Jev or reproduce Jev calibration". Neither is the vendor, and neither
speaks for the vendor. plumbline's `local_logits` arm is this class by
construction, and it is the null hypothesis the calibrated claims have to beat.

**`none`.** No probability at all. The arm is excluded from calibration
entirely, never imputed and never defaulted to zero, and renders as not reported.
Its accuracy is the floor a probability-reporting system has to clear before its
probabilities are worth discussing.

The grouping is not cosmetic. A restricted softmax and a calibrated claim are
different kinds of number, and a table that lists them together invites a
comparison that the numbers do not support. So the report separates the groups,
labels each one, and says in its own words that figures in different groups are
not comparable.

## The mandatory fit and eval split

A temperature is fitted on one half of the rows and reported on the other. The
split is disjoint, the disjointness is checked rather than trusted, and the seed
and both sizes print in the report, including when the verdict is a refusal,
because the procedure is part of the result.

Fitting and reporting on the same rows manufactures an improvement that does not
survive new data, and it manufactures it reliably enough that the number looks
like a measurement. Every metric in a recalibration result, before and after, is
computed on the evaluation rows only, so the pair is comparable and neither half
has seen the fit.

Below 200 held-out rows plumbline refuses to recalibrate at all. A temperature
fitted on fewer rows carries uncertainty larger than the correction it claims to
make, and it arrives looking authoritative. The report says the rule it failed
and gives no temperature.

## A temperature per predicted label, when one temperature is the wrong shape

A model can be well calibrated on most labels and overconfident on one. One
temperature cannot reach that: flattening enough for the skewed label
over-flattens the honest ones, and the global fit's verdict says so, either as a
refusal because the change was no larger than chance or as a partial fit whose
residual stays above the floor. After exactly those two verdicts, and never
otherwise, the report tries the smallest correction that can reach it: one
temperature per predicted label.

It is keyed on the label the model predicted, because that is all a caller knows
when the correction is applied. Each label's temperature is fitted on the fit
rows the model predicted that label for, in the same form as the global fit, and
every held-out row is scaled by its own label's temperature. Scaling never
changes which label is on top, so a row stays in its label's group. The split is
the global fit's, so both are judged on the same held-out rows, and the verdict
is the same rule: inside the floor is recommended, a material improvement that
stops short is partial, and anything else is refused with no temperature
printed. The report prints the per-label temperatures in a block of their own,
says they are a different correction and not comparable with the global
temperature, and names which of the two to apply.

A label with fewer than 100 fit rows is not fitted. It keeps its probabilities as
they came, and the report names it with its counts. A temperature fitted on a
handful of rows is the noise of those rows, and a per-label fit has one of those
per label.

### When it helps, and what it costs

`scripts/per_label_study.py` measures both on the seeded mock (accuracy 0.75,
four options, five seeds). Each cell is post-scaling ECE on the held-out half
over the floor's 95th percentile, so 1.0 or below is inside the floor, with how
often each verdict would ship a correction:

| rows | fit rows per label | one temperature | per label | per label ships |
|---|---|---|---|---|
| 500 | 49 | 1.05x | not fitted | 0% |
| 1,000 | 103 | 1.22x | 0.77x | 60% |
| 2,000 | 216 | 1.93x | 0.83x | 100% |
| 4,000 | 461 | 2.09x | 0.86x | 100% |
| 8,000 | 952 | 3.30x | 0.67x | 100% |

That is a per-label bias: the model sharpened by a temperature of 0.45 whenever
it says billing, and honest otherwise. From about 100 fit rows per label the
per-label fit lands inside the floor, while the global fit falls further behind
as rows are added, the same pattern as the top-line penalty above: a fixed bias
that a shrinking floor exposes.

The cost is measured where one temperature is the right shape: every answer
sharpened by 0.5. There the four extra parameters can only add noise, and they
leave 5 to 15 percent more ECE on the held-out rows than one temperature does
(0.77x against 0.72x at 1,000 rows, 0.67x against 0.58x at 8,000), all inside the
floor. The report never pays that cost, since it tries per-label only after one
temperature was diagnosed as the wrong shape.

The row gate is set more strictly than the mock requires. At 500 rows with the
gate lowered to 40, the per-label fit still reached 0.69x, and the verdict
shipped it on only two of five seeds. But the mock is per-label scaling's best
case, since its distortion is exactly a per-label temperature and a real model's
is not. So the default waits for 100 rows a label, where the result above is not
in doubt.

What it does not do: vector or matrix scaling, which fit more parameters than
most datasets here can support; a correction keyed on the gold label, which no
caller knows at inference time; or feeding the per-label temperatures into the
cascade, which still scores on the global temperature when one was emitted.

## The binary Brier formulation

The Brier score plumbline reports by default is
`mean((prob_selected - correct) ** 2)`: the two-class form applied to the top
label, where `correct` is 1 when the selected label was right and 0 otherwise.

It is not the multiclass Brier score and the two are never compared. Where a full
distribution exists plumbline also computes the multiclass form,
`mean over cases of sum_k (p_k - y_k) ** 2`, in the unnormalized convention with
range `[0, 2]`. An adapter with no distribution gets the binary figure only, and
the multiclass column renders as not reported rather than as zero.

Both forms are read against a null, like everything else here. A model that
reports 0.6 on every case cannot score a binary Brier below 0.24 however well
calibrated it is, so the floor is computed from the same parametric bootstrap
that produces the ECE floor: outcomes redrawn from the reported probabilities,
Brier scored on each resample. Exceeding the floor means worse than a calibrated
model of the same sharpness, which is a different and more useful claim than
"not zero".

## One request per case

Every case is one request. Nothing is sampled repeatedly and voted on, nothing is
retried to get a better-looking answer, and nothing is asked twice to reduce
variance, because a number produced that way is not the number the system would
give in production, and it is the production number plumbline is measuring.

Transport failures are retried, with backoff, because a connection reset is not
an answer. Decisions are not. A refusal (an option that is not a single token,
a generator that answered off-label, a model that declined) is recorded once
and never retried, since retrying would understate exactly the failure rate the
report is there to show.

"Transport failure" is narrow on purpose, because every retry of a paid call is
paid for. A dropped connection, a timeout, a rate limit (429), a request timeout
or conflict (408, 409, 425), and a server error (5xx) are retried, waiting as
long as a `Retry-After` header asks, up to 60 seconds. Everything else fails on
its first attempt: any other 4xx (a bad key does not become good), an answer to
a different question than the one asked, a checkpoint that is not the one
pinned, a missing optional dependency, and any error the runner does not
recognise. The SDKs' own retries are switched off, so the runner is the only
layer that retries and the artifact's `attempts` is the number of calls made.

A cache hit replaces the call entirely and is recorded
as a hit, contributing to neither cost nor latency, because it measures disk.

## Binning scheme and the ECE floor

ECE is the count-weighted mean gap between predicted probability and observed
accuracy, over ten equal-width bins by default. Equal-count binning is available
and moves the number, which is one reason the bin count and scheme print beside
every figure. Equal-count bins never split a tie: rows with the same predicted
probability always share a bin, so the figure does not depend on the order the
rows arrived in, and with heavy ties the bins are near-equal rather than equal.

### Why the floor is not zero

A perfectly calibrated model does not score ECE 0 on a finite sample. Each bin's
observed accuracy is a binomial draw around its mean predicted probability, and
the absolute gaps do not cancel: they add. So ECE has a positive expectation
under perfect calibration, set by the row count, the bin count, and how the
predictions are spread across the range.

plumbline measures that floor rather than assuming it. The parametric bootstrap
holds the observed probabilities fixed and redraws each outcome from
Bernoulli(p_i), so every resample is perfectly calibrated by construction, and
the spread of ECE across resamples is the noise floor for this exact sample. The
floor prints next to the measurement, always, and "distinguishable" means the
measurement is above the floor's 95th percentile.

Printing the floor rather than a p-value is deliberate: the reader needs to know
how big a number has to be here before it means anything, and that quantity is
useful even when the measurement is nowhere near it.

### The measured result: MCE cannot see gross overconfidence at 500 rows

MCE is the largest gap over bins holding at least ten rows. Being a maximum, its
null distribution is wide, and the wide null swallows real miscalibration.

Measured on the mock at 500 rows, accuracy 0.75, ten equal-width bins, 2000
bootstrap draws:

| injected skew | ECE | ECE floor p95 | ECE vs floor | MCE | MCE floor p95 | MCE vs floor |
|---|---|---|---|---|---|---|
| none (T = 1.0) | 0.0253 | 0.0526 | 0.48x | 0.0999 | 0.2135 | 0.47x |
| overconfident (T = 0.5) | 0.1722 | 0.0330 | 5.22x | 0.2504 | 0.2552 | 0.98x |
| grossly overconfident (T = 0.35) | 0.2128 | 0.0224 | 9.49x | 0.3626 | 0.3374 | 1.07x |

Read the middle row. A model whose reported probabilities have been sharpened by
a temperature of 0.5 is overconfident in a way any user would notice, ECE
catches it at more than five times its floor, and MCE lands at 0.98 times its
own floor: inside the band, so MCE reports the question as unanswerable on this
many rows while ECE answers it plainly. Even at T = 0.35, where ECE is nine
times its floor, MCE clears its floor by seven percent.

That is why MCE is demoted to a diagnostics block and never printed beside ECE.
It is not wrong, and it is not useless on larger samples, but at the few hundred
rows a real private dataset holds it cannot answer the question it appears to
answer. A reader who sees "MCE 0.25" next to "ECE 0.17" will weigh them equally
unless the report stops them, so the report stops them.

## Requested and reported model strings

plumbline records what it asked for and what answered, separately, on every run.

A model alias can resolve somewhere other than where the config pointed. Asking
for `jev-latest` and being answered by `jev-1.13` is normal, and a result labeled
only with the alias becomes unreadable the moment the alias moves. So the
artifact carries `model_requested` and `model_reported`, and the report prints
both.

Pricing follows what answered, not what was asked for, because billing does. A
local checkpoint is pinned by commit and the revision that actually loaded is
recorded next to the one that was requested; a pinned commit that resolves
elsewhere is refused outright, since every number measured under it would be
attributed to the wrong weights.
