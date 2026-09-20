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
