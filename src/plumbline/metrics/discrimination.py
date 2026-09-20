"""Discrimination: how well a score separates correct answers from wrong ones.

This is where ``confidence`` is evaluated, and the only place it is. A score can
rank cases perfectly and still be badly calibrated, and it can be perfectly
calibrated and rank nothing. AUROC answers the ranking question, which is the
one confidence was built for: the TypeSafe docs present it as a gate, "the
answer tells you what; confidence tells you whether to act".

No ECE, no Brier, no reliability bins are computed here, and none are computed
against confidence anywhere else either.

Confidence against prob_selected
--------------------------------
The docs' confidence formula, ``(n * peak - 1) / (n - 1)``, divides out the
option count. Within a fixed label set it is an increasing affine function of
``prob_selected``, so the two produce identical rankings and therefore identical
AUROC, to the bit. Across a dataset whose cases carry differing numbers of
options they come apart, because a top probability of 0.5 is chance among two
options and a clear read among eight, and only the normalized statistic knows
the difference.

:func:`auroc` accepts either series so that both halves of that claim can be
measured rather than asserted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from plumbline.types import ConfidenceSeries, ProbabilitySeries

#: The sweep the report prints: 0.50 to 0.95 in steps of 0.05.
DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))


@dataclass(frozen=True)
class SweepRow:
    """What happens if you act only on scores at or above ``threshold``."""

    threshold: float
    covered: int
    coverage: float
    accuracy_covered: float | None
    accuracy_overall_if_escalation_is_correct: float


def _values(series: ProbabilitySeries | ConfidenceSeries) -> tuple[float, ...]:
    if isinstance(series, ProbabilitySeries):
        return series.require_reportable()
    if isinstance(series, ConfidenceSeries):
        return series.require_reportable()
    raise TypeError(
        f"expected a ProbabilitySeries or ConfidenceSeries, got {type(series).__name__}"
    )


def _checked(
    series: ProbabilitySeries | ConfidenceSeries, correct: Sequence[bool]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    scores = _values(series)
    if len(scores) != len(correct):
        raise ValueError(f"{len(scores)} scores against {len(correct)} outcomes; these must match")
    if not scores:
        raise ValueError("no rows to score")
    return np.asarray(scores, dtype=np.float64), np.asarray(correct, dtype=bool)


def auroc(series: ProbabilitySeries | ConfidenceSeries, correct: Sequence[bool]) -> float:
    """Area under the ROC curve for the score separating correct from incorrect.

    Computed from mid-ranks, so tied scores contribute 0.5 rather than being
    broken arbitrarily. This matters: an adapter that returns the same confidence
    for every case should score 0.5, not 1.0 or 0.0 depending on sort order.

    Equivalently, the probability that a randomly chosen correct case is scored
    above a randomly chosen incorrect one.

    Raises if either class is empty, because AUROC is undefined there rather than
    being 0.5.
    """
    scores, outcomes = _checked(series, correct)
    n_correct = int(outcomes.sum())
    n_wrong = len(outcomes) - n_correct
    if n_correct == 0 or n_wrong == 0:
        raise ValueError(
            "AUROC needs at least one correct and one incorrect case. This run has "
            f"{n_correct} correct and {n_wrong} incorrect."
        )

    # Mann-Whitney U from mid-ranks.
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    sorted_scores = scores[order]
    start = 0
    for position in range(1, len(sorted_scores) + 1):
        if position == len(sorted_scores) or sorted_scores[position] != sorted_scores[start]:
            if position - start > 1:
                tied = order[start:position]
                ranks[tied] = ranks[tied].mean()
            start = position

    rank_sum = float(ranks[outcomes].sum())
    return (rank_sum - n_correct * (n_correct + 1) / 2) / (n_correct * n_wrong)


def threshold_sweep(
    series: ProbabilitySeries | ConfidenceSeries,
    correct: Sequence[bool],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[SweepRow]:
    """Accuracy and coverage at each threshold, the gating question in table form.

    ``coverage`` is the fraction of traffic handled without escalation, meaning
    the score was at or above the threshold. ``accuracy_covered`` is accuracy on
    that portion, and is None when nothing clears the bar.
    ``accuracy_overall_if_escalation_is_correct`` assumes whatever you escalate
    to answers correctly, which is the optimistic bound, and the report says so.
    """
    scores, outcomes = _checked(series, correct)
    total = len(scores)

    rows: list[SweepRow] = []
    for threshold in thresholds:
        selected = scores >= threshold
        covered = int(selected.sum())
        rows.append(
            SweepRow(
                threshold=float(threshold),
                covered=covered,
                coverage=covered / total,
                accuracy_covered=float(outcomes[selected].mean()) if covered else None,
                accuracy_overall_if_escalation_is_correct=float(
                    (outcomes[selected].sum() + (total - covered)) / total
                ),
            )
        )
    return rows
