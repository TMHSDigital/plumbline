"""Threshold selection: what a recalibrated probability is actually worth.

This is where the metrics stop being metrics. Given a threshold, traffic below it
escalates to something more expensive and traffic above it is handled by the
cheap model. The user supplies two numbers that no benchmark can know: what an
escalation costs and what being wrong costs. Those two numbers decide the
threshold, and the threshold decides the bill.

Run this on recalibrated probabilities. A threshold picked against a raw
overconfident probability is set at the wrong place, because 0.9 from an
overconfident model is not 0.9.

The accuracy figure assuming escalation is correct is the optimistic bound and
is named that way. Whatever you escalate to has its own error rate, and this
number does not know it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from plumbline.types import ConfidenceSeries, ProbabilitySeries


@dataclass(frozen=True)
class CascadeRow:
    """What acting at ``threshold`` costs and buys."""

    threshold: float
    covered: int
    coverage: float
    escalated: int
    errors_covered: int
    accuracy_covered: float | None
    accuracy_overall_if_escalation_is_correct: float
    total_cost_usd: float
    cost_per_case_usd: float


#: The threshold no score reaches: every case escalates.
ESCALATE_ALL = math.inf


def _values(series: ProbabilitySeries | ConfidenceSeries) -> tuple[float, ...]:
    if not isinstance(series, ProbabilitySeries | ConfidenceSeries):
        raise TypeError(f"expected a probability or confidence series, got {type(series).__name__}")
    return series.require_reportable()


def _checked(
    series: ProbabilitySeries | ConfidenceSeries, correct: Sequence[bool]
) -> tuple[np.ndarray, np.ndarray]:
    values = _values(series)
    if len(values) != len(correct):
        raise ValueError(f"{len(values)} scores against {len(correct)} outcomes; these must match")
    if not values:
        raise ValueError("no rows to score")
    return np.asarray(values, dtype=np.float64), np.asarray(correct, dtype=bool)


def candidate_thresholds(series: ProbabilitySeries | ConfidenceSeries) -> tuple[float, ...]:
    """Every threshold that can change the outcome, and no others.

    Cost as a function of the threshold is a step function that only moves where a
    row crosses it, so scanning the distinct observed values finds the exact
    minimum rather than the best point on an arbitrary grid.

    The last candidate is :data:`ESCALATE_ALL`, above every score. Without it
    the highest observed value still covers the rows that reach it, so
    escalating everything was never considered even when it is the cheapest
    choice.
    """
    return (0.0, *sorted(set(_values(series))), ESCALATE_ALL)


def cascade_sweep(
    series: ProbabilitySeries | ConfidenceSeries,
    correct: Sequence[bool],
    cost_escalation_usd: float,
    cost_error_usd: float,
    thresholds: Sequence[float] | None = None,
) -> list[CascadeRow]:
    """Coverage, accuracy, and total cost at each threshold.

    Cost counts only what the threshold changes: an error on covered traffic costs
    ``cost_error_usd``, and an escalation costs ``cost_escalation_usd``. The cheap
    model's own per-case cost is deliberately excluded, because it is paid on every
    case at every threshold and so cannot move the optimum. Add it back when
    quoting an absolute bill.

    Escalated traffic is assumed to answer correctly, which is why escalation
    carries no error cost here.
    """
    if cost_escalation_usd < 0 or cost_error_usd < 0:
        raise ValueError("costs must be non-negative")
    scores, outcomes = _checked(series, correct)
    total = len(scores)
    grid = tuple(thresholds) if thresholds is not None else candidate_thresholds(series)

    rows: list[CascadeRow] = []
    for threshold in grid:
        selected = scores >= threshold
        covered = int(selected.sum())
        escalated = total - covered
        errors = int((~outcomes[selected]).sum())
        cost = errors * cost_error_usd + escalated * cost_escalation_usd
        rows.append(
            CascadeRow(
                threshold=float(threshold),
                covered=covered,
                coverage=covered / total,
                escalated=escalated,
                errors_covered=errors,
                accuracy_covered=float(outcomes[selected].mean()) if covered else None,
                accuracy_overall_if_escalation_is_correct=(covered - errors + escalated) / total,
                total_cost_usd=cost,
                cost_per_case_usd=cost / total,
            )
        )
    return rows


def optimal_threshold(
    series: ProbabilitySeries | ConfidenceSeries,
    correct: Sequence[bool],
    cost_escalation_usd: float,
    cost_error_usd: float,
) -> CascadeRow:
    """The threshold minimizing expected cost, found exactly.

    Ties go to the lower threshold, which is the one covering more traffic. When
    escalation costs at least as much as an error, nothing is worth escalating and
    the answer is a threshold of zero.
    """
    rows = cascade_sweep(series, correct, cost_escalation_usd, cost_error_usd)
    return min(rows, key=lambda row: (row.total_cost_usd, row.threshold))
