"""Threshold selection: the point where the metrics turn into a bill.

Hand-computable cases first, so the cost arithmetic is pinned independently of
any mock, then the two degenerate regimes that bracket every real answer, then
the integration case that shows why the threshold has to be picked against a
recalibrated probability rather than a raw one.
"""

from __future__ import annotations

import pytest

from plumbline.adapters.mock import MockAdapter
from plumbline.metrics import cascade, recalibration
from plumbline.types import ConfidenceSeries, ProbabilitySeries, apply_temperature
from tests.helpers import gold_by_text, make_cases, measure

LABELS = ("billing", "returns", "shipping", "other")


def a_series(values: tuple[float, ...]) -> ProbabilitySeries:
    return ProbabilitySeries(values=values, semantics="calibrated_claim")


# Hand-computable cost


def test_cost_counts_covered_errors_and_escalations() -> None:
    # Four rows. At threshold 0.5 three are covered, one of which is wrong, and
    # one escalates. Cost is 1 error at $2.00 plus 1 escalation at $0.50.
    series = a_series((0.9, 0.8, 0.6, 0.2))
    outcomes = [True, True, False, False]

    row = cascade.cascade_sweep(series, outcomes, 0.50, 2.00, thresholds=[0.5])[0]
    assert row.covered == 3
    assert row.escalated == 1
    assert row.errors_covered == 1
    assert row.total_cost_usd == pytest.approx(2.50)
    assert row.cost_per_case_usd == pytest.approx(0.625)


def test_coverage_and_accuracy_describe_the_covered_portion() -> None:
    series = a_series((0.9, 0.8, 0.6, 0.2))
    outcomes = [True, True, False, False]

    row = cascade.cascade_sweep(series, outcomes, 0.50, 2.00, thresholds=[0.7])[0]
    assert row.coverage == pytest.approx(0.5)
    assert row.accuracy_covered == pytest.approx(1.0)
    # Two covered and correct, two escalated and assumed correct.
    assert row.accuracy_overall_if_escalation_is_correct == pytest.approx(1.0)


def test_the_escalation_bound_is_labeled_optimistic_because_it_is() -> None:
    """Escalated traffic is assumed correct. Whatever you escalate to is not."""
    series = a_series((0.9, 0.1))
    row = cascade.cascade_sweep(series, [False, False], 0.10, 1.00, thresholds=[0.5])[0]
    assert row.accuracy_covered == 0.0
    assert row.accuracy_overall_if_escalation_is_correct == pytest.approx(0.5)


def test_nothing_covered_reports_no_accuracy_rather_than_zero() -> None:
    row = cascade.cascade_sweep(a_series((0.1, 0.2)), [True, False], 0.5, 2.0, thresholds=[0.9])[0]
    assert row.covered == 0
    assert row.accuracy_covered is None


# The two regimes that bracket every real answer


def test_free_escalation_escalates_every_row_that_would_have_been_wrong() -> None:
    """With escalation free, the only cost left is covered errors, so it drives to zero.

    Not "escalate everything". Coverage that costs nothing is kept, because ties
    break toward the lower threshold and therefore toward less escalation. Here
    the two correct rows stay on the cheap model and the two wrong ones leave.
    """
    series = a_series((0.9, 0.8, 0.6, 0.2))
    best = cascade.optimal_threshold(series, [True, True, False, False], 0.0, 1.00)
    assert best.total_cost_usd == 0.0
    assert best.errors_covered == 0
    assert best.coverage == pytest.approx(0.5)


def test_escalation_dearer_than_an_error_escalates_nothing() -> None:
    series = a_series((0.9, 0.8, 0.6, 0.2))
    best = cascade.optimal_threshold(series, [True, True, False, False], 5.00, 1.00)
    assert best.coverage == 1.0
    assert best.threshold == 0.0


def test_the_optimum_sits_between_the_two_when_costs_are_balanced() -> None:
    series = a_series((0.95, 0.9, 0.55, 0.5))
    best = cascade.optimal_threshold(series, [True, True, False, False], 0.20, 1.00)
    assert 0.0 < best.coverage < 1.0
    assert best.total_cost_usd == pytest.approx(0.40)


def test_the_sweep_is_exhaustive_over_thresholds_that_can_change_anything() -> None:
    """Cost is a step function, so the distinct observed values contain the optimum."""
    series = a_series((0.9, 0.8, 0.6, 0.2))
    thresholds = cascade.candidate_thresholds(series)
    assert thresholds == (0.0, 0.2, 0.6, 0.8, 0.9, cascade.ESCALATE_ALL)


def test_the_optimum_is_never_beaten_by_a_fine_grid() -> None:
    cases = make_cases(1000, labels=LABELS)
    run = measure(MockAdapter(gold_by_text(cases), seed=7, accuracy=0.75), cases)
    best = cascade.optimal_threshold(run.probabilities, run.outcomes, 0.02, 0.30)

    grid = [index / 1000 for index in range(1001)]
    swept = cascade.cascade_sweep(run.probabilities, run.outcomes, 0.02, 0.30, thresholds=grid)
    assert best.total_cost_usd <= min(row.total_cost_usd for row in swept)


def test_coverage_falls_as_the_threshold_rises() -> None:
    cases = make_cases(1000, labels=LABELS)
    run = measure(MockAdapter(gold_by_text(cases), seed=7, accuracy=0.75), cases)
    coverages = [
        row.coverage for row in cascade.cascade_sweep(run.probabilities, run.outcomes, 0.02, 0.30)
    ]
    assert coverages == sorted(coverages, reverse=True)


# Why the threshold has to come off a recalibrated probability


def test_an_overconfident_model_picks_a_different_threshold_after_recalibration() -> None:
    """0.9 from an overconfident model is not 0.9, so it buys a different decision.

    The same rows, the same costs, and the same optimum in substance, but the
    number a user would paste into a config differs once the probabilities have
    been corrected. Picking a threshold against the raw column sets it in the
    wrong place.
    """
    cases = make_cases(4000, labels=LABELS)
    run = measure(
        MockAdapter(gold_by_text(cases), seed=7, accuracy=0.75, calibration_temperature=0.5),
        cases,
    )

    result = recalibration.recalibrate(
        run.probabilities,
        run.outcomes,
        distributions=run.distributions,
        gold_labels=run.gold_labels,
        n_boot_ci=40,
        n_boot_floor=200,
    )
    eval_rows = result.split.eval_indices

    raw = ProbabilitySeries(
        values=tuple(run.probabilities.values[index] for index in eval_rows),
        semantics=run.probabilities.semantics,
    )
    corrected_distributions = [
        apply_temperature(run.predictions[index].distribution, result.temperature)
        for index in eval_rows
    ]
    corrected = ProbabilitySeries(
        values=tuple(
            distribution[max(distribution, key=lambda label: distribution[label])]
            for distribution in corrected_distributions
        ),
        semantics=run.probabilities.semantics,
    )
    outcomes = [run.outcomes[index] for index in eval_rows]

    raw_best = cascade.optimal_threshold(raw, outcomes, 0.02, 0.30)
    corrected_best = cascade.optimal_threshold(corrected, outcomes, 0.02, 0.30)

    assert raw_best.threshold != corrected_best.threshold
    assert corrected_best.threshold < raw_best.threshold


# Guards


def test_a_confidence_series_is_allowed_here_because_gating_is_its_job() -> None:
    """Unlike calibration, a cascade may gate on confidence. It ranks, and that is enough."""
    rows = cascade.cascade_sweep(
        ConfidenceSeries(values=(0.9, 0.2)), [True, False], 0.1, 1.0, thresholds=[0.5]
    )
    assert rows[0].covered == 1


@pytest.mark.parametrize(("escalation", "error"), [(-0.1, 1.0), (1.0, -0.1)])
def test_negative_costs_are_rejected(escalation: float, error: float) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        cascade.cascade_sweep(a_series((0.5, 0.6)), [True, False], escalation, error)


def test_mismatched_column_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="must match"):
        cascade.cascade_sweep(a_series((0.5, 0.6)), [True], 0.1, 1.0)


def test_escalating_every_case_is_a_candidate_when_it_is_cheapest() -> None:
    """With every case wrong and errors dear, covering anything costs more (#34)."""
    series = ProbabilitySeries(values=(0.5, 0.7, 0.9), semantics="calibrated_claim")
    best = cascade.optimal_threshold(series, [False, False, False], 1.0, 100.0)

    assert best.covered == 0 and best.escalated == 3
    assert best.total_cost_usd == pytest.approx(3.0)
