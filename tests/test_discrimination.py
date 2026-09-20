"""Discrimination, including the confidence against prob_selected identity.

The claim under test has two halves. Within a fixed label set the docs'
confidence formula is an increasing affine function of ``prob_selected``, so the
two rank identically and AUROC is equal to the bit. Across a corpus whose cases
carry differing option counts they come apart. Both halves are asserted, not
noted.
"""

from __future__ import annotations

import numpy as np
import pytest

from plumbline.adapters.mock import MockAdapter
from plumbline.metrics import discrimination
from plumbline.types import ConfidenceSeries, ProbabilitySeries
from tests.helpers import gold_by_text, make_cases, make_variable_width_cases, measure


def a_score_series(values: tuple[float, ...]) -> ConfidenceSeries:
    return ConfidenceSeries(values=values)


# Hand-computable AUROC


def test_perfect_separation_scores_one() -> None:
    assert (
        discrimination.auroc(a_score_series((0.9, 0.8, 0.7, 0.6)), [True, True, False, False])
        == 1.0
    )


def test_perfectly_wrong_separation_scores_zero() -> None:
    assert (
        discrimination.auroc(a_score_series((0.9, 0.8, 0.7, 0.6)), [False, False, True, True])
        == 0.0
    )


def test_a_constant_score_scores_one_half() -> None:
    """An adapter that reports the same confidence every time ranks nothing.

    Handled by mid-ranks. Breaking ties by sort order would score this 1.0 or
    0.0 depending on how the rows happened to be ordered.
    """
    assert (
        discrimination.auroc(a_score_series((0.5, 0.5, 0.5, 0.5)), [True, True, False, False])
        == 0.5
    )


def test_partial_ties_are_split_rather_than_broken() -> None:
    # One correct at 0.8, incorrect at 0.8 and 0.2. Half credit plus full credit.
    assert discrimination.auroc(
        a_score_series((0.8, 0.8, 0.2)), [True, False, False]
    ) == pytest.approx(0.75)


def test_auroc_refuses_when_a_class_is_empty() -> None:
    with pytest.raises(ValueError, match="at least one correct and one incorrect"):
        discrimination.auroc(a_score_series((0.9, 0.8)), [True, True])


def test_auroc_accepts_a_probability_series_too() -> None:
    """Both series are valid ranking signals, which is what makes the parity test possible."""
    series = ProbabilitySeries(values=(0.9, 0.8, 0.7, 0.6), semantics="calibrated_claim")
    assert discrimination.auroc(series, [True, True, False, False]) == 1.0


# The parity and divergence assertions


def test_confidence_and_prob_selected_rank_identically_at_fixed_option_count() -> None:
    """Affine and increasing in prob_selected, so the AUROCs are equal exactly.

    Not approximately. The report must say so rather than presenting the two as
    independent evidence on a fixed-width corpus.
    """
    cases = make_cases(4000)
    run = measure(MockAdapter(gold_by_text(cases), seed=7, accuracy=0.75), cases)

    assert discrimination.auroc(run.probabilities, run.outcomes) == discrimination.auroc(
        run.confidences, run.outcomes
    )


def test_confidence_and_prob_selected_diverge_when_option_counts_vary() -> None:
    """Widths 2, 3, 5, and 8 in one corpus, where the normalization starts to bite.

    Direction matters and is recorded here. This mock makes ``prob_selected`` the
    true probability of correctness by construction, so it is already the optimal
    ranker and normalizing away the option count can only lose information.
    Confidence therefore scores lower here, by roughly 0.03 across seeds.

    That is a property of this fixture, not a general result. See
    ``test_normalization_wins_when_the_raw_probability_is_option_count_inflated``
    for the case that runs the other way, which is the one that matters for a
    restricted softmax.
    """
    cases = make_variable_width_cases(4000)
    for seed in (1, 2, 3):
        run = measure(MockAdapter(gold_by_text(cases), seed=seed, accuracy=0.6), cases)
        by_probability = discrimination.auroc(run.probabilities, run.outcomes)
        by_confidence = discrimination.auroc(run.confidences, run.outcomes)

        assert abs(by_probability - by_confidence) > 0.01
        assert by_probability > by_confidence


def test_normalization_wins_when_the_raw_probability_is_option_count_inflated() -> None:
    """The case confidence exists for, built directly rather than through the mock.

    Two sub-populations, two options and thirty-two, where the true chance of
    being correct is the option-count-normalized statistic rather than the raw
    top probability. This is the shape a restricted softmax has: its number is
    conditional on the options supplied, so a narrow label set inflates it.

    The normalized score is the better ranker here, consistently but not
    dramatically: the gap runs 0.029 to 0.044 across draws, measured before this
    margin was chosen. The threshold below sits under the observed minimum with
    room to spare rather than being tuned up to whatever passed.
    """
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        raw: list[float] = []
        normalized: list[float] = []
        outcomes: list[bool] = []

        for width, size in ((2, 2000), (32, 2000)):
            chance = 1.0 / width
            top = rng.uniform(chance, 1.0, size=size)
            confidence = (width * top - 1.0) / (width - 1.0)
            raw.extend(top.tolist())
            normalized.extend(confidence.tolist())
            outcomes.extend((rng.random(size) < confidence).tolist())

        by_raw = discrimination.auroc(ProbabilitySeries(tuple(raw), "restricted_softmax"), outcomes)
        by_normalized = discrimination.auroc(ConfidenceSeries(tuple(normalized)), outcomes)

        assert by_normalized > by_raw + 0.015


# Threshold sweep


def test_the_sweep_runs_from_one_half_to_ninety_five_hundredths() -> None:
    assert discrimination.DEFAULT_THRESHOLDS[0] == 0.5
    assert discrimination.DEFAULT_THRESHOLDS[-1] == 0.95
    assert len(discrimination.DEFAULT_THRESHOLDS) == 10


def test_coverage_falls_as_the_threshold_rises() -> None:
    cases = make_cases(2000)
    run = measure(MockAdapter(gold_by_text(cases), seed=7, accuracy=0.75), cases)
    rows = discrimination.threshold_sweep(run.confidences, run.outcomes)

    coverages = [row.coverage for row in rows]
    assert coverages == sorted(coverages, reverse=True)
    assert all(0.0 <= coverage <= 1.0 for coverage in coverages)


def test_gating_on_a_useful_signal_raises_accuracy_on_what_is_kept() -> None:
    cases = make_cases(2000)
    run = measure(MockAdapter(gold_by_text(cases), seed=7, accuracy=0.75), cases)
    rows = discrimination.threshold_sweep(run.confidences, run.outcomes)

    kept = [row for row in rows if row.accuracy_covered is not None and row.covered > 50]
    assert kept[-1].accuracy_covered > kept[0].accuracy_covered
    assert kept[0].accuracy_covered > run.accuracy


def test_a_threshold_nothing_clears_reports_no_accuracy_rather_than_zero() -> None:
    rows = discrimination.threshold_sweep(
        a_score_series((0.1, 0.2)), [True, False], thresholds=[0.9]
    )
    assert rows[0].covered == 0
    assert rows[0].coverage == 0.0
    assert rows[0].accuracy_covered is None


def test_the_escalation_bound_is_the_optimistic_one() -> None:
    """Everything escalated is assumed correct, which the report states plainly."""
    rows = discrimination.threshold_sweep(
        a_score_series((0.9, 0.1)), [True, False], thresholds=[0.5]
    )
    assert rows[0].covered == 1
    assert rows[0].accuracy_covered == 1.0
    assert rows[0].accuracy_overall_if_escalation_is_correct == 1.0


def test_a_mismatched_row_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="must match"):
        discrimination.auroc(a_score_series((0.5, 0.5)), [True])


def test_discrimination_refuses_missing_confidence_values() -> None:
    with pytest.raises(ValueError, match="missing"):
        discrimination.auroc(ConfidenceSeries(values=(0.9, None)), [True, False])
