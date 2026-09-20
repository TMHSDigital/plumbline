"""Calibration metrics, tested against mocks whose miscalibration is known.

The assertions here are written against the floor, never against a magic
epsilon. A perfectly calibrated model does not score ECE 0 on a finite sample,
so "ECE is small" is not a testable claim and tuning a threshold until it passes
would only encode the tuning.
"""

from __future__ import annotations

import statistics

import pytest

from plumbline.adapters.mock import MockAdapter
from plumbline.metrics import calibration
from plumbline.types import (
    ConfidenceSeries,
    NotCalibratableError,
    ProbabilitySeries,
)
from tests.helpers import gold_by_text, make_cases, measure

ACCURACY = 0.75
CALIBRATED = 1.0
OVERCONFIDENT = 0.5
UNDERCONFIDENT = 2.0


def measured(n_cases: int, temperature: float, *, seed: int = 7, accuracy: float = ACCURACY):
    cases = make_cases(n_cases)
    adapter = MockAdapter(
        gold_by_text(cases),
        seed=seed,
        accuracy=accuracy,
        calibration_temperature=temperature,
    )
    return measure(adapter, cases)


def ratios(run, *, n_boot: int = 800, floor_seed: int = 11) -> tuple[float, float]:
    """Measured ECE and MCE, each divided by its own calibrated-model p95."""
    floors = calibration.calibration_floor(run.probabilities, n_boot=n_boot, seed=floor_seed)
    return (
        calibration.ece(run.probabilities, run.outcomes) / floors["ece"].p95,
        calibration.mce(run.probabilities, run.outcomes) / floors["mce"].p95,
    )


# Hand-computable values, so the formulas are pinned before any mock is involved


def a_series(values: tuple[float, ...]) -> ProbabilitySeries:
    return ProbabilitySeries(values=values, semantics="calibrated_claim")


def test_ece_matches_a_hand_computed_value() -> None:
    series = a_series((0.1, 0.1, 0.9, 0.9))
    outcomes = [False, False, True, False]
    # Bin [0.1, 0.2): n=2, mean 0.1, accuracy 0.0, gap 0.1
    # Bin [0.9, 1.0): n=2, mean 0.9, accuracy 0.5, gap 0.4
    assert calibration.ece(series, outcomes) == pytest.approx(0.5 * 0.1 + 0.5 * 0.4)


def test_mce_is_the_largest_gap_not_the_average() -> None:
    series = a_series((0.1, 0.1, 0.9, 0.9))
    outcomes = [False, False, True, False]
    assert calibration.mce(series, outcomes, min_bin_count=1) == pytest.approx(0.4)


def test_brier_matches_a_hand_computed_value() -> None:
    series = a_series((0.1, 0.1, 0.9, 0.9))
    outcomes = [False, False, True, False]
    assert calibration.brier(series, outcomes) == pytest.approx((0.01 + 0.01 + 0.01 + 0.81) / 4)


def test_multiclass_brier_matches_a_hand_computed_value() -> None:
    assert calibration.multiclass_brier([{"a": 0.7, "b": 0.3}], ["a"]) == pytest.approx(0.09 + 0.09)


def test_reliability_bins_account_for_every_row() -> None:
    run = measured(1000, CALIBRATED)
    bins = calibration.reliability(run.probabilities, run.outcomes)
    assert sum(single.count for single in bins) == 1000
    for single in bins:
        assert single.lower <= single.mean_probability <= single.upper
        assert 0.0 <= single.accuracy <= 1.0


# The floor, which is the point of this phase


@pytest.mark.parametrize("n_cases", [500, 4000])
def test_a_calibrated_mock_lands_inside_its_own_noise_floor(n_cases: int) -> None:
    """ECE near 0.016 at n=4000 is not evidence of miscalibration. It is the floor."""
    run = measured(n_cases, CALIBRATED)
    floors = calibration.calibration_floor(run.probabilities, n_boot=800, seed=11)
    measured_ece = calibration.ece(run.probabilities, run.outcomes)

    assert not calibration.is_distinguishable(measured_ece, floors["ece"])
    assert "not distinguishable" in calibration.verdict(measured_ece, floors["ece"])


@pytest.mark.parametrize("n_cases", [500, 4000])
@pytest.mark.parametrize("temperature", [OVERCONFIDENT, UNDERCONFIDENT])
def test_a_skewed_mock_lands_far_outside_the_floor(n_cases: int, temperature: float) -> None:
    run = measured(n_cases, temperature)
    floors = calibration.calibration_floor(run.probabilities, n_boot=800, seed=11)
    measured_ece = calibration.ece(run.probabilities, run.outcomes)

    assert calibration.is_distinguishable(measured_ece, floors["ece"])
    assert measured_ece > 2.5 * floors["ece"].p95
    assert "distinguishable from sampling noise" in calibration.verdict(measured_ece, floors["ece"])


def test_the_floor_grows_as_the_dataset_shrinks() -> None:
    """The reason a 500 row result needs the floor printed next to it."""
    large = calibration.calibration_floor(
        measured(4000, CALIBRATED).probabilities, n_boot=800, seed=11
    )
    small = calibration.calibration_floor(
        measured(500, CALIBRATED).probabilities, n_boot=800, seed=11
    )
    assert small["ece"].p95 > 2 * large["ece"].p95


def test_synthetic_floor_lets_a_run_be_sized_before_it_is_paid_for() -> None:
    small = calibration.synthetic_floor(200, accuracy=0.8, n_boot=400)
    large = calibration.synthetic_floor(5000, accuracy=0.8, n_boot=400)
    assert small["ece"].p95 > large["ece"].p95
    assert large["ece"].n == 5000


def test_mce_is_a_far_weaker_detector_than_ece() -> None:
    """MCE is a maximum over bins, so one sparse bin decides it.

    Measured here: a mock reporting probabilities near 0.91 while answering
    correctly 75 percent of the time is caught by ECE at four to fourteen times
    its floor, and is not reliably caught by MCE at all, at either sample size.
    This is why the report prints MCE with its floor attached and does not let
    MCE alone carry a conclusion.
    """
    for n_cases in (500, 4000):
        for seed in (1, 2, 3):
            run = measured(n_cases, OVERCONFIDENT, seed=seed)
            ece_ratio, mce_ratio = ratios(run)
            assert ece_ratio > 3.0, f"ECE missed the skew at n={n_cases}, seed={seed}"
            assert mce_ratio < 2.0
            assert ece_ratio > 3.0 * mce_ratio


def test_mce_on_a_calibrated_mock_stays_inside_its_floor() -> None:
    run = measured(4000, CALIBRATED)
    floors = calibration.calibration_floor(run.probabilities, n_boot=800, seed=11)
    assert not calibration.is_distinguishable(
        calibration.mce(run.probabilities, run.outcomes), floors["mce"]
    )


def test_mce_refuses_rather_than_reporting_a_sparse_bin() -> None:
    series = a_series((0.05, 0.95))
    with pytest.raises(ValueError, match="no bin holds at least"):
        calibration.mce(series, [False, True], min_bin_count=10)


# Brier


def test_a_calibrated_brier_equals_the_mean_of_p_times_one_minus_p() -> None:
    """The exact target for a calibrated model, so this is not a tuned tolerance."""
    run = measured(4000, CALIBRATED)
    expected = statistics.fmean(
        prediction.prob_selected * (1 - prediction.prob_selected) for prediction in run.predictions
    )
    assert calibration.brier(run.probabilities, run.outcomes) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("temperature", [OVERCONFIDENT, UNDERCONFIDENT])
def test_skew_makes_brier_worse_because_it_is_a_proper_scoring_rule(temperature: float) -> None:
    calibrated = measured(4000, CALIBRATED)
    skewed = measured(4000, temperature)
    assert calibration.brier(skewed.probabilities, skewed.outcomes) > calibration.brier(
        calibrated.probabilities, calibrated.outcomes
    )


def test_multiclass_brier_equals_its_identity_on_a_calibrated_model() -> None:
    """For a calibrated model the score is exactly ``1 - E[sum_k p_k ** 2]``."""
    run = measured(4000, CALIBRATED)
    identity = 1 - statistics.fmean(
        sum(value * value for value in prediction.distribution.values())
        for prediction in run.predictions
    )
    assert calibration.multiclass_brier(run.distributions, run.gold_labels) == pytest.approx(
        identity, abs=0.02
    )


def test_multiclass_brier_refuses_a_missing_distribution() -> None:
    with pytest.raises(NotCalibratableError, match="report no distribution"):
        calibration.multiclass_brier([{"a": 1.0}, None], ["a", "a"])


# Binning


@pytest.mark.parametrize("binning", ["equal_width", "equal_count"])
def test_both_binning_schemes_are_supported(binning: str) -> None:
    run = measured(1000, CALIBRATED)
    assert calibration.ece(run.probabilities, run.outcomes, binning=binning) >= 0.0


def test_equal_count_bins_are_near_equally_populated() -> None:
    run = measured(1000, CALIBRATED)
    counts = [
        single.count
        for single in calibration.reliability(
            run.probabilities, run.outcomes, binning="equal_count"
        )
    ]
    assert max(counts) - min(counts) <= 1


def test_equal_width_bins_are_sparse_at_the_edges() -> None:
    """The documented cost of the default scheme, made visible."""
    run = measured(1000, CALIBRATED)
    counts = [
        single.count
        for single in calibration.reliability(
            run.probabilities, run.outcomes, binning="equal_width"
        )
    ]
    assert max(counts) > 5 * min(counts)


def test_an_unknown_binning_scheme_is_rejected() -> None:
    run = measured(100, CALIBRATED)
    with pytest.raises(ValueError, match="equal_width"):
        calibration.ece(run.probabilities, run.outcomes, binning="quantile")


# The refusals that keep the distinction real


@pytest.mark.parametrize(
    "function",
    [
        calibration.ece,
        calibration.mce,
        calibration.brier,
        calibration.reliability,
        calibration.calibration_floor,
    ],
)
def test_calibration_functions_reject_a_confidence_series(function) -> None:
    """The central guard. Confidence is not a probability of correctness.

    Enforced at the type boundary rather than by a naming convention, so there
    is no call site at which the wrong column can be passed by accident.
    """
    confidences = ConfidenceSeries(values=(0.9, 0.8, 0.7, 0.6))
    with pytest.raises(NotCalibratableError, match="confidence is not a probability"):
        if function is calibration.calibration_floor:
            function(confidences)
        else:
            function(confidences, [True, False, True, False])


def test_calibration_refuses_an_adapter_that_reports_no_probability() -> None:
    series = ProbabilitySeries(values=(None, None, None), semantics="none")
    with pytest.raises(NotCalibratableError, match="not reported"):
        calibration.ece(series, [True, False, True])


def test_calibration_refuses_missing_values_rather_than_imputing_zero() -> None:
    series = ProbabilitySeries(values=(0.9, None, 0.7), semantics="calibrated_claim")
    with pytest.raises(NotCalibratableError, match="missing"):
        calibration.ece(series, [True, False, True])


def test_a_restricted_softmax_is_measurable_but_stays_labeled() -> None:
    """It is not excluded from calibration. It is grouped apart from a claim."""
    cases = make_cases(500)
    adapter = MockAdapter(
        gold_by_text(cases), seed=7, accuracy=ACCURACY, probability_semantics="restricted_softmax"
    )
    run = measure(adapter, cases)
    assert run.probabilities.semantics == "restricted_softmax"
    assert calibration.ece(run.probabilities, run.outcomes) >= 0.0


def test_a_mismatched_row_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="must match"):
        calibration.ece(a_series((0.5, 0.5)), [True])


def test_semantics_none_with_a_value_present_is_a_contradiction() -> None:
    with pytest.raises(ValueError, match="every value must be None"):
        ProbabilitySeries(values=(0.5,), semantics="none")
