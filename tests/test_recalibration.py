"""Temperature scaling: what it recovers, what it cannot, and what it must refuse.

Three claims are under test.

The split is enforced in code, not by discipline. Fitting and reporting on the
same rows would manufacture an improvement that does not survive new data, so
``Split`` refuses overlapping index sets and every reported metric comes from the
held-out half.

The fit is an interval. A calibrated model produces an interval spanning 1.0,
and the correct response is to say no recalibration is justified rather than to
ship a temperature fitted to noise.

The residual is real. The mock's own skew is a pure temperature, so recovering
it is nearly tautological: the model class is correctly specified and a good fit
is guaranteed. The misspecified fixtures are the ones that say something about
real data, and they say something uncomfortable.
"""

from __future__ import annotations

import pytest

from plumbline.adapters.mock import MockAdapter
from plumbline.metrics import calibration, recalibration
from plumbline.metrics.recalibration import Split
from plumbline.types import (
    InsufficientDataError,
    ProbabilitySeries,
    apply_temperature,
)
from tests.helpers import (
    Measured,
    gold_by_text,
    make_cases,
    measure,
    per_label_skew,
    piecewise_skew,
    redistort,
)

LABELS = ("billing", "returns", "shipping", "other")

# Bootstraps are the slow part, so tests run modest counts. Everything is seeded,
# so a lower count changes the numbers but never makes a test flaky.
N_BOOT_CI = 80
N_BOOT_FLOOR = 300

_RUNS: dict[tuple, Measured] = {}


def run_for(n_cases: int, temperature: float, seed: int = 7) -> Measured:
    key = (n_cases, temperature, seed)
    if key not in _RUNS:
        cases = make_cases(n_cases, labels=LABELS)
        adapter = MockAdapter(
            gold_by_text(cases),
            seed=seed,
            accuracy=0.75,
            calibration_temperature=temperature,
        )
        _RUNS[key] = measure(adapter, cases)
    return _RUNS[key]


def recalibrate(run: Measured, *, multiclass: bool = True, **overrides):
    options: dict = {"n_boot_ci": N_BOOT_CI, "n_boot_floor": N_BOOT_FLOOR}
    options.update(overrides)
    if multiclass:
        options.setdefault("distributions", run.distributions)
        options.setdefault("gold_labels", run.gold_labels)
    return recalibration.recalibrate(run.probabilities, run.outcomes, **options)


# The split, which is the invariant everything else rests on


def test_a_split_may_not_share_a_row_between_its_halves() -> None:
    with pytest.raises(ValueError, match="appear in both halves"):
        Split(fit_indices=(0, 1, 2), eval_indices=(2, 3, 4))


def test_a_split_may_not_repeat_a_row() -> None:
    with pytest.raises(ValueError, match="may not repeat"):
        Split(fit_indices=(0, 0, 1), eval_indices=(2, 3))


def test_a_split_may_not_leave_a_half_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Split(fit_indices=(0, 1), eval_indices=())


def test_make_split_uses_every_row_exactly_once() -> None:
    split = make = recalibration.make_split(1000, seed=3)
    assert set(make.fit_indices) | set(split.eval_indices) == set(range(1000))
    assert len(split.fit_indices) + len(split.eval_indices) == 1000


@pytest.mark.parametrize(("fraction", "expected_fit"), [(0.5, 500), (0.3, 300), (0.8, 800)])
def test_make_split_honours_the_requested_fraction(fraction: float, expected_fit: int) -> None:
    split = recalibration.make_split(1000, fit_fraction=fraction, seed=3)
    assert len(split.fit_indices) == expected_fit


def test_recalibration_refuses_too_few_evaluation_rows() -> None:
    """A temperature fitted on a handful of rows arrives looking like a measurement."""
    run = run_for(300, 0.5)
    with pytest.raises(InsufficientDataError, match="at least 200 held-out evaluation rows"):
        recalibrate(run)


def test_every_reported_metric_comes_from_the_held_out_rows() -> None:
    """Recompute the before figure independently on the eval half and compare.

    This is the assertion that proves the split is honored rather than merely
    constructed. If the reported "before" had been computed on all rows, or on
    the fit rows, these would not match.
    """
    run = run_for(4000, 0.5)
    split = recalibration.make_split(4000, seed=0)
    result = recalibrate(run, split=split)

    eval_series = ProbabilitySeries(
        values=tuple(run.probabilities.values[index] for index in split.eval_indices),
        semantics=run.probabilities.semantics,
    )
    eval_outcomes = [run.outcomes[index] for index in split.eval_indices]

    assert result.before.ece == pytest.approx(calibration.ece(eval_series, eval_outcomes))
    assert result.n_eval == len(split.eval_indices)
    assert result.n_fit == len(split.fit_indices)
    assert set(result.split.fit_indices).isdisjoint(result.split.eval_indices)


def test_a_split_indexing_rows_that_do_not_exist_is_rejected() -> None:
    run = run_for(4000, 0.5)
    with pytest.raises(ValueError, match="rows that do not exist"):
        recalibrate(run, split=Split(fit_indices=tuple(range(2000)), eval_indices=(9999,)))


# The well specified case, which proves the machinery and little else


@pytest.mark.parametrize("injected", [0.5, 0.7, 2.0])
def test_the_interval_covers_the_temperature_that_was_injected(injected: float) -> None:
    """The mock skews by ``softmax(log p / T_inject)``, so the correction is ``1 / T_inject``.

    Asserted against the interval rather than the point estimate. The point
    estimate is fitted on 2000 rows and does not land on the truth exactly, which
    is the whole reason the interval is reported.
    """
    result = recalibrate(run_for(4000, injected))
    low, high = result.temperature_ci
    assert low <= 1.0 / injected <= high
    assert result.justified


def test_a_well_specified_fit_returns_calibration_to_the_floor() -> None:
    result = recalibrate(run_for(4000, 0.5))
    assert result.after.ece < result.before.ece / 5
    assert result.fit_is_complete
    assert "accounts for the miscalibration" in result.summary()


def test_scaling_never_moves_a_predicted_label() -> None:
    """Monotone in log p, so recalibration changes probabilities and not decisions."""
    run = run_for(4000, 0.5)
    for prediction in run.predictions[:200]:
        assert prediction.distribution is not None
        for temperature in (0.3, 0.9, 1.0, 2.5, 8.0):
            scaled = apply_temperature(prediction.distribution, temperature)
            assert max(scaled, key=lambda label: scaled[label]) == prediction.label


# The interval discipline


@pytest.mark.parametrize("n_cases", [500, 1000, 4000])
def test_a_calibrated_model_yields_an_interval_spanning_one(n_cases: int) -> None:
    """The honest answer is that no correction is warranted, at any of these sizes."""
    result = recalibrate(run_for(n_cases, 1.0))
    low, high = result.temperature_ci
    assert low <= 1.0 <= high
    assert not result.justified
    assert result.recommendation == "refused"
    assert result.temperature_to_use is None
    assert "nothing to correct" in result.summary()


def test_a_genuinely_skewed_model_yields_an_interval_clear_of_one() -> None:
    result = recalibrate(run_for(4000, 0.5))
    low, _ = result.temperature_ci
    assert low > 1.0
    assert result.justified
    assert "no recalibration is justified" not in result.summary()


# Misspecification, which is what real data looks like


def test_a_piecewise_distortion_improves_a_lot_but_never_reaches_the_floor() -> None:
    """One temperature above mid-range and another below.

    A single fitted T is pulled between the two and fits neither. Measured across
    mock seeds it removes about 70 percent of the ECE and leaves the rest: post
    scaling ECE stays at 1.3 to 1.7 times the floor. The report has to say that,
    or a reader will take the large improvement as a finished job.
    """
    for seed in (1, 2):
        run = redistort(run_for(4000, 1.0, seed=seed), piecewise_skew())
        result = recalibrate(run)

        assert result.after.ece < 0.5 * result.before.ece
        assert not result.fit_is_complete
        assert result.residual_ratio > 1.2
        assert result.recommendation == "partial"
        assert result.temperature_to_use == result.temperature
        assert "wrong shape of correction" in result.summary()


def test_a_per_label_bias_is_not_reliably_improved_at_all() -> None:
    """One label overconfident, the rest honest. A global T cannot reach this.

    Flattening enough to fix the skewed label over-flattens the other three, so
    the net effect on ECE is a coin flip. Measured across mock and split seeds the
    post-scaling ECE runs from 0.77 to 1.60 times the pre-scaling value: sometimes
    better, more often worse. Recalibration is never complete here.

    This is stronger than "improves but does not finish", and it is the case a
    user most needs warned about, because the fitted T still arrives with a tight
    interval and looks authoritative.
    """
    ratios = []
    for mock_seed in (1, 2, 3):
        run = redistort(run_for(4000, 1.0, seed=mock_seed), per_label_skew(("billing",)))
        for split_seed in (0, 5):
            result = recalibrate(run, seed=split_seed)
            ratios.append(result.after.ece / result.before.ece)
            assert not result.fit_is_complete
            assert result.residual_ratio > 1.3

    assert max(ratios) > 1.15, "expected at least one seed where scaling made ECE worse"


def test_a_misspecified_fit_still_reports_a_tight_interval() -> None:
    """The interval measures sampling noise in the fit, not whether the model fits.

    A narrow interval is not evidence that temperature scaling was the right
    correction. Only the residual says that, which is why both are reported.
    """
    run = redistort(run_for(4000, 1.0, seed=1), per_label_skew(("billing",)))
    result = recalibrate(run)
    low, high = result.temperature_ci
    assert high - low < 0.35
    assert result.justified
    assert not result.fit_is_complete


# The binary form, for adapters that report no distribution


def test_the_binary_form_is_used_when_no_distribution_is_available() -> None:
    result = recalibrate(run_for(4000, 0.5), multiclass=False)
    assert result.method == "binary"
    assert result.after.multiclass_brier is None


def test_the_binary_form_is_itself_misspecified_against_a_multiclass_skew() -> None:
    """Scaling the top probability alone is an approximation of the real correction.

    The mock's distortion is a pure temperature on the whole distribution, so the
    multiclass form inverts it and lands inside the floor. The binary form, given
    the identical data, leaves post-scaling ECE at five times the floor on the
    underconfident mock. An adapter that reports no distribution therefore gets a
    worse correction even when the underlying miscalibration is simple.
    """
    run = run_for(4000, 2.0)
    multiclass = recalibrate(run)
    binary = recalibrate(run, multiclass=False)

    assert multiclass.fit_is_complete
    assert not binary.fit_is_complete
    assert binary.after.ece > 5 * multiclass.after.ece


def test_multiclass_is_preferred_whenever_a_distribution_exists() -> None:
    result = recalibrate(run_for(4000, 0.5))
    assert result.method == "multiclass"
    assert result.after.multiclass_brier is not None


# Input guards


def test_mismatched_column_lengths_are_rejected() -> None:
    series = ProbabilitySeries(values=(0.9, 0.8), semantics="calibrated_claim")
    with pytest.raises(ValueError, match="must match"):
        recalibration.recalibrate(series, [True])


def test_recalibration_refuses_an_adapter_reporting_no_probability() -> None:
    from plumbline.types import NotCalibratableError

    series = ProbabilitySeries(values=(None,) * 400, semantics="none")
    with pytest.raises(NotCalibratableError, match="not reported"):
        recalibration.recalibrate(series, [True] * 400)


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_a_non_positive_temperature_is_rejected(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        apply_temperature({"a": 0.6, "b": 0.4}, temperature)


# The recommendation gate, which is what protects a user from the fitted number


def test_a_pure_temperature_skew_is_recommended() -> None:
    result = recalibrate(run_for(4000, 0.5))
    assert result.recommendation == "recommended"
    assert result.reason == "fit_recovers_calibration"
    assert result.temperature_to_use == pytest.approx(result.temperature)
    assert "Recommended:" in result.summary()


def test_an_already_calibrated_model_is_refused_with_nothing_to_correct() -> None:
    """Not a failure of the fit. There was no miscalibration to remove."""
    result = recalibrate(run_for(4000, 1.0))
    assert result.recommendation == "refused"
    assert result.reason == "already_calibrated"
    assert result.already_calibrated
    assert result.temperature_to_use is None


def test_a_piecewise_distortion_is_partial_and_still_emits_a_temperature() -> None:
    result = recalibrate(redistort(run_for(4000, 1.0, seed=1), piecewise_skew()))
    assert result.recommendation == "partial"
    assert result.reason == "residual_above_floor"
    assert result.temperature_to_use is not None
    assert "Partial:" in result.summary()
    assert "some miscalibration survives it" in result.summary()


def test_a_per_label_bias_is_refused_on_the_seeds_where_scaling_degrades() -> None:
    """The path that protects users, tested across the seed range that produced it.

    On these fixtures the fitted interval is tight and clear of 1.0, so a reader
    following the interval alone would ship a temperature that makes calibration
    worse. The gate refuses instead, and names the cause.
    """
    refusals = 0
    for mock_seed in (1, 2, 3):
        run = redistort(run_for(4000, 1.0, seed=mock_seed), per_label_skew(("billing",)))
        for split_seed in (0, 5):
            result = recalibrate(run, seed=split_seed)
            if result.improvement <= result.noise_scale:
                refusals += 1
                assert result.recommendation == "refused"
                assert result.reason == "no_material_improvement"
                assert result.temperature_to_use is None
                assert "does not fit this model's miscalibration" in result.summary()
                assert "per-label bias" in result.summary()
                # The interval alone would have said go ahead.
                assert result.justified

    assert refusals >= 4, f"expected the degrading seeds to refuse, got {refusals}"


def test_the_refusal_names_a_next_step_rather_than_stopping() -> None:
    run = redistort(run_for(4000, 1.0, seed=2), per_label_skew(("billing",)))
    summary = recalibrate(run, seed=0).summary()
    assert "vector scaling" in summary
    assert "plumbline fits neither" in summary


def test_an_interval_spanning_one_is_refused_even_when_ece_improves() -> None:
    """A defensive path the fixtures do not reach, so it is exercised directly.

    Material improvement almost always drags the interval clear of 1.0, so this
    combination is rare. It is still a refusal: without an interval that excludes
    1.0 there is no evidence a correction is warranted.
    """
    result = _result_with(justified=False, before_ece=0.20, after_ece=0.02)
    assert result.reason == "interval_spans_one"
    assert result.recommendation == "refused"
    assert result.temperature_to_use is None


def test_the_reasons_are_checked_in_priority_order() -> None:
    """Nothing to correct outranks everything else, including a bad fit."""
    result = _result_with(justified=True, before_ece=0.005, after_ece=0.40)
    assert result.reason == "already_calibrated"


def _result_with(
    *, justified: bool, before_ece: float, after_ece: float
) -> recalibration.RecalibrationResult:
    """Build a result directly, to reach branches the fixtures do not produce."""
    from plumbline.metrics.calibration import FloorBand

    def band(p95: float) -> dict[str, FloorBand]:
        return {
            "ece": FloorBand(metric="ece", mean=p95 * 0.6, p95=p95, n=2000, n_bins=10, n_boot=300)
        }

    def metrics(value: float) -> recalibration.MetricSet:
        return recalibration.MetricSet(ece=value, mce=None, brier=0.2, multiclass_brier=None)

    return recalibration.RecalibrationResult(
        method="multiclass",
        semantics="calibrated_claim",
        had_distribution=True,
        temperature=1.4,
        temperature_ci=(0.9, 1.9) if not justified else (1.2, 1.6),
        justified=justified,
        split=Split(fit_indices=(0, 1), eval_indices=(2, 3)),
        seed=0,
        before=metrics(before_ece),
        after=metrics(after_ece),
        floor=band(0.02),
        floor_before=band(0.02),
    )


# The consequence of reporting no distribution


def test_the_binary_path_carries_the_no_distribution_warning() -> None:
    """Stated in one line wherever it applies, not left in a methodology file."""
    summary = recalibrate(run_for(4000, 2.0), multiclass=False).summary()
    assert "worse off twice" in summary
    assert "excluded from multiclass Brier" in summary
    assert "Noul answer carries this limitation" in summary


def test_the_multiclass_path_does_not_carry_the_warning() -> None:
    assert "worse off twice" not in recalibrate(run_for(4000, 2.0)).summary()
