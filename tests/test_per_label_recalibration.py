"""Per-label temperature scaling, the fallback for a diagnosed per-label bias (#4).

A model can be well calibrated on most labels and overconfident on one. One
global temperature cannot reach that: flattening enough for the skewed label
over-flattens the honest ones. A temperature per predicted label can, and it
has to earn its place by the same gate as the global fit: fitted on one half,
judged on the other, refused when it does not recover calibration.
"""

from __future__ import annotations

import dataclasses

from plumbline.adapters.mock import MockAdapter
from plumbline.metrics import recalibration
from plumbline.report import markdown
from plumbline.runner import execute
from plumbline.types import Prediction, apply_temperature
from tests.helpers import Measured, per_label_skew, redistort
from tests.test_recalibration import LABELS, N_BOOT_CI, N_BOOT_FLOOR, recalibrate, run_for


def per_label(run: Measured, **overrides: object) -> recalibration.PerLabelResult:
    options: dict = {
        "n_boot_ci": N_BOOT_CI,
        "n_boot_floor": N_BOOT_FLOOR,
        "distributions": run.distributions,
        "gold_labels": run.gold_labels,
    }
    options.update(overrides)
    return recalibration.recalibrate_per_label(
        run.probabilities,
        run.outcomes,
        predicted_labels=[prediction.label for prediction in run.predictions],
        **options,
    )


def billing_biased(seed: int) -> Measured:
    return redistort(run_for(4000, 1.0, seed=seed), per_label_skew(("billing",)))


def test_a_per_label_bias_that_defeats_the_global_fit_is_mostly_recovered_per_label() -> None:
    """Most of the way, not all of it, and the test says which.

    Conditioning on the predicted label is a selection: the rows a model calls
    billing are not a calibrated sample of anything, so the fitted temperature
    for billing (about 2.0) falls short of the exact inverse of the injected
    skew (2.2). What per-label buys is measured against what the global fit
    leaves, on the same held-out rows.
    """
    for seed in (1, 2):
        run = billing_biased(seed)
        global_fit = recalibrate(run)
        assert not global_fit.fit_is_complete

        result = per_label(run)

        assert result.recommendation != "refused", result.summary()
        assert result.improvement_is_material
        assert result.residual_ratio < 1.25, result.summary()
        assert result.after.ece < 0.8 * global_fit.after.ece


def test_the_skewed_label_gets_the_correction_and_the_honest_ones_do_not() -> None:
    result = per_label(billing_biased(1))
    by_label = {fit.label: fit for fit in result.labels}

    billing = by_label["billing"]
    assert billing.temperature is not None and billing.temperature > 1.5
    for label in ("returns", "shipping", "other"):
        honest = by_label[label]
        assert honest.temperature is not None and abs(honest.temperature - 1.0) < 0.25


def test_it_judges_on_the_same_held_out_rows_as_the_global_fit() -> None:
    run = billing_biased(1)

    assert per_label(run, seed=3).split == recalibrate(run, seed=3).split


def test_a_label_below_the_row_gate_keeps_its_probabilities_and_says_why() -> None:
    result = per_label(billing_biased(1), min_label_rows=5000)

    assert all(fit.temperature is None for fit in result.labels)
    assert all(fit.n_fit < 5000 for fit in result.labels)
    assert result.recommendation == "refused"
    assert "5000" in result.summary()


def test_an_honest_model_is_not_given_a_per_label_correction() -> None:
    # The overfitting check: with nothing per-label to find, several free
    # parameters must not manufacture a recommendation.
    result = per_label(run_for(4000, 1.0, seed=1))

    assert result.recommendation == "refused"


class BillingOverconfident(MockAdapter):
    """The mock, with its answers sharpened whenever it says billing."""

    def classify(self, text: str, labels: list[str], **kwargs: object) -> Prediction:
        answer = super().classify(text, labels, **kwargs)  # type: ignore[arg-type]
        if answer.label != "billing" or answer.distribution is None:
            return answer
        sharpened = apply_temperature(answer.distribution, 0.45)
        return dataclasses.replace(
            answer, distribution=sharpened, prob_selected=sharpened[answer.label]
        )


def a_report(adapter_type: type[MockAdapter], temperature: float = 1.0) -> str:
    from tests.helpers import gold_by_text, make_cases

    cases = make_cases(4000, labels=LABELS)
    adapter = adapter_type(
        gold_by_text(cases), seed=1, accuracy=0.75, calibration_temperature=temperature
    )
    result = execute.run(adapter, cases, workers=1)
    return markdown.render([result], options=markdown.ReportOptions(n_boot=N_BOOT_FLOOR))


def test_the_report_tries_per_label_after_the_global_fit_is_refused() -> None:
    document = a_report(BillingOverconfident)
    section = document.split("#### Per-label fallback", 1)

    assert len(section) == 2, "no per-label block after a per-label bias"
    block = section[1].split("\n#### ", 1)[0]
    assert "| billing |" in block
    assert "Apply" in block
    assert "not comparable" in block


def test_the_report_does_not_try_per_label_when_one_temperature_is_the_right_shape() -> None:
    document = a_report(MockAdapter, temperature=0.5)

    assert "#### Per-label fallback" not in document
