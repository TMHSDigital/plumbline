"""The mock must produce the accuracy and the calibration skew it was configured with.

Everything the metrics package later claims rests on these tests. If the mock is
not what it says it is, a passing ECE test in Phase 2 proves nothing.
"""

from __future__ import annotations

import math
import random

import pytest

from plumbline.adapters.mock import MockAdapter
from plumbline.types import docs_confidence
from tests.helpers import (
    accuracy_of,
    gold_by_text,
    make_cases,
    make_variable_width_cases,
    max_bin_gap,
    run,
)

N_CASES = 4000


def build(cases, **overrides) -> MockAdapter:
    config: dict = {"accuracy": 0.8, "seed": 7}
    config.update(overrides)
    return MockAdapter(gold_by_text(cases), **config)


# Accuracy


@pytest.mark.parametrize("target", [0.30, 0.55, 0.80, 0.95])
def test_accuracy_matches_configuration(target: float) -> None:
    cases = make_cases(N_CASES)
    results = run(build(cases, accuracy=target), cases)
    assert accuracy_of(results) == pytest.approx(target, abs=0.03)


@pytest.mark.parametrize("width", [2, 3, 8])
def test_accuracy_holds_for_any_number_of_labels(width: int) -> None:
    labels = tuple(f"label_{index}" for index in range(width))
    cases = make_cases(N_CASES, labels=labels)
    results = run(build(cases, accuracy=0.7), cases)
    assert accuracy_of(results) == pytest.approx(0.7, abs=0.03)


@pytest.mark.parametrize("temperature", [0.4, 1.0, 2.5])
def test_accuracy_is_independent_of_calibration_skew(temperature: float) -> None:
    """Temperature is monotone on the log probabilities, so it cannot move the argmax.

    Skew is excluded from the per-case seed as well, so this holds exactly and
    case by case rather than only on average.
    """
    cases = make_cases(N_CASES)
    baseline = run(build(cases, accuracy=0.75, calibration_temperature=1.0), cases)
    skewed = run(build(cases, accuracy=0.75, calibration_temperature=temperature), cases)

    assert accuracy_of(skewed) == pytest.approx(0.75, abs=0.03)
    assert [pred.label for _, pred in skewed] == [pred.label for _, pred in baseline]


def test_below_chance_accuracy_is_refused() -> None:
    cases = make_cases(4, labels=("a", "b", "c", "d"))
    adapter = build(cases, accuracy=0.10)
    with pytest.raises(ValueError, match="below chance"):
        adapter.classify(cases[0].text, list(cases[0].labels))


# Calibration skew


def test_calibrated_mock_is_calibrated() -> None:
    """At temperature 1 the reported probability is the outcome frequency."""
    cases = make_cases(N_CASES)
    results = run(build(cases, accuracy=0.75, calibration_temperature=1.0), cases)
    pairs = [(pred.prob_selected, pred.label == case.gold_label) for case, pred in results]

    mean_probability = sum(probability for probability, _ in pairs) / len(pairs)
    assert mean_probability == pytest.approx(accuracy_of(results), abs=0.02)
    assert max_bin_gap(pairs, n_bins=5) < 0.05


def test_calibrated_mock_is_calibrated_over_the_whole_distribution() -> None:
    """Every label's probability is P(that label is gold), not only the top one.

    This is what lets Phase 2 compute a multiclass Brier score against a known
    target rather than against an assumption.
    """
    cases = make_cases(N_CASES)
    results = run(build(cases, accuracy=0.75, calibration_temperature=1.0), cases)
    pairs = [
        (probability, label == case.gold_label)
        for case, pred in results
        for label, probability in pred.distribution.items()
    ]
    assert max_bin_gap(pairs, n_bins=10) < 0.04


def test_overconfident_mock_overstates_its_probabilities() -> None:
    cases = make_cases(N_CASES)
    adapter = build(cases, accuracy=0.75, calibration_temperature=0.5)
    assert adapter.skew == "overconfident"

    results = run(adapter, cases)
    mean_probability = sum(pred.prob_selected for _, pred in results) / len(results)
    assert mean_probability - accuracy_of(results) > 0.05


def test_underconfident_mock_understates_its_probabilities() -> None:
    cases = make_cases(N_CASES)
    adapter = build(cases, accuracy=0.75, calibration_temperature=2.0)
    assert adapter.skew == "underconfident"

    results = run(adapter, cases)
    mean_probability = sum(pred.prob_selected for _, pred in results) / len(results)
    assert accuracy_of(results) - mean_probability > 0.05


def test_skew_is_monotone_in_temperature() -> None:
    """Lower temperature reports higher probabilities, case by case, always."""
    cases = make_cases(200)
    sharp = run(build(cases, accuracy=0.75, calibration_temperature=0.5), cases)
    flat = run(build(cases, accuracy=0.75, calibration_temperature=2.0), cases)
    for (_, sharp_pred), (_, flat_pred) in zip(sharp, flat, strict=True):
        assert sharp_pred.label == flat_pred.label
        assert sharp_pred.prob_selected > flat_pred.prob_selected


def test_temperature_one_reports_the_calibrated_distribution_unchanged() -> None:
    cases = make_cases(100)
    for _, pred in run(build(cases, calibration_temperature=1.0), cases):
        assert pred.distribution == pytest.approx(pred.raw["calibrated_distribution"])


@pytest.mark.parametrize("temperature", [0.25, 0.5, 2.0, 4.0])
def test_injected_temperature_is_exactly_recoverable(temperature: float) -> None:
    """Phase 3 fits this out, so Phase 1 proves the distortion is the one claimed.

    Dividing the reported log probabilities by ``1 / temperature`` must return
    the calibrated distribution exactly.
    """
    cases = make_cases(100)
    for _, pred in run(build(cases, calibration_temperature=temperature), cases):
        logits = {label: math.log(value) for label, value in pred.distribution.items()}
        rescaled = {label: value * temperature for label, value in logits.items()}
        peak = max(rescaled.values())
        weights = {label: math.exp(value - peak) for label, value in rescaled.items()}
        total = sum(weights.values())
        recovered = {label: value / total for label, value in weights.items()}
        assert recovered == pytest.approx(pred.raw["calibrated_distribution"], abs=1e-9)


# Determinism


def test_same_seed_same_answers_regardless_of_order() -> None:
    cases = make_cases(300)
    first = {case.id: pred for case, pred in run(build(cases), cases)}

    shuffled = list(cases)
    random.Random(99).shuffle(shuffled)
    second = {case.id: pred for case, pred in run(build(cases), shuffled)}

    assert first == second


def test_different_seeds_give_different_answers() -> None:
    cases = make_cases(300)
    first = [pred.prob_selected for _, pred in run(build(cases, seed=1), cases)]
    second = [pred.prob_selected for _, pred in run(build(cases, seed=2), cases)]
    assert first != second


# The reported record


def test_distribution_is_well_formed_and_the_label_is_its_argmax() -> None:
    cases = make_cases(300)
    for case, pred in run(build(cases), cases):
        assert set(pred.distribution) == set(case.labels)
        assert sum(pred.distribution.values()) == pytest.approx(1.0, abs=1e-9)
        assert pred.label == max(pred.distribution, key=lambda label: pred.distribution[label])
        assert pred.prob_selected == pytest.approx(pred.distribution[pred.label])


def test_confidence_is_the_documented_statistic_and_stays_in_range() -> None:
    cases = make_cases(300)
    for _, pred in run(build(cases), cases):
        assert 0.0 <= pred.confidence <= 1.0
        assert pred.confidence == pytest.approx(docs_confidence(pred.distribution))


def test_latency_is_positive_and_varies() -> None:
    cases = make_cases(300)
    latencies = [pred.latency_ms for _, pred in run(build(cases), cases)]
    assert all(latency > 0 for latency in latencies)
    assert len(set(latencies)) > 1


def test_model_reported_may_differ_from_model_requested() -> None:
    """A model alias can resolve elsewhere, so plumbline records both."""
    cases = make_cases(5)
    adapter = build(cases, model_requested="mock-alias", model_reported="mock-1.13")
    _, pred = run(adapter, cases)[0]
    assert adapter.model_requested == "mock-alias"
    assert pred.model_reported == "mock-1.13"


def test_tokens_are_absent_when_not_reported() -> None:
    """Absent tokens must stay absent, so cost later reads None and not zero."""
    cases = make_cases(20)
    for _, pred in run(build(cases, report_tokens=False), cases):
        assert pred.input_tokens is None
        assert pred.output_tokens is None
        assert pred.cost_usd is None


# probability_semantics


@pytest.mark.parametrize("semantics", ["calibrated_claim", "restricted_softmax", "none"])
def test_every_semantics_class_can_be_constructed(semantics: str) -> None:
    cases = make_cases(50)
    adapter = build(cases, probability_semantics=semantics)
    assert adapter.probability_semantics == semantics
    assert len(run(adapter, cases)) == 50


def test_semantics_none_reports_a_label_and_nothing_else() -> None:
    cases = make_cases(N_CASES)
    adapter = build(cases, accuracy=0.75, probability_semantics="none")
    results = run(adapter, cases)

    assert accuracy_of(results) == pytest.approx(0.75, abs=0.03)
    for _, pred in results:
        assert pred.prob_selected is None
        assert pred.distribution is None
        assert pred.confidence is None


def test_restricted_softmax_mock_is_otherwise_identical_to_a_calibrated_one() -> None:
    """Semantics is a label on the number, not a change to it.

    The whole point of grouping by semantics in the report is that two numbers
    can be numerically identical and still must not be compared, because one
    carries a calibration claim and the other does not.
    """
    cases = make_cases(200)
    claimed = run(build(cases, probability_semantics="calibrated_claim"), cases)
    restricted = run(build(cases, probability_semantics="restricted_softmax"), cases)
    for (_, left), (_, right) in zip(claimed, restricted, strict=True):
        assert left.prob_selected == right.prob_selected
        assert left.label == right.label


def test_unknown_semantics_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="probability_semantics"):
        MockAdapter({}, probability_semantics="calibrated")  # type: ignore[arg-type]


# Refusals and bad input


def test_missing_answer_key_raises_rather_than_scoring_wrong() -> None:
    adapter = MockAdapter({})
    with pytest.raises(KeyError, match="gold_by_text"):
        adapter.classify("a case it has never seen", ["a", "b"])


def test_gold_label_outside_the_label_set_raises() -> None:
    adapter = MockAdapter({"text": "missing"})
    with pytest.raises(ValueError, match="is not among the labels"):
        adapter.classify("text", ["a", "b"])


@pytest.mark.parametrize("labels", [["only"], ["a", "a"]])
def test_degenerate_label_sets_are_rejected(labels: list[str]) -> None:
    adapter = MockAdapter({"text": labels[0]})
    with pytest.raises(ValueError):
        adapter.classify("text", labels)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accuracy", 1.5),
        ("accuracy", -0.1),
        ("calibration_temperature", 0.0),
        ("calibration_temperature", -1.0),
        ("concentration", 0.0),
        ("latency_p50_ms", -1.0),
    ],
)
def test_invalid_configuration_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        MockAdapter({}, **{field: value})


# Variable-width corpora, which Phase 2 needs


def test_variable_width_corpus_runs_and_keeps_its_widths() -> None:
    cases = make_variable_width_cases(400)
    results = run(build(cases, accuracy=0.6), cases)
    assert accuracy_of(results) == pytest.approx(0.6, abs=0.06)
    assert len({len(pred.distribution) for _, pred in results}) > 1
