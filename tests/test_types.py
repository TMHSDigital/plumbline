"""Record-level guards, including the ones that keep a bad adapter from lying."""

from __future__ import annotations

import pytest

from plumbline.types import PROBABILITY_SEMANTICS, Case, Prediction, docs_confidence


def a_prediction(**overrides) -> Prediction:
    fields: dict = {
        "label": "a",
        "prob_selected": 0.7,
        "distribution": {"a": 0.7, "b": 0.3},
        "confidence": 0.4,
        "latency_ms": 12.0,
        "cost_usd": None,
        "input_tokens": 100,
        "output_tokens": 2,
        "model_reported": "mock-1",
    }
    fields.update(overrides)
    return Prediction(**fields)


def test_semantics_has_exactly_three_classes() -> None:
    assert PROBABILITY_SEMANTICS == ("calibrated_claim", "restricted_softmax", "none")


# Prediction


def test_a_well_formed_prediction_is_accepted() -> None:
    assert a_prediction().prob_selected == 0.7


def test_none_probabilities_are_allowed_and_not_defaulted() -> None:
    pred = a_prediction(prob_selected=None, distribution=None, confidence=None)
    assert (pred.prob_selected, pred.distribution, pred.confidence) == (None, None, None)


@pytest.mark.parametrize("field", ["prob_selected", "confidence"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_probabilities_outside_the_unit_interval_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        a_prediction(**{field: value})


def test_negative_latency_is_rejected() -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        a_prediction(latency_ms=-1.0)


def test_a_distribution_missing_the_selected_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing from distribution"):
        a_prediction(label="c")


def test_a_distribution_that_does_not_sum_to_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected approximately 1"):
        a_prediction(distribution={"a": 0.7, "b": 0.9})


def test_approximate_sums_are_accepted_because_the_api_only_promises_approximate() -> None:
    assert a_prediction(distribution={"a": 0.7, "b": 0.3001}).label == "a"


# Case


def test_a_well_formed_case_is_accepted() -> None:
    case = Case(id="1", text="body", labels=("a", "b"), gold_label="a")
    assert case.gold_label == "a"


@pytest.mark.parametrize(
    ("labels", "gold", "message"),
    [
        (("a",), "a", "at least 2 labels"),
        (("a", "a"), "a", "duplicates"),
        (("a", "b"), "c", "not one of"),
    ],
)
def test_malformed_cases_are_rejected(labels, gold, message) -> None:
    with pytest.raises(ValueError, match=message):
        Case(id="1", text="body", labels=labels, gold_label=gold)


# docs_confidence


def test_a_point_mass_is_full_confidence() -> None:
    assert docs_confidence({"a": 1.0, "b": 0.0}) == pytest.approx(1.0)


@pytest.mark.parametrize("width", [2, 3, 5, 8])
def test_a_flat_distribution_is_zero_confidence(width: int) -> None:
    flat = {f"l{index}": 1.0 / width for index in range(width)}
    assert docs_confidence(flat) == pytest.approx(0.0, abs=1e-12)


def test_confidence_normalizes_for_option_count_and_prob_selected_does_not() -> None:
    """The divergence Phase 2 turns into an assertion.

    The same top probability means different things at different widths: 0.5 of
    two options is chance, 0.5 of eight options is a clear read. Confidence says
    so and the raw probability cannot.
    """
    narrow = {"a": 0.5, "b": 0.5}
    wide = {"a": 0.5, **{f"l{index}": 0.5 / 7 for index in range(7)}}
    assert docs_confidence(narrow) == pytest.approx(0.0, abs=1e-12)
    assert docs_confidence(wide) > 0.4


def test_confidence_needs_at_least_two_options() -> None:
    with pytest.raises(ValueError, match="at least 2 options"):
        docs_confidence({"a": 1.0})
