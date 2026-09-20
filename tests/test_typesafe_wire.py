"""The Jev wire adapter, exercised against a fake client rather than the network.

The token-count tests are the point of this file. ``Usage.input_tokens`` and
``Usage.output_tokens`` are ``int | None`` in the SDK and default to None, so
the path where the API reports no counts is a real path, not a hypothetical,
and it has to end in "not reported" rather than in zero.
"""

from __future__ import annotations

from typing import Any

import pytest
from typesafe_sdk import ChoiceAnswer, SystemOneResponse, Usage

from plumbline.adapters import registry
from plumbline.adapters.typesafe_wire import (
    QUESTION_NAME,
    TypeSafeWireAdapter,
    WireContractError,
)
from plumbline.metrics.cost import Pricing, cost_of, summarize

LABELS = ["billing", "technical", "sales"]

PRICING = Pricing(input_usd_per_million=1.0, output_usd_per_million=5.0)


def a_response(
    *,
    choice: str = "billing",
    probabilities: dict[str, float] | None = None,
    confidence: float = 0.82,
    input_tokens: int | None = 120,
    output_tokens: int | None = 12,
    model: str = "jev-1.2",
) -> SystemOneResponse:
    return SystemOneResponse(
        model=model,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        answers={
            QUESTION_NAME: ChoiceAnswer(
                choice=choice,
                confidence=confidence,
                probabilities=(
                    probabilities
                    if probabilities is not None
                    else {"billing": 0.7, "technical": 0.2, "sales": 0.1}
                ),
            )
        },
    )


class FakeClient:
    """Stands in for TypeSafeClient. Records what it was asked."""

    def __init__(self, response: SystemOneResponse | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def system_one(self, state: Any, questions: Any, **kwargs: Any) -> SystemOneResponse:
        self.calls.append({"state": state, "questions": questions, **kwargs})
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def close(self) -> None:
        self.closed = True


def an_adapter(response: SystemOneResponse | Exception, **config: Any) -> TypeSafeWireAdapter:
    return TypeSafeWireAdapter(client=FakeClient(response), **config)  # type: ignore[arg-type]


# The answer


def test_a_choice_answer_becomes_a_prediction() -> None:
    adapter = an_adapter(a_response())

    prediction = adapter.classify("I was charged twice", LABELS)

    assert prediction.label == "billing"
    assert prediction.distribution == {"billing": 0.7, "technical": 0.2, "sales": 0.1}
    assert prediction.prob_selected == 0.7
    assert prediction.confidence == 0.82
    assert prediction.model_reported == "jev-1.2"
    assert prediction.latency_ms >= 0.0


def test_the_case_text_and_labels_reach_the_request() -> None:
    adapter = an_adapter(a_response())
    client: FakeClient = adapter._client  # type: ignore[assignment]

    adapter.classify("I was charged twice", LABELS)

    sent = client.calls[0]
    assert sent["state"] == {"document": "I was charged twice"}
    assert sorted(sent["questions"][QUESTION_NAME].criteria) == sorted(LABELS)


def test_confidence_is_carried_but_never_used_as_a_probability() -> None:
    """The SDK reports both. They are different quantities and stay separate."""
    adapter = an_adapter(a_response(confidence=0.95))

    prediction = adapter.classify("text", LABELS)

    assert prediction.confidence == 0.95
    assert prediction.prob_selected == 0.7


def test_the_requested_model_and_the_reported_model_are_both_kept() -> None:
    adapter = an_adapter(a_response(model="jev-1.2"), model_requested="jev-latest")

    prediction = adapter.classify("text", LABELS)

    assert adapter.model_requested == "jev-latest"
    assert prediction.model_reported == "jev-1.2"


# Token counts, and the cost that depends on them


def test_reported_token_counts_are_carried_through_unchanged() -> None:
    adapter = an_adapter(a_response(input_tokens=120, output_tokens=12))

    prediction = adapter.classify("text", LABELS)

    assert prediction.input_tokens == 120
    assert prediction.output_tokens == 12
    assert cost_of(prediction.input_tokens, prediction.output_tokens, PRICING) == pytest.approx(
        (120 * 1.0 + 12 * 5.0) / 1_000_000
    )


def test_unreported_tokens_stay_none_rather_than_becoming_zero() -> None:
    """The SDK defaults both counts to None. None must not turn into free."""
    adapter = an_adapter(a_response(input_tokens=None, output_tokens=None))

    prediction = adapter.classify("text", LABELS)

    assert prediction.input_tokens is None
    assert prediction.output_tokens is None
    assert prediction.cost_usd is None
    assert cost_of(prediction.input_tokens, prediction.output_tokens, PRICING) is None


def test_half_a_token_count_is_not_a_token_count() -> None:
    """One count present and one absent yields no cost, not a partial one."""
    adapter = an_adapter(a_response(input_tokens=120, output_tokens=None))

    prediction = adapter.classify("text", LABELS)

    assert prediction.input_tokens == 120
    assert prediction.output_tokens is None
    assert cost_of(prediction.input_tokens, prediction.output_tokens, PRICING) is None


def test_a_run_of_unpriced_cases_reports_no_cost_rather_than_zero() -> None:
    """End to end: unreported tokens reach the summary as a refusal to report."""
    adapter = an_adapter(a_response(input_tokens=None, output_tokens=None))
    predictions = [adapter.classify(f"text {i}", LABELS) for i in range(4)]

    summary = summarize(
        [cost_of(p.input_tokens, p.output_tokens, PRICING) for p in predictions],
        correct=[True, True, False, True],
    )

    assert summary.total_usd is None
    assert summary.per_correct_usd is None
    assert summary.priced_cases == 0
    assert summary.unpriced_cases == 4
    assert "not reported" in summary.note


def test_the_adapter_never_computes_cost_itself() -> None:
    """Pricing lives in config, so an adapter reporting a price would be stale."""
    adapter = an_adapter(a_response())

    assert adapter.classify("text", LABELS).cost_usd is None


# The wire contract


def test_a_label_set_that_was_not_asked_about_is_refused() -> None:
    adapter = an_adapter(
        a_response(probabilities={"billing": 0.6, "technical": 0.4})  # "sales" missing
    )

    with pytest.raises(WireContractError, match="Missing"):
        adapter.classify("text", LABELS)


def test_an_unexpected_label_is_refused_rather_than_filtered() -> None:
    adapter = an_adapter(
        a_response(
            probabilities={"billing": 0.4, "technical": 0.2, "sales": 0.1, "other": 0.3},
        )
    )

    with pytest.raises(WireContractError, match="Unexpected"):
        adapter.classify("text", LABELS)


def test_a_distribution_is_never_renormalized_to_hide_a_bad_sum() -> None:
    """A distribution that does not sum to one is surfaced, not rescaled."""
    adapter = an_adapter(a_response(probabilities={"billing": 0.5, "technical": 0.2, "sales": 0.1}))

    with pytest.raises(ValueError, match="sums to"):
        adapter.classify("text", LABELS)


def test_a_missing_answer_is_refused() -> None:
    response = SystemOneResponse(model="jev-1", usage=Usage(), answers={})

    with pytest.raises(WireContractError, match="no choice answer"):
        an_adapter(response).classify("text", LABELS)


def test_empty_labels_are_rejected_before_any_call() -> None:
    adapter = an_adapter(a_response())
    client: FakeClient = adapter._client  # type: ignore[assignment]

    with pytest.raises(ValueError, match="labels must not be empty"):
        adapter.classify("text", [])

    assert client.calls == []


def test_a_transport_error_propagates_for_the_runner_to_retry() -> None:
    adapter = an_adapter(RuntimeError("connection reset"))

    with pytest.raises(RuntimeError, match="connection reset"):
        adapter.classify("text", LABELS)


# Configuration, caching, and credentials


def test_the_semantics_label_records_the_claim_and_can_be_overridden() -> None:
    assert an_adapter(a_response()).probability_semantics == "calibrated_claim"
    assert an_adapter(a_response(), probability_semantics="none").probability_semantics == "none"


def test_an_unknown_semantics_label_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="probability_semantics"):
        an_adapter(a_response(), probability_semantics="calibrated")  # type: ignore[arg-type]


def test_answer_changing_config_is_in_the_cache_key() -> None:
    adapter = an_adapter(a_response(), instructions="Pick one.", base_url="https://x.invalid")

    assert adapter.call_params["instructions"] == "Pick one."
    assert adapter.call_params["base_url"] == "https://x.invalid"


def test_the_api_key_is_not_part_of_the_cache_key() -> None:
    """A cache key is written to disk. A credential does not change the answer."""
    adapter = an_adapter(a_response(), api_key="ts-secret-value")

    assert "ts-secret-value" not in repr(dict(adapter.call_params))


def test_the_api_key_never_reaches_the_raw_record() -> None:
    adapter = an_adapter(a_response(), api_key="ts-secret-value")

    prediction = adapter.classify("text", LABELS)

    assert "ts-secret-value" not in repr(prediction.raw)
    assert prediction.raw["usage"] == {"input_tokens": 120, "output_tokens": 12}


def test_a_supplied_client_is_not_closed_by_the_adapter() -> None:
    """The caller owns what the caller built."""
    adapter = an_adapter(a_response())
    client: FakeClient = adapter._client  # type: ignore[assignment]

    adapter.close()

    assert client.closed is False


def test_the_adapter_is_registered_under_its_transport_name() -> None:
    assert "typesafe_wire" in registry.available()
