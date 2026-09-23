"""The generative control arm, exercised against a fake client.

This arm reports no probability at all, which is the point of having it: it
measures what you give up by asking a text generator to classify, and it is the
floor that any probability-reporting system has to beat on accuracy before its
probabilities matter.

Everything here is about refusing to turn a generation into a measurement it is
not. An answer that is not one of the options is recorded as a refusal rather
than scored, and a truncated answer is never read as a label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from plumbline import config
from plumbline.adapters import registry
from plumbline.adapters.generative import (
    GenerativeAdapter,
    ModelRefusedError,
    OffLabelAnswerError,
    TruncatedAnswerError,
)
from plumbline.runner import execute
from plumbline.types import CaseRefusedError, NotCalibratableError
from tests.helpers import make_cases

LABELS = ["billing", "returns", "shipping"]


@dataclass
class FakeBlock:
    text: str
    type: str = "text"


@dataclass
class FakeUsage:
    input_tokens: int | None = 120
    output_tokens: int | None = 3


@dataclass
class FakeResponse:
    content: list[FakeBlock]
    model: str = "claude-opus-5"
    stop_reason: str = "end_turn"
    stop_details: Any = None
    usage: FakeUsage = field(default_factory=FakeUsage)


@dataclass
class FakeStopDetails:
    category: str = "cyber"
    explanation: str = "declined"


class FakeMessages:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]


class FakeClient:
    def __init__(self, *responses: FakeResponse) -> None:
        self.messages = FakeMessages(list(responses) or [answering("billing")])


def answering(text: str, **overrides: Any) -> FakeResponse:
    return FakeResponse(content=[FakeBlock(text=text)], **overrides)


def an_adapter(client: FakeClient | None = None, **overrides: Any) -> GenerativeAdapter:
    return GenerativeAdapter(client=client or FakeClient(), **overrides)


# What it reports, and what it refuses to report


def test_the_written_label_is_the_prediction() -> None:
    prediction = an_adapter(FakeClient(answering("returns"))).classify("a ticket", LABELS)

    assert prediction.label == "returns"


def test_it_reports_no_probability_and_invents_none() -> None:
    prediction = an_adapter().classify("a ticket", LABELS)

    assert prediction.prob_selected is None
    assert prediction.distribution is None
    assert prediction.confidence is None


def test_the_semantics_say_there_is_no_probability_here() -> None:
    assert an_adapter().probability_semantics == "none"


def test_the_semantics_cannot_be_overridden_by_config() -> None:
    with pytest.raises(TypeError):
        GenerativeAdapter(
            client=FakeClient(),
            probability_semantics="calibrated_claim",  # type: ignore[call-arg]
        )


def test_the_arm_is_excluded_from_calibration_rather_than_scored_as_zero() -> None:
    cases = make_cases(4, labels=tuple(LABELS))
    adapter = an_adapter(FakeClient(answering("billing")))

    result = execute.run(adapter, cases, workers=1)

    with pytest.raises(NotCalibratableError, match="reports no probability"):
        result.probabilities().require_reportable()


def test_the_tokens_and_the_model_that_answered_are_recorded() -> None:
    client = FakeClient(answering("billing", model="claude-opus-5-something"))
    prediction = an_adapter(client).classify("a ticket", LABELS)

    assert GenerativeAdapter.reports_tokens is True
    assert prediction.input_tokens == 120
    assert prediction.output_tokens == 3
    assert prediction.model_reported == "claude-opus-5-something"
    assert prediction.cost_usd is None  # derived from the pricing table, not here


# Reading the answer


@pytest.mark.parametrize("written", ["billing", " billing ", '"billing"', "billing.", "Billing"])
def test_an_answer_is_read_through_its_punctuation_and_case(written: str) -> None:
    prediction = an_adapter(FakeClient(answering(written))).classify("a ticket", LABELS)

    assert prediction.label == "billing"


def test_an_answer_that_is_not_an_option_is_refused_rather_than_scored() -> None:
    """Guessing which option it meant would score the harness, not the model."""
    client = FakeClient(answering("I think this is a warranty question"))

    with pytest.raises(OffLabelAnswerError) as refusal:
        an_adapter(client).classify("a ticket", LABELS)

    assert "warranty" in str(refusal.value)
    assert isinstance(refusal.value, CaseRefusedError)


def test_an_off_label_answer_is_not_retried() -> None:
    """One request per case. A retry would hide how often this arm misses."""
    client = FakeClient(answering("something else entirely"))
    cases = make_cases(2, labels=tuple(LABELS))

    result = execute.run(an_adapter(client), cases, workers=1)

    assert all(record.refused for record in result.records)
    assert len(client.messages.calls) == 2


def test_an_answer_cut_off_by_the_token_cap_is_never_read_as_a_label() -> None:
    client = FakeClient(answering("bill", stop_reason="max_tokens"))

    with pytest.raises(TruncatedAnswerError, match="max_tokens"):
        an_adapter(client).classify("a ticket", LABELS)


def test_a_model_refusal_is_recorded_with_its_category() -> None:
    client = FakeClient(
        answering("", stop_reason="refusal", stop_details=FakeStopDetails(category="cyber"))
    )

    with pytest.raises(ModelRefusedError, match="cyber"):
        an_adapter(client).classify("a ticket", LABELS)


def test_an_empty_answer_is_refused() -> None:
    with pytest.raises(CaseRefusedError):
        an_adapter(FakeClient(answering("   "))).classify("a ticket", LABELS)


# What is sent


def test_the_request_names_every_option_and_asks_for_one_of_them() -> None:
    client = FakeClient()
    an_adapter(client).classify("the invoice is wrong", LABELS)

    sent = client.messages.calls[0]
    prompt = f"{sent['system']} {sent['messages'][0]['content']}"
    assert "the invoice is wrong" in prompt
    assert all(label in prompt for label in LABELS)


def test_no_sampling_parameters_are_sent() -> None:
    """The current models reject temperature and top_p outright."""
    client = FakeClient()
    an_adapter(client).classify("a ticket", LABELS)

    sent = client.messages.calls[0]
    assert "temperature" not in sent
    assert "top_p" not in sent


def test_the_cache_key_covers_what_changes_the_answer() -> None:
    params = an_adapter(instructions="Pick one.").call_params

    assert params["instructions"] == "Pick one."
    assert "max_tokens" in params
    assert "effort" in params


# Running it


def test_it_is_registered_so_a_run_is_config_rather_than_code() -> None:
    adapter = registry.create("generative", client=FakeClient())

    assert isinstance(adapter, GenerativeAdapter)
    assert adapter.name == "generative"


def test_the_artifact_marks_the_arm_as_reporting_no_probability(tmp_path: Path) -> None:
    cases = make_cases(3, labels=tuple(LABELS))
    result = execute.run(an_adapter(FakeClient(answering("billing"))), cases, workers=1)

    stored = json.loads(result.write(tmp_path / "results").read_text(encoding="utf-8"))

    assert stored["probability_semantics"] == "none"
    assert all(record["prediction"]["prob_selected"] is None for record in stored["records"])


def test_a_run_is_priced_from_the_shipped_table() -> None:
    cases = make_cases(3, labels=tuple(LABELS))
    adapter = an_adapter(FakeClient(answering("billing", model="claude-opus-5")))

    result = execute.run(adapter, cases, pricing_table=config.DEFAULT_PRICING_TABLE, workers=1)

    assert {record.cost_basis for record in result.records} == {"priced"}
    assert all(record.cost_usd is not None for record in result.records)


def test_the_shipped_anthropic_pricing_carries_its_source_and_read_date() -> None:
    entry = config.DEFAULT_PRICING_TABLE["claude-opus-5"]

    assert entry.input_usd_per_million == 5.0
    assert entry.output_usd_per_million == 25.0
    assert "anthropic" in entry.source.lower()
    assert entry.as_of.isoformat() in entry.statement(entry.as_of)


def test_the_sdk_client_is_built_with_its_own_retries_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner owns retries; an SDK retrying underneath multiplies billed calls (#26)."""
    from plumbline.adapters import generative

    built: dict[str, Any] = {}

    def capture(**kwargs: Any) -> object:
        built.update(kwargs)
        return object()

    monkeypatch.setattr(generative.anthropic, "Anthropic", capture)
    adapter = generative.GenerativeAdapter(api_key="not-a-key")
    assert adapter.client is not None
    assert built["max_retries"] == 0


def test_an_endpoint_set_in_the_environment_reaches_the_cache_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proxied run must never be served a direct run's cached answers (#40)."""
    from plumbline.adapters import generative
    from plumbline.runner.cache import cache_key

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    direct = generative.GenerativeAdapter(api_key="not-a-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy.example")
    proxied = generative.GenerativeAdapter(api_key="not-a-key")

    assert direct.base_url is None
    assert proxied.base_url == "http://proxy.example"
    assert cache_key(direct, "t", ["a", "b"]) != cache_key(proxied, "t", ["a", "b"])
