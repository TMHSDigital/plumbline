"""The local logit-readout arm, exercised against a fake readout.

This adapter is the null hypothesis of the whole tool: a restricted softmax over
the option tokens, with no calibration claim attached to it. Everything here is
about keeping that claim honest: the probabilities are a softmax over the
option tokens and nothing else, the checkpoint is pinned and recorded, and a
case whose options do not map to single tokens is refused rather than quietly
turned into a different question.

Nothing here loads a model. The readout is injected.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from plumbline.adapters import registry
from plumbline.adapters.local_logits import (
    CheckpointMismatchError,
    LocalLogitsAdapter,
)
from plumbline.runner import execute
from plumbline.types import CaseRefusedError
from tests.helpers import make_cases

PINNED = "a" * 40
OTHER_SHA = "b" * 40
LABELS = ["billing", "returns", "shipping"]
SCORES = {"billing": 2.0, "returns": 1.0, "shipping": 0.0}


class FakeReadout:
    """A tokenizer and a next-token logit row, without a checkpoint behind them.

    One token per option by default. ``multi_token`` makes an option tokenize
    into several pieces, which is the case the adapter must refuse; ``aliases``
    makes two options share a token id, which it must refuse for the same
    reason.
    """

    def __init__(
        self,
        scores: dict[str, float] = SCORES,
        *,
        resolved_revision: str = PINNED,
        multi_token: tuple[str, ...] = (),
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.scores = dict(scores)
        self.resolved_revision = resolved_revision
        self.multi_token = multi_token
        self.aliases = dict(aliases or {})
        self.prompts: list[str] = []
        self._ids = {label: index for index, label in enumerate(sorted(self.scores))}

    def token_ids(self, text: str) -> list[int]:
        label = text.strip()
        if label in self.multi_token:
            return [90, 91, 92]
        return [self._ids[self.aliases.get(label, label)]]

    def option_logits(self, prompt: str, token_ids: list[int]) -> list[float]:
        self.prompts.append(prompt)
        by_id = {self._ids[label]: score for label, score in self.scores.items()}
        return [by_id[token_id] for token_id in token_ids]


def an_adapter(readout: FakeReadout | None = None, **overrides: object) -> LocalLogitsAdapter:
    config: dict = {"model_requested": "tiny-model", "revision": PINNED}
    config.update(overrides)
    return LocalLogitsAdapter(readout=readout or FakeReadout(), **config)


# What the numbers are


def test_the_probabilities_are_a_softmax_over_the_option_tokens_only() -> None:
    prediction = an_adapter().classify("a ticket", LABELS)

    total = math.exp(2.0) + math.exp(1.0) + math.exp(0.0)
    assert prediction.distribution is not None
    assert prediction.distribution["billing"] == pytest.approx(math.exp(2.0) / total)
    assert prediction.distribution["shipping"] == pytest.approx(math.exp(0.0) / total)
    assert sum(prediction.distribution.values()) == pytest.approx(1.0)


def test_the_selected_label_is_the_highest_scoring_option() -> None:
    prediction = an_adapter().classify("a ticket", LABELS)

    assert prediction.label == "billing"
    assert prediction.distribution is not None
    assert prediction.prob_selected == pytest.approx(prediction.distribution["billing"])


def test_the_semantics_say_restricted_softmax_and_nothing_stronger() -> None:
    """The null hypothesis arm may never be filed under a calibrated claim."""
    assert an_adapter().probability_semantics == "restricted_softmax"


def test_the_semantics_cannot_be_overridden_by_config() -> None:
    with pytest.raises(TypeError):
        LocalLogitsAdapter(
            model_requested="tiny-model",
            revision=PINNED,
            readout=FakeReadout(),
            probability_semantics="calibrated_claim",  # type: ignore[call-arg]
        )


def test_no_vendor_confidence_is_invented() -> None:
    """There is no vendor here, so there is no vendor statistic to report."""
    assert an_adapter().classify("a ticket", LABELS).confidence is None


def test_it_reports_no_tokens_and_therefore_no_cost() -> None:
    prediction = an_adapter().classify("a ticket", LABELS)

    assert LocalLogitsAdapter.reports_tokens is False
    assert prediction.input_tokens is None
    assert prediction.output_tokens is None
    assert prediction.cost_usd is None


def test_the_prompt_carries_the_case_text_and_every_option() -> None:
    readout = FakeReadout()
    an_adapter(readout).classify("the invoice is wrong", LABELS)

    prompt = readout.prompts[0]
    assert "the invoice is wrong" in prompt
    assert all(label in prompt for label in LABELS)


# Refusing rather than truncating


def test_an_option_that_is_not_a_single_token_is_refused() -> None:
    readout = FakeReadout(multi_token=("shipping",))

    with pytest.raises(CaseRefusedError) as refusal:
        an_adapter(readout).classify("a ticket", LABELS)

    message = str(refusal.value)
    assert "shipping" in message
    assert "3 tokens" in message
    assert "truncat" in message


def test_the_refusal_names_every_offending_option_not_just_the_first() -> None:
    readout = FakeReadout(multi_token=("returns", "shipping"))

    with pytest.raises(CaseRefusedError) as refusal:
        an_adapter(readout).classify("a ticket", LABELS)

    assert "returns" in str(refusal.value)
    assert "shipping" in str(refusal.value)


def test_two_options_sharing_one_token_are_refused() -> None:
    """Two options that read back as the same token cannot be told apart."""
    readout = FakeReadout(aliases={"returns": "billing"})

    with pytest.raises(CaseRefusedError) as refusal:
        an_adapter(readout).classify("a ticket", LABELS)

    assert "same token" in str(refusal.value)


def test_a_refusal_is_recorded_as_a_refusal_and_not_retried() -> None:
    cases = make_cases(3, labels=tuple(LABELS))
    adapter = an_adapter(FakeReadout(multi_token=("shipping",)))

    result = execute.run(adapter, cases, workers=1)

    assert all(record.refused for record in result.records)
    assert all(record.prediction is None for record in result.records)
    assert all(record.attempts == 1 for record in result.records)


# The checkpoint


def test_a_checkpoint_with_no_revision_is_refused() -> None:
    with pytest.raises(ValueError, match="revision"):
        LocalLogitsAdapter(model_requested="tiny-model", revision="", readout=FakeReadout())


def test_a_moving_reference_is_refused_unless_it_is_asked_for_explicitly() -> None:
    """A branch can move under a run; a commit cannot."""
    with pytest.raises(ValueError, match="commit"):
        LocalLogitsAdapter(model_requested="tiny-model", revision="main", readout=FakeReadout())

    adapter = LocalLogitsAdapter(
        model_requested="tiny-model",
        revision="main",
        readout=FakeReadout(resolved_revision="main"),
        allow_unpinned_revision=True,
    )
    prediction = adapter.classify("a ticket", LABELS)
    assert prediction.raw["revision_pinned_by_commit"] is False


def test_the_revision_that_answered_is_recorded_next_to_the_one_asked_for() -> None:
    prediction = an_adapter().classify("a ticket", LABELS)

    assert prediction.model_reported == f"tiny-model@{PINNED}"
    assert prediction.raw["revision_requested"] == PINNED
    assert prediction.raw["revision_resolved"] == PINNED


def test_a_checkpoint_that_resolved_somewhere_else_is_refused() -> None:
    readout = FakeReadout(resolved_revision=OTHER_SHA)

    with pytest.raises(CheckpointMismatchError) as mismatch:
        an_adapter(readout).classify("a ticket", LABELS)

    assert PINNED in str(mismatch.value)
    assert OTHER_SHA in str(mismatch.value)


def test_the_cache_key_covers_what_changes_the_answer() -> None:
    params = an_adapter(instructions="Pick one.").call_params

    assert params["instructions"] == "Pick one."
    assert "prompt_template" in params


# Running it


def test_it_is_registered_so_a_run_is_config_rather_than_code() -> None:
    adapter = registry.create(
        "local_logits",
        model_requested="tiny-model",
        revision=PINNED,
        readout=FakeReadout(),
    )

    assert isinstance(adapter, LocalLogitsAdapter)
    assert adapter.name == "local_logits"


def test_the_artifact_marks_the_arm_as_a_restricted_softmax(tmp_path: Path) -> None:
    """The report groups on this field, so the null hypothesis stays labeled."""
    cases = make_cases(4, labels=tuple(LABELS))
    result = execute.run(an_adapter(), cases, workers=1)

    stored = json.loads(result.write(tmp_path / "results").read_text(encoding="utf-8"))

    assert stored["probability_semantics"] == "restricted_softmax"
    assert stored["revision"] == PINNED
    assert {record["cost_basis"] for record in stored["records"]} == {"adapter_reports_no_tokens"}


# Running for real


def test_concurrent_first_cases_load_the_checkpoint_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run's workers all reach the first case together. Each loading its own
    copy of the checkpoint is several gigabytes per worker, onto one GPU."""
    import threading
    import time

    from plumbline.adapters import local_logits

    loads: list[int] = []

    class SlowToLoad(FakeReadout):
        def __init__(self, **_kwargs: object) -> None:
            loads.append(1)
            time.sleep(0.05)  # long enough for every worker to arrive meanwhile
            super().__init__()

    monkeypatch.setattr(local_logits, "TransformersReadout", SlowToLoad)
    adapter = LocalLogitsAdapter(model_requested="tiny-model", revision=PINNED)
    start = threading.Barrier(8)

    def first_case() -> None:
        start.wait()
        adapter.classify("text", LABELS)

    workers = [threading.Thread(target=first_case) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert len(loads) == 1


def test_the_device_reaches_the_readout_and_the_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    from plumbline.adapters import local_logits

    seen: dict[str, object] = {}

    class Recorded(FakeReadout):
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)
            super().__init__()

    monkeypatch.setattr(local_logits, "TransformersReadout", Recorded)
    adapter = registry.create("local_logits", model_requested="m", revision=PINNED, device="cuda")

    result = execute.run(adapter, make_cases(3, labels=LABELS), workers=1)

    assert seen["device"] == "cuda"
    assert result.config["device"] == "cuda"
