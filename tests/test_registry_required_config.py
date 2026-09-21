"""An adapter built without the settings it requires refuses, rather than crashing.

Found by the fresh-install pass: ``plumbline run --adapter local_logits`` with no
model printed a bare ``TypeError`` traceback from ``__init__``. A traceback is
the right output for a bug and the wrong output for "you left an option off".
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from plumbline.adapters import registry
from plumbline.adapters.base import Adapter
from plumbline.types import Prediction, QuestionType, UnknownAdapterError


class _Stub(Adapter):
    """A concrete adapter whose settings all have defaults."""

    def __init__(self, *, model_requested: str = "stub-1") -> None:
        self.name = "stub"
        self.model_requested = model_requested
        self.revision = None
        self.probability_semantics = "calibrated_claim"

    def classify(
        self, text: str, labels: list[str], *, question_type: QuestionType = "choice"
    ) -> Prediction:
        return Prediction(label=labels[0], prob_selected=1.0)


class _Exploding(_Stub):
    """A concrete adapter that fails for a reason of its own."""

    def __init__(self, *, fine: str = "ok") -> None:
        raise TypeError("something genuinely wrong in here")


@pytest.fixture
def registered() -> Iterator[None]:
    """Register the test adapters, and take them out again afterwards."""
    registry.register("stub_for_test", _Stub)
    registry.register("exploding_for_test", _Exploding)
    try:
        yield
    finally:
        for name in ("stub_for_test", "exploding_for_test"):
            registry._REGISTRY.pop(name, None)


def test_a_missing_required_setting_is_named_rather_than_raising_typeerror() -> None:
    with pytest.raises(UnknownAdapterError) as refused:
        registry.create("local_logits")

    message = str(refused.value)
    assert "'model_requested'" in message
    assert "'revision'" in message


def test_only_the_settings_actually_missing_are_named() -> None:
    with pytest.raises(UnknownAdapterError) as refused:
        registry.create("local_logits", model_requested="some/model")

    message = str(refused.value)
    assert "'revision'" in message
    # The one that was supplied is not listed as missing.
    assert "'model_requested'" not in message


def test_the_settings_are_named_as_settings_not_as_invented_flags() -> None:
    """The CLI calls this one --model, so --model-requested would be a dead end."""
    with pytest.raises(UnknownAdapterError) as refused:
        registry.create("local_logits")

    assert "--model-requested" not in str(refused.value)


def test_the_reason_given_is_not_borrowed_from_a_different_adapter() -> None:
    """``mock`` requires a gold map, which has nothing to do with checkpoints."""
    with pytest.raises(UnknownAdapterError) as refused:
        registry.create("mock")

    message = str(refused.value)
    assert "'gold_by_text'" in message
    assert "checkpoint" not in message
    assert "revision" not in message


def test_an_adapter_whose_settings_all_have_defaults_builds(registered: None) -> None:
    assert registry.create("stub_for_test").model_requested == "stub-1"


def test_a_typeerror_from_inside_init_is_not_relabelled_as_missing_config(
    registered: None,
) -> None:
    """A real fault inside an adapter must not be reported as a missing option."""
    with pytest.raises(TypeError, match="genuinely wrong"):
        registry.create("exploding_for_test")


def test_an_unknown_name_still_lists_what_is_available() -> None:
    with pytest.raises(UnknownAdapterError, match="no adapter registered"):
        registry.create("not_a_real_adapter")
