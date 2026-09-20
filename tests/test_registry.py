"""The registry keeps adapter selection in config rather than in code."""

from __future__ import annotations

import pytest

from plumbline.adapters import registry
from plumbline.adapters.mock import MockAdapter
from plumbline.types import UnknownAdapterError


def test_mock_is_registered() -> None:
    assert "mock" in registry.available()


def test_create_builds_a_configured_adapter() -> None:
    adapter = registry.create(
        "mock",
        gold_by_text={"text": "a"},
        accuracy=0.9,
        probability_semantics="restricted_softmax",
    )
    assert isinstance(adapter, MockAdapter)
    assert adapter.accuracy == 0.9
    assert adapter.probability_semantics == "restricted_softmax"


def test_unknown_name_names_what_is_available() -> None:
    with pytest.raises(UnknownAdapterError, match="mock"):
        registry.create("nimble")


def test_duplicate_registration_is_an_error() -> None:
    with pytest.raises(ValueError, match="already registered"):
        registry.register("mock", MockAdapter)
