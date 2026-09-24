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


def test_a_broken_optional_sdk_breaks_only_the_adapter_that_needs_it() -> None:
    """With anthropic unimportable, the mock must still run and generative must say why (#47).

    Run in a fresh interpreter, so the SDK can be made to fail to import without
    disturbing the modules this test session already loaded.
    """
    import subprocess
    import sys

    script = """
import sys
sys.modules["anthropic"] = None  # makes `import anthropic` raise ImportError
from plumbline.adapters import registry
assert "generative" in registry.available()
registry.create("mock", gold_by_text={"t": "a"})
try:
    registry.create("generative")
except Exception as error:
    print(type(error).__name__, "|", error)
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert "PlumblineError" in done.stdout and "anthropic" in done.stdout
