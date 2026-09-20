"""Name to adapter-factory mapping, so runs are configured rather than coded.

There are three transports and a mock. Adding a vendor is almost always adding
config: a Jev-compatible service is a ``base_url`` on ``typesafe_wire``, and a
new open checkpoint is a HuggingFace model id and revision on ``local_logits``.
Registering a new name here is for a genuinely new transport, which is rare.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from plumbline.adapters.base import Adapter
from plumbline.types import UnknownAdapterError

AdapterFactory = Callable[..., Adapter]

_REGISTRY: dict[str, AdapterFactory] = {}


def register(name: str, factory: AdapterFactory) -> None:
    """Bind ``name`` to a factory. Re-registering the same name is an error."""
    if name in _REGISTRY:
        raise ValueError(f"adapter {name!r} is already registered")
    _REGISTRY[name] = factory


def create(name: str, **config: Any) -> Adapter:
    """Build a configured adapter by registered name."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise UnknownAdapterError(
            f"no adapter registered as {name!r}. Available: {available()!r}"
        ) from None
    return factory(**config)


def available() -> tuple[str, ...]:
    """Registered adapter names, sorted."""
    return tuple(sorted(_REGISTRY))


def _register_builtins() -> None:
    from plumbline.adapters.mock import MockAdapter

    register("mock", MockAdapter)


_register_builtins()
