"""Name to adapter-factory mapping, so runs are configured rather than coded.

There are three transports and a mock. Adding a vendor is almost always adding
config: a Jev-compatible service is a ``base_url`` on ``typesafe_wire``, and a
new open checkpoint is a HuggingFace model id and revision on ``local_logits``.
Registering a new name here is for a genuinely new transport, which is rare.
"""

from __future__ import annotations

import inspect
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

    missing = _missing_arguments(factory, config)
    if missing:
        # Without this the CLI prints a bare TypeError traceback, which reads as
        # a crash rather than as "you left an option off". An adapter that needs
        # a checkpoint and a pinned revision is the common case: neither has a
        # safe default, so the adapter is right to require them and the caller
        # deserves to be told which one to pass.
        #
        # Settings are named as settings, not as command line flags. This module
        # is a library and does not know what any front end calls them: the CLI
        # renames model_requested to --model, so inventing --model-requested
        # here would send the reader to a flag that does not exist.
        #
        # The reason is stated generically too. Which settings are required is
        # the adapter's business, and a message that explained one adapter's
        # reasons would be wrong for every other adapter.
        raise UnknownAdapterError(
            f"adapter {name!r} requires {missing!r}, and "
            f"{'that setting was' if len(missing) == 1 else 'those settings were'} not "
            "supplied. An adapter requires a setting when no default would be safe, "
            "because a guessed value changes what the run measures without saying so. "
            "Pass it as configuration; from the command line, see `plumbline run --help` "
            "for the option that carries it."
        )
    return factory(**config)


def _missing_arguments(factory: AdapterFactory, config: dict[str, Any]) -> list[str]:
    """Required parameters of ``factory`` that ``config`` does not supply.

    Signature inspection rather than catching ``TypeError`` around the call: a
    ``TypeError`` raised from inside an adapter's ``__init__`` is a real fault
    and must not be relabelled as a missing option.
    """
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):  # a builtin or C callable has no signature
        return []
    return [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in config
    ]


def available() -> tuple[str, ...]:
    """Registered adapter names, sorted."""
    return tuple(sorted(_REGISTRY))


def _register_builtins() -> None:
    from plumbline.adapters.generative import GenerativeAdapter
    from plumbline.adapters.local_logits import LocalLogitsAdapter
    from plumbline.adapters.mock import MockAdapter
    from plumbline.adapters.typesafe_wire import TypeSafeWireAdapter

    register("generative", GenerativeAdapter)
    register("local_logits", LocalLogitsAdapter)
    register("mock", MockAdapter)
    register("typesafe_wire", TypeSafeWireAdapter)


_register_builtins()
