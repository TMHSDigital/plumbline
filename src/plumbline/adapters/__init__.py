"""Adapters, organized by transport rather than by vendor.

The adapter classes are imported on first access, not with the package, so a
broken or missing SDK affects only the adapter that needs it.
"""

from __future__ import annotations

import importlib
from typing import Any

from plumbline.adapters.base import Adapter, check_probability_semantics
from plumbline.adapters.registry import available, create, register

_LAZY = {
    "GenerativeAdapter": "plumbline.adapters.generative",
    "LocalLogitsAdapter": "plumbline.adapters.local_logits",
    "MockAdapter": "plumbline.adapters.mock",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Adapter",
    "GenerativeAdapter",
    "LocalLogitsAdapter",
    "MockAdapter",
    "available",
    "check_probability_semantics",
    "create",
    "register",
]
