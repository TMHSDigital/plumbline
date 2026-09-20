"""Adapters, organized by transport rather than by vendor."""

from plumbline.adapters.base import Adapter, check_probability_semantics
from plumbline.adapters.generative import GenerativeAdapter
from plumbline.adapters.local_logits import LocalLogitsAdapter
from plumbline.adapters.mock import MockAdapter
from plumbline.adapters.registry import available, create, register

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
