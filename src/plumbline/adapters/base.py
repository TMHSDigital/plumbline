"""The adapter interface every system under test implements.

Adapters are organized by transport, not by vendor. There are three real
transports plus a mock:

``typesafe_wire``
    The Jev wire format over HTTP. Covers hosted Jev and any Jev-compatible
    endpoint reached by overriding ``base_url``.
``local_logits``
    Restricted-vocabulary logit readout, parameterized by checkpoint.
``generative``
    Text-generating control arm.

A new vendor is almost never a new adapter. A Jev-compatible service is a
``base_url``. A new open checkpoint is a ``local_logits`` config. Contributors
add config, not code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from plumbline.types import PROBABILITY_SEMANTICS, Prediction, ProbabilitySemantics


def check_probability_semantics(value: str) -> ProbabilitySemantics:
    """Reject an unrecognized semantics label at construction time.

    The report groups by this field and refuses to compare across groups without
    saying so. A typo that slipped through would put a restricted softmax in the
    calibrated column, which is the exact confusion plumbline exists to prevent.
    """
    if value not in PROBABILITY_SEMANTICS:
        raise ValueError(
            f"probability_semantics must be one of {PROBABILITY_SEMANTICS!r}, got {value!r}"
        )
    return value


class Adapter(ABC):
    """One configured system under test.

    ``model_reported`` and ``revision`` are recorded rather than trusted. A model
    alias can resolve somewhere other than where the config points, so the
    methodology records what actually answered alongside what was asked for.
    """

    name: str
    model_requested: str
    revision: str | None  # pinned checkpoint, for local models
    probability_semantics: ProbabilitySemantics

    @property
    def call_params(self) -> Mapping[str, object]:
        """Call parameters that change the answer, for the Phase 4 cache key.

        Extends the minimum interface. The cache key is a hash of the adapter
        name, model_requested, revision, text, sorted labels, and these. An
        adapter with no such parameters leaves this empty.
        """
        return {}

    @abstractmethod
    def classify(self, text: str, labels: list[str]) -> Prediction: ...
