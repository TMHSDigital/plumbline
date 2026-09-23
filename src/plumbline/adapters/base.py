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

from plumbline.types import (
    PROBABILITY_SEMANTICS,
    Prediction,
    ProbabilitySemantics,
    QuestionType,
)


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

    supported_question_types: tuple[QuestionType, ...] = ("choice",)
    """Question types this adapter asks natively.

    An adapter that is handed a type outside this set either refuses the case or
    asks it in the nearest way it can, and says which in ``raw["asked_as"]``. The
    runner records both what the row is and how it was asked, so a yes/no row
    answered as a two-option choice is never compared with one answered as a
    noul without that difference on the page.
    """

    reports_tokens: bool = True
    """Whether this transport can ever report token counts.

    False says the adapter cannot report cost at all, which is what a local
    checkpoint is: its real cost is hardware and wall-clock, and no token count
    exists to price. That is a different fact from an API which normally reports
    counts and returned none on this call, and the runner keeps the two apart in
    the artifact so a blank cost column can be read rather than guessed at.
    """

    label_order_matters: bool = False
    """Whether the order the options arrive in changes the question asked.

    True for an adapter whose prompt lists the options, since option order is a
    known source of position bias. The cache then keys on the order as given; for
    every other adapter it sorts them, so reordering the options reuses answers.
    """

    uses_label_descriptions: bool = False
    """Whether this adapter sends option descriptions with the question.

    When True the runner passes a case's ``label_descriptions`` to ``classify``
    as ``descriptions``, and they join the cache key. When False they are not
    sent, and the report says so for any run whose rows carried them.
    """

    @property
    def call_params(self) -> Mapping[str, object]:
        """Call parameters that change the answer, for the Phase 4 cache key.

        Extends the minimum interface. The cache key is a hash of the adapter
        name, model_requested, revision, text, sorted labels, and these. An
        adapter with no such parameters leaves this empty.
        """
        return {}

    @abstractmethod
    def classify(
        self,
        text: str,
        labels: list[str],
        *,
        question_type: QuestionType = "choice",
    ) -> Prediction:
        """Answer one case. ``question_type`` is what the dataset says it asks."""
