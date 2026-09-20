"""Core record types shared by adapters, the runner, and the metrics package.

The central invariant of plumbline lives here: ``prob_selected``, ``confidence``,
and ``probability_semantics`` are three different things and are never collapsed
into one another.

``prob_selected``
    P(the selected label is correct), as reported by the system under test. For
    a TypeSafe Choice answer this is ``probabilities[answer.choice]``. For a
    Noul answer it is ``noul`` itself. This is the only quantity ECE, MCE, and
    Brier may be computed against.

``confidence``
    A vendor summary statistic derived from the shape of the distribution. The
    TypeSafe docs describe it as "a statistic computed from the probability
    distribution the answer already gives you", collapsing that shape "into a
    single number from 0 to 1, so you can threshold on it". It is a gating and
    ranking signal. It is not a probability of correctness, and no calibration
    metric is ever computed against it.

``probability_semantics``
    What kind of number ``prob_selected`` is, so the report never silently
    compares a restricted softmax against a calibrated claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

ProbabilitySemantics = Literal["calibrated_claim", "restricted_softmax", "none"]
"""How a reported probability should be read.

``calibrated_claim``
    The vendor asserts calibrated probabilities. Calibration metrics are
    meaningful on their own terms and may be compared to each other.

``restricted_softmax``
    A softmax over the declared label tokens only, with no calibration claim
    attached. SemIf states the constraint directly: "Returned probabilities are
    conditional on the supplied options. Calibrate and validate them on the
    workload where they will make decisions." Measuring whether such a number
    behaves like a calibrated one is a first-class output of plumbline, not an
    aside, so these are grouped separately in the report.

``none``
    No probability is available. The adapter reports None, is excluded from
    calibration entirely, is never imputed or defaulted to zero, and renders in
    the report as "not reported".
"""

PROBABILITY_SEMANTICS: tuple[ProbabilitySemantics, ...] = get_args(ProbabilitySemantics)

#: Tolerance applied when checking that a reported distribution sums to one. The
#: TypeSafe docs say Choice probabilities "sum to approximately 1", so an exact
#: check would reject valid responses.
DISTRIBUTION_SUM_TOLERANCE = 1e-3


class PlumblineError(Exception):
    """Base class for every error plumbline raises on purpose."""


class CaseRefusedError(PlumblineError):
    """An adapter cannot answer this case and declines rather than degrading it.

    Raised, for example, when a local checkpoint cannot map an option to a single
    token. Refusing is correct: truncating the option would silently change the
    question being asked. The runner records a refusal as a refusal.
    """


class UnknownAdapterError(PlumblineError):
    """A name was requested from the registry that nothing is registered under."""


@dataclass(frozen=True)
class Case:
    """One labeled row of a dataset.

    Mirrors the JSONL record that ``datasets/loader.py`` reads: ``id``, ``text``,
    ``labels``, ``gold_label``, and optional ``label_descriptions``.
    """

    id: str
    text: str
    labels: tuple[str, ...]
    gold_label: str
    label_descriptions: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if len(self.labels) < 2:
            raise ValueError(f"case {self.id!r}: need at least 2 labels, got {len(self.labels)}")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"case {self.id!r}: labels contain duplicates: {self.labels!r}")
        if self.gold_label not in self.labels:
            raise ValueError(
                f"case {self.id!r}: gold_label {self.gold_label!r} is not one of {self.labels!r}"
            )


@dataclass(frozen=True)
class Prediction:
    """One adapter's answer to one case.

    ``cost_usd`` is left as None by adapters that do not receive a cost from the
    vendor. Cost is derived in ``metrics/cost.py`` from the token counts and a
    per-model pricing table in config, so that pricing is never hardcoded in an
    adapter and never silently goes stale.
    """

    label: str
    prob_selected: float | None  # P(selected label correct), or None
    distribution: dict[str, float] | None  # full distribution when available
    confidence: float | None  # vendor summary stat, NOT a probability
    latency_ms: float
    cost_usd: float | None  # None when tokens unreported
    input_tokens: int | None
    output_tokens: int | None
    model_reported: str | None  # what the API says it used
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("prob_selected", "confidence"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms!r}")
        if self.distribution is not None:
            if self.label not in self.distribution:
                raise ValueError(
                    f"label {self.label!r} is missing from distribution "
                    f"{sorted(self.distribution)!r}"
                )
            total = sum(self.distribution.values())
            if abs(total - 1.0) > DISTRIBUTION_SUM_TOLERANCE:
                raise ValueError(f"distribution sums to {total!r}, expected approximately 1")


def docs_confidence(distribution: Mapping[str, float]) -> float:
    """Confidence as the TypeSafe docs describe it, for mocks and for comparison.

    The confidence page's explorer approximates the statistic as
    ``(n * peak - 1) / (n - 1)``, which maps a flat distribution (peak ``1/n``)
    to 0 and a point mass to 1.

    Treat this as unverified. The docs present it as an approximation used by a
    demo widget, not as the published definition. Phase 5 records the confidence
    the live API actually returns alongside the distribution it returns, so the
    real statistic can be measured rather than assumed.

    The normalization matters for discrimination. Because the transform divides
    out the option count, confidence and ``prob_selected`` rank identically
    within a fixed label set but diverge across a dataset whose cases have
    differing numbers of options. That divergence is where confidence carries
    information that ``prob_selected`` does not.
    """
    n = len(distribution)
    if n < 2:
        raise ValueError(f"confidence needs at least 2 options, got {n}")
    peak = max(distribution.values())
    return min(1.0, max(0.0, (n * peak - 1.0) / (n - 1.0)))
