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

import math
from collections.abc import Mapping, Sequence
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

QuestionType = Literal["choice", "noul", "score"]
"""What kind of question a case actually asks.

``choice``
    Pick one of n declared options. The default, and what most of a
    classification dataset is.

``noul``
    A yes/no question. The answer is one probability, that the answer is yes,
    with no distribution and no confidence behind it. Asking it as a two-option
    choice is a different question: it invites a distribution over two strings
    rather than a single probability of a statement being true, and the cleanest
    calibration target in the API is the bare probability.

``score``
    An ordinal level, such as 0 to 3. plumbline v0.1 has no ordinal support:
    flattening levels into unordered options throws away the ordering, and
    reporting rank-blind metrics on them would be worse than reporting nothing.
    Such cases are loaded, marked, and excluded from scored results.
"""

QUESTION_TYPES: tuple[QuestionType, ...] = get_args(QuestionType)

#: Question types plumbline can score in v0.1. A case outside this set is loaded
#: and carried so that nothing is silently lost, and excluded from every metric.
SUPPORTED_QUESTION_TYPES: tuple[QuestionType, ...] = ("choice", "noul")

#: Option names read as "yes" and as "no". A yes/no question has to be pinned to
#: its two outcomes before a bare P(yes) can be attached to either of them.
AFFIRMATIVE_LABELS: tuple[str, ...] = ("yes", "true")
NEGATIVE_LABELS: tuple[str, ...] = ("no", "false")


def yes_no_labels(labels: Sequence[str]) -> tuple[str, str] | None:
    """The (affirmative, negative) pair in ``labels``, or None if it is not one.

    Returning None rather than guessing is the point. Which of "approve" and
    "escalate" is the yes is a question about someone's dataset, not something
    an adapter may decide on their behalf, and a wrong guess inverts every
    probability it touches.
    """
    if len(labels) != 2:
        return None
    affirmative = [label for label in labels if label.strip().casefold() in AFFIRMATIVE_LABELS]
    negative = [label for label in labels if label.strip().casefold() in NEGATIVE_LABELS]
    if len(affirmative) != 1 or len(negative) != 1:
        return None
    return affirmative[0], negative[0]


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


class InsufficientDataError(PlumblineError):
    """There are too few rows to answer the question that was asked.

    Recalibration raises this rather than fitting a temperature to a handful of
    rows. A temperature fitted on noise is worse than no recalibration, because
    it arrives looking like a measurement.
    """


class DatasetError(PlumblineError):
    """A dataset could not be read, or a row in it could not be scored.

    Raised for the file as a whole: missing, empty, or demanded complete when
    it is not. A single bad row is a refusal recorded in the load report rather
    than an exception, so one typo does not stop the other 110 rows from running.
    """


class ArtifactError(PlumblineError):
    """A results artifact could not be read back.

    Missing, unreadable, or not the shape a run writes. Separate from
    ``DatasetError`` because the remedy is different: a dataset is the user's
    input and they can fix a row, while an artifact is plumbline's own output
    and a broken one usually means the wrong path was named.
    """


class NotCalibratableError(PlumblineError):
    """A calibration metric was asked for on something that is not a probability.

    Raised when a confidence series reaches a calibration function, when an
    adapter reports no probability at all, or when values are missing. plumbline
    refuses rather than imputing, because a zero in a calibration table reads as
    a measurement and a blank reads as an absence.
    """


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
    question_type: QuestionType = "choice"
    """What the row actually asks, so an adapter can ask it that way.

    Carried from the dataset rather than inferred from the options: two options
    named yes and no may be a genuine yes/no question or may be a choice between
    two strings, and only the dataset knows which.
    """

    @property
    def is_scoreable(self) -> bool:
        """Whether v0.1 can turn this case into a number it is willing to report."""
        return self.question_type in SUPPORTED_QUESTION_TYPES

    def __post_init__(self) -> None:
        if len(self.labels) < 2:
            raise ValueError(f"case {self.id!r}: need at least 2 labels, got {len(self.labels)}")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"case {self.id!r}: labels contain duplicates: {self.labels!r}")
        if self.gold_label not in self.labels:
            raise ValueError(
                f"case {self.id!r}: gold_label {self.gold_label!r} is not one of {self.labels!r}"
            )
        if self.question_type not in QUESTION_TYPES:
            raise ValueError(
                f"case {self.id!r}: question_type must be one of {QUESTION_TYPES!r}, "
                f"got {self.question_type!r}"
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
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError(f"latency_ms must be finite and non-negative, got {self.latency_ms!r}")
        if self.distribution is not None:
            # Checked one by one: NaN compares false with everything, and a
            # negative entry can hide behind a sum that still comes to 1.
            for option, probability in self.distribution.items():
                if not (math.isfinite(probability) and 0.0 <= probability <= 1.0):
                    raise ValueError(
                        f"distribution gives {option!r} a probability of {probability!r}, "
                        "which is not a finite number in [0, 1]"
                    )
            if self.label not in self.distribution:
                raise ValueError(
                    f"label {self.label!r} is missing from distribution "
                    f"{sorted(self.distribution)!r}"
                )
            total = sum(self.distribution.values())
            if abs(total - 1.0) > DISTRIBUTION_SUM_TOLERANCE:
                raise ValueError(f"distribution sums to {total!r}, expected approximately 1")

    @property
    def tied_for_top(self) -> tuple[str, ...]:
        """The options sharing the highest probability, when two or more do.

        On such a row the answer was chosen by the vendor's tie-break rather
        than by a margin. Probabilities on a two-decimal grid make that
        ordinary, and ``label`` is always the vendor's pick, never a recomputed
        argmax. Empty when nothing tied or there is no distribution.
        """
        if not self.distribution:
            return ()
        peak = max(self.distribution.values())
        tied = sorted(option for option, value in self.distribution.items() if value == peak)
        return tuple(tied) if len(tied) > 1 else ()


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


@dataclass(frozen=True)
class ProbabilitySeries:
    """A column of ``prob_selected`` values, tagged with what kind of number it is.

    The separation between probability and confidence is enforced by the type
    system rather than by a naming convention. A calibration function accepts
    this type and nothing else, so there is no call site at which a confidence
    column can be passed to ECE by mistake.
    """

    values: tuple[float | None, ...]
    semantics: ProbabilitySemantics

    def __post_init__(self) -> None:
        if self.semantics == "none" and any(value is not None for value in self.values):
            raise ValueError("semantics 'none' means every value must be None")
        for value in self.values:
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"probability must lie in [0, 1], got {value!r}")

    def __len__(self) -> int:
        return len(self.values)

    @property
    def is_reportable(self) -> bool:
        """Whether a calibration metric may be computed from this column at all."""
        return self.semantics != "none" and all(value is not None for value in self.values)

    def require_reportable(self) -> tuple[float, ...]:
        """Return the values, or refuse and say exactly why."""
        if self.semantics == "none":
            raise NotCalibratableError(
                "this adapter reports no probability, so it is excluded from calibration. "
                "The report renders it as 'not reported', never as zero."
            )
        missing = sum(1 for value in self.values if value is None)
        if missing:
            raise NotCalibratableError(
                f"{missing} of {len(self.values)} probabilities are missing. Drop the "
                "failed cases explicitly rather than letting them be imputed."
            )
        return tuple(value for value in self.values if value is not None)


@dataclass(frozen=True)
class ConfidenceSeries:
    """A column of vendor ``confidence`` values.

    Deliberately not a :class:`ProbabilitySeries`. Confidence is evaluated as a
    discrimination and gating signal: AUROC, plus accuracy and coverage across a
    threshold sweep. It never reaches ECE, MCE, Brier, or a reliability diagram.
    """

    values: tuple[float | None, ...]

    def __post_init__(self) -> None:
        for value in self.values:
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"confidence must lie in [0, 1], got {value!r}")

    def __len__(self) -> int:
        return len(self.values)

    @property
    def is_reportable(self) -> bool:
        return all(value is not None for value in self.values)

    def require_reportable(self) -> tuple[float, ...]:
        missing = sum(1 for value in self.values if value is None)
        if missing:
            raise ValueError(f"{missing} of {len(self.values)} confidence values are missing")
        return tuple(value for value in self.values if value is not None)


def probability_series(
    predictions: Sequence[Prediction], semantics: ProbabilitySemantics
) -> ProbabilitySeries:
    """Pull ``prob_selected`` out of a run, tagged with the adapter's semantics."""
    return ProbabilitySeries(
        values=tuple(prediction.prob_selected for prediction in predictions),
        semantics=semantics,
    )


def confidence_series(predictions: Sequence[Prediction]) -> ConfidenceSeries:
    """Pull ``confidence`` out of a run."""
    return ConfidenceSeries(values=tuple(prediction.confidence for prediction in predictions))


#: Probabilities below this are clamped before taking a log, so a zero entry in a
#: distribution does not produce a negative infinity.
_PROBABILITY_FLOOR = 1e-16


def apply_temperature(distribution: Mapping[str, float], temperature: float) -> dict[str, float]:
    """Rescale a distribution by temperature: ``softmax(log p / T)``.

    ``T`` below 1 sharpens, above 1 flattens, and exactly 1 is the identity. The
    map is monotone in ``log p``, so the argmax never moves and a predicted label
    never changes under recalibration.

    One implementation, used both by the mock that injects a known skew and by
    the recalibrator that fits one out, so the two can never drift apart.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")
    if temperature == 1.0:
        return dict(distribution)

    scaled = {
        label: math.log(max(value, _PROBABILITY_FLOOR)) / temperature
        for label, value in distribution.items()
    }
    peak = max(scaled.values())
    weights = {label: math.exp(value - peak) for label, value in scaled.items()}
    total = sum(weights.values())
    return {label: value / total for label, value in weights.items()}
