"""A deterministic, seeded mock with configurable accuracy and calibration skew.

This adapter is what the metrics tests run against, and it is what makes the
metrics trustworthy. If ECE cannot recover a known miscalibration injected here,
no number plumbline prints about a real model means anything.

How a case is constructed
-------------------------
1. Draw ``p_top``, the probability mass on the most likely label, from a Beta
   distribution rescaled onto ``[1/n, 1]`` whose mean is the configured
   accuracy. Below-chance accuracy is not representable and is refused.
2. Draw the remaining ``n - 1`` probabilities, summing to ``1 - p_top`` and each
   no larger than ``p_top``, and sort the whole vector descending.
3. Sample a position ``j`` from that vector and place the gold label there. Fill
   the other positions with the other labels in a random order.
4. The predicted label is whatever sits at position 0.

Step 3 is the point of the design. It makes the mock calibrated in both senses
at once: the top probability equals P(the top label is gold), so binary Brier
and ECE have an exact target, and every label's probability equals P(that label
is gold), so multiclass Brier does too. Accuracy converges to ``E[p_top]``, which
is the configured accuracy.

Miscalibration is then injected as a temperature on the log probabilities, which
is exactly the distortion Phase 3 fits out. ``calibration_temperature < 1``
sharpens the distribution and yields an overconfident mock;
``calibration_temperature > 1`` flattens it and yields an underconfident one.
Because the map is monotone, the predicted label is unchanged, so accuracy is
independent of the skew. Temperature scaling fitted on the reported
probabilities should recover ``1 / calibration_temperature``.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping

from plumbline.adapters.base import Adapter, check_probability_semantics
from plumbline.types import (
    Prediction,
    ProbabilitySemantics,
    QuestionType,
    apply_temperature,
    docs_confidence,
)

#: Smoothing added to every label before the temperature transform, so that no
#: probability is exactly zero and every log is finite.
_EPSILON = 1e-12

#: How many times to redraw the tail of the distribution before falling back to
#: splitting the remaining mass evenly. The fallback always satisfies the
#: constraint that no tail entry exceeds the top entry, because p_top >= 1/n.
_TAIL_DRAW_ATTEMPTS = 8


class MockAdapter(Adapter):
    """A fake system under test whose behavior is known in advance.

    Args:
        gold_by_text: Maps a case's ``text`` to its gold label. The adapter
            interface deliberately does not pass the gold label to ``classify``,
            so the mock is told the answer key up front instead. A case whose
            text is absent raises, rather than quietly scoring as wrong.
        accuracy: Target fraction of cases answered correctly. Must be at least
            ``1 / len(labels)`` for every case, since the predicted label is the
            argmax of a distribution whose top entry is at least chance.
        calibration_temperature: Temperature applied to the log probabilities
            before reporting. 1.0 reports the calibrated distribution unchanged,
            below 1.0 is overconfident, above 1.0 is underconfident.
        probability_semantics: Declared as a constructor argument rather than
            hardcoded, so the report's grouping and its refusal to compare across
            groups can be tested against mocks in all three classes before any
            real adapter exists. Under ``"none"`` the adapter still predicts a
            label but reports no probability, no distribution, and no confidence.
        seed: Global seed. Each case draws from its own generator seeded by this
            value together with the case text and labels, so results do not
            depend on the order cases are run in or on how many workers run them.
        concentration: Beta concentration for ``p_top``. Lower values spread the
            predicted probabilities more widely across the range, which is what
            populates the reliability bins.
        report_tokens: When False, token counts are None, which is the path that
            makes downstream cost None rather than zero. It sets the adapter's
            ``reports_tokens`` capability too, so the mock stands in for a local
            checkpoint that cannot report cost at all rather than for an API
            that returned no counts on one call.
        model_reported: What the fake API claims it used. Defaults to
            ``model_requested``. Setting it to something else exercises the case
            plumbline records for real adapters, where a model alias resolves
            somewhere other than where the config pointed.
    """

    def __init__(
        self,
        gold_by_text: Mapping[str, str],
        *,
        name: str = "mock",
        accuracy: float = 0.8,
        calibration_temperature: float = 1.0,
        probability_semantics: ProbabilitySemantics = "calibrated_claim",
        seed: int = 0,
        concentration: float = 6.0,
        report_tokens: bool = True,
        model_requested: str = "mock-1",
        model_reported: str | None = None,
        latency_p50_ms: float = 40.0,
        latency_sigma: float = 0.5,
    ) -> None:
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"accuracy must lie in [0, 1], got {accuracy!r}")
        if calibration_temperature <= 0.0:
            raise ValueError(
                f"calibration_temperature must be positive, got {calibration_temperature!r}"
            )
        if concentration <= 0.0:
            raise ValueError(f"concentration must be positive, got {concentration!r}")
        if latency_p50_ms < 0.0:
            raise ValueError(f"latency_p50_ms must be non-negative, got {latency_p50_ms!r}")

        self.name = name
        self.model_requested = model_requested
        self.revision = None
        self.probability_semantics = check_probability_semantics(probability_semantics)

        self.gold_by_text = dict(gold_by_text)
        self.accuracy = accuracy
        self.calibration_temperature = calibration_temperature
        self.seed = seed
        self.concentration = concentration
        self.report_tokens = report_tokens
        self.reports_tokens = report_tokens
        self.model_reported = model_reported if model_reported is not None else model_requested
        self.latency_p50_ms = latency_p50_ms
        self.latency_sigma = latency_sigma

        self._call_params: dict[str, object] = {
            "accuracy": accuracy,
            "calibration_temperature": calibration_temperature,
            "seed": seed,
            "concentration": concentration,
        }

    @property
    def call_params(self) -> Mapping[str, object]:
        return self._call_params

    @property
    def skew(self) -> str:
        """The injected calibration skew, as the report would name it."""
        if math.isclose(self.calibration_temperature, 1.0):
            return "calibrated"
        return "overconfident" if self.calibration_temperature < 1.0 else "underconfident"

    def classify(
        self,
        text: str,
        labels: list[str],
        *,
        question_type: QuestionType = "choice",
    ) -> Prediction:
        """Answer as a choice, whatever the row is, and record that it did.

        The mock has one behaviour and no transport behind it, so a yes/no row
        is answered as a two-option choice here. The runner records ``asked_as``
        from ``raw`` so the difference reaches the report.
        """
        if len(labels) < 2:
            raise ValueError(f"need at least 2 labels, got {len(labels)}")
        if len(set(labels)) != len(labels):
            raise ValueError(f"labels contain duplicates: {labels!r}")
        if text not in self.gold_by_text:
            raise KeyError(
                "MockAdapter has no gold label for this case. Pass every case text in "
                "gold_by_text; the mock cannot hit a target accuracy without the answer key."
            )
        gold = self.gold_by_text[text]
        if gold not in labels:
            raise ValueError(f"gold label {gold!r} is not among the labels {labels!r}")

        n = len(labels)
        chance = 1.0 / n
        if self.accuracy < chance:
            raise ValueError(
                f"accuracy {self.accuracy!r} is below chance for {n} labels ({chance:.4f}). "
                "The predicted label is the argmax of the distribution, so the mock cannot "
                "be made to perform worse than chance by construction."
            )

        rng = self._case_rng(text, labels)
        calibrated = self._draw_calibrated_distribution(rng, labels, gold, n, chance)
        reported = apply_temperature(calibrated, self.calibration_temperature)
        predicted = max(reported, key=lambda label: reported[label])

        latency_ms = self._draw_latency(rng)
        input_tokens, output_tokens = self._draw_tokens(rng, text)

        raw = {
            "asked_as": "choice",
            "question_type": question_type,
            "gold_label": gold,
            "calibrated_distribution": calibrated,
            "calibration_temperature": self.calibration_temperature,
            "skew": self.skew,
        }

        if self.probability_semantics == "none":
            return Prediction(
                label=predicted,
                prob_selected=None,
                distribution=None,
                confidence=None,
                latency_ms=latency_ms,
                cost_usd=None,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model_reported=self.model_reported,
                raw=raw,
            )

        return Prediction(
            label=predicted,
            prob_selected=reported[predicted],
            distribution=reported,
            confidence=docs_confidence(reported),
            latency_ms=latency_ms,
            cost_usd=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_reported=self.model_reported,
            raw=raw,
        )

    def _case_rng(self, text: str, labels: list[str]) -> random.Random:
        """Seed a generator from the case itself, not from call order.

        ``calibration_temperature`` is deliberately excluded. The skew is a
        post-hoc distortion of how a draw is reported, not part of the draw, so
        two mocks differing only in temperature see the identical underlying
        distribution and predict the identical label on every case. That is what
        makes accuracy exactly, rather than approximately, independent of skew.
        """
        key = "\u0000".join(
            [
                str(self.seed),
                self.name,
                self.model_requested,
                str(self.revision),
                f"{self.accuracy!r}",
                f"{self.concentration!r}",
                text,
                *sorted(labels),
            ]
        )
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return random.Random(int.from_bytes(digest, "big"))

    def _draw_calibrated_distribution(
        self,
        rng: random.Random,
        labels: list[str],
        gold: str,
        n: int,
        chance: float,
    ) -> dict[str, float]:
        p_top = self._draw_p_top(rng, chance)
        tail = self._draw_tail(rng, p_top, n)
        ordered = [p_top, *sorted(tail, reverse=True)]

        # Place the gold label at a position sampled from the distribution
        # itself. This is what makes P(label is gold) equal that label's
        # probability, for every label and not only the top one.
        gold_position = _sample_index(rng, ordered)
        others = [label for label in labels if label != gold]
        rng.shuffle(others)

        arrangement: list[str] = []
        for position in range(n):
            arrangement.append(gold if position == gold_position else others.pop())

        distribution = dict(zip(arrangement, ordered, strict=True))
        return _smooth(distribution)

    def _draw_p_top(self, rng: random.Random, chance: float) -> float:
        """Draw the top probability so that its mean is the configured accuracy."""
        span = 1.0 - chance
        if span <= 0.0:
            return 1.0
        mean = (self.accuracy - chance) / span
        mean = min(1.0, max(0.0, mean))
        if mean <= 1e-9:
            unit = 0.0
        elif mean >= 1.0 - 1e-9:
            unit = 1.0
        else:
            unit = rng.betavariate(mean * self.concentration, (1.0 - mean) * self.concentration)
        p_top = chance + span * unit
        # Keep strictly inside the open interval so no log is degenerate.
        return min(1.0 - 1e-9, max(chance, p_top))

    def _draw_tail(self, rng: random.Random, p_top: float, n: int) -> list[float]:
        """Split the remaining mass, with no entry exceeding the top entry."""
        remaining = 1.0 - p_top
        even = remaining / (n - 1)
        for _ in range(_TAIL_DRAW_ATTEMPTS):
            weights = [rng.expovariate(1.0) for _ in range(n - 1)]
            total = sum(weights)
            if total <= 0.0:
                continue
            tail = [remaining * weight / total for weight in weights]
            if max(tail) <= p_top:
                return tail
        # An even split always satisfies the constraint, because p_top >= 1/n.
        return [even] * (n - 1)

    def _draw_latency(self, rng: random.Random) -> float:
        if self.latency_p50_ms == 0.0:
            return 0.0
        return rng.lognormvariate(math.log(self.latency_p50_ms), self.latency_sigma)

    def _draw_tokens(self, rng: random.Random, text: str) -> tuple[int | None, int | None]:
        if not self.report_tokens:
            return None, None
        input_tokens = max(1, len(text) // 4) + rng.randint(8, 24)
        output_tokens = rng.randint(1, 4)
        return input_tokens, output_tokens


def _smooth(distribution: dict[str, float]) -> dict[str, float]:
    """Nudge every probability off zero and renormalize."""
    nudged = {label: value + _EPSILON for label, value in distribution.items()}
    total = sum(nudged.values())
    return {label: value / total for label, value in nudged.items()}


def _sample_index(rng: random.Random, weights: list[float]) -> int:
    """Sample a position proportionally to ``weights``, which sum to one."""
    target = rng.random()
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if target < cumulative:
            return index
    return len(weights) - 1
