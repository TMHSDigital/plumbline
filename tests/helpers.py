"""Shared fixtures: synthetic corpora and small statistics helpers.

The binning helpers here are written independently of ``plumbline.metrics``, and
deliberately so. A Phase 1 test that proved the mock using the Phase 2 metrics
would be checking two unverified things against each other.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from plumbline.adapters.base import Adapter
from plumbline.types import Case, Prediction

if TYPE_CHECKING:
    from plumbline.types import ConfidenceSeries, ProbabilitySeries


def make_cases(
    n_cases: int,
    labels: Sequence[str] = ("billing", "returns", "shipping", "other"),
    *,
    seed: int = 1234,
    prefix: str = "case",
) -> list[Case]:
    """Build a synthetic labeled corpus with a uniformly random gold label."""
    rng = random.Random(seed)
    labels = tuple(labels)
    return [
        Case(
            id=f"{prefix}-{index}",
            text=f"{prefix}-{index}: synthetic ticket body {rng.random():.12f}",
            labels=labels,
            gold_label=rng.choice(labels),
        )
        for index in range(n_cases)
    ]


def make_variable_width_cases(
    n_cases: int,
    widths: Sequence[int] = (2, 3, 5, 8),
    *,
    seed: int = 4321,
) -> list[Case]:
    """Build a corpus whose cases have differing numbers of options.

    Confidence divides out the option count and ``prob_selected`` does not, so
    the two rank identically within a fixed label set and diverge here. Phase 2
    asserts both halves of that claim.
    """
    rng = random.Random(seed)
    cases: list[Case] = []
    for index in range(n_cases):
        width = widths[index % len(widths)]
        labels = tuple(f"label_{position}" for position in range(width))
        cases.append(
            Case(
                id=f"var-{index}",
                text=f"var-{index}: synthetic body {rng.random():.12f}",
                labels=labels,
                gold_label=rng.choice(labels),
            )
        )
    return cases


def gold_by_text(cases: Iterable[Case]) -> dict[str, str]:
    """The answer key the mock needs, since ``classify`` is not given the gold."""
    return {case.text: case.gold_label for case in cases}


def run(adapter: Adapter, cases: Iterable[Case]) -> list[tuple[Case, Prediction]]:
    """Classify every case, keeping the case alongside its prediction."""
    return [(case, adapter.classify(case.text, list(case.labels))) for case in cases]


def accuracy_of(results: Sequence[tuple[Case, Prediction]]) -> float:
    return sum(1 for case, pred in results if pred.label == case.gold_label) / len(results)


def equal_count_bins(
    pairs: Sequence[tuple[float, bool]], n_bins: int
) -> list[tuple[int, float, float]]:
    """Bin (probability, outcome) pairs by equal count.

    Returns one ``(count, mean probability, observed frequency)`` triple per bin.
    """
    ordered = sorted(pairs, key=lambda pair: pair[0])
    size = len(ordered) // n_bins
    bins: list[tuple[int, float, float]] = []
    for index in range(n_bins):
        start = index * size
        stop = len(ordered) if index == n_bins - 1 else start + size
        chunk = ordered[start:stop]
        if not chunk:
            continue
        mean_probability = sum(probability for probability, _ in chunk) / len(chunk)
        frequency = sum(1 for _, outcome in chunk if outcome) / len(chunk)
        bins.append((len(chunk), mean_probability, frequency))
    return bins


def max_bin_gap(pairs: Sequence[tuple[float, bool]], n_bins: int = 5) -> float:
    """Largest absolute gap between mean predicted probability and observed rate."""
    return max(
        abs(mean_probability - frequency)
        for _, mean_probability, frequency in equal_count_bins(pairs, n_bins)
    )


def measure(adapter: Adapter, cases: Sequence[Case]) -> Measured:
    """Run an adapter and package what the metrics functions need."""
    from plumbline.types import confidence_series
    from plumbline.types import probability_series as _probability_series

    results = run(adapter, cases)
    predictions = [prediction for _, prediction in results]
    outcomes = [prediction.label == case.gold_label for case, prediction in results]
    return Measured(
        cases=list(cases),
        predictions=predictions,
        outcomes=outcomes,
        probabilities=_probability_series(predictions, adapter.probability_semantics),
        confidences=confidence_series(predictions),
    )


@dataclass(frozen=True)
class Measured:
    cases: list[Case]
    predictions: list[Prediction]
    outcomes: list[bool]
    probabilities: ProbabilitySeries
    confidences: ConfidenceSeries

    @property
    def accuracy(self) -> float:
        return sum(self.outcomes) / len(self.outcomes)

    @property
    def distributions(self) -> list[dict[str, float] | None]:
        return [prediction.distribution for prediction in self.predictions]

    @property
    def gold_labels(self) -> list[str]:
        return [case.gold_label for case in self.cases]
