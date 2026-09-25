"""Rank-aware figures for ordinal score rows.

A score row asks for a level, 0 to K minus 1, and the levels are ordered. Every
choice metric is rank-blind: wrong by one level and wrong by three cost the
same. The three figures here are not:

- **Mean absolute error** of the expected score, in levels, read against a
  permutation null: the arm's own answers shuffled across the rows.
- **Ranked probability score**, the ordinal counterpart of the Brier score.
- **Cumulative calibration error**: at every threshold between two levels, the
  predicted probability that the level is at or below it against how often it
  was, pooled and binned as ECE bins a choice column.

The second and third are read against a calibrated-model floor built the way
every floor here is: the predicted distributions held fixed and each row's gold
level redrawn from its own distribution. METHODOLOGY, "Ordinal score questions
are scored by rank", is the design.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from plumbline.metrics.calibration import (
    DEFAULT_BINNING,
    DEFAULT_N_BINS,
    DEFAULT_N_BOOT,
    Binning,
    FloorBand,
    _judgment,
    bin_assignments,
    is_distinguishable,
)


@dataclass(frozen=True)
class ScoreAnswer:
    """One score row, ready to be measured."""

    levels: tuple[int, ...]
    """The row's levels, ascending."""
    probabilities: tuple[float, ...] | None
    """One probability per level, in the order of ``levels``, or None when the arm
    returned a level and no distribution."""
    expected: float
    """The expected score: the vendor's own, or the probability-weighted mean."""
    gold: int


def levels_of(labels: Sequence[str]) -> tuple[int, ...]:
    """A score row's options as ordered integer levels."""
    try:
        levels = sorted(int(label) for label in labels)
    except ValueError:
        raise ValueError(
            f"score levels must be integers, got {list(labels)!r}. A score row's options "
            "are its levels, 0 to K minus 1."
        ) from None
    if len(set(levels)) != len(levels):
        raise ValueError(f"score levels repeat: {list(labels)!r}")
    return tuple(levels)


def answer_from(
    labels: Sequence[str],
    distribution: Mapping[str, float] | None,
    label: str,
    gold_label: str,
    *,
    expected: float | None = None,
) -> ScoreAnswer:
    """A score answer from what a prediction carries.

    ``expected`` is the vendor's expected score when it returned one; it is read
    as given. Otherwise it is the probability-weighted mean of the levels, or,
    for an arm that returned no distribution, the level it answered.
    """
    levels = levels_of(labels)
    by_level = {int(option): option for option in labels}
    probabilities = (
        tuple(float(distribution.get(by_level[level], 0.0)) for level in levels)
        if distribution
        else None
    )
    if expected is None:
        expected = (
            float(np.dot(probabilities, levels)) if probabilities is not None else float(int(label))
        )
    return ScoreAnswer(
        levels=levels, probabilities=probabilities, expected=expected, gold=int(gold_label)
    )


def _with_distributions(answers: Sequence[ScoreAnswer]) -> list[ScoreAnswer]:
    return [answer for answer in answers if answer.probabilities is not None]


def cumulative_events(
    answers: Sequence[ScoreAnswer],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """One event per row per threshold: P(level <= threshold), and whether it was."""
    predicted: list[float] = []
    observed: list[float] = []
    for answer in _with_distributions(answers):
        assert answer.probabilities is not None
        cumulative = np.clip(np.cumsum(answer.probabilities)[:-1], 0.0, 1.0)
        predicted.extend(float(value) for value in cumulative)
        observed.extend(1.0 if answer.gold <= level else 0.0 for level in answer.levels[:-1])
    return np.asarray(predicted, dtype=np.float64), np.asarray(observed, dtype=np.float64)


def _event_weights(answers: Sequence[ScoreAnswer]) -> NDArray[np.float64]:
    """Each event's weight in the ranked probability score: 1 / (thresholds * rows)."""
    rows = _with_distributions(answers)
    return np.concatenate(
        [
            np.full(len(answer.levels) - 1, 1.0 / ((len(answer.levels) - 1) * len(rows)))
            for answer in rows
        ]
    )


def ranked_probability_score(answers: Sequence[ScoreAnswer]) -> float:
    """Mean over rows of the thresholds' squared cumulative gaps, over their count."""
    predicted, observed = cumulative_events(answers)
    if not predicted.size:
        raise ValueError("no score answer carries a distribution")
    return float(np.dot((predicted - observed) ** 2, _event_weights(answers)))


def cumulative_calibration_error(
    answers: Sequence[ScoreAnswer],
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = DEFAULT_BINNING,
) -> float:
    """ECE over the pooled threshold events: the count-weighted mean per-bin gap."""
    predicted, observed = cumulative_events(answers)
    if not predicted.size:
        raise ValueError("no score answer carries a distribution")
    return float(_binned_gap(predicted, observed[None, :], n_bins, binning)[0])


def _binned_gap(
    predicted: NDArray[np.float64],
    observed: NDArray[np.float64],
    n_bins: int,
    binning: Binning,
) -> NDArray[np.float64]:
    """ECE of fixed predictions against each row of ``observed``, one per draw."""
    indices, _ = bin_assignments(predicted, n_bins, binning)
    one_hot = np.zeros((predicted.size, n_bins))
    one_hot[np.arange(predicted.size), indices] = 1.0
    predicted_sums = predicted @ one_hot
    gaps: NDArray[np.float64] = np.abs(observed @ one_hot - predicted_sums).sum(axis=1)
    return gaps / predicted.size


def score_floor(
    answers: Sequence[ScoreAnswer],
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = DEFAULT_BINNING,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict[str, FloorBand]:
    """What a calibrated arm scores with these distributions at this row count.

    Each row's gold level is redrawn from its own predicted distribution, so every
    resample is calibrated by construction, and both figures are recomputed.
    """
    rows = _with_distributions(answers)
    if not rows:
        raise ValueError("no score answer carries a distribution")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    predicted, _ = cumulative_events(rows)
    rng = np.random.default_rng(seed)
    uniforms = rng.random((n_boot, len(rows)))
    observed = np.empty((n_boot, predicted.size))
    position = 0
    for column, answer in enumerate(rows):
        assert answer.probabilities is not None
        thresholds = len(answer.levels) - 1
        cumulative = np.cumsum(answer.probabilities)
        drawn = np.minimum(
            np.searchsorted(cumulative, uniforms[:, column], side="right"), thresholds
        )
        observed[:, position : position + thresholds] = (
            drawn[:, None] <= np.arange(thresholds)[None, :]
        )
        position += thresholds
    rps = ((predicted[None, :] - observed) ** 2) @ _event_weights(rows)
    calibration = _binned_gap(predicted, observed, n_bins, binning)

    def band(metric: str, values: NDArray[np.float64]) -> FloorBand:
        return FloorBand(
            metric=metric,
            mean=float(values.mean()),
            p95=float(np.percentile(values, 95)),
            n=len(rows),
            n_bins=n_bins,
            n_boot=n_boot,
        )

    return {"rps": band("rps", rps), "cumulative ece": band("cumulative ece", calibration)}


@dataclass(frozen=True)
class ErrorFigure:
    """Mean absolute error in levels, read against the permutation null."""

    value: float
    n: int
    null_mean: float
    null_p05: float
    null_p95: float

    @property
    def is_better_than_null(self) -> bool:
        return self.value < self.null_p05

    @property
    def is_worse_than_null(self) -> bool:
        return self.value > self.null_p95

    def statement(self) -> str:
        head = f"Mean absolute error {self.value:.4f} levels over {self.n} rows, against a "
        if self.is_better_than_null:
            return (
                head + f"permutation null of {self.null_mean:.4f} (5th percentile "
                f"{self.null_p05:.4f}): better than the permutation null at this sample size."
            )
        if self.is_worse_than_null:
            return (
                head + f"permutation null of {self.null_mean:.4f} (95th percentile "
                f"{self.null_p95:.4f}): worse than the permutation null at this sample size. "
                "The same answers shuffled across the rows would rarely score this badly, so "
                "they run against the levels, which usually means the scale is reversed "
                "between the dataset and the arm."
            )
        return (
            head + f"permutation null of {self.null_mean:.4f} (5th percentile "
            f"{self.null_p05:.4f}): INCONCLUSIVE at this sample size. The same answers "
            "shuffled across the rows would often score this well, so this dataset cannot "
            "tell whether they carry information about the level. This is not a result in "
            "either direction. Collect more rows to make the question answerable."
        )


def mean_absolute_error_figure(
    answers: Sequence[ScoreAnswer], n_boot: int = DEFAULT_N_BOOT, seed: int = 0
) -> ErrorFigure:
    """How far the expected scores land from the gold levels, against shuffled answers."""
    if not answers:
        raise ValueError("no score answers to measure")
    expected = np.asarray([answer.expected for answer in answers], dtype=np.float64)
    gold = np.asarray([answer.gold for answer in answers], dtype=np.float64)
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(
        [np.abs(rng.permutation(expected) - gold).mean() for _ in range(n_boot)], dtype=np.float64
    )
    return ErrorFigure(
        value=float(np.abs(expected - gold).mean()),
        n=len(answers),
        null_mean=float(shuffled.mean()),
        null_p05=float(np.percentile(shuffled, 5)),
        null_p95=float(np.percentile(shuffled, 95)),
    )


@dataclass(frozen=True)
class OrdinalFigure:
    """A ranked probability score or cumulative calibration error with its floor."""

    metric: str
    value: float
    n: int
    floor: FloorBand
    detail: str = ""

    @property
    def is_distinguishable(self) -> bool:
        return is_distinguishable(self.value, self.floor)

    def statement(self) -> str:
        name = {"rps": "Ranked probability score", "cumulative ece": "Cumulative calibration error"}
        return (
            f"{name[self.metric]} {self.value:.4f} over {self.n} rows{self.detail}, against a "
            f"calibrated-model floor of {self.floor.mean:.4f} (95th percentile "
            f"{self.floor.p95:.4f}): {_judgment(self.value, self.floor)}"
        )


def score_statements(
    answers: Sequence[ScoreAnswer],
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = DEFAULT_BINNING,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> list[str]:
    """The lines a report prints for an arm's score rows, each with its null."""
    if not answers:
        return []
    lines = [mean_absolute_error_figure(answers, n_boot=n_boot, seed=seed).statement()]
    rows = _with_distributions(answers)
    if not rows:
        lines.append(
            "Ranked probability score and cumulative calibration error: not reported. This "
            "arm returned no distribution over the levels, so there is nothing to score by "
            "rank beyond the error above."
        )
        return lines
    floors = score_floor(rows, n_bins=n_bins, binning=binning, n_boot=n_boot, seed=seed + 1)
    counts = {len(answer.levels) - 1 for answer in rows}
    per_row = (
        f"{counts.pop()} thresholds a row"
        if len(counts) == 1
        else f"{sum(len(answer.levels) - 1 for answer in rows)} threshold events"
    )
    lines.append(
        OrdinalFigure("rps", ranked_probability_score(rows), len(rows), floors["rps"]).statement()
    )
    lines.append(
        OrdinalFigure(
            "cumulative ece",
            cumulative_calibration_error(rows, n_bins, binning),
            len(rows),
            floors["cumulative ece"],
            detail=f" ({per_row}, {n_bins} {binning.replace('_', ' ')} bins)",
        ).statement()
    )
    if len(rows) < len(answers):
        lines.append(
            f"{len(answers) - len(rows)} of {len(answers)} score rows returned no "
            "distribution, so the two figures above cover the rest and the error covers all."
        )
    return lines
