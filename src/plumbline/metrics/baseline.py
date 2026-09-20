"""Null bands for the figures whose null is not a calibration floor.

Accuracy and AUROC are read the same way ECE is: against what the number would
be if nothing were happening, at this exact sample size. A bare 0.72 accuracy is
unreadable -- it is excellent on eight-way options and it is nothing on two-way
options -- and a bare AUROC of 0.58 over 100 rows is well inside what an
uninformative score column produces by chance.

Both nulls are simulated rather than assumed. Accuracy's null draws an answer
uniformly from each case's own options, so the mix of option widths in the
dataset is what sets chance rather than a single assumed 1/n. AUROC's null
permutes the outcomes against the same scores, which is the distribution of the
statistic when the score carries no information about the outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from plumbline.metrics.discrimination import auroc
from plumbline.types import ConfidenceSeries, ProbabilitySeries

DEFAULT_N_BOOT = 2000


@dataclass(frozen=True)
class NullBand:
    """What a metric does when nothing is happening, at one sample size.

    ``mean`` is the expected value under the null and ``p95`` is what it exceeds
    one time in twenty. A measurement inside the band is not a finding.
    """

    metric: str
    null: str  # how the null was built, in one word: "chance", "permutation"
    mean: float
    p95: float
    n: int
    n_boot: int


#: How each metric is spelled in a report line.
METRIC_NAMES = {"accuracy": "Accuracy", "auroc": "AUROC"}


@dataclass(frozen=True)
class Figure:
    """A measured value with its sample size and its null, never one alone."""

    metric: str
    value: float
    n: int
    band: NullBand
    beats: str  # what clearing the band means, in three words
    note: str = ""

    @property
    def is_distinguishable(self) -> bool:
        """Whether the value is outside what the null produces at this size."""
        return self.value > self.band.p95

    def statement(self) -> str:
        judgment = (
            f"{self.beats} at this sample size."
            if self.is_distinguishable
            else (
                f"not distinguishable from {self.band.null} at this sample size. "
                "Collect more rows before reading anything into it."
            )
        )
        name = METRIC_NAMES.get(self.metric, self.metric.upper())
        line = (
            f"{name} {self.value:.4f} over {self.n} rows, against a "
            f"{self.band.null} null of {self.band.mean:.4f} "
            f"(95th percentile {self.band.p95:.4f}): {judgment}"
        )
        return f"{line} {self.note}".strip()


def chance_band(
    option_counts: Sequence[int],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> NullBand:
    """What guessing scores on these cases, given each case's own option count.

    Conditioning on the widths matters. A dataset of yes/no rows has a chance
    accuracy of 0.5 and a dataset of eight-way rows has 0.125; a corpus mixing
    them has neither, and quoting a single 1/n for it would flatter or punish
    the model depending on which rows happened to be included.
    """
    if not option_counts:
        raise ValueError("no cases to build a chance band from")
    if any(count < 2 for count in option_counts):
        raise ValueError("every case needs at least 2 options")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")

    probabilities = 1.0 / np.asarray(option_counts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = (rng.random((n_boot, len(probabilities))) < probabilities).mean(axis=1)
    return NullBand(
        metric="accuracy",
        null="chance",
        # The expectation is known exactly -- it is the mean of 1/n over the
        # cases -- so it is computed rather than estimated. Only the spread,
        # which is what the sample size controls, needs the simulation.
        mean=float(probabilities.mean()),
        p95=float(np.percentile(draws, 95)),
        n=len(probabilities),
        n_boot=n_boot,
    )


def accuracy_figure(
    correct: Sequence[bool],
    option_counts: Sequence[int],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> Figure:
    """Accuracy with the row count and the chance band it has to clear."""
    if len(correct) != len(option_counts):
        raise ValueError(
            f"{len(correct)} outcomes against {len(option_counts)} option counts; these must match"
        )
    band = chance_band(option_counts, n_boot=n_boot, seed=seed)
    return Figure(
        metric="accuracy",
        value=sum(1 for outcome in correct if outcome) / len(correct),
        n=len(correct),
        band=band,
        beats="better than chance",
    )


def auroc_band(
    scores: Sequence[float],
    correct: Sequence[bool],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> NullBand:
    """What AUROC does when the score column says nothing about the outcome.

    Built by permuting the outcomes against the same scores, so the band keeps
    the observed ties and the observed class balance -- both of which move it.
    """
    if len(scores) != len(correct):
        raise ValueError(f"{len(scores)} scores against {len(correct)} outcomes; these must match")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")

    values = np.asarray(scores, dtype=np.float64)
    outcomes = np.asarray(correct, dtype=bool)
    ranks = _mid_ranks(values)
    positives = int(outcomes.sum())
    negatives = len(outcomes) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            "AUROC is undefined when every case is correct or every case is wrong, so "
            "there is no null band to build either."
        )

    rng = np.random.default_rng(seed)
    # Permuting the outcome labels is the same as drawing which rank positions
    # are positives, so the draw is a shuffle of ranks rather than a re-scoring.
    draws = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        drawn = rng.permutation(len(ranks))[:positives]
        rank_sum = ranks[drawn].sum()
        draws[index] = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    return NullBand(
        metric="auroc",
        null="permutation",
        mean=float(draws.mean()),
        p95=float(np.percentile(draws, 95)),
        n=len(values),
        n_boot=n_boot,
    )


def auroc_figure(
    series: ProbabilitySeries | ConfidenceSeries,
    correct: Sequence[bool],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> Figure:
    """AUROC with the row count and the permutation null it has to clear."""
    values = series.require_reportable()
    return Figure(
        metric="auroc",
        value=auroc(series, correct),
        n=len(values),
        band=auroc_band(values, correct, n_boot=n_boot, seed=seed),
        beats="separates correct from incorrect",
    )


def _mid_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ranks with ties averaged, matching how :func:`auroc` handles them."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    sorted_values = values[order]
    start = 0
    for index in range(1, len(sorted_values) + 1):
        if index == len(sorted_values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                tied = order[start:index]
                ranks[tied] = ranks[tied].mean()
            start = index
    return ranks
