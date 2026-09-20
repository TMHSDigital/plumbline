"""Calibration metrics, computed against ``prob_selected`` and nothing else.

Every function here takes a :class:`ProbabilitySeries`. A
:class:`ConfidenceSeries` is refused at the type boundary, because confidence is
a summary of distribution shape rather than a probability of correctness, and a
reliability diagram drawn against it would be meaningless while looking exactly
like a real one.

Why ECE has a floor
-------------------
ECE is a sum of absolute differences between per-bin accuracy and per-bin mean
probability. Absolute value does not cancel sampling noise, it accumulates it.
A perfectly calibrated model measured on a finite sample therefore scores an ECE
of roughly ``O(sqrt(1 / n_bin))``, not zero. At 4000 rows and 10 bins that floor
sits near 0.015 to 0.02. At 500 rows it is around three times larger, which is
the size of many real private datasets and larger than plenty of published ECE
differences.

So plumbline never prints a bare ECE. It prints the value together with the
floor for that sample size and bin count, and a sentence saying whether the two
are distinguishable. :func:`calibration_floor` builds that null by parametric
bootstrap: hold the observed predicted probabilities fixed, redraw each outcome
from ``Bernoulli(p_i)``, and read off the distribution of ECE that a perfectly
calibrated model of exactly this shape and size would produce.

MCE is worse. It is a maximum over bins rather than a weighted mean, so it is
biased further upward and is dominated by whichever bin is sparsest. It gets the
same floor treatment, and bins below ``min_bin_count`` are excluded outright.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from plumbline.types import (
    ConfidenceSeries,
    NotCalibratableError,
    ProbabilitySeries,
)

Binning = Literal["equal_width", "equal_count"]

DEFAULT_N_BINS = 10
DEFAULT_BINNING: Binning = "equal_width"

#: Bins holding fewer rows than this are excluded from MCE. A maximum over bins
#: is decided by its sparsest bin, and a bin of three rows can only report an
#: accuracy of 0, 1/3, 2/3, or 1.
DEFAULT_MIN_BIN_COUNT = 10

DEFAULT_N_BOOT = 2000


@dataclass(frozen=True)
class Bin:
    """One reliability bin: how many rows, what was predicted, what happened."""

    index: int
    lower: float
    upper: float
    count: int
    mean_probability: float
    accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.accuracy - self.mean_probability)


@dataclass(frozen=True)
class FloorBand:
    """The null distribution of a calibration metric at a given sample size.

    ``mean`` is what a perfectly calibrated model scores on average. ``p95`` is
    the value it exceeds only one time in twenty. A measurement at or below
    ``p95`` is not distinguishable from calibrated at this sample size.
    """

    metric: str
    mean: float
    p95: float
    n: int
    n_bins: int
    n_boot: int


def _guard(series: ProbabilitySeries | ConfidenceSeries) -> tuple[float, ...]:
    """Refuse anything that is not a probability, by type and then by content."""
    if isinstance(series, ConfidenceSeries):
        raise NotCalibratableError(
            "confidence is not a probability of correctness. It is a statistic "
            "computed from the shape of the distribution, and the vendor docs say a "
            "different computation may suit a given case better. ECE, MCE, Brier, and "
            "reliability diagrams are computed against prob_selected only. Use "
            "plumbline.metrics.discrimination for confidence."
        )
    if not isinstance(series, ProbabilitySeries):
        raise TypeError(
            f"expected a ProbabilitySeries, got {type(series).__name__}. Wrap the column "
            "so its semantics travel with it."
        )
    return series.require_reportable()


def _checked(
    series: ProbabilitySeries | ConfidenceSeries, correct: Sequence[bool]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = _guard(series)
    if len(values) != len(correct):
        raise ValueError(
            f"{len(values)} probabilities against {len(correct)} outcomes; these must match"
        )
    if not values:
        raise ValueError("no rows to score")
    return np.asarray(values, dtype=np.float64), np.asarray(correct, dtype=np.float64)


def bin_assignments(
    probabilities: NDArray[np.float64],
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = DEFAULT_BINNING,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Assign each probability to a bin, returning the indices and the edges.

    ``equal_width`` cuts [0, 1] into ``n_bins`` equal slices. It is the default
    because the bins mean the same thing across adapters, which is what makes two
    reliability diagrams comparable. Its cost is that the upper bins are sparse
    for a model whose probabilities cluster high, and a sparse bin is exactly
    where ECE's finite-sample noise lives.

    ``equal_count`` cuts at quantiles instead, so every bin is equally populated
    and the noise is spread evenly. Its cost is that the bin boundaries differ
    between adapters, so the diagrams no longer line up.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")

    if binning == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        indices = np.clip(np.floor(probabilities * n_bins).astype(np.intp), 0, n_bins - 1)
        return indices, edges

    if binning == "equal_count":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(probabilities, quantiles)
        edges[0], edges[-1] = 0.0, 1.0
        # Rows are split by rank, so ties that straddle a boundary land together
        # in whichever bin the sort puts them. Counts are then near-equal rather
        # than exactly equal, which is the honest behavior for tied predictions.
        order = np.argsort(probabilities, kind="stable")
        indices = np.empty(len(probabilities), dtype=np.intp)
        chunks = np.array_split(order, n_bins)
        for position, chunk in enumerate(chunks):
            indices[chunk] = position
        return indices, edges

    raise ValueError(f"binning must be 'equal_width' or 'equal_count', got {binning!r}")


def reliability(
    series: ProbabilitySeries,
    correct: Sequence[bool],
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = DEFAULT_BINNING,
) -> list[Bin]:
    """Per bin: how many rows, the mean predicted probability, and the accuracy."""
    probabilities, outcomes = _checked(series, correct)
    indices, edges = bin_assignments(probabilities, n_bins, binning)

    counts = np.bincount(indices, minlength=n_bins)
    summed_probability = np.bincount(indices, weights=probabilities, minlength=n_bins)
    summed_correct = np.bincount(indices, weights=outcomes, minlength=n_bins)

    bins: list[Bin] = []
    for index in range(n_bins):
        count = int(counts[index])
        if count == 0:
            continue
        bins.append(
            Bin(
                index=index,
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                mean_probability=float(summed_probability[index] / count),
                accuracy=float(summed_correct[index] / count),
            )
        )
    return bins


def ece(
    series: ProbabilitySeries,
    correct: Sequence[bool],
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = DEFAULT_BINNING,
) -> float:
    """Expected calibration error: the count-weighted mean per-bin gap.

    ``sum_b (n_b / N) * |accuracy_b - mean_probability_b|``.

    Read this next to :func:`calibration_floor`, never on its own.
    """
    bins = reliability(series, correct, n_bins, binning)
    total = sum(single.count for single in bins)
    return sum(single.count * single.gap for single in bins) / total


def mce(
    series: ProbabilitySeries,
    correct: Sequence[bool],
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = DEFAULT_BINNING,
    min_bin_count: int = DEFAULT_MIN_BIN_COUNT,
) -> float:
    """Maximum calibration error over bins holding at least ``min_bin_count`` rows.

    Raises if no bin qualifies. That is not a metric of zero, it is a dataset too
    small to support the question, and saying so is the useful answer.
    """
    bins = [
        single
        for single in reliability(series, correct, n_bins, binning)
        if single.count >= min_bin_count
    ]
    if not bins:
        raise ValueError(
            f"no bin holds at least {min_bin_count} rows, so MCE is undefined here. "
            "Reduce n_bins, use equal_count binning, or collect more rows."
        )
    return max(single.gap for single in bins)


def brier(series: ProbabilitySeries, correct: Sequence[bool]) -> float:
    """Binary Brier score: ``mean((prob_selected - correct) ** 2)``.

    This is the two-class form applied to the top label, because that is the only
    probability some adapters report. It is **not** the multiclass Brier score
    and the two are not comparable. A reader will assume otherwise, so the report
    labels it explicitly. Where a full distribution exists,
    :func:`multiclass_brier` is computed and reported as a separate figure.
    """
    probabilities, outcomes = _checked(series, correct)
    return float(np.mean((probabilities - outcomes) ** 2))


def multiclass_brier(
    distributions: Sequence[Mapping[str, float] | None],
    gold_labels: Sequence[str],
) -> float:
    """Multiclass Brier score: ``mean over cases of sum_k (p_k - y_k) ** 2``.

    The unnormalized sum-of-squares convention, so the range is [0, 2] and a
    perfect model scores 0. Some sources halve this. plumbline does not, and the
    report says which convention it used.

    Refuses if any case lacks a distribution, rather than scoring it as zero.
    """
    if len(distributions) != len(gold_labels):
        raise ValueError(
            f"{len(distributions)} distributions against {len(gold_labels)} gold labels"
        )
    missing = sum(1 for distribution in distributions if distribution is None)
    if missing:
        raise NotCalibratableError(
            f"{missing} of {len(distributions)} cases report no distribution, so the "
            "multiclass Brier score is not available for this adapter."
        )

    total = 0.0
    for distribution, gold in zip(distributions, gold_labels, strict=True):
        assert distribution is not None
        if gold not in distribution:
            raise ValueError(f"gold label {gold!r} is absent from the distribution")
        total += sum(
            (probability - (1.0 if label == gold else 0.0)) ** 2
            for label, probability in distribution.items()
        )
    return total / len(distributions)


def calibration_floor(
    series: ProbabilitySeries,
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = DEFAULT_BINNING,
    min_bin_count: int = DEFAULT_MIN_BIN_COUNT,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> dict[str, FloorBand]:
    """The null distribution of ECE and MCE for a perfectly calibrated model.

    Parametric bootstrap. The observed predicted probabilities are held fixed and
    only the outcomes are redrawn, each from ``Bernoulli(p_i)``. Every resample is
    therefore perfectly calibrated by construction, and the spread of ECE across
    resamples is the noise floor for this exact sample size, bin count, and shape
    of predicted probabilities.

    Conditioning on the observed probabilities matters. The floor depends heavily
    on how the predictions are spread across the range, not only on how many
    there are, which is why this is preferred over :func:`synthetic_floor`.

    Returns a band for ``"ece"`` and ``"brier"``, and, when any bin qualifies,
    for ``"mce"``.
    """
    probabilities = np.asarray(_guard(series), dtype=np.float64)
    return _bootstrap_floor(probabilities, n_bins, binning, min_bin_count, n_boot, seed)


def synthetic_floor(
    n: int,
    n_bins: int = DEFAULT_N_BINS,
    accuracy: float = 0.8,
    n_boot: int = DEFAULT_N_BOOT,
    binning: Binning = DEFAULT_BINNING,
    min_bin_count: int = DEFAULT_MIN_BIN_COUNT,
    concentration: float = 6.0,
    seed: int = 0,
) -> dict[str, FloorBand]:
    """A floor from summary numbers alone, for planning a run before making it.

    Use this to answer "how many rows do I need before an ECE difference of 0.02
    would mean anything?". It invents a probability vector of size ``n`` whose
    mean is ``accuracy``, so it is only as good as that assumption.
    ``concentration`` sets how tightly the invented probabilities cluster, and it
    moves the floor, which is precisely why :func:`calibration_floor` on real
    predictions is the one the report prints.
    """
    if not 0.0 < accuracy < 1.0:
        raise ValueError(f"accuracy must lie strictly inside (0, 1), got {accuracy!r}")
    rng = np.random.default_rng(seed)
    probabilities = rng.beta(accuracy * concentration, (1.0 - accuracy) * concentration, size=n)
    return _bootstrap_floor(probabilities, n_bins, binning, min_bin_count, n_boot, seed + 1)


def _bootstrap_floor(
    probabilities: NDArray[np.float64],
    n_bins: int,
    binning: Binning,
    min_bin_count: int,
    n_boot: int,
    seed: int,
) -> dict[str, FloorBand]:
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    n = len(probabilities)
    indices, _ = bin_assignments(probabilities, n_bins, binning)

    counts = np.bincount(indices, minlength=n_bins).astype(np.float64)
    occupied = counts > 0
    mean_probability = np.zeros(n_bins)
    np.divide(
        np.bincount(indices, weights=probabilities, minlength=n_bins),
        counts,
        out=mean_probability,
        where=occupied,
    )

    # One-hot the bin assignment once, then every resample is a single matmul.
    membership = np.zeros((n, n_bins), dtype=np.float64)
    membership[np.arange(n), indices] = 1.0

    rng = np.random.default_rng(seed)
    outcomes = (rng.random((n_boot, n)) < probabilities).astype(np.float64)
    accuracies = np.zeros((n_boot, n_bins))
    np.divide(outcomes @ membership, counts, out=accuracies, where=occupied)

    gaps = np.abs(accuracies - mean_probability)
    gaps[:, ~occupied] = 0.0

    ece_draws = (gaps * counts).sum(axis=1) / n
    brier_draws = ((outcomes - probabilities) ** 2).mean(axis=1)
    bands = {
        "ece": FloorBand(
            metric="ece",
            mean=float(ece_draws.mean()),
            p95=float(np.percentile(ece_draws, 95)),
            n=n,
            n_bins=n_bins,
            n_boot=n_boot,
        ),
        # The same resampled outcomes, scored as Brier. A calibrated model of
        # this sharpness does not score zero, and this is what it does score.
        "brier": FloorBand(
            metric="brier",
            mean=float(brier_draws.mean()),
            p95=float(np.percentile(brier_draws, 95)),
            n=n,
            n_bins=n_bins,
            n_boot=n_boot,
        ),
    }

    qualifying = counts >= min_bin_count
    if qualifying.any():
        mce_draws = gaps[:, qualifying].max(axis=1)
        bands["mce"] = FloorBand(
            metric="mce",
            mean=float(mce_draws.mean()),
            p95=float(np.percentile(mce_draws, 95)),
            n=n,
            n_bins=n_bins,
            n_boot=n_boot,
        )
    return bands


#: How each metric is spelled in a report line.
METRIC_NAMES = {
    "ece": "ECE",
    "mce": "MCE",
    "brier": "Brier",
    "multiclass brier": "Multiclass Brier",
}


@dataclass(frozen=True)
class CalibrationFigure:
    """A calibration number, the sample it came from, and its floor, together.

    ECE on its own is unreadable. The floor it has to clear depends on the row
    count, so 0.03 over 5,000 rows and 0.03 over 80 rows are different findings
    and a report that prints only the number invites the wrong one. This type
    exists so that a figure cannot be rendered without the count beside it.
    """

    metric: str
    value: float
    n: int
    floor: FloorBand
    n_bins: int | None = None
    binning: Binning | None = None

    @property
    def is_distinguishable(self) -> bool:
        return is_distinguishable(self.value, self.floor)

    def statement(self) -> str:
        """The line a report prints: the number, the rows, and the floor."""
        bins = (
            f" ({self.n_bins} {self.binning.replace('_', ' ')} bins)"
            if self.n_bins is not None and self.binning is not None
            else ""
        )
        name = METRIC_NAMES.get(self.metric, self.metric.upper())
        return (
            f"{name} {self.value:.4f} over {self.n} rows{bins}, against a "
            f"calibrated-model floor of {self.floor.mean:.4f} "
            f"(95th percentile {self.floor.p95:.4f}): {_judgment(self.value, self.floor)}"
        )


def ece_figure(
    series: ProbabilitySeries,
    correct: Sequence[bool],
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = DEFAULT_BINNING,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> CalibrationFigure:
    """ECE with the row count and the calibrated-null floor attached.

    The way a report should ask for ECE. :func:`ece` returns the bare number for
    arithmetic that needs one; anything a reader sees goes through here.
    """
    value = ece(series, correct, n_bins, binning)
    floor = calibration_floor(series, n_bins, binning, n_boot=n_boot, seed=seed)["ece"]
    return CalibrationFigure(
        metric="ece",
        value=value,
        n=floor.n,
        n_bins=n_bins,
        binning=binning,
        floor=floor,
    )


def mce_figure(
    series: ProbabilitySeries,
    correct: Sequence[bool],
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = DEFAULT_BINNING,
    min_bin_count: int = DEFAULT_MIN_BIN_COUNT,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> CalibrationFigure:
    """MCE with its floor, for the diagnostics block.

    Never printed beside ECE. A maximum over bins is decided by one bin, it
    cannot detect gross overconfidence spread evenly across the range, and at a
    few hundred rows its floor is wide enough to swallow most real differences.
    It raises rather than returning a number when no bin qualifies.
    """
    value = mce(series, correct, n_bins, binning, min_bin_count)
    bands = calibration_floor(series, n_bins, binning, min_bin_count, n_boot=n_boot, seed=seed)
    return CalibrationFigure(
        metric="mce",
        value=value,
        n=bands["ece"].n,
        n_bins=n_bins,
        binning=binning,
        floor=bands["mce"],
    )


def brier_figure(
    series: ProbabilitySeries,
    correct: Sequence[bool],
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = DEFAULT_BINNING,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> CalibrationFigure:
    """Binary Brier against what a calibrated model of the same sharpness scores.

    The floor here is not zero and is not a constant: a model that reports 0.6
    on every case cannot score below 0.24 however well calibrated it is. The
    band is the distribution of Brier when the outcomes are redrawn from the
    reported probabilities, so exceeding it means worse than calibrated rather
    than merely imperfect.
    """
    value = brier(series, correct)
    floor = calibration_floor(series, n_bins, binning, n_boot=n_boot, seed=seed)["brier"]
    return CalibrationFigure(metric="brier", value=value, n=floor.n, floor=floor)


def multiclass_brier_figure(
    distributions: Sequence[Mapping[str, float]],
    gold_labels: Sequence[str],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> CalibrationFigure:
    """Multiclass Brier with the null a calibrated model of this shape produces."""
    value = multiclass_brier(distributions, gold_labels)
    floor = multiclass_brier_floor(distributions, n_boot=n_boot, seed=seed)
    return CalibrationFigure(metric="multiclass brier", value=value, n=floor.n, floor=floor)


def multiclass_brier_floor(
    distributions: Sequence[Mapping[str, float]],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> FloorBand:
    """The null distribution of multiclass Brier for these reported distributions.

    Same parametric bootstrap as the calibration floor, one level up: the gold
    label of each case is redrawn from that case's own reported distribution, so
    every resample is perfectly calibrated by construction and the spread is what
    this many rows of this shape produce on their own.
    """
    if not distributions:
        raise NotCalibratableError("no distributions, so there is no null to build")
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")

    rng = np.random.default_rng(seed)
    totals = np.zeros(n_boot, dtype=np.float64)
    for distribution in distributions:
        values = np.asarray(list(distribution.values()), dtype=np.float64)
        # sum_k (p_k - y_k)^2 collapses to sum_k p_k^2 + 1 - 2 p_gold.
        drawn = values[np.searchsorted(np.cumsum(values), rng.random(n_boot))]
        totals += float((values**2).sum()) + 1.0 - 2.0 * drawn

    draws = totals / len(distributions)
    return FloorBand(
        metric="multiclass brier",
        mean=float(draws.mean()),
        p95=float(np.percentile(draws, 95)),
        n=len(distributions),
        n_bins=0,  # Not a binned metric; the field is carried for one shape of band.
        n_boot=n_boot,
    )


def is_distinguishable(measured: float, floor: FloorBand) -> bool:
    """Whether a measured value exceeds what calibrated noise alone would produce."""
    return measured > floor.p95


def verdict(measured: float, floor: FloorBand) -> str:
    """One plain sentence for the report, next to the number.

    On a 500 row private dataset this sentence is the difference between a
    finding and a coincidence, which is the case plumbline exists for.
    """
    name = floor.metric.upper()
    common = (
        f"{name} {measured:.4f} against a calibrated-model floor of "
        f"{floor.mean:.4f} (95th percentile {floor.p95:.4f}) at n={floor.n}, "
        f"{floor.n_bins} bins"
    )
    return f"{common}: {_judgment(measured, floor)}"


def _judgment(measured: float, floor: FloorBand) -> str:
    """The trailing clause of a verdict, so one wording serves every caller."""
    if is_distinguishable(measured, floor):
        return "miscalibration is distinguishable from sampling noise."
    return (
        "not distinguishable from a perfectly calibrated model at this sample size. "
        "Collect more rows before reading anything into it."
    )
