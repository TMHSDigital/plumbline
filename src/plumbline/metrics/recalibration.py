"""Temperature scaling fitted on the user's own data. The differentiating feature.

A leaderboard has to report raw calibration to stay comparable across systems. A
tool pointed at one user's data can do something no leaderboard can: fit a
correction on that data and hand back the number that goes into production code.

Three disciplines make that honest rather than harmful.

**The split is mandatory and enforced in code.** Fitting and reporting on the
same rows is the one mistake that would make this tool actively harmful, because
it manufactures an improvement that will not survive contact with new data.
There is no public entry point here that takes a temperature and evaluates it on
the rows it was fitted on. :class:`Split` refuses overlapping index sets at
construction, and :func:`recalibrate` builds the split itself.

**The fit is reported as an interval, not a point.** At 200 to 500 eval rows a
temperature carries real variance, and a reader handed "T=1.47" will paste 1.47
into a config file. The fit is bootstrapped and a 95 percent interval reported.
When that interval spans 1.0, the honest statement is that no recalibration is
justified at this sample size, and :meth:`RecalibrationResult.summary` says
exactly that instead of shipping a temperature fitted to noise. This is the same
discipline as the ECE floor, applied to the fit rather than to the metric.

**The residual is reported.** Temperature scaling is a one-parameter family. Real
miscalibration is not a pure temperature: a single label can be systematically
overconfident while the rest are fine, or the distortion can differ above and
below mid-range. In those cases a fitted T improves ECE without returning it to
the floor, and a user needs to see what was left on the table rather than
believing the fit was complete.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]
from scipy.special import logsumexp  # type: ignore[import-untyped]

from plumbline.metrics.calibration import (
    DEFAULT_BINNING,
    DEFAULT_MIN_BIN_COUNT,
    DEFAULT_N_BINS,
    Binning,
    FloorBand,
    brier,
    calibration_floor,
    ece,
    mce,
    multiclass_brier,
)
from plumbline.types import (
    InsufficientDataError,
    ProbabilitySeries,
    apply_temperature,
)

Method = Literal["multiclass", "binary"]

#: Below this many evaluation rows, recalibration refuses. A temperature fitted
#: and judged on fewer rows than this reports an improvement that is mostly the
#: noise of the split itself.
DEFAULT_MIN_EVAL_ROWS = 200

#: Search bounds for the fit. Wide enough to cover any real miscalibration and
#: narrow enough that hitting an edge is a signal that something is wrong.
TEMPERATURE_BOUNDS = (0.05, 20.0)

DEFAULT_N_BOOT_CI = 400

_PROBABILITY_CLIP = 1e-12


@dataclass(frozen=True)
class Split:
    """Which rows fit the temperature and which rows judge it.

    Disjointness is checked here rather than trusted, because this is the single
    invariant whose violation would invalidate every number downstream.
    """

    fit_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        fit, evaluate = set(self.fit_indices), set(self.eval_indices)
        if len(fit) != len(self.fit_indices) or len(evaluate) != len(self.eval_indices):
            raise ValueError("a split may not repeat a row")
        if not fit or not evaluate:
            raise ValueError("both halves of the split must be non-empty")
        overlap = fit & evaluate
        if overlap:
            raise ValueError(
                f"{len(overlap)} rows appear in both halves of the split. Fitting a "
                "temperature and reporting it on the same rows manufactures an "
                "improvement that will not survive new data."
            )


def make_split(n: int, fit_fraction: float = 0.5, seed: int = 0) -> Split:
    """A random disjoint split. Default 50/50, with the seed recorded in the report."""
    if n < 2:
        raise ValueError(f"need at least 2 rows to split, got {n}")
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError(f"fit_fraction must lie strictly inside (0, 1), got {fit_fraction!r}")

    order = np.random.default_rng(seed).permutation(n)
    cut = max(1, min(n - 1, round(n * fit_fraction)))
    return Split(
        fit_indices=tuple(int(index) for index in order[:cut]),
        eval_indices=tuple(int(index) for index in order[cut:]),
    )


@dataclass(frozen=True)
class MetricSet:
    """Calibration metrics for one set of rows, before or after scaling."""

    ece: float
    mce: float | None
    brier: float
    multiclass_brier: float | None


@dataclass(frozen=True)
class RecalibrationResult:
    """Everything the report needs to state what the fit did and did not achieve."""

    method: Method
    temperature: float
    temperature_ci: tuple[float, float]
    justified: bool
    split: Split
    seed: int
    before: MetricSet
    after: MetricSet
    floor: dict[str, FloorBand]

    @property
    def n_fit(self) -> int:
        return len(self.split.fit_indices)

    @property
    def n_eval(self) -> int:
        return len(self.split.eval_indices)

    @property
    def residual_ratio(self) -> float:
        """Post-scaling ECE as a multiple of what a calibrated model would score."""
        return self.after.ece / self.floor["ece"].p95

    @property
    def fit_is_complete(self) -> bool:
        """Whether scaling brought ECE back inside the calibrated noise band."""
        return self.after.ece <= self.floor["ece"].p95

    def summary(self) -> str:
        lines = [
            f"Temperature scaling, {self.method} form, fitted on {self.n_fit} rows and "
            f"evaluated on {self.n_eval} held-out rows (seed {self.seed}).",
            f"Fitted T = {self.temperature:.3f}, 95 percent interval "
            f"[{self.temperature_ci[0]:.3f}, {self.temperature_ci[1]:.3f}].",
        ]
        if not self.justified:
            lines.append(
                "That interval spans 1.0, so no recalibration is justified at this "
                "sample size. Do not hardcode this temperature. Collect more rows."
            )
        lines.append(
            f"ECE {self.before.ece:.4f} before, {self.after.ece:.4f} after, against a "
            f"calibrated-model floor of {self.floor['ece'].mean:.4f} "
            f"(95th percentile {self.floor['ece'].p95:.4f})."
        )
        if self.fit_is_complete:
            lines.append(
                "Post-scaling ECE is inside the floor, so a single temperature accounts "
                "for the miscalibration that was present."
            )
        else:
            lines.append(
                f"Post-scaling ECE is still {self.residual_ratio:.1f} times the floor. A "
                "single temperature does not account for all of this miscalibration; "
                "some of it is structure that one parameter cannot reach, such as a "
                "per-label bias or a distortion that differs across the range."
            )
        return " ".join(lines)


def apply_temperature_binary(probability: float, temperature: float) -> float:
    """Temperature on a lone top-label probability: ``sigmoid(logit(p) / T)``.

    Used only when no full distribution is available. It is itself a misspecified
    model of a multiclass distortion, so it leaves residual even on a mock whose
    skew is a pure temperature. Prefer the multiclass form wherever the adapter
    reports a distribution.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")
    clipped = min(1.0 - _PROBABILITY_CLIP, max(_PROBABILITY_CLIP, probability))
    logit = math.log(clipped / (1.0 - clipped))
    return 1.0 / (1.0 + math.exp(-logit / temperature))


def _stack(
    distributions: Sequence[Mapping[str, float]], gold_labels: Sequence[str]
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Pack ragged distributions into a log-probability matrix padded with -inf.

    Padding with negative infinity lets a corpus of mixed option counts share one
    matrix, because ``logsumexp`` ignores those entries and ``-inf / T`` stays
    ``-inf`` for any positive T.
    """
    width = max(len(distribution) for distribution in distributions)
    log_probs = np.full((len(distributions), width), -np.inf)
    gold_positions = np.empty(len(distributions), dtype=np.intp)

    for row, (distribution, gold) in enumerate(zip(distributions, gold_labels, strict=True)):
        labels = sorted(distribution)
        if gold not in distribution:
            raise ValueError(f"gold label {gold!r} is absent from the distribution")
        for column, label in enumerate(labels):
            log_probs[row, column] = math.log(max(distribution[label], _PROBABILITY_CLIP))
        gold_positions[row] = labels.index(gold)

    return log_probs, gold_positions


def _nll_multiclass(
    log_probs: NDArray[np.float64], gold_positions: NDArray[np.intp], temperature: float
) -> float:
    scaled = log_probs / temperature
    chosen = scaled[np.arange(len(gold_positions)), gold_positions]
    return float(np.mean(logsumexp(scaled, axis=1) - chosen))


def _nll_binary(
    probabilities: NDArray[np.float64], outcomes: NDArray[np.float64], temperature: float
) -> float:
    clipped = np.clip(probabilities, _PROBABILITY_CLIP, 1.0 - _PROBABILITY_CLIP)
    logits = np.log(clipped / (1.0 - clipped)) / temperature
    # log(1 + exp(-x)) computed stably.
    return float(
        np.mean(np.logaddexp(0.0, -logits) * outcomes + np.logaddexp(0.0, logits) * (1 - outcomes))
    )


def _minimize(objective: Callable[[float], float]) -> float:
    result = minimize_scalar(objective, bounds=TEMPERATURE_BOUNDS, method="bounded")
    return float(result.x)


def fit_temperature_multiclass(
    distributions: Sequence[Mapping[str, float]], gold_labels: Sequence[str]
) -> float:
    """Fit T by minimizing multiclass NLL, applying it to the whole distribution.

    This is the form to use wherever a distribution exists. Scaling the entire
    distribution is what the correction actually is; scaling only the top
    probability is an approximation to it.
    """
    log_probs, gold_positions = _stack(distributions, gold_labels)
    return _minimize(lambda t: _nll_multiclass(log_probs, gold_positions, t))


def fit_temperature_binary(probabilities: Sequence[float], correct: Sequence[bool]) -> float:
    """Fit T on the top-label probability alone, for adapters reporting no distribution."""
    values = np.asarray(probabilities, dtype=np.float64)
    outcomes = np.asarray(correct, dtype=np.float64)
    return _minimize(lambda t: _nll_binary(values, outcomes, t))


def bootstrap_temperature_ci(
    fit_multiclass: bool,
    distributions: Sequence[Mapping[str, float]] | None,
    gold_labels: Sequence[str] | None,
    probabilities: Sequence[float] | None,
    correct: Sequence[bool] | None,
    n_boot: int = DEFAULT_N_BOOT_CI,
    seed: int = 0,
) -> tuple[float, float]:
    """A 95 percent interval for the fitted temperature, by resampling the fit rows.

    An interval spanning 1.0 means the data do not establish that any
    recalibration is needed.
    """
    rng = np.random.default_rng(seed)
    if fit_multiclass:
        assert distributions is not None and gold_labels is not None
        n = len(distributions)
        draws = [
            fit_temperature_multiclass(
                [distributions[index] for index in rows],
                [gold_labels[index] for index in rows],
            )
            for rows in (rng.integers(0, n, size=n) for _ in range(n_boot))
        ]
    else:
        assert probabilities is not None and correct is not None
        n = len(probabilities)
        draws = [
            fit_temperature_binary(
                [probabilities[index] for index in rows],
                [correct[index] for index in rows],
            )
            for rows in (rng.integers(0, n, size=n) for _ in range(n_boot))
        ]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _metrics(
    series: ProbabilitySeries,
    correct: Sequence[bool],
    distributions: Sequence[Mapping[str, float]] | None,
    gold_labels: Sequence[str] | None,
    n_bins: int,
    binning: Binning,
    min_bin_count: int,
) -> MetricSet:
    try:
        maximum: float | None = mce(series, correct, n_bins, binning, min_bin_count)
    except ValueError:
        maximum = None
    return MetricSet(
        ece=ece(series, correct, n_bins, binning),
        mce=maximum,
        brier=brier(series, correct),
        multiclass_brier=(
            multiclass_brier(list(distributions), list(gold_labels))
            if distributions is not None and gold_labels is not None
            else None
        ),
    )


def recalibrate(
    series: ProbabilitySeries,
    correct: Sequence[bool],
    *,
    distributions: Sequence[Mapping[str, float] | None] | None = None,
    gold_labels: Sequence[str] | None = None,
    split: Split | None = None,
    fit_fraction: float = 0.5,
    seed: int = 0,
    min_eval_rows: int = DEFAULT_MIN_EVAL_ROWS,
    n_bins: int = DEFAULT_N_BINS,
    binning: Binning = DEFAULT_BINNING,
    min_bin_count: int = DEFAULT_MIN_BIN_COUNT,
    n_boot_ci: int = DEFAULT_N_BOOT_CI,
    n_boot_floor: int = 800,
) -> RecalibrationResult:
    """Fit a temperature on one half of the data and report it on the other.

    Uses the multiclass form when ``distributions`` and ``gold_labels`` are
    supplied for every row, and the binary top-label form otherwise. Every metric
    in the result, before and after, is computed on the evaluation rows only, so
    the before and after numbers are comparable to each other and neither has
    seen the fit.
    """
    probabilities = series.require_reportable()
    if len(probabilities) != len(correct):
        raise ValueError(
            f"{len(probabilities)} probabilities against {len(correct)} outcomes; these must match"
        )

    use_multiclass = (
        distributions is not None
        and gold_labels is not None
        and all(distribution is not None for distribution in distributions)
    )
    if use_multiclass:
        assert distributions is not None and gold_labels is not None
        if not (len(distributions) == len(gold_labels) == len(probabilities)):
            raise ValueError("distributions, gold labels, and probabilities must align")

    split = split if split is not None else make_split(len(probabilities), fit_fraction, seed)
    if max(max(split.fit_indices), max(split.eval_indices)) >= len(probabilities):
        raise ValueError("the split indexes rows that do not exist")

    if len(split.eval_indices) < min_eval_rows:
        raise InsufficientDataError(
            f"recalibration needs at least {min_eval_rows} held-out evaluation rows and "
            f"this split has {len(split.eval_indices)}. Fitting a temperature on fewer "
            "rows produces a number whose uncertainty is larger than the correction it "
            "claims to make, and it arrives looking like a measurement. Collect more "
            "labeled rows, or skip recalibration and report raw calibration only."
        )

    fit_rows, eval_rows = split.fit_indices, split.eval_indices

    if use_multiclass:
        assert distributions is not None and gold_labels is not None
        # use_multiclass already established that no entry is None, so positions
        # line up with the probability and outcome columns.
        complete: list[Mapping[str, float]] = [
            distribution for distribution in distributions if distribution is not None
        ]
        temperature = fit_temperature_multiclass(
            [complete[index] for index in fit_rows], [gold_labels[index] for index in fit_rows]
        )
        low, high = bootstrap_temperature_ci(
            True,
            [complete[index] for index in fit_rows],
            [gold_labels[index] for index in fit_rows],
            None,
            None,
            n_boot_ci,
            seed + 1,
        )
        scaled = [apply_temperature(complete[index], temperature) for index in eval_rows]
        after_probabilities = tuple(
            distribution[max(distribution, key=lambda label: distribution[label])]
            for distribution in scaled
        )
        eval_distributions_before: list[Mapping[str, float]] | None = [
            complete[index] for index in eval_rows
        ]
        eval_distributions_after: list[Mapping[str, float]] | None = list(scaled)
        eval_gold: list[str] | None = [gold_labels[index] for index in eval_rows]
        method: Method = "multiclass"
    else:
        temperature = fit_temperature_binary(
            [probabilities[index] for index in fit_rows], [correct[index] for index in fit_rows]
        )
        low, high = bootstrap_temperature_ci(
            False,
            None,
            None,
            [probabilities[index] for index in fit_rows],
            [correct[index] for index in fit_rows],
            n_boot_ci,
            seed + 1,
        )
        after_probabilities = tuple(
            apply_temperature_binary(probabilities[index], temperature) for index in eval_rows
        )
        eval_distributions_before = None
        eval_distributions_after = None
        eval_gold = None
        method = "binary"

    eval_correct = [correct[index] for index in eval_rows]
    before_series = ProbabilitySeries(
        values=tuple(probabilities[index] for index in eval_rows), semantics=series.semantics
    )
    after_series = ProbabilitySeries(values=after_probabilities, semantics=series.semantics)

    return RecalibrationResult(
        method=method,
        temperature=temperature,
        temperature_ci=(low, high),
        justified=not (low <= 1.0 <= high),
        split=split,
        seed=seed,
        before=_metrics(
            before_series,
            eval_correct,
            eval_distributions_before,
            eval_gold,
            n_bins,
            binning,
            min_bin_count,
        ),
        after=_metrics(
            after_series,
            eval_correct,
            eval_distributions_after,
            eval_gold,
            n_bins,
            binning,
            min_bin_count,
        ),
        floor=calibration_floor(
            after_series,
            n_bins=n_bins,
            binning=binning,
            min_bin_count=min_bin_count,
            n_boot=n_boot_floor,
            seed=seed + 2,
        ),
    )
