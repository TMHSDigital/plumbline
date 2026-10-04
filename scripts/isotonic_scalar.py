"""Does a monotone fit beat temperature on a scalar-only column? (issue #8)

METHODOLOGY records that an adapter reporting only its top-line probability
recalibrates worse than one reporting the full distribution, and that the penalty
is a property of the transport rather than of the sample size. The question left
open on that issue is whether a different correction does better on the lone
scalar: isotonic regression needs no distribution and makes no assumption that
the distortion is a temperature, at the price of needing more rows and being able
to overfit.

The mock injects a known temperature into its log probabilities, then each run
is split in half, as :func:`plumbline.metrics.recalibration.recalibrate` does.
Two corrections are fitted on the first half from the top-line probability and
the outcome alone, and judged on the second: the binary temperature
plumbline ships, and isotonic regression by pool-adjacent-violators. The table
prints the floor's mean on the held-out rows, then the held-out ECE after each
correction with its ratio to the floor of the corrected probabilities, averaged
over seeds. A ratio of 1.0 is as calibrated as sampling allows.

    uv run python scripts/isotonic_scalar.py [seeds]

This is a study, not a feature: nothing here changes what the report fits.
"""

from __future__ import annotations

import sys

import numpy as np

from plumbline.adapters.mock import MockAdapter
from plumbline.metrics.calibration import calibration_floor, ece
from plumbline.metrics.recalibration import (
    apply_temperature_binary,
    fit_temperature_binary,
    make_split,
)
from plumbline.runner import execute
from plumbline.types import Case, ProbabilitySeries

LABELS = ("billing", "returns", "shipping", "other")
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CLAMP = 1e-3


def cases(n: int, seed: int) -> list[Case]:
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, len(LABELS), size=n)
    return [
        Case(id=f"r{index}", text=f"row {seed} {index}", labels=LABELS, gold_label=LABELS[g])
        for index, g in enumerate(gold)
    ]


def fit_isotonic(probabilities: np.ndarray, correct: np.ndarray):
    """Pool-adjacent-violators on outcome against probability, as a step function."""
    order = np.argsort(probabilities, kind="stable")
    x, y = probabilities[order], correct[order].astype(np.float64)
    means: list[float] = []
    weights: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for value, outcome in zip(x, y, strict=True):
        means.append(outcome)
        weights.append(1.0)
        lows.append(value)
        highs.append(value)
        while len(means) > 1 and means[-2] > means[-1]:
            total = weights[-2] + weights[-1]
            means[-2] = (means[-2] * weights[-2] + means[-1] * weights[-1]) / total
            weights[-2] = total
            highs[-2] = highs[-1]
            for stack in (means, weights, lows, highs):
                stack.pop()
    knots_x = np.array(
        [point for low, high in zip(lows, highs, strict=True) for point in (low, high)]
    )
    knots_y = np.repeat(np.array(means), 2)

    def apply(values: np.ndarray) -> np.ndarray:
        return np.clip(np.interp(values, knots_x, knots_y), CLAMP, 1.0 - CLAMP)

    return apply


def one_run(n: int, temperature: float, seed: int) -> dict[str, tuple[float, float]]:
    rows = cases(n, seed)
    adapter = MockAdapter(
        {case.text: case.gold_label for case in rows},
        accuracy=0.75,
        calibration_temperature=temperature,
        seed=seed,
    )
    result = execute.run(adapter, rows, workers=1)
    kept = [
        (p, bool(ok))
        for p, ok in zip(result.probabilities().values, result.outcomes, strict=True)
        if p is not None
    ]
    probabilities = np.array([p for p, _ in kept])
    correct = np.array([ok for _, ok in kept])
    split = make_split(len(kept), seed=seed)
    fit, held = list(split.fit_indices), list(split.eval_indices)

    t = fit_temperature_binary(probabilities[fit].tolist(), correct[fit].tolist())
    isotonic = fit_isotonic(probabilities[fit], correct[fit])
    corrected = {
        "raw": probabilities[held],
        "temperature": np.array([apply_temperature_binary(p, t) for p in probabilities[held]]),
        "isotonic": isotonic(probabilities[held]),
    }
    figures = {}
    for name, values in corrected.items():
        series = ProbabilitySeries(values=tuple(values.tolist()), semantics="calibrated_claim")
        floor = calibration_floor(series, n_boot=300, seed=seed)["ece"].mean
        figures[name] = (ece(series, correct[held].tolist()), floor)
    return figures


def mean_ece(runs: list[dict[str, tuple[float, float]]], name: str) -> float:
    return float(np.mean([run[name][0] for run in runs]))


def mean_ratio(runs: list[dict[str, tuple[float, float]]], name: str) -> float:
    return float(np.mean([run[name][0] / run[name][1] for run in runs]))


print(f"{'rows':>6} {'injected':>9} {'floor':>7} {'raw':>8} {'temperature':>17} {'isotonic':>17}")
for temperature, label in ((0.5, "T = 0.5"), (2.0, "T = 2.0")):
    for n in (200, 500, 2000, 5000, 20000):
        runs = [one_run(n, temperature, seed) for seed in range(SEEDS)]

        floor = float(np.mean([run["raw"][1] for run in runs]))
        print(
            f"{n:>6} {label:>9} {floor:>7.4f} {mean_ece(runs, 'raw'):>8.4f} "
            f"{mean_ece(runs, 'temperature'):>8.4f} ({mean_ratio(runs, 'temperature'):>5.2f}x) "
            f"{mean_ece(runs, 'isotonic'):>8.4f} ({mean_ratio(runs, 'isotonic'):>5.2f}x)",
            flush=True,
        )
