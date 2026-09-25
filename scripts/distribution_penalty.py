"""How the penalty for reporting no distribution changes with row count (issue #8).

METHODOLOGY measures that an adapter reporting only its top-line probability
recalibrates worse than one reporting the full distribution, even when the
miscalibration is a pure temperature. This asks whether that is a small-sample
artifact that vanishes with more rows, or a property of the answer shape.

The mock injects a known temperature into its log probabilities. Each run is
recalibrated twice on the same rows: once with the distributions (the
multiclass form) and once with the top-line probability alone (the binary
form). It prints the floor's mean on the held-out rows, then post-scaling ECE
for each form with its ratio to that floor, averaged over seeds; a ratio of
1.0 is as calibrated as sampling allows.

    uv run python scripts/distribution_penalty.py [seeds]

METHODOLOGY quotes the table this prints at 5 seeds.
"""

from __future__ import annotations

import sys

import numpy as np

from plumbline.adapters.mock import MockAdapter
from plumbline.metrics.recalibration import recalibrate
from plumbline.runner import execute
from plumbline.types import Case, ProbabilitySeries

LABELS = ("billing", "returns", "shipping", "other")
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def cases(n: int, seed: int) -> list[Case]:
    rng = np.random.default_rng(seed)
    gold = rng.integers(0, len(LABELS), size=n)
    return [
        Case(id=f"r{index}", text=f"row {seed} {index}", labels=LABELS, gold_label=LABELS[g])
        for index, g in enumerate(gold)
    ]


def scaled(n: int, temperature: float, seed: int, multiclass: bool) -> tuple[float, float]:
    rows = cases(n, seed)
    adapter = MockAdapter(
        {case.text: case.gold_label for case in rows},
        accuracy=0.75,
        calibration_temperature=temperature,
        seed=seed,
    )
    result = execute.run(adapter, rows, workers=1)
    fit = recalibrate(
        ProbabilitySeries(
            values=tuple(p for p in result.probabilities().values if p is not None),
            semantics="calibrated_claim",
        ),
        result.outcomes,
        distributions=result.distributions() if multiclass else None,
        gold_labels=result.gold_labels() if multiclass else None,
        seed=seed,
        n_boot_ci=50,
        n_boot_floor=300,
    )
    return fit.after.ece, fit.floor["ece"].mean


print(f"{'rows':>6} {'injected':>9} {'floor':>7} {'multiclass':>17} {'binary':>17}")
for temperature, name in ((0.5, "T = 0.5"), (2.0, "T = 2.0")):
    for n in (500, 2000, 5000, 20000):
        multi = np.array([scaled(n, temperature, seed, True) for seed in range(SEEDS)])
        binary = np.array([scaled(n, temperature, seed, False) for seed in range(SEEDS)])
        floor = multi[:, 1].mean()
        print(
            f"{n:>6} {name:>9} {floor:>7.4f} "
            f"{multi[:, 0].mean():>8.4f} ({(multi[:, 0] / multi[:, 1]).mean():>5.2f}x) "
            f"{binary[:, 0].mean():>8.4f} ({(binary[:, 0] / binary[:, 1]).mean():>5.2f}x)",
            flush=True,
        )
