"""When a temperature per predicted label helps, and what it costs (issue #4).

Two cases on the seeded mock (accuracy 0.75, four options), at several sizes,
over seeds. Every figure is on the held-out half, as the report's are.

- A per-label bias: the model is sharpened (temperature 0.45) whenever it says
  "billing" and honest otherwise. One temperature cannot reach this; the
  question is how many rows a label needs before its own temperature can.
- A global skew: every answer sharpened by the same temperature (0.5). Here one
  temperature is the right shape, so any held-out loss from fitting four is the
  price of the extra parameters, which is what overfitting means here.

Each cell is post-scaling ECE over the floor's 95th percentile, averaged over
seeds (at or below 1.0 is inside the floor), with how often each fit's verdict
would ship a correction.

    uv run python scripts/per_label_study.py [seeds]

METHODOLOGY quotes the tables this prints at 5 seeds.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np

from plumbline.adapters.mock import MockAdapter
from plumbline.metrics.recalibration import recalibrate, recalibrate_per_label
from plumbline.runner import execute
from plumbline.types import Case, Prediction, ProbabilitySeries, apply_temperature

LABELS = ("billing", "returns", "shipping", "other")
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SIZES = (500, 1000, 2000, 4000, 8000)
BOOT = {"n_boot_ci": 100, "n_boot_floor": 300}


def rows(n: int, seed: int) -> list[Case]:
    rng = np.random.default_rng(seed)
    return [
        Case(id=f"r{i}", text=f"row {seed} {i}", labels=LABELS, gold_label=LABELS[g])
        for i, g in enumerate(rng.integers(0, len(LABELS), size=n))
    ]


def sharpen(prediction: Prediction, temperature: float) -> Prediction:
    assert prediction.distribution is not None
    scaled = apply_temperature(prediction.distribution, temperature)
    return dataclasses.replace(
        prediction, distribution=scaled, prob_selected=scaled[prediction.label]
    )


def measured(n: int, seed: int, case: str) -> tuple[list[Prediction], list[bool], list[str]]:
    cases = rows(n, seed)
    adapter = MockAdapter(
        {c.text: c.gold_label for c in cases},
        accuracy=0.75,
        calibration_temperature=0.5 if case == "global" else 1.0,
        seed=seed,
    )
    result = execute.run(adapter, cases, workers=1)
    predictions = [r.prediction for r in result.records if r.prediction is not None]
    if case == "per-label":
        predictions = [sharpen(p, 0.45) if p.label == "billing" else p for p in predictions]
    gold = [r.gold_label for r in result.records if r.prediction is not None]
    return predictions, [p.label == g for p, g in zip(predictions, gold, strict=True)], gold


def study(n: int, seed: int, case: str) -> tuple[float, float, bool, bool, int]:
    predictions, correct, gold = measured(n, seed, case)
    series = ProbabilitySeries(
        values=tuple(p.prob_selected or 0.0 for p in predictions), semantics="calibrated_claim"
    )
    common = {"distributions": [p.distribution for p in predictions], "gold_labels": gold}
    one = recalibrate(series, correct, seed=seed, **common, **BOOT)
    many = recalibrate_per_label(
        series,
        correct,
        predicted_labels=[p.label for p in predictions],
        seed=seed,
        **common,
        **BOOT,
    )
    smallest = min(entry.n_fit for entry in many.labels)
    return (
        one.after.ece / one.floor["ece"].p95,
        many.after.ece / many.floor["ece"].p95,
        one.recommendation != "refused",
        many.recommendation != "refused",
        smallest,
    )


for case in ("per-label", "global"):
    print(f"\n{case} skew")
    header = ("rows", 6), ("label rows", 10), ("one T", 7), ("per label", 10)
    print(" ".join(f"{name:>{width}}" for name, width in header), f"{'ships 1':>8} {'ships K':>8}")
    for n in SIZES:
        results = [study(n, seed, case) for seed in range(SEEDS)]
        one, many, ship_one, ship_many, smallest = (
            np.array(column) for column in zip(*results, strict=True)
        )
        print(
            f"{n:>6} {int(smallest.min()):>10} {one.mean():>6.2f}x {many.mean():>9.2f}x "
            f"{ship_one.mean():>8.0%} {ship_many.mean():>8.0%}",
            flush=True,
        )
