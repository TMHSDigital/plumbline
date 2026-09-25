"""Does a two-decimal grid raise the ECE floor? (issue #9)

A perfectly calibrated model has true probabilities p; the vendor sends
q = round(p, 2). plumbline reads ECE(q, outcomes) against calibration_floor(q),
the floor for a model calibrated at q. If rounding costs calibration the floor
does not model, a calibrated model reported on the grid would clear the floor's
95th percentile more often than the nominal 5 percent.

    uv run python scripts/quantization_floor.py [trials]

METHODOLOGY quotes the table this prints at 200 trials. It takes several
minutes, since every trial bootstraps its own floor.
"""

import sys

import numpy as np

from plumbline.metrics.calibration import calibration_floor, ece
from plumbline.types import ProbabilitySeries

TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(20260925)
print(f"{'n':>6} {'bins':>4} {'grid':>5} {'mean ECE':>9} {'floor mean':>10} {'over p95':>9}")
for n_bins, grid in [(10, 0.01), (10, 0.05), (20, 0.01)]:
    for n in (40, 105, 500, 2000, 10000):
        measured, means, over = [], [], 0
        for trial in range(TRIALS):
            p = np.clip(rng.beta(0.8 * 6, 0.2 * 6, size=n), 0.0, 1.0)
            outcomes = rng.random(n) < p
            q = np.clip(np.round(p / grid) * grid, 0.0, 1.0)
            series = ProbabilitySeries(
                values=tuple(float(v) for v in q), semantics="calibrated_claim"
            )
            value = ece(series, list(outcomes), n_bins=n_bins)
            band = calibration_floor(series, n_bins=n_bins, n_boot=400, seed=trial)["ece"]
            measured.append(value)
            means.append(band.mean)
            over += value > band.p95
        print(
            f"{n:>6} {n_bins:>4} {grid:>5} {np.mean(measured):>9.4f} {np.mean(means):>10.4f} "
            f"{over / TRIALS:>9.1%}",
            flush=True,
        )
