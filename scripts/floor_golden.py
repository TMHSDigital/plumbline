"""Export, or check, the golden values the site's floor calculator is held to.

The site reimplements :func:`plumbline.metrics.calibration.synthetic_floor` in
JavaScript (``site/floor.js``). If the two disagree, the site misreports what the
tool says, which is worse than having no site. This script writes the Python's
answers for a spread of configurations to ``site/floor-golden.json``, and
``scripts/check_floor_parity.mjs`` fails CI when the JavaScript disagrees.

Run with no arguments to rewrite the fixture. Run with ``--check`` to confirm the
committed fixture still matches what the Python produces today, so a change to
the floor cannot leave the JavaScript agreeing with a stale answer.

    uv run python scripts/floor_golden.py
    uv run python scripts/floor_golden.py --check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from plumbline.metrics.calibration import CalibrationFigure, synthetic_floor

FIXTURE = Path(__file__).resolve().parent.parent / "site" / "floor-golden.json"

#: Absolute tolerance on every floor value, in both directions of the check.
#: The JavaScript reproduces numpy's random stream draw for draw, so the only
#: differences left are summation order (numpy sums pairwise) and last-bit
#: differences between C's and JavaScript's pow, log, and exp: around 1e-16.
#: One desynchronised draw moves the result by around 1e-3. 1e-9 sits far from
#: both, and five orders below the fourth decimal the page prints.
TOLERANCE = 1e-9

#: (rows, bins, accuracy). Covers the ordinary range, the 105-row example, very
#: small n, more bins than rows, accuracies whose gamma shape falls below 1
#: (numpy takes a different sampling path there), and one large n.
CASES: list[tuple[int, int, float]] = [
    (1, 10, 0.8),
    (5, 10, 0.8),
    (5, 20, 0.5),
    (10, 50, 0.8),
    (20, 10, 0.9),
    (30, 5, 0.7),
    (50, 15, 0.8),
    (100, 10, 0.5),
    (105, 10, 0.7714),
    (105, 10, 0.8),
    (200, 20, 0.6),
    (250, 100, 0.8),
    (300, 10, 0.95),
    (500, 10, 0.8),
    (500, 15, 0.9),
    (500, 5, 0.3),
    (750, 10, 0.99),
    (1000, 10, 0.75),
    (1000, 20, 0.85),
    (1000, 50, 0.95),
    (2000, 10, 0.8),
    (2000, 15, 0.1),
    (4000, 10, 0.8),
    (5000, 20, 0.97),
    (10000, 10, 0.8),
    (10000, 30, 0.9),
    (20000, 10, 0.65),
]

#: Measured ECE values each case's verdict sentence is exported for, so the
#: wording, the formatting, and the side of the line a value falls on are all
#: checked, not only the floor.
MEASURED = [0.00125, 0.005, 0.02, 0.03125, 0.05, 0.074, 0.1, 0.2]


def _formatting_cases() -> dict[str, str]:
    """Values on and beside a rounding tie, with what the report prints for them.

    The report formats with ``:.4f``, which rounds the exact binary value, half
    to even. A value that looks like a tie in decimal, such as 0.00125, is not
    one in binary and rounds by which side of the tie it really lies on; the
    only exact ties are odd multiples of 1/32. The measured values above held
    none of either, which is how a formatter that rounded every near tie to
    even passed this check.
    """
    values = {j / 32 for j in range(1, 64, 2)}
    for k in range(0, 2000, 7):
        tie = (k + 0.5) / 10000
        values.update({tie, math.nextafter(tie, 0.0), math.nextafter(tie, 1.0)})
    return {repr(value): f"{value:.4f}" for value in sorted(values)}


def _pcg64_state(seed: int) -> dict[str, str]:
    state = np.random.default_rng(seed).bit_generator.state["state"]
    return {"state": str(state["state"]), "inc": str(state["inc"])}


def build() -> dict[str, Any]:
    cases = []
    for n, n_bins, accuracy in CASES:
        band = synthetic_floor(n, n_bins=n_bins, accuracy=accuracy)["ece"]
        statements = {
            repr(measured): CalibrationFigure(
                metric="ece",
                value=measured,
                n=band.n,
                floor=band,
                n_bins=n_bins,
                binning="equal_width",
            ).statement()
            for measured in MEASURED
        }
        cases.append(
            {
                "n": n,
                "n_bins": n_bins,
                "accuracy": accuracy,
                "mean": band.mean,
                "p95": band.p95,
                "statements": statements,
            }
        )
    return {
        "about": (
            "Golden values from plumbline.metrics.calibration.synthetic_floor, "
            "which site/floor.js must reproduce. Written by scripts/floor_golden.py."
        ),
        "function": "synthetic_floor(n, n_bins, accuracy), all other arguments default",
        "defaults": {"binning": "equal_width", "n_boot": 2000, "concentration": 6.0, "seed": 0},
        "numpy": np.__version__,
        "tolerance": TOLERANCE,
        "pcg64_initial_state": {"0": _pcg64_state(0), "1": _pcg64_state(1)},
        "cases": cases,
        "fixed4": _formatting_cases(),
    }


def _check(fresh: dict[str, Any]) -> list[str]:
    if not FIXTURE.is_file():
        return [f"{FIXTURE} does not exist; run this script without --check"]
    committed = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problems = []
    if committed["pcg64_initial_state"] != fresh["pcg64_initial_state"]:
        problems.append("numpy's seeded PCG64 states differ from the fixture")
    if committed.get("fixed4") != fresh["fixed4"]:
        problems.append("the :.4f formatting cases differ from the fixture")
    if len(committed["cases"]) != len(fresh["cases"]):
        problems.append("the fixture holds a different set of cases; regenerate it")
        return problems
    for old, new in zip(committed["cases"], fresh["cases"], strict=True):
        label = f"n={new['n']} bins={new['n_bins']} accuracy={new['accuracy']}"
        if (old["n"], old["n_bins"], old["accuracy"]) != (
            new["n"],
            new["n_bins"],
            new["accuracy"],
        ):
            problems.append(f"{label}: case list changed; regenerate the fixture")
            continue
        for key in ("mean", "p95"):
            if abs(old[key] - new[key]) > TOLERANCE:
                problems.append(f"{label}: {key} is {new[key]!r} now, fixture says {old[key]!r}")
        if old["statements"] != new["statements"]:
            problems.append(f"{label}: verdict wording changed")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed fixture no longer matches the Python",
    )
    args = parser.parse_args()

    fresh = build()
    if args.check:
        problems = _check(fresh)
        for problem in problems:
            print(problem, file=sys.stderr)
        if problems:
            print(
                "site/floor-golden.json is stale. Regenerate it with "
                "`uv run python scripts/floor_golden.py`, then make site/floor.js agree.",
                file=sys.stderr,
            )
            return 1
        print(f"fixture current: {len(fresh['cases'])} cases match the Python")
        return 0

    FIXTURE.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(fresh['cases'])} cases to {FIXTURE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
