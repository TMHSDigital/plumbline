"""Run the whole pipeline over the public fixture, with a mock in the model's chair.

    python examples/smoke_public_dataset.py

Loader, runner, metrics, report, artifact, end to end on a file somebody else
wrote. It makes no network call and spends nothing, and the numbers it prints
are meaningless: a seeded mock is answering, so its accuracy and its ECE are
properties of the mock's configuration and nothing else. What it proves is that
the pieces compose on real shapes (structured states, options that are yes/no
on some rows and five-way on others, gold labels written as numbers, ordinal
rows that v0.1 will not score) before a live run turns mistakes into money.

The same run is available as ``plumbline run --format jevbench``. This file stays
because it is the shortest readable path through the library, and because
nothing in plumbline imports it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from plumbline.adapters.mock import MockAdapter
from plumbline.config import DEFAULT_PRICING_TABLE
from plumbline.datasets import LoadReport, load_jevbench
from plumbline.report import ReportOptions, render
from plumbline.runner import execute

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO / "datasets/public/jevbench-hard.jsonl"
DEFAULT_RESULTS = REPO / "results"

HEADER = (
    "This is a smoke test, not a result. A seeded mock answered every case, so\n"
    "the accuracy and calibration below describe the mock's configuration and\n"
    "say nothing about any model. What is being checked is that the loader, the\n"
    "runner, the metrics, the report, and the artifact compose on a real file.\n"
)


@dataclass(frozen=True)
class Smoked:
    """Everything the run produced, so a test can assert on the same object."""

    load: LoadReport
    result: execute.RunResult
    summary: str
    artifact: Path


def smoke(
    dataset: Path,
    results_dir: Path,
    *,
    seed: int = 7,
    accuracy: float = 0.8,
    n_boot: int = 2000,
) -> Smoked:
    """Load, run, measure, write, and render one report."""
    load = load_jevbench(dataset)
    # Scoreable only: the ordinal score rows are loaded and marked, and v0.1
    # will not turn them into numbers, so they are never sent either.
    cases = list(load.scoreable)

    adapter = MockAdapter(
        {case.text: case.gold_label for case in cases},
        seed=seed,
        accuracy=accuracy,
    )
    result = execute.run(
        adapter,
        cases,
        pricing_table=DEFAULT_PRICING_TABLE,
        workers=4,
        extra_config={"dataset": str(dataset), "note": "smoke test, mock adapter"},
    )
    artifact = result.write(results_dir)

    document = render([result], load=load, options=ReportOptions(n_boot=n_boot))
    return Smoked(
        load=load,
        result=result,
        summary=f"{HEADER}\n{document}\nArtifact: {artifact}\n",
        artifact=artifact,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--accuracy", type=float, default=0.8)
    parser.add_argument("--boot", type=int, default=2000, dest="n_boot")
    args = parser.parse_args(argv)

    smoked = smoke(
        args.dataset,
        args.results,
        seed=args.seed,
        accuracy=args.accuracy,
        n_boot=args.n_boot,
    )
    print(smoked.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
