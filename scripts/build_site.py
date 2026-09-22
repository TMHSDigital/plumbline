"""Assemble the site into one directory, ready to deploy. Runs in CI, not by hand.

Copies ``site/`` into the output directory, then writes what the page derives
its worked example from:

``example-run.json``
    The 105 predicted probabilities and outcomes behind ``docs/example-report.md``,
    produced by re-running the command that report records, together with the
    figures the report prints. The page recomputes the ECE and its floor from the
    rows, in JavaScript, and ``scripts/check_floor_parity.mjs`` fails the build if
    what it derives differs from the report by a single character.

The example is regenerated from the **mock** adapter and nothing else. The
command is read from the report, and anything other than ``--adapter mock`` is
refused, so a vendor run can never be published through this path. The run
writes into a fresh temporary directory and only the artifact written there is
read; no existing results directory or artifact is ever opened.

    uv run python scripts/build_site.py --out _site
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from plumbline.cli import app
from plumbline.metrics.calibration import synthetic_floor
from plumbline.runner.execute import RunResult

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
EXAMPLE_REPORT = ROOT / "docs" / "example-report.md"

#: The only adapter whose output this script will publish.
ALLOWED_ADAPTER = "mock"

ECE_LINE = re.compile(
    r"^- (ECE (?P<ece>\d\.\d{4}) over (?P<n>\d+) rows \((?P<bins>\d+) equal width bins\), "
    r"against a calibrated-model floor of (?P<mean>\d\.\d{4}) "
    r"\(95th percentile (?P<p95>\d\.\d{4})\): .+)$"
)
ACCURACY_LINE = re.compile(r"^- (Accuracy (?P<accuracy>\d\.\d{4}) over (?P<n>\d+) rows, .+)$")
COMMAND_ROW = re.compile(r"^\| Command \| `(?P<command>plumbline run [^`]+)` \|$")


class BuildError(Exception):
    """A reason the site must not deploy, stated for whoever reads the CI log."""


def _report_lines(text: str) -> list[str]:
    """The lines of the tool's own output: everything after the report heading."""
    lines = text.splitlines()
    try:
        start = lines.index("# plumbline report")
    except ValueError as missing:
        raise BuildError("no '# plumbline report' heading in the example report") from missing
    return lines[start:]


def _single(pattern: re.Pattern[str], lines: list[str], what: str) -> re.Match[str]:
    found = [match for line in lines if (match := pattern.match(line))]
    if len(found) != 1:
        raise BuildError(f"expected exactly one {what} line in the report, found {len(found)}")
    return found[0]


def _example_arguments(command: str, results: Path, report: Path) -> list[str]:
    """The report's recorded command, pointed at a scratch directory."""
    words = shlex.split(command)
    if words[:2] != ["plumbline", "run"]:
        raise BuildError(f"the example command is not a plumbline run: {command}")
    arguments = words[1:]  # the subcommand stays: the app has several
    at = arguments.index("--adapter") if "--adapter" in arguments else -1
    if at < 0 or at + 1 >= len(arguments):
        raise BuildError("the example command does not name its adapter")
    adapter = arguments[at + 1]
    if adapter != ALLOWED_ADAPTER:
        raise BuildError(
            f"the example command uses the {adapter!r} adapter. Only {ALLOWED_ADAPTER!r} "
            "output is published by this build, so the site never carries a vendor's rows."
        )
    for flag in ("--report", "--results"):
        if flag in arguments:
            at = arguments.index(flag)
            del arguments[at : at + 2]
    return [*arguments, "--results", str(results), "--report", str(report)]


def build_example() -> dict[str, Any]:
    document = EXAMPLE_REPORT.read_text(encoding="utf-8")
    command = _single(COMMAND_ROW, document.splitlines(), "Command")["command"]
    printed = _report_lines(document)
    ece_match = _single(ECE_LINE, printed, "ECE")
    accuracy_match = _single(ACCURACY_LINE, printed, "Accuracy")

    with tempfile.TemporaryDirectory(prefix="plumbline-site-example-") as scratch:
        results = Path(scratch) / "results"
        regenerated_report = Path(scratch) / "report.md"
        arguments = _example_arguments(command, results, regenerated_report)

        # The dataset path in the command is relative to the repository root.
        here = Path.cwd()
        os.chdir(ROOT)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                app(arguments, standalone_mode=False)
        finally:
            os.chdir(here)

        artifacts = list(results.glob("*.json"))
        if len(artifacts) != 1:
            raise BuildError(f"expected one artifact from the example run, found {len(artifacts)}")
        run = RunResult.read(artifacts[0])
        regenerated = _report_lines(regenerated_report.read_text(encoding="utf-8"))

    if run.adapter_name != ALLOWED_ADAPTER:
        raise BuildError(f"the example artifact came from {run.adapter_name!r}, not the mock")

    fresh_line = _single(ECE_LINE, regenerated, "ECE").group(1)
    if fresh_line != ece_match.group(1):
        raise BuildError(
            "docs/example-report.md no longer matches what its own command produces.\n"
            f"  report says: {ece_match.group(1)}\n"
            f"  run gives:   {fresh_line}\n"
            "Regenerate the example report before deploying."
        )

    probabilities = list(run.probabilities().values)
    outcomes = run.outcomes
    n_bins = int(ece_match["bins"])
    accuracy = sum(outcomes) / len(outcomes)
    summary = synthetic_floor(len(probabilities), n_bins=n_bins, accuracy=accuracy)["ece"]

    return {
        "about": (
            "The rows behind docs/example-report.md, regenerated at deploy time from the "
            "mock adapter by the command that report records. Written by "
            "scripts/build_site.py."
        ),
        "command": command,
        "adapter": run.adapter_name,
        "model_reported": run.model_reported,
        "n_bins": n_bins,
        "probabilities": probabilities,
        "correct": [1 if outcome else 0 for outcome in outcomes],
        "report": {
            "ece_line": ece_match.group(1),
            "accuracy_line": accuracy_match.group(1),
            "ece": float(ece_match["ece"]),
            "n": int(ece_match["n"]),
            "floor_mean": float(ece_match["mean"]),
            "floor_p95": float(ece_match["p95"]),
            "accuracy": float(accuracy_match["accuracy"]),
        },
        "python": {
            "accuracy": accuracy,
            "summary_floor": {"mean": summary.mean, "p95": summary.p95},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "_site", help="output directory")
    args = parser.parse_args()
    out: Path = args.out.resolve()
    # The output directory is deleted and rebuilt, so it must be a directory of
    # its own and never the repository, the site sources, or anything above them.
    if out in (ROOT, SITE) or out in ROOT.parents or SITE in out.parents:
        print(f"site build refused: {out} is not a safe output directory", file=sys.stderr)
        return 1

    try:
        example = build_example()
    except BuildError as problem:
        print(f"site build refused: {problem}", file=sys.stderr)
        return 1

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SITE, out)
    (out / "example-run.json").write_text(json.dumps(example, indent=1) + "\n", encoding="utf-8")
    print(f"site assembled in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
