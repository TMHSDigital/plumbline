"""Assemble the site into one directory, ready to deploy. Runs in CI, not by hand.

Copies ``site/`` into the output directory, then writes what the page derives
its worked example from:

``example-run.json``
    The 105 predicted probabilities and outcomes behind ``docs/example-report.md``,
    produced by re-running the command that report records, together with the
    figures the report prints. The page recomputes the ECE and its floor from the
    rows, in JavaScript, and ``scripts/check_floor_parity.mjs`` fails the build if
    what it derives differs from the report by a single character.

``docs/``
    The repository's markdown docs as they are at the commit being built: each
    one copied verbatim as ``<slug>.md`` and rendered to ``<slug>.html`` by
    ``scripts/render_docs.mjs``, with a line naming the commit and build time it
    came from. Nothing is written by hand and nothing is fetched at runtime, so
    the pages cannot drift from ``main``: a deploy re-renders them.

The example is regenerated from the **mock** adapter and nothing else. The
command is read from the report, and anything other than ``--adapter mock`` is
refused, so a vendor run can never be published through this path. The run
writes into a fresh temporary directory and only the artifact written there is
read; no existing results directory or artifact is ever opened.

    uv run python scripts/build_site.py --out _site

Rendering needs ``node`` on the PATH (the runner's own; nothing is installed).
"""

from __future__ import annotations

import argparse
import contextlib
import html
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plumbline.cli import app
from plumbline.metrics.calibration import synthetic_floor
from plumbline.runner.execute import RunResult

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
EXAMPLE_REPORT = ROOT / "docs" / "example-report.md"
RENDERER = ROOT / "scripts" / "render_docs.mjs"
REPO = "TMHSDigital/plumbline"

#: The only adapter whose output this script will publish.
ALLOWED_ADAPTER = "mock"


@dataclass(frozen=True)
class Doc:
    source: str  # path in the repository
    slug: str  # docs/<slug>.html and docs/<slug>.md on the site
    label: str  # its name in the navigation
    blurb: str  # one line on the docs index


#: The docs the site hosts, in navigation order. A relative link between two of
#: these becomes a link between their pages; a link to any other file goes to
#: GitHub at the commit being built.
DOCS = (
    Doc(
        "README.md",
        "readme",
        "README",
        "What plumbline is, how to run it, and what it does not do.",
    ),
    Doc(
        "METHODOLOGY.md",
        "methodology",
        "Methodology",
        "How each figure is computed, and the floor each is reported against.",
    ),
    Doc(
        "docs/example-report.md",
        "example-report",
        "Example report",
        "Unedited output of one mock run: the report the explainer derives its example from.",
    ),
    Doc("docs/PLAN.md", "plan", "Plan", "What is built, what is deliberately not, and why."),
    Doc("CHANGELOG.md", "changelog", "Changelog", "Notable changes per release."),
    Doc(
        "CONTRIBUTING.md",
        "contributing",
        "Contributing",
        "How to work on the code and what CI checks.",
    ),
    Doc(
        "SECURITY.md",
        "security",
        "Security",
        "How to report a vulnerability, and what the scanners cover.",
    ),
    Doc(
        "datasets/public/README.md",
        "dataset",
        "Dataset",
        "The vendored JevBench fixture: where it comes from and its license.",
    ),
)

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


def _git(*arguments: str) -> str:
    try:
        done = subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as failed:
        raise BuildError(f"git {' '.join(arguments)} failed: {failed}") from failed
    return done.stdout


@dataclass(frozen=True)
class Provenance:
    sha: str
    built: datetime
    #: Hosted docs whose working copy differs from the commit. Always empty in
    #: CI; a local build says so on the page rather than claiming the commit.
    modified: frozenset[str]

    def line(self, doc: Doc) -> str:
        short = self.sha[:7]
        at = self.built.strftime("%Y-%m-%d %H:%M UTC")
        source = f"https://github.com/{REPO}/blob/{self.sha}/{doc.source}"
        commit = f"https://github.com/{REPO}/commit/{self.sha}"
        dirty = (
            " <strong>plus uncommitted local changes</strong>"
            if doc.source in self.modified
            else ""
        )
        return (
            f'Rendered from <a href="{source}"><code>{html.escape(doc.source)}</code></a> '
            f'at commit <a href="{commit}"><code>{short}</code></a>{dirty}, built {at}. '
            f'<a href="{doc.slug}.md">Markdown source</a>.'
        )


def provenance() -> Provenance:
    sha = _git("rev-parse", "HEAD").strip()
    status = _git("status", "--porcelain", "--", *(doc.source for doc in DOCS))
    modified = frozenset(line[3:].strip() for line in status.splitlines())
    # SOURCE_DATE_EPOCH makes the build reproducible when it is set.
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    built = datetime.fromtimestamp(int(epoch), UTC) if epoch else datetime.now(UTC)
    return Provenance(sha, built, modified)


def render_docs(prov: Provenance) -> dict[str, dict[str, str]]:
    """Markdown to HTML fragments, by the vendored renderer under the runner's Node."""
    job = {
        "repo": REPO,
        "sha": prov.sha,
        "tracked": _git("ls-files").splitlines(),
        "docs": [
            {
                "source": doc.source,
                "slug": doc.slug,
                "markdown": (ROOT / doc.source).read_text(encoding="utf-8"),
            }
            for doc in DOCS
        ],
    }
    try:
        done = subprocess.run(
            ["node", str(RENDERER)],
            input=json.dumps(job),
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError as missing:
        raise BuildError(f"could not run node to render the docs: {missing}") from missing
    if done.returncode != 0:
        raise BuildError(f"the docs did not render:\n{done.stderr.strip()}")
    rendered: dict[str, dict[str, str]] = json.loads(done.stdout)
    return rendered


def _nav(current: str | None, up: str) -> str:
    items = [f'<li><a href="{up}">The ECE floor</a></li>']
    for doc in DOCS:
        mark = ' aria-current="page"' if doc.slug == current else ""
        items.append(f'<li><a href="{doc.slug}.html"{mark}>{html.escape(doc.label)}</a></li>')
    joined = "\n    ".join(items)
    return f'<nav aria-label="Documentation">\n  <ul>\n    {joined}\n  </ul>\n</nav>'


# The explainer's plumb-bob icon, inline so the page makes no request for it.
ICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E"
    "%3Cpath d='M8 1v9' stroke='%23555' stroke-width='1.5'/%3E"
    "%3Cpath d='M5 10h6l-3 5z' fill='%23555'/%3E%3C/svg%3E"
)

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} | plumbline</title>
<meta name="description" content="{description}">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="{icon}">
<link rel="stylesheet" href="../base.css">
<link rel="stylesheet" href="../docs.css">
</head>
<body>
<a class="skip" href="#doc">Skip to the document</a>
<header>
<p class="kicker"><a href="../">plumbline</a></p>
{nav}
</header>
<main id="doc">
{body}
</main>
<footer>
<p>Every page here is rendered from the repository at deploy time; none is edited by hand.
<a href="https://github.com/{repo}">Source on GitHub</a>, Apache-2.0 licensed.
No analytics, no trackers, no external requests.</p>
</footer>
</body>
</html>
"""


def write_docs(out: Path, prov: Provenance, rendered: dict[str, dict[str, str]]) -> None:
    docs = out / "docs"
    docs.mkdir()
    for doc in DOCS:
        shutil.copyfile(ROOT / doc.source, docs / f"{doc.slug}.md")
        body = (
            f'<p class="provenance">{prov.line(doc)}</p>\n'
            f'<article class="prose">\n{rendered[doc.slug]["html"]}</article>'
        )
        page = PAGE.format(
            title=html.escape(doc.label),
            description=html.escape(doc.blurb, quote=True),
            nav=_nav(doc.slug, "../"),
            body=body,
            repo=REPO,
            icon=ICON,
        )
        (docs / f"{doc.slug}.html").write_text(page, encoding="utf-8")

    listing = "\n".join(
        f'  <li><a href="{doc.slug}.html">{html.escape(doc.label)}</a> '
        f'<span class="muted small">{html.escape(doc.source)}</span><br>'
        f"{html.escape(doc.blurb)}</li>"
        for doc in DOCS
    )
    at = prov.built.strftime("%Y-%m-%d %H:%M UTC")
    index = (
        '<article class="prose">\n<h1>Documentation</h1>\n'
        "<p>The repository's own markdown, rendered from commit "
        f'<a href="https://github.com/{REPO}/commit/{prov.sha}"><code>{prov.sha[:7]}</code></a> '
        f"at {at}.</p>\n"
        f'<ul class="doc-list">\n{listing}\n</ul>\n</article>'
    )
    (docs / "index.html").write_text(
        PAGE.format(
            title="Documentation",
            description="plumbline's documentation, rendered from the repository.",
            nav=_nav(None, "../"),
            body=index,
            repo=REPO,
            icon=ICON,
        ),
        encoding="utf-8",
    )


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
        prov = provenance()
        rendered = render_docs(prov)
    except BuildError as problem:
        print(f"site build refused: {problem}", file=sys.stderr)
        return 1

    if out.exists():
        shutil.rmtree(out)
    # vendor/ holds the markdown renderer, which runs here at build time; the
    # pages it produces need no script, so it is not shipped.
    shutil.copytree(SITE, out, ignore=shutil.ignore_patterns("vendor"))
    (out / "example-run.json").write_text(json.dumps(example, indent=1) + "\n", encoding="utf-8")
    write_docs(out, prov, rendered)
    print(f"site assembled in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
