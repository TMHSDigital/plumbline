"""Assemble the site into one directory, ready to deploy. Runs in CI, not by hand.

Copies ``site/`` into the output directory, then writes what the page derives
its worked example from:

``example-run.json``
    The 105 predicted probabilities and outcomes behind ``docs/example-report.md``,
    produced by re-running the command that report records, together with the
    figures the report prints. The page recomputes the ECE and its floor from the
    rows, in JavaScript, and ``scripts/check_floor_parity.mjs`` fails the build if
    what it derives differs from the report by a single character. Before any
    of that, every line of the committed report is compared with what the
    command prints today (only the generation date and path separators are
    normalized), and the build refuses on any difference.

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
from typing import Any, TypedDict

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
    group: str  # its heading in the docs sidebar, one of GROUPS


USING, PROJECT = "Using plumbline", "Project"

#: The docs the site hosts, in navigation order: the sidebar lists them top to
#: bottom under their group, and previous and next follow the same order. A
#: relative link between two of these becomes a link between their pages; a
#: link to any other file goes to GitHub at the commit being built.
DOCS = (
    Doc(
        "README.md",
        "readme",
        "README",
        "What plumbline is, how to run it, and what it does not do.",
        USING,
    ),
    Doc(
        "METHODOLOGY.md",
        "methodology",
        "Methodology",
        "How each figure is computed, and the floor each is reported against.",
        USING,
    ),
    Doc(
        "docs/example-report.md",
        "example-report",
        "Example report",
        "One seeded mock run, rechecked against its command on every build.",
        USING,
    ),
    Doc(
        "datasets/public/README.md",
        "dataset",
        "Dataset",
        "The vendored JevBench fixture: where it comes from and its license.",
        USING,
    ),
    Doc(
        "docs/PLAN.md",
        "plan",
        "Plan",
        "What is built, what is deliberately not, and why.",
        PROJECT,
    ),
    Doc("CHANGELOG.md", "changelog", "Changelog", "Notable changes per release.", PROJECT),
    Doc(
        "CONTRIBUTING.md",
        "contributing",
        "Contributing",
        "How to work on the code and what CI checks.",
        PROJECT,
    ),
    Doc(
        "SECURITY.md",
        "security",
        "Security",
        "How to report a vulnerability, and what the scanners cover.",
        PROJECT,
    ),
)

#: The docs sidebar's groups, in order.
GROUPS = (USING, PROJECT)


class Heading(TypedDict):
    level: int
    id: str
    text: str


class Section(TypedDict):
    id: str | None  # None for text before the first heading
    heading: str
    level: int
    text: str


class Rendered(TypedDict):
    """What ``scripts/render_docs.mjs`` returns for one doc."""

    html: str
    headings: list[Heading]
    sections: list[Section]


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


#: The report's generation date, the one line that legitimately differs between
#: the committed report and a rerun of its command on a later day.
GENERATED_LINE = re.compile(r"^Generated \d{4}-\d{2}-\d{2} against ")


def _comparable(line: str) -> str:
    """A report line with what depends on when and where it ran taken out.

    The date is dropped, and the dataset path's separators are made POSIX: the
    committed report may have been written on Windows and CI runs on Linux.
    Nothing else is normalized. The mock's latencies are simulated from its
    seed, so even the latency line must match exactly.
    """
    if GENERATED_LINE.match(line):
        return GENERATED_LINE.sub("Generated <date> against ", line)
    return line.replace("datasets\\public\\", "datasets/public/")


def _require_same_report(committed: list[str], regenerated: list[str]) -> None:
    """Every line of the committed report must be what its command prints today."""
    ours = [_comparable(line) for line in committed]
    theirs = [_comparable(line) for line in regenerated]
    if ours == theirs:
        return
    differing = [
        (number, mine, fresh)
        for number, (mine, fresh) in enumerate(zip(ours, theirs, strict=False), start=1)
        if mine != fresh
    ]
    detail = "".join(
        f"\n  line {number}\n    report says: {mine}\n    run gives:   {fresh}"
        for number, mine, fresh in differing[:3]
    )
    if len(ours) != len(theirs):
        detail += f"\n  the report has {len(ours)} lines and the run printed {len(theirs)}"
    raise BuildError(
        "docs/example-report.md no longer matches what its own command produces."
        f"{detail}\nRegenerate the example report before deploying."
    )


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

    _require_same_report(printed, regenerated)

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
            f'<a href="{doc.slug}.md">Markdown source</a>. '
            f'<a href="https://github.com/{REPO}/edit/main/{doc.source}">Edit on GitHub</a>.'
        )


def provenance() -> Provenance:
    sha = _git("rev-parse", "HEAD").strip()
    status = _git("status", "--porcelain", "--", *(doc.source for doc in DOCS))
    modified = frozenset(line[3:].strip() for line in status.splitlines())
    # SOURCE_DATE_EPOCH makes the build reproducible when it is set.
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    built = datetime.fromtimestamp(int(epoch), UTC) if epoch else datetime.now(UTC)
    return Provenance(sha, built, modified)


def render_docs(prov: Provenance) -> dict[str, Rendered]:
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
    rendered: dict[str, Rendered] = json.loads(done.stdout)
    return rendered


# The plumb-bob mark, drawn inline so the header makes no request for it.
MARK = (
    '<svg class="mark" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" '
    'focusable="false"><path d="M8 1v9" stroke="currentColor" stroke-width="1.5"/>'
    '<path d="M5 10h6l-3 5z" fill="currentColor"/></svg>'
)


def site_header(root: str, current: str | None) -> str:
    """The bar at the top of every page.

    ``root`` is the site's root relative to the page ("", "../", or the absolute
    path on the 404 page). ``current`` is "calculator", "docs", or a doc slug,
    and marks the matching link. The search and theme buttons are hidden until
    ``site.js`` runs, so a reader without scripts never sees a dead control.
    """
    links = (
        ("calculator", root, "Calculator"),
        ("docs", f"{root}docs/", "Docs"),
        ("example-report", f"{root}docs/example-report.html", "Example report"),
    )
    items = []
    for key, href, label in links:
        mark = ' aria-current="page"' if key == current else ""
        items.append(f'<li><a href="{href}"{mark}>{label}</a></li>')
    items.append(f'<li><a href="https://github.com/{REPO}">GitHub</a></li>')
    joined = "\n        ".join(items)
    return f"""<header class="site-header">
  <div class="bar">
    <a class="brand" href="{root}">{MARK}<span>plumbline</span></a>
    <nav class="primary" aria-label="Site">
      <ul>
        {joined}
      </ul>
    </nav>
    <div class="tools">
      <button type="button" class="search-open" hidden aria-haspopup="dialog">
        Search <kbd>/</kbd>
      </button>
      <button type="button" class="theme-toggle" hidden>Theme: auto</button>
    </div>
  </div>
</header>"""


def docs_sidebar(current: str | None, prefix: str = "") -> str:
    """Every hosted doc, by group, with the current one marked.

    ``prefix`` is the docs directory relative to the page: empty on a doc page,
    absolute on the 404 page.
    """
    groups = []
    for group in GROUPS:
        items = []
        for doc in DOCS:
            if doc.group != group:
                continue
            mark = ' aria-current="page"' if doc.slug == current else ""
            items.append(
                f'<li><a href="{prefix}{doc.slug}.html"{mark}>{html.escape(doc.label)}</a></li>'
            )
        joined = "\n      ".join(items)
        groups.append(
            f'<p class="group">{html.escape(group)}</p>\n    <ul>\n      {joined}\n    </ul>'
        )
    body = "\n    ".join(groups)
    return (
        '<details class="collapsible side docs-nav" open>\n'
        "  <summary>Documentation</summary>\n"
        f'  <nav aria-label="Documentation">\n    <p><a href="{prefix or "./"}">All docs</a></p>\n'
        f"    {body}\n  </nav>\n</details>"
    )


def page_toc(headings: list[Heading]) -> str:
    """The page's h2 and h3 headings, h3 nested under its h2.

    Empty for a page with fewer than two h2s, where a contents list would only
    repeat the page's one heading.
    """
    groups: list[tuple[Heading, list[Heading]]] = []
    for heading in headings:
        if heading["level"] == 2:
            groups.append((heading, []))
        elif heading["level"] == 3 and groups:
            groups[-1][1].append(heading)
    if len(groups) < 2:
        return ""

    def link(heading: Heading) -> str:
        return f'<a href="#{heading["id"]}">{html.escape(heading["text"])}</a>'

    items = []
    for h2, h3s in groups:
        nested = "".join(f"<li>{link(h3)}</li>" for h3 in h3s)
        items.append(f"<li>{link(h2)}" + (f"<ul>{nested}</ul>" if nested else "") + "</li>")
    return (
        '<details class="collapsible side toc" open>\n'
        "  <summary>On this page</summary>\n"
        '  <nav aria-label="On this page">\n'
        '    <p class="label">On this page</p>\n'
        f"    <ul>{''.join(items)}</ul>\n"
        '    <p class="toc-top"><a href="#doc">Back to top</a></p>\n'
        "  </nav>\n</details>"
    )


def prev_next(slug: str) -> str:
    """Links to the docs either side of this one, in sidebar order."""
    at = next(i for i, doc in enumerate(DOCS) if doc.slug == slug)
    links = []
    if at > 0:
        before = DOCS[at - 1]
        links.append(
            f'<a class="prev" href="{before.slug}.html"><span>Previous</span>'
            f"{html.escape(before.label)}</a>"
        )
    if at < len(DOCS) - 1:
        after = DOCS[at + 1]
        links.append(
            f'<a class="next" href="{after.slug}.html"><span>Next</span>'
            f"{html.escape(after.label)}</a>"
        )
    return '<nav class="pager" aria-label="Previous and next">\n' + "\n".join(links) + "\n</nav>"


# The explainer's plumb-bob icon, inline so the page makes no request for it.
ICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E"
    "%3Cpath d='M8 1v9' stroke='%23555' stroke-width='1.5'/%3E"
    "%3Cpath d='M5 10h6l-3 5z' fill='%23555'/%3E%3C/svg%3E"
)

#: Where Pages serves the site. Canonical and Open Graph URLs are absolute, and
#: the 404 page links by absolute path, because it is served at any depth.
SITE_URL = "https://tmhsdigital.github.io/plumbline/"
SITE_PATH = "/plumbline/"
OG_IMAGE_ALT = (
    "A calibration claim has a floor: ECE 0.074 on 105 rows sits below the floor's "
    "95th percentile of 0.111, so the result is inconclusive."
)


def social_meta(title: str, description: str, url: str) -> str:
    """Canonical, Open Graph, and Twitter card tags for one page.

    ``site/index.html`` carries the same tags written out by hand, and
    ``scripts/check_site_links.mjs`` checks every page has them and that its
    canonical and og:url name the page itself.
    """
    title, description = html.escape(title), html.escape(description)
    image = f"{SITE_URL}og.png"
    alt = html.escape(OG_IMAGE_ALT)
    return "\n".join(
        (
            f'<link rel="canonical" href="{url}">',
            '<meta property="og:type" content="website">',
            '<meta property="og:site_name" content="plumbline">',
            f'<meta property="og:title" content="{title}">',
            f'<meta property="og:description" content="{description}">',
            f'<meta property="og:url" content="{url}">',
            f'<meta property="og:image" content="{image}">',
            '<meta property="og:image:width" content="1200">',
            '<meta property="og:image:height" content="630">',
            f'<meta property="og:image:alt" content="{alt}">',
            '<meta name="twitter:card" content="summary_large_image">',
            f'<meta name="twitter:title" content="{title}">',
            f'<meta name="twitter:description" content="{description}">',
            f'<meta name="twitter:image" content="{image}">',
            f'<meta name="twitter:image:alt" content="{alt}">',
        )
    )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} | plumbline</title>
<meta name="description" content="{description}">
{meta}
<meta name="color-scheme" content="light dark">
<link rel="icon" href="{icon}">
<script src="{root}theme.js"></script>
<link rel="stylesheet" href="{root}base.css">
<link rel="stylesheet" href="{root}docs.css">
</head>
<body>
<a class="skip" href="#doc">Skip to the document</a>
{header}
<div class="docs-layout">
{sidebar}
<main id="doc">
{body}
</main>
{toc}
</div>
<footer>
<p>Every page here is rendered from the repository at deploy time; none is edited by hand.
<a href="https://github.com/{repo}">Source on GitHub</a>, Apache-2.0 licensed.
No analytics, no trackers, no external requests.</p>
</footer>
<script src="{root}site.js" defer></script>
</body>
</html>
"""


def _page(
    title: str,
    description: str,
    meta: str,
    root: str,
    *,
    body: str,
    doc: str | None = None,
    header: str | None = "docs",
    docs_prefix: str = "",
    toc: str = "",
) -> str:
    """One doc-shaped page: header, docs sidebar, the body, and its contents list."""
    return PAGE.format(
        title=html.escape(title),
        description=html.escape(description, quote=True),
        meta=meta,
        icon=ICON,
        root=root,
        header=site_header(root, header),
        sidebar=docs_sidebar(doc, docs_prefix),
        body=body,
        toc=toc,
        repo=REPO,
    )


def write_docs(out: Path, prov: Provenance, rendered: dict[str, Rendered]) -> None:
    docs = out / "docs"
    docs.mkdir()
    for doc in DOCS:
        shutil.copyfile(ROOT / doc.source, docs / f"{doc.slug}.md")
        body = (
            f'<p class="provenance">{prov.line(doc)}</p>\n'
            f'<article class="prose">\n{rendered[doc.slug]["html"]}</article>\n'
            f"{prev_next(doc.slug)}"
        )
        meta = social_meta(f"{doc.label} | plumbline", doc.blurb, f"{SITE_URL}docs/{doc.slug}.html")
        toc = page_toc(rendered[doc.slug]["headings"])
        # The example report has its own link in the header; every other doc is "Docs".
        header = "example-report" if doc.slug == "example-report" else "docs"
        page = _page(
            doc.label, doc.blurb, meta, "../", body=body, doc=doc.slug, header=header, toc=toc
        )
        (docs / f"{doc.slug}.html").write_text(page, encoding="utf-8")

    listing = []
    for group in GROUPS:
        items = "\n".join(
            f'  <li><a href="{doc.slug}.html">{html.escape(doc.label)}</a> '
            f'<span class="muted small">{html.escape(doc.source)}</span><br>'
            f"{html.escape(doc.blurb)}</li>"
            for doc in DOCS
            if doc.group == group
        )
        listing.append(f'<h2>{html.escape(group)}</h2>\n<ul class="doc-list">\n{items}\n</ul>')
    at = prov.built.strftime("%Y-%m-%d %H:%M UTC")
    index = (
        '<article class="prose">\n<h1>Documentation</h1>\n'
        "<p>The repository's own markdown, rendered from commit "
        f'<a href="https://github.com/{REPO}/commit/{prov.sha}"><code>{prov.sha[:7]}</code></a> '
        f"at {at}.</p>\n" + "\n".join(listing) + "\n</article>"
    )
    description = "plumbline's documentation, rendered from the repository."
    meta = social_meta("Documentation | plumbline", description, f"{SITE_URL}docs/")
    page = _page("Documentation", description, meta, "../", body=index)
    (docs / "index.html").write_text(page, encoding="utf-8")


def write_404(out: Path) -> None:
    """Pages serves this for any missing path, at any depth, so it links absolutely."""
    body = (
        '<article class="prose">\n<h1>Not found</h1>\n'
        "<p>There is no page at this address.</p>\n"
        f'<p><a href="{SITE_PATH}">Go to the explainer</a>, or '
        f'<a href="{SITE_PATH}docs/">browse the documentation</a>.</p>\n</article>'
    )
    meta = '<meta name="robots" content="noindex">'
    page = _page(
        "Not found",
        "No page at this address.",
        meta,
        SITE_PATH,
        body=body,
        header=None,
        docs_prefix=f"{SITE_PATH}docs/",
    )
    (out / "404.html").write_text(page, encoding="utf-8")


#: Where the shared header goes in ``site/index.html``, which is written by hand.
HEADER_SLOT = "<!-- site:header -->"


def explainer_page(source: str) -> str:
    """The explainer as written, with the shared header in its slot."""
    if source.count(HEADER_SLOT) != 1:
        raise BuildError(f"site/index.html must contain {HEADER_SLOT} exactly once")
    return source.replace(HEADER_SLOT, site_header("", "calculator"))


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
        explainer = explainer_page((SITE / "index.html").read_text(encoding="utf-8"))
    except BuildError as problem:
        print(f"site build refused: {problem}", file=sys.stderr)
        return 1

    if out.exists():
        shutil.rmtree(out)
    # vendor/ holds the markdown renderer, which runs here at build time; the
    # pages it produces are finished HTML, so it is not shipped.
    shutil.copytree(SITE, out, ignore=shutil.ignore_patterns("vendor"))
    (out / "example-run.json").write_text(json.dumps(example, indent=1) + "\n", encoding="utf-8")
    (out / "index.html").write_text(explainer, encoding="utf-8")
    write_docs(out, prov, rendered)
    write_404(out)
    print(f"site assembled in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
