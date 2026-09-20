"""The command line: load a dataset, run an arm over it, render the report.

Three commands and no cleverness. ``run`` does one arm over one dataset and
writes both an artifact and a report; ``report`` renders artifacts that already
exist, so a finished run is never repeated to get a document out of it; and
``adapters`` says what can be run at all.

Two defaults are deliberately absent. There is no default dataset and no default
results directory on ``report``: a relative default resolves against whatever
directory the process happened to start in, and the artifact carries per-case
records. Where they land is the caller's decision, not the shell's.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from plumbline import __version__
from plumbline.adapters import registry
from plumbline.adapters.base import Adapter
from plumbline.config import DEFAULT_PRICING_TABLE
from plumbline.datasets import loader
from plumbline.metrics.cost import PricingTable
from plumbline.report import markdown
from plumbline.runner import execute
from plumbline.runner.cache import Cache
from plumbline.types import Case, DatasetError, PlumblineError, ProbabilitySemantics

app = typer.Typer(
    help="Choose and configure a decision model on your own labeled data.",
    no_args_is_help=True,
    add_completion=False,
)

FORMATS = ("jsonl", "jevbench")


@app.command()
def run(
    dataset: Annotated[Path, typer.Argument(help="JSONL file of labeled cases.")],
    adapter: Annotated[str, typer.Option(help="Registered adapter name.")] = "mock",
    model: Annotated[str | None, typer.Option(help="Model to request.")] = None,
    revision: Annotated[str | None, typer.Option(help="Pinned checkpoint commit.")] = None,
    results: Annotated[Path, typer.Option(help="Directory the artifact is written to.")] = Path(
        "results"
    ),
    report: Annotated[Path | None, typer.Option(help="Write the report here too.")] = None,
    data_format: Annotated[str, typer.Option("--format", help="jsonl or jevbench.")] = "jsonl",
    strict: Annotated[bool, typer.Option(help="Refuse to run unless every row loaded.")] = False,
    limit: Annotated[int | None, typer.Option(help="Run only the first N cases.")] = None,
    workers: Annotated[int, typer.Option(help="Concurrent requests.")] = 8,
    cache_dir: Annotated[Path | None, typer.Option("--cache", help="Cache directory.")] = None,
    max_cost_usd: Annotated[float | None, typer.Option(help="Abort above this.")] = None,
    max_cases: Annotated[int | None, typer.Option(help="Abort above this many.")] = None,
    semantics: Annotated[
        str | None, typer.Option(help="Override probability_semantics (mock only).")
    ] = None,
    seed: Annotated[int, typer.Option(help="Mock seed.")] = 7,
    accuracy: Annotated[float, typer.Option(help="Mock target accuracy.")] = 0.8,
    n_boot: Annotated[int, typer.Option("--boot", help="Bootstrap draws per null.")] = 2000,
) -> None:
    """Run one adapter over one dataset, and write what it found."""
    load = _load(dataset, data_format)
    typer.echo(load.statement())
    for refusal in load.refusals:
        typer.echo(f"  refused {refusal}")
    if strict:
        _guard(load.require_complete)

    cases = list(load.scoreable)[:limit] if limit else list(load.scoreable)
    if not cases:
        _fail("no scoreable rows in this dataset, so there is nothing to run.")

    built = _build(
        adapter,
        cases,
        model=model,
        revision=revision,
        semantics=semantics,
        seed=seed,
        accuracy=accuracy,
    )
    result = _guard(
        lambda: execute.run(
            built,
            cases,
            cache=Cache(cache_dir) if cache_dir else None,
            guard=execute.CostGuard(max_cost_usd=max_cost_usd, max_cases=max_cases),
            pricing_table=_pricing(),
            workers=workers,
            extra_config={"dataset": str(dataset), "format": data_format},
        )
    )

    artifact = result.write(results)
    typer.echo(f"artifact: {artifact}")

    document = markdown.render([result], load=load, options=markdown.ReportOptions(n_boot=n_boot))
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(document, encoding="utf-8")
        typer.echo(f"report: {report}")
    else:
        typer.echo("")
        typer.echo(document)


@app.command()
def report(
    artifacts: Annotated[list[Path], typer.Argument(help="Artifact JSON files.")],
    out: Annotated[Path | None, typer.Option("--out", help="Write here instead of stdout.")] = None,
    n_boot: Annotated[int, typer.Option("--boot", help="Bootstrap draws per null.")] = 2000,
) -> None:
    """Render one document from runs that already happened."""
    results = [_guard(partial(execute.RunResult.read, path)) for path in artifacts]
    document = markdown.render(results, options=markdown.ReportOptions(n_boot=n_boot))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(document, encoding="utf-8")
        typer.echo(f"report: {out}")
    else:
        typer.echo(document)


@app.command()
def adapters() -> None:
    """List the transports this install can run."""
    for name in registry.available():
        typer.echo(name)


@app.command()
def version() -> None:
    """Print the plumbline version."""
    typer.echo(__version__)


def _load(dataset: Path, data_format: str) -> loader.LoadReport:
    if data_format not in FORMATS:
        _fail(f"--format must be one of {list(FORMATS)!r}, got {data_format!r}")
    read = loader.load_jevbench if data_format == "jevbench" else loader.load_jsonl
    return _guard(lambda: read(dataset))


def _build(
    adapter: str,
    cases: list[Case],
    *,
    model: str | None,
    revision: str | None,
    semantics: str | None,
    seed: int,
    accuracy: float,
) -> Adapter:
    """Construct the adapter, passing only what that adapter actually takes."""
    config: dict[str, object] = {}
    if model is not None:
        config["model_requested"] = model
    if revision is not None:
        config["revision"] = revision

    if adapter == "mock":
        # The mock is told the answer key up front, because classify() is never
        # given the gold label. It is a demo arm, not a system under test.
        config["gold_by_text"] = {case.text: case.gold_label for case in cases}
        config["seed"] = seed
        config["accuracy"] = accuracy
        if semantics is not None:
            config["probability_semantics"] = _semantics(semantics)
    elif semantics is not None:
        _fail("--semantics applies to the mock only; a real adapter declares its own.")

    return _guard(lambda: registry.create(adapter, **config))


def _semantics(value: str) -> ProbabilitySemantics:
    from plumbline.types import PROBABILITY_SEMANTICS

    if value not in PROBABILITY_SEMANTICS:
        _fail(f"--semantics must be one of {list(PROBABILITY_SEMANTICS)!r}, got {value!r}")
    return value


def _pricing() -> PricingTable:
    """The shipped table. Every entry carries the date its price was read."""
    return DEFAULT_PRICING_TABLE


def _guard[T](call: Callable[[], T]) -> T:
    """Run a step, turning plumbline's own errors into a message and an exit code.

    A traceback is the right output for a bug and the wrong output for "that
    dataset has a typo on line 12", so the deliberate errors print as one line.
    """
    try:
        return call()
    except (PlumblineError, DatasetError, ValueError) as refused:
        _fail(str(refused))


def _fail(message: str) -> NoReturn:
    typer.echo(message)
    raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover - exercised through the installed script
    sys.exit(app())
