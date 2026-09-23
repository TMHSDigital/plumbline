"""The command line: load a dataset, run an arm over it, render the report.

Four commands and no cleverness. ``run`` does one arm over one dataset and
writes both an artifact and a report; ``report`` renders artifacts that already
exist, so a finished run is never repeated to get a document out of it;
``adapters`` says what can be run at all; and ``version`` says which build this is.

Two defaults are deliberately absent. There is no default dataset and no default
results directory on ``report``: a relative default resolves against whatever
directory the process happened to start in, and the artifact carries per-case
records. Where they land is the caller's decision, not the shell's.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import partial
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from plumbline import __version__, config
from plumbline.adapters import registry
from plumbline.adapters.base import Adapter
from plumbline.config import DEFAULT_PRICING_TABLE
from plumbline.datasets import loader
from plumbline.metrics.cost import PricingTable
from plumbline.report import markdown
from plumbline.runner import execute
from plumbline.runner.cache import Cache
from plumbline.types import (
    PROBABILITY_SEMANTICS,
    Case,
    DatasetError,
    PlumblineError,
    ProbabilitySemantics,
)

app = typer.Typer(
    help="Measure whether a decision model's probabilities are trustworthy on your own "
    "labeled data.",
    no_args_is_help=True,
    add_completion=False,
)

FORMATS = ("jsonl", "jevbench")


@app.command()
def run(
    dataset: Annotated[Path, typer.Argument(help="JSONL file of labeled cases.")],
    adapter: Annotated[
        str, typer.Option(help=f"One of: {', '.join(registry.available())}.")
    ] = "mock",
    model: Annotated[str | None, typer.Option(help="Model to request.")] = None,
    revision: Annotated[str | None, typer.Option(help="Pinned checkpoint commit.")] = None,
    results: Annotated[Path, typer.Option(help="Directory the artifact is written to.")] = Path(
        "results"
    ),
    report: Annotated[Path | None, typer.Option(help="Write the report here too.")] = None,
    data_format: Annotated[str, typer.Option("--format", help="jsonl or jevbench.")] = "jsonl",
    strict: Annotated[bool, typer.Option(help="Refuse to run unless every row loaded.")] = False,
    limit: Annotated[int | None, typer.Option(min=1, help="Run only the first N cases.")] = None,
    workers: Annotated[int, typer.Option(min=1, help="Concurrent requests.")] = 8,
    cache_dir: Annotated[Path | None, typer.Option("--cache", help="Cache directory.")] = None,
    max_cost_usd: Annotated[float | None, typer.Option(help="Abort above this.")] = None,
    max_cases: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Refuse the run if it covers more cases than this. A guard, not a "
            "truncation: use --limit to run fewer.",
        ),
    ] = None,
    semantics: Annotated[
        str | None,
        typer.Option(
            help="Override probability_semantics, mock only: one of "
            f"{', '.join(PROBABILITY_SEMANTICS)}."
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Mock seed.")] = 7,
    accuracy: Annotated[float, typer.Option(help="Mock target accuracy.")] = 0.8,
    escalation_cost: Annotated[
        float | None,
        typer.Option("--escalation-cost", help="What one escalation costs, in USD."),
    ] = None,
    error_cost: Annotated[
        float | None, typer.Option("--error-cost", help="What one wrong answer costs, in USD.")
    ] = None,
    pricing: Annotated[
        Path | None,
        typer.Option("--pricing", help="JSON pricing table to lay over the shipped one."),
    ] = None,
    n_boot: Annotated[int, typer.Option("--boot", min=1, help="Bootstrap draws per null.")] = 2000,
) -> None:
    """Run one adapter over one dataset, and write what it found.

    Status lines and errors go to stderr, so stdout carries only the report when
    no --report is given. The exit code is 1 when no case produced a prediction,
    after the artifact and the report are written.
    """
    # Everything that can be checked before a call goes out is checked here,
    # so a mistake costs nothing.
    if report is not None and report.is_dir():
        _fail(f"--report {report} is a directory; give it a file path, such as {report}/report.md")
    load = _load(dataset, data_format)
    _status(load.statement())
    for refusal in load.refusals:
        _status(f"  refused {refusal}")
    if strict:
        _guard(load.require_complete)

    cases = list(load.scoreable)[:limit] if limit is not None else list(load.scoreable)
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
            pricing_table=_pricing(pricing),
            workers=workers,
            extra_config={"dataset": str(dataset), "format": data_format},
        )
    )

    artifact = result.write(results)
    _status(f"artifact: {artifact}")

    document = markdown.render(
        [result],
        load=load,
        options=markdown.ReportOptions(
            n_boot=n_boot,
            cost_escalation_usd=escalation_cost,
            cost_error_usd=error_cost,
        ),
    )
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(document, encoding="utf-8")
        _status(f"report: {report}")
    else:
        typer.echo(document)

    if not result.successes:
        # The artifact and the report are written first, since they hold the
        # per-case errors; then the run says it produced nothing, in its exit code.
        reasons = {record.error for record in result.records if record.error}
        why = f" All for the same reason: {reasons.pop()}" if len(reasons) == 1 else ""
        _fail(f"every case failed or was refused, so there is nothing to measure.{why}")


@app.command()
def report(
    artifacts: Annotated[list[Path], typer.Argument(help="Artifact JSON files.")],
    out: Annotated[Path | None, typer.Option("--out", help="Write here instead of stdout.")] = None,
    escalation_cost: Annotated[
        float | None,
        typer.Option("--escalation-cost", help="What one escalation costs, in USD."),
    ] = None,
    error_cost: Annotated[
        float | None, typer.Option("--error-cost", help="What one wrong answer costs, in USD.")
    ] = None,
    n_boot: Annotated[int, typer.Option("--boot", min=1, help="Bootstrap draws per null.")] = 2000,
) -> None:
    """Render one document from runs that already happened."""
    results = [_guard(partial(execute.RunResult.read, path)) for path in artifacts]
    document = markdown.render(
        results,
        options=markdown.ReportOptions(
            n_boot=n_boot,
            cost_escalation_usd=escalation_cost,
            cost_error_usd=error_cost,
        ),
    )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(document, encoding="utf-8")
        _status(f"report: {out}")
    else:
        typer.echo(document)


@app.command()
def adapters() -> None:
    """List the transports this install can run."""
    for name in registry.available():
        missing = [module for module in _EXTRAS.get(name, ()) if find_spec(module) is None]
        typer.echo(f"{name}  (needs the local extra: uv sync --extra local)" if missing else name)


#: Adapters whose dependencies are an optional extra, and the modules it installs.
_EXTRAS = {"local_logits": ("torch", "transformers")}


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

    try:
        return registry.create(adapter, **config)
    except (PlumblineError, ValueError) as refused:
        _fail(str(refused))
    except TypeError:
        given = [f"--{name.replace('_requested', '')}" for name in config if name in _OPTIONS]
        _fail(f"the {adapter} adapter does not take {', '.join(given) or 'these settings'}.")
    except Exception as unavailable:  # an SDK that cannot start, such as a missing key
        variable = _KEY_VARIABLES.get(adapter)
        hint = f" Set {variable} in the environment." if variable else ""
        _fail(f"could not set up the {adapter} adapter: {unavailable}.{hint}")


#: Settings the CLI passes to an adapter from its own options.
_OPTIONS = ("model_requested", "revision")

#: Where each hosted adapter reads its key, for the message when it is missing.
_KEY_VARIABLES = {"typesafe_wire": "TYPESAFE_API_KEY", "generative": "ANTHROPIC_API_KEY"}


def _semantics(value: str) -> ProbabilitySemantics:
    if value not in PROBABILITY_SEMANTICS:
        _fail(f"--semantics must be one of {list(PROBABILITY_SEMANTICS)!r}, got {value!r}")
    return value


def _pricing(supplied: Path | None) -> PricingTable:
    """The shipped table, with the operator's own entries over it if they gave any.

    plumbline ships no figures for a vendor whose terms make its prices
    confidential, so for those a cost column exists only when the operator reads
    the published tariff and supplies it here. Every entry, shipped or supplied,
    carries the date its price was read.
    """
    if supplied is None:
        return DEFAULT_PRICING_TABLE
    return config.merged_pricing_table(_guard(lambda: config.load_pricing_file(supplied)))


def _guard[T](call: Callable[[], T]) -> T:
    """Run a step, turning plumbline's own errors into a message and an exit code.

    A traceback is the right output for a bug and the wrong output for "that
    dataset has a typo on line 12", so the deliberate errors print as one line.
    """
    try:
        return call()
    except (PlumblineError, DatasetError, ValueError) as refused:
        _fail(str(refused))


def _status(message: str) -> None:
    """A line about the run rather than its result, so it goes to stderr."""
    typer.echo(message, err=True)


def _fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover - exercised through the installed script
    sys.exit(app())
