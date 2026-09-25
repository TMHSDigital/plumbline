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

import dataclasses
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
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="Send requests here instead of the vendor's endpoint, for an adapter that "
            "takes one. Recorded in the artifact and part of the cache key.",
        ),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option(
            "--timeout", help="Seconds one request may take, for an adapter that takes it."
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option(
            "--device",
            help="Where a local checkpoint runs, such as cpu or cuda, for an adapter that "
            "takes it. Recorded in the artifact.",
        ),
    ] = None,
    semantics: Annotated[
        str | None,
        typer.Option(
            help="Override the probability_semantics the adapter declares, for one that "
            f"takes it: one of {', '.join(PROBABILITY_SEMANTICS)}. The report says so."
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Load, check, and price the run, print what it would do, and send nothing.",
        ),
    ] = False,
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
    after the artifact and the report are written. With --dry-run, stdout carries
    the plan instead, and nothing is sent or written.
    """
    # Everything that can be checked before a call goes out is checked here,
    # so a mistake costs nothing.
    if report is not None and report.is_dir():
        _fail(f"--report {report} is a directory; give it a file path, such as {report}/report.md")
    if timeout is not None and not timeout > 0:
        _fail(f"--timeout must be a number of seconds above 0, got {timeout}")
    load = dataclasses.replace(_load(dataset, data_format), source=_shown(dataset))
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
        base_url=base_url,
        timeout=timeout,
        device=device,
        semantics=semantics,
        seed=seed,
        accuracy=accuracy,
    )
    guard = execute.CostGuard(max_cost_usd=max_cost_usd, max_cases=max_cases)
    table = _pricing(pricing)
    if dry_run:
        planned = _guard(lambda: execute.plan(built, cases, guard=guard, pricing_table=table))
        typer.echo(_plan_text(built, planned, guard, semantics_set=semantics is not None))
        return

    extra: dict[str, object] = {"dataset": _shown(dataset), "format": data_format}
    if semantics is not None:
        extra["semantics_set_by"] = "operator"
    result = _guard(
        lambda: execute.run(
            built,
            cases,
            cache=Cache(cache_dir) if cache_dir else None,
            guard=guard,
            pricing_table=table,
            workers=workers,
            extra_config=extra,
            load=load.summary(),
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
    allow_mixed: Annotated[
        bool,
        typer.Option(
            "--allow-mixed",
            help="Allow runs over different datasets in one document; each arm names its own.",
        ),
    ] = False,
) -> None:
    """Render one document from runs that already happened."""
    results = [_guard(partial(execute.RunResult.read, path)) for path in artifacts]
    document = _guard(
        lambda: markdown.render(
            results,
            load=_stored_load(results),
            options=markdown.ReportOptions(
                n_boot=n_boot,
                cost_escalation_usd=escalation_cost,
                cost_error_usd=error_cost,
                allow_mixed_datasets=allow_mixed,
            ),
        )
    )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(document, encoding="utf-8")
        _status(f"report: {out}")
    else:
        typer.echo(document)


def _stored_load(results: list[execute.RunResult]) -> loader.LoadSummary | None:
    """The Dataset section a rebuilt report can print, from what the runs stored.

    Only when every run is over one dataset and they all say the same thing
    about loading it. Otherwise there is no single section that is true of the
    document, and printing one run's counts over another's figures would be
    worse than printing none.
    """
    if len({result.dataset_hash for result in results}) != 1:
        return None
    first = results[0].load
    return first if all(result.load == first for result in results) else None


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


def _shown(dataset: Path) -> str:
    """The dataset as a report or an artifact names it.

    Relative to the working directory when it is inside it, as the caller most
    likely typed it; otherwise the file name alone. An absolute path in a report
    that gets shared names the user and their directory layout, and the dataset
    hash already says which rows these were.
    """
    try:
        return str(dataset.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return dataset.name


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
    base_url: str | None = None,
    timeout: float | None = None,
    device: str | None = None,
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
    if base_url is not None:
        config["base_url"] = base_url
    if timeout is not None:
        config["timeout"] = timeout
    if device is not None:
        config["device"] = device
    if semantics is not None:
        config["probability_semantics"] = _semantics(semantics)

    if adapter == "mock":
        # The mock is told the answer key up front, because classify() is never
        # given the gold label. It is a demo arm, not a system under test.
        config["gold_by_text"] = {case.text: case.gold_label for case in cases}
        config["seed"] = seed
        config["accuracy"] = accuracy

    # Name the exact option an adapter will not take, before building it: the
    # generative arm reports no probability, so it has no semantics to override.
    refused_options = [
        flag
        for setting, flag in _OPTIONS.items()
        if setting in config and not _guard(partial(registry.accepts, adapter, setting))
    ]
    if refused_options:
        _fail(f"the {adapter} adapter does not take {', '.join(refused_options)}.")

    try:
        return registry.create(adapter, **config)
    except (PlumblineError, ValueError) as refused:
        _fail(str(refused))
    except TypeError:
        given = [flag for setting, flag in _OPTIONS.items() if setting in config]
        _fail(f"the {adapter} adapter does not take {', '.join(given) or 'these settings'}.")
    except Exception as unavailable:  # an SDK that cannot start, such as a missing key
        variable = _KEY_VARIABLES.get(adapter)
        hint = f" Set {variable} in the environment." if variable else ""
        _fail(f"could not set up the {adapter} adapter: {unavailable}.{hint}")


#: Settings the CLI passes to an adapter from its own options, and the option
#: each comes from.
_OPTIONS = {
    "model_requested": "--model",
    "revision": "--revision",
    "base_url": "--base-url",
    "timeout": "--timeout",
    "device": "--device",
    "probability_semantics": "--semantics",
}


def _plan_text(
    adapter: Adapter, planned: execute.Plan, guard: execute.CostGuard, *, semantics_set: bool
) -> str:
    """What a dry run prints: what would be sent, to whom, and what it would cost."""
    endpoint = getattr(adapter, "base_url", None)
    timeout = getattr(adapter, "timeout", None)
    device = getattr(adapter, "device", None)
    semantics = adapter.probability_semantics + (", set by --semantics" if semantics_set else "")
    if planned.estimated_cost_usd is not None and planned.pricing is not None:
        cost = (
            f"about {planned.estimated_cost_usd:.4f} USD, priced by `{planned.pricing_key}` "
            f"as read on {planned.pricing.as_of.isoformat()}. The estimate is rough: it counts "
            "the case text, the option names, and a fixed overhead."
        )
    else:
        cost = (
            f"not estimated: `{adapter.model_requested}` is not in the pricing table. Pass "
            "--pricing with an entry for it to cost the run in advance."
        )
    cost_limit = (
        f"max cost {guard.max_cost_usd} USD" if guard.max_cost_usd is not None else "no cost limit"
    )
    case_limit = f"max cases {guard.max_cases}" if guard.max_cases is not None else "no case limit"
    revision = f", revision `{adapter.revision}`" if adapter.revision else ""
    return "\n".join(
        [
            "dry run: nothing was sent and nothing was written.",
            "",
            f"- {planned.cases} cases for the {adapter.name} adapter, model "
            f"`{adapter.model_requested}`{revision}.",
            f"- Endpoint: `{endpoint}`." if endpoint else "- Endpoint: the adapter's default.",
            f"- Timeout: {timeout:g} s per request."
            if timeout
            else "- Timeout: the adapter's default.",
            *([f"- Device: {device}."] if device else []),
            f"- Probability semantics: {semantics}.",
            f"- Cost: {cost}",
            f"- Guard: {cost_limit}, {case_limit}; the run would start.",
        ]
    )


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
