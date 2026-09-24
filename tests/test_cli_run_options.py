"""The run options that change what is sent, and the dry run that sends nothing.

Everything here stays off the network. The hosted adapters are built with a
dummy key and only ever reach the dry run, which returns before any call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from plumbline import cli
from plumbline.adapters.mock import MockAdapter
from plumbline.datasets.loader import LoadSummary
from plumbline.runner import execute
from plumbline.types import Case

runner = CliRunner()


def a_dataset(path: Path, n_rows: int = 12, broken_rows: int = 0) -> Path:
    labels = ["billing", "returns", "shipping", "other"]
    lines = [
        json.dumps(
            {
                "id": f"case-{index}",
                "text": f"ticket body number {index}",
                "labels": labels,
                "gold_label": labels[index % len(labels)],
            }
        )
        for index in range(n_rows)
    ]
    # A gold label that is not among the row's options is refused on load.
    lines += [
        json.dumps({"id": f"bad-{index}", "text": "t", "labels": ["a", "b"], "gold_label": "c"})
        for index in range(broken_rows)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def invoke(*args: str):
    return runner.invoke(cli.app, list(args), catch_exceptions=False)


@pytest.fixture
def no_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any classify call fails the test: a dry run must not make one."""

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a dry run called the adapter")

    monkeypatch.setattr(MockAdapter, "classify", refuse)


def test_a_dry_run_prices_the_run_and_sends_nothing(tmp_path: Path, no_calls: None) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl")
    results = tmp_path / "results"

    done = invoke("run", str(dataset), "--dry-run", "--results", str(results))

    assert done.exit_code == 0, done.stderr
    assert "dry run" in done.stdout and "nothing was sent" in done.stdout
    assert "12 cases" in done.stdout
    assert "not in the pricing table" in done.stdout
    assert not results.exists(), "a dry run wrote an artifact"


def test_a_dry_run_names_the_estimate_and_the_entry_that_priced_it(
    tmp_path: Path, no_calls: None
) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl")
    pricing = tmp_path / "pricing.json"
    pricing.write_text(
        json.dumps(
            {
                "demo-model": {
                    "input_usd_per_million": 1.0,
                    "output_usd_per_million": 2.0,
                    "source": "https://example.invalid/prices",
                    "as_of": "2026-09-01",
                }
            }
        ),
        encoding="utf-8",
    )

    done = invoke(
        "run", str(dataset), "--dry-run", "--model", "demo-model", "--pricing", str(pricing)
    )

    assert done.exit_code == 0, done.stderr
    assert "USD" in done.stdout and "`demo-model`" in done.stdout
    assert "2026-09-01" in done.stdout


def test_a_dry_run_refuses_what_the_run_would_refuse(tmp_path: Path, no_calls: None) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl")

    done = invoke("run", str(dataset), "--dry-run", "--max-cases", "5")

    assert done.exit_code == 1
    assert "max_cases is 5" in done.stderr


def test_an_endpoint_and_a_timeout_reach_the_adapter_and_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "dummy-not-a-real-key")
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    dataset = a_dataset(tmp_path / "d.jsonl")

    done = invoke(
        "run",
        str(dataset),
        "--adapter",
        "typesafe_wire",
        "--base-url",
        "http://localhost:8911",
        "--timeout",
        "12.5",
        "--semantics",
        "restricted_softmax",
        "--dry-run",
    )

    assert done.exit_code == 0, done.stderr
    assert "`http://localhost:8911`" in done.stdout
    assert "12.5 s" in done.stdout
    assert "restricted_softmax, set by --semantics" in done.stdout
    assert "dummy-not-a-real-key" not in done.stdout + done.stderr


@pytest.mark.parametrize(
    ("adapter", "flag", "value"),
    [
        ("mock", "--base-url", "http://localhost:1"),
        ("mock", "--timeout", "5"),
        ("generative", "--semantics", "calibrated_claim"),
    ],
)
def test_a_setting_the_adapter_does_not_take_is_named_exactly(
    tmp_path: Path, adapter: str, flag: str, value: str
) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl")

    done = invoke("run", str(dataset), "--adapter", adapter, flag, value, "--dry-run")

    assert done.exit_code == 1
    assert f"the {adapter} adapter does not take {flag}" in done.stderr


def test_a_timeout_must_be_positive(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl")

    done = invoke("run", str(dataset), "--timeout", "0", "--dry-run")

    assert done.exit_code == 1
    assert "--timeout" in done.stderr


def test_a_report_rebuilt_from_artifacts_keeps_its_dataset_section(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl", broken_rows=2)
    results = tmp_path / "results"
    at_run = tmp_path / "at-run.md"
    invoke("run", str(dataset), "--results", str(results), "--report", str(at_run), "--boot", "50")
    artifact = next(results.glob("*.json"))

    rebuilt = invoke("report", str(artifact), "--boot", "50")

    assert rebuilt.exit_code == 0, rebuilt.stderr

    def dataset_section(text: str) -> str:
        return text.split("## Dataset", 1)[1].split("\n## ", 1)[0]

    written = at_run.read_text(encoding="utf-8")
    assert "14 rows read" in dataset_section(written) and "2 refused" in dataset_section(written)
    assert dataset_section(rebuilt.stdout) == dataset_section(written)


def test_an_artifact_from_before_the_load_was_stored_still_reports(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl")
    results = tmp_path / "results"
    invoke("run", str(dataset), "--results", str(results), "--boot", "50")
    artifact = next(results.glob("*.json"))
    stored = json.loads(artifact.read_text(encoding="utf-8"))
    del stored["load"]
    artifact.write_text(json.dumps(stored), encoding="utf-8")

    rebuilt = invoke("report", str(artifact), "--boot", "50")

    assert rebuilt.exit_code == 0, rebuilt.stderr
    assert "## Dataset" not in rebuilt.stdout


def test_the_load_summary_round_trips_through_the_artifact(tmp_path: Path) -> None:
    summary = LoadSummary(
        source="d.jsonl",
        rows_read=5,
        loaded=4,
        refusals=("line 5: gold label 'c' is not among the options",),
        notes=("a note.",),
        unsupported_by_type={"score": 1},
    )
    cases = [Case(id=f"c{i}", text=f"t{i}", labels=("a", "b"), gold_label="a") for i in range(3)]
    result = execute.run(
        MockAdapter(gold_by_text={case.text: case.gold_label for case in cases}),
        cases,
        load=summary,
    )

    reread = execute.RunResult.read(result.write(tmp_path))

    assert reread.load == summary
    assert reread.load is not None and reread.load.statement() == summary.statement()


def test_the_timeout_an_adapter_used_is_in_the_artifact() -> None:
    class Patient(MockAdapter):
        timeout = 42.0

    cases = [Case(id="c", text="t", labels=("a", "b"), gold_label="a")]

    result = execute.run(Patient(gold_by_text={"t": "a"}), cases)

    assert result.config["timeout_seconds"] == 42.0
