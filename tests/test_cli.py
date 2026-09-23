"""The command line: the entry point pyproject has been promising all along.

``[project.scripts]`` declares ``plumbline = "plumbline.cli:app"``. Until this
module existed, a fresh install put a ``plumbline`` command on the path that
failed at import. The first test here is the one that closes that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from plumbline import __version__, cli

runner = CliRunner()
FIXTURE = Path(__file__).resolve().parent.parent / "datasets/public/jevbench-hard.jsonl"


def a_dataset(path: Path, n_rows: int = 24) -> Path:
    labels = ["billing", "returns", "shipping", "other"]
    rows = [
        {
            "id": f"case-{index}",
            "text": f"ticket body number {index}",
            "labels": labels,
            "gold_label": labels[index % len(labels)],
        }
        for index in range(n_rows)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def invoke(*args: str):
    return runner.invoke(cli.app, list(args), catch_exceptions=False)


def test_the_entry_point_pyproject_declares_actually_exists() -> None:
    """plumbline = "plumbline.cli:app". A fresh install must not ship a stub."""
    from importlib.metadata import entry_points

    assert callable(cli.app)
    declared = [
        entry
        for entry in entry_points(group="console_scripts")
        if entry.name == "plumbline" and entry.value.startswith("plumbline.cli")
    ]
    assert declared, "no console_scripts entry point named plumbline"
    assert declared[0].load() is cli.app


def test_it_lists_the_transports_it_can_run() -> None:
    result = invoke("adapters")

    assert result.exit_code == 0
    for name in ("mock", "typesafe_wire", "local_logits", "generative"):
        assert name in result.stdout


def test_it_reports_its_version() -> None:
    """Exactly, not as a substring.

    This asserted ``"0.1.0" in stdout``, which is true of ``0.1.0.dev0`` as
    well, so a release tagged v0.1.0 shipped a package still calling itself
    ``0.1.0.dev0`` and the test stayed green. Comparing against ``__version__``
    also pins the CLI to the package rather than to a literal that has to be
    remembered in two places.
    """
    result = invoke("version")

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_a_run_writes_an_artifact_and_a_report(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl")
    report = tmp_path / "report.md"

    result = invoke(
        "run",
        str(dataset),
        "--adapter",
        "mock",
        "--results",
        str(tmp_path / "results"),
        "--report",
        str(report),
        "--boot",
        "100",
    )

    assert result.exit_code == 0
    assert list((tmp_path / "results").glob("*.json"))
    text = report.read_text(encoding="utf-8")
    assert "# plumbline report" in text
    assert "over 24 rows" in text


def test_a_run_prints_what_it_loaded_and_what_it_refused(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl", n_rows=6)
    with dataset.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "bad", "text": "x", "labels": ["a", "b"], "gold": "a"}))
        handle.write("\n")

    result = invoke("run", str(dataset), "--results", str(tmp_path / "r"), "--boot", "100")

    assert result.exit_code == 0
    assert "6 loaded" in result.output
    assert "1 refused" in result.output


def test_strict_refuses_to_run_a_dataset_with_an_unscoreable_row(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl", n_rows=4)
    with dataset.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "typo",
                    "text": "x",
                    "labels": ["a", "b"],
                    "gold_label": "c",
                }
            )
            + "\n"
        )

    result = invoke(
        "run", str(dataset), "--results", str(tmp_path / "r"), "--strict", "--boot", "100"
    )

    assert result.exit_code != 0
    assert "typo" in result.output


def test_the_jevbench_loader_is_selectable(tmp_path: Path) -> None:
    result = invoke(
        "run",
        str(FIXTURE),
        "--format",
        "jevbench",
        "--results",
        str(tmp_path / "r"),
        "--limit",
        "12",
        "--boot",
        "100",
    )

    assert result.exit_code == 0
    assert "111 rows read" in result.output


def test_a_report_can_be_rendered_from_a_stored_artifact(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl")
    invoke("run", str(dataset), "--results", str(tmp_path / "results"), "--boot", "100")
    artifact = next((tmp_path / "results").glob("*.json"))

    result = invoke("report", str(artifact), "--boot", "100")

    assert result.exit_code == 0
    assert "# plumbline report" in result.stdout
    assert "Calibrated claims" in result.stdout


def test_a_report_of_two_artifacts_keeps_them_in_labeled_groups(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl")
    for semantics in ("calibrated_claim", "restricted_softmax"):
        invoke(
            "run",
            str(dataset),
            "--results",
            str(tmp_path / "results"),
            "--semantics",
            semantics,
            "--boot",
            "100",
        )
    artifacts = sorted((tmp_path / "results").glob("*.json"))

    result = invoke("report", *[str(path) for path in artifacts], "--boot", "100")

    assert "## Calibrated claims" in result.stdout
    assert "## Restricted softmax" in result.stdout


def test_a_missing_dataset_fails_with_the_path_it_looked_at(tmp_path: Path) -> None:
    result = invoke("run", str(tmp_path / "nope.jsonl"), "--results", str(tmp_path / "r"))

    assert result.exit_code != 0
    assert "nope.jsonl" in result.output


def test_an_unknown_adapter_names_the_ones_that_exist(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl", n_rows=4)

    result = invoke("run", str(dataset), "--adapter", "telepathy", "--results", str(tmp_path / "r"))

    assert result.exit_code != 0
    assert "mock" in result.output


@pytest.mark.parametrize("command", ["run", "report", "adapters", "version"])
def test_every_command_documents_itself(command: str) -> None:
    result = invoke(command, "--help")

    assert result.exit_code == 0
    assert result.stdout.strip()


def test_the_cascade_sentence_appears_when_the_two_costs_are_supplied(tmp_path: Path) -> None:
    """The two numbers no benchmark can know are flags, not defaults."""
    dataset = a_dataset(tmp_path / "d.jsonl", n_rows=400)
    report_path = tmp_path / "report.md"

    result = invoke(
        "run",
        str(dataset),
        "--results",
        str(tmp_path / "results"),
        "--report",
        str(report_path),
        "--escalation-cost",
        "0.02",
        "--error-cost",
        "1.00",
        "--boot",
        "100",
    )

    assert result.exit_code == 0
    text = report_path.read_text(encoding="utf-8")
    assert "stays on the cheap arm" in text
    assert "versus $" in text


def test_without_the_costs_the_report_says_they_are_yours_to_supply(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl", n_rows=40)
    report_path = tmp_path / "report.md"

    invoke(
        "run",
        str(dataset),
        "--results",
        str(tmp_path / "results"),
        "--report",
        str(report_path),
        "--boot",
        "100",
    )

    text = report_path.read_text(encoding="utf-8")
    assert "--escalation-cost" in text


# Validation before anything is sent (#41), and what the exit code and the
# streams say afterwards (#42)


@pytest.mark.parametrize(
    "option",
    [["--limit", "0"], ["--limit", "-1"], ["--boot", "0"], ["--workers", "0"]],
    ids=["limit-0", "limit-negative", "boot-0", "workers-0"],
)
def test_a_count_that_must_be_positive_is_refused_before_the_run(
    tmp_path: Path, option: list[str]
) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl", n_rows=6)
    result = invoke("run", str(dataset), "--results", str(tmp_path / "r"), *option)

    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert not (tmp_path / "r").exists()


def test_an_option_the_adapter_does_not_take_is_named_not_a_traceback(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl", n_rows=6)
    result = runner.invoke(
        cli.app,
        ["run", str(dataset), "--adapter", "typesafe_wire", "--revision", "abc"],
        env={"TYPESAFE_API_KEY": "not-a-key"},
    )

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "revision" in result.output


def test_a_missing_api_key_is_one_line_naming_the_variable(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl", n_rows=6)
    result = runner.invoke(
        cli.app,
        ["run", str(dataset), "--adapter", "typesafe_wire"],
        env={"TYPESAFE_API_KEY": None},
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "TYPESAFE_API_KEY" in result.output


def test_a_report_path_that_is_a_directory_is_refused_before_the_run(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl", n_rows=6)
    (tmp_path / "out").mkdir()
    result = invoke(
        "run", str(dataset), "--results", str(tmp_path / "r"), "--report", str(tmp_path / "out")
    )

    assert result.exit_code != 0
    assert "directory" in result.output
    assert not (tmp_path / "r").exists()


def test_status_and_errors_go_to_stderr_and_only_the_report_to_stdout(tmp_path: Path) -> None:
    dataset = a_dataset(tmp_path / "d.jsonl", n_rows=24)
    result = invoke("run", str(dataset), "--results", str(tmp_path / "r"), "--boot", "100")

    assert result.exit_code == 0
    assert result.stdout.lstrip().startswith("# plumbline report")
    assert "rows read" in result.stderr and "artifact:" in result.stderr

    wrong = invoke("run", str(dataset), "--format", "csv")
    assert wrong.exit_code != 0
    assert "--format" in wrong.stderr and wrong.stdout == ""


def test_a_run_where_every_case_fails_exits_non_zero_and_says_why(tmp_path: Path) -> None:
    """The mock refuses an accuracy below chance on every case: one reason, said once."""
    dataset = a_dataset(tmp_path / "d.jsonl", n_rows=12)
    report = tmp_path / "report.md"
    result = invoke(
        "run",
        str(dataset),
        "--results",
        str(tmp_path / "r"),
        "--report",
        str(report),
        "--accuracy",
        "0.05",
        "--boot",
        "100",
    )

    assert result.exit_code == 1
    assert list((tmp_path / "r").glob("*.json")), "the artifact is still written"
    assert "every case failed" in result.stderr.lower()
    assert "accuracy" in result.stderr
    assert "for the same reason" in report.read_text(encoding="utf-8")


def test_a_dataset_outside_the_working_directory_is_named_not_located(tmp_path: Path) -> None:
    """An absolute path in a shared report names the user and their layout (#46)."""
    dataset = a_dataset(tmp_path / "d.jsonl", n_rows=12)
    report = tmp_path / "report.md"
    result = invoke(
        "run",
        str(dataset),
        "--results",
        str(tmp_path / "r"),
        "--report",
        str(report),
        "--boot",
        "100",
    )

    assert result.exit_code == 0
    assert str(tmp_path) not in report.read_text(encoding="utf-8")
    artifact = json.loads(next((tmp_path / "r").glob("*.json")).read_text(encoding="utf-8"))
    assert artifact["config"]["dataset"] == "d.jsonl"


def test_report_refuses_mixed_datasets_unless_allowed(tmp_path: Path) -> None:
    for name, rows in (("a", 12), ("b", 13)):
        invoke(
            "run",
            str(a_dataset(tmp_path / f"{name}.jsonl", n_rows=rows)),
            "--results",
            str(tmp_path / "r"),
            "--boot",
            "100",
        )
    artifacts = [str(path) for path in sorted((tmp_path / "r").glob("*.json"))]

    refused = invoke("report", *artifacts, "--boot", "100")
    allowed = invoke("report", *artifacts, "--boot", "100", "--allow-mixed")

    assert refused.exit_code == 1 and "different datasets" in refused.output
    assert allowed.exit_code == 0 and "2 datasets" in allowed.stdout
