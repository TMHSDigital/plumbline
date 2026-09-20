"""The whole pipeline over a real file, with a mock in the model's chair.

The numbers this produces are meaningless: a seeded mock is answering. The point
is that loader, runner, metrics, and artifact compose on a file somebody else
wrote, with its real shapes and its real edge cases, before a live call costs
anything.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "examples/smoke_public_dataset.py"
FIXTURE = REPO / "datasets/public/jevbench-hard.jsonl"


@pytest.fixture(scope="module")
def smoke() -> ModuleType:
    """Import the example script the way a reader would run it."""
    spec = importlib.util.spec_from_file_location("smoke_public_dataset", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so the dataclass in it can resolve its own
    # annotations, exactly as a normal import would.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoked(smoke: ModuleType, tmp_path_factory: pytest.TempPathFactory):
    directory = tmp_path_factory.mktemp("smoke")
    return smoke.smoke(FIXTURE, directory, n_boot=200)


def test_every_row_of_the_public_fixture_reaches_the_runner(smoked) -> None:
    assert smoked.load.rows_read == 111
    assert smoked.load.row_count == 111
    assert len(smoked.result.records) == 111
    assert not smoked.result.failures


def test_the_summary_states_what_was_loaded_and_what_was_refused(smoked) -> None:
    assert "111 rows read" in smoked.summary
    assert "111 loaded" in smoked.summary
    assert "0 refused" in smoked.summary


def test_the_summary_puts_the_row_count_next_to_the_calibration_figure(smoked) -> None:
    assert "ECE" in smoked.summary
    assert "over 111 rows" in smoked.summary


def test_the_summary_says_why_there_is_no_cost(smoked) -> None:
    """A mock is not priced, and the summary says that rather than showing zero."""
    assert "not priced" in smoked.summary
    assert "$0" not in smoked.summary


def test_the_summary_refuses_to_let_a_mock_be_read_as_a_result(smoked) -> None:
    assert "mock" in smoked.summary.lower()
    assert "not a result" in smoked.summary.lower()


def test_the_artifact_lands_on_disk_with_the_rows_it_covered(smoked) -> None:
    stored = json.loads(smoked.artifact.read_text(encoding="utf-8"))

    assert stored["dataset_rows"] == 111
    assert stored["dataset_hash"]
    assert len(stored["records"]) == 111
    assert stored["probability_semantics"] == "calibrated_claim"


def test_the_script_runs_as_a_script(smoke: ModuleType, tmp_path: Path, capsys) -> None:
    exit_code = smoke.main(["--dataset", str(FIXTURE), "--results", str(tmp_path), "--boot", "200"])

    assert exit_code == 0
    assert "ECE" in capsys.readouterr().out
