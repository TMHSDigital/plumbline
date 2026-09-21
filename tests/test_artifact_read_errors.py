"""Reading a results artifact back refuses with a sentence, not a traceback.

Found by the fresh-install pass, alongside the adapter-config bug and in the
same class as it: ``plumbline report results/nope.json`` printed a raw
``FileNotFoundError`` traceback. Naming the wrong file is an ordinary mistake,
and a stack trace is the right output for a bug and the wrong output for a typo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plumbline.adapters.mock import MockAdapter
from plumbline.runner import execute
from plumbline.runner.execute import RunResult
from plumbline.types import ArtifactError
from tests.helpers import gold_by_text, make_cases


def test_a_missing_artifact_says_so(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="no artifact at"):
        RunResult.read(tmp_path / "absent.json")


def test_a_file_that_is_not_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "notjson.md"
    path.write_text("# a readme, not an artifact\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="not valid JSON"):
        RunResult.read(path)


def test_json_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ArtifactError, match="not an artifact object"):
        RunResult.read(path)


def test_json_of_the_wrong_shape_names_what_was_missing(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")

    with pytest.raises(ArtifactError, match="not a plumbline artifact"):
        RunResult.read(path)


def test_a_directory_is_refused_rather_than_crashing(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        RunResult.read(tmp_path)


def test_a_real_artifact_still_round_trips(tmp_path: Path) -> None:
    """The error handling must not have broken the path that works."""
    cases = make_cases(6)
    adapter = MockAdapter(gold_by_text=gold_by_text(cases), seed=7)
    original = execute.run(adapter, cases, workers=1)

    restored = RunResult.read(original.write(tmp_path))

    assert restored.adapter_name == original.adapter_name
    assert restored.dataset_hash == original.dataset_hash
    assert len(restored.records) == len(original.records)
