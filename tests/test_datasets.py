"""Dataset loading: what is read, what is refused, and what is said about both.

The rule the rest of this file exists to protect: a row whose gold label is not
one of its own options is refused, never loaded. Scoring such a row is
guaranteed to mark every system wrong on it, which reads as a model failure and
is a dataset typo. plumbline would rather run 110 rows and say so than run 111
and quietly hand out a penalty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plumbline.datasets import loader
from plumbline.types import DatasetError

PUBLIC_FIXTURE = Path(__file__).resolve().parent.parent / "datasets/public/jevbench-hard.jsonl"


def a_row(**overrides: object) -> dict:
    row = {
        "id": "case-1",
        "text": "the invoice is wrong",
        "labels": ["billing", "returns", "shipping"],
        "gold_label": "billing",
    }
    row.update(overrides)
    return row


def write_jsonl(path: Path, rows: list[object]) -> Path:
    path.write_text(
        "\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


# plumbline's own JSONL


def test_a_well_formed_file_loads_every_row(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "d.jsonl",
        [a_row(id="a"), a_row(id="b", gold_label="returns")],
    )

    report = loader.load_jsonl(path)

    assert report.row_count == 2
    assert report.rows_read == 2
    assert [case.id for case in report.cases] == ["a", "b"]
    assert report.is_complete


def test_a_gold_label_that_is_not_one_of_the_options_is_refused(tmp_path: Path) -> None:
    """The headline rule. A typo here is otherwise scored as a model failure."""
    path = write_jsonl(tmp_path / "d.jsonl", [a_row(), a_row(id="typo", gold_label="biling")])

    report = loader.load_jsonl(path)

    assert [case.id for case in report.cases] == ["case-1"]
    assert len(report.refusals) == 1
    refusal = report.refusals[0]
    assert refusal.case_id == "typo"
    assert refusal.row == 2
    assert "biling" in refusal.reason
    assert "billing" in refusal.reason


def test_a_refusal_never_reaches_the_cases(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "d.jsonl", [a_row(id="typo", gold_label="nope")])

    report = loader.load_jsonl(path)

    assert report.cases == ()
    assert report.row_count == 0
    assert not report.is_complete


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"labels": ["only"]}, "at least 2"),
        ({"labels": ["a", "a", "b"], "gold_label": "a"}, "duplicate"),
        ({"text": "   "}, "text"),
        ({"labels": "billing"}, "labels"),
    ],
)
def test_a_row_that_cannot_be_a_case_is_refused_with_the_reason(
    tmp_path: Path, overrides: dict, expected: str
) -> None:
    path = write_jsonl(tmp_path / "d.jsonl", [a_row(**overrides)])

    report = loader.load_jsonl(path)

    assert report.cases == ()
    assert expected in report.refusals[0].reason


def test_a_missing_field_is_refused_by_name(tmp_path: Path) -> None:
    row = a_row()
    del row["gold_label"]
    path = write_jsonl(tmp_path / "d.jsonl", [row])

    report = loader.load_jsonl(path)

    assert "gold_label" in report.refusals[0].reason


def test_a_malformed_line_is_refused_with_its_line_number(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "d.jsonl", [a_row(), "{not json", a_row(id="c")])

    report = loader.load_jsonl(path)

    assert report.refusals[0].row == 2
    assert [case.id for case in report.cases] == ["case-1", "c"]


def test_blank_lines_are_skipped_rather_than_refused(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "d.jsonl", [a_row(), "", "   ", a_row(id="c")])

    report = loader.load_jsonl(path)

    assert report.row_count == 2
    assert report.refusals == ()


def test_a_repeated_id_is_refused_because_results_are_keyed_by_it(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "d.jsonl", [a_row(id="same"), a_row(id="same")])

    report = loader.load_jsonl(path)

    assert report.row_count == 1
    assert "duplicate id" in report.refusals[0].reason


def test_an_empty_file_is_an_error_rather_than_an_empty_run(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "d.jsonl", [])

    with pytest.raises(DatasetError, match="no rows"):
        loader.load_jsonl(path)


def test_a_missing_file_says_which_path_it_looked_at(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"missing\.jsonl"):
        loader.load_jsonl(tmp_path / "missing.jsonl")


def test_the_report_states_what_it_read_and_what_it_refused(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "d.jsonl", [a_row(), a_row(id="typo", gold_label="nope")])

    statement = loader.load_jsonl(path).statement()

    assert "2 rows read" in statement
    assert "1 loaded" in statement
    assert "1 refused" in statement


def test_a_caller_can_demand_a_clean_load(tmp_path: Path) -> None:
    """A run that must cover the whole dataset should not discover a gap later."""
    path = write_jsonl(tmp_path / "d.jsonl", [a_row(), a_row(id="typo", gold_label="nope")])
    report = loader.load_jsonl(path)

    with pytest.raises(DatasetError, match="typo"):
        report.require_complete()


# The JevBench public fixture


def test_the_public_fixture_loads_every_row() -> None:
    report = loader.load_jevbench(PUBLIC_FIXTURE)

    assert report.rows_read == 111
    assert report.row_count == 111
    assert report.refusals == ()


def test_a_numeric_gold_is_matched_to_its_option_and_the_coercion_is_recorded() -> None:
    """Six scored rows write gold as a JSON number against string options.

    ``1`` and ``"1"`` are the same option, so the row is loaded rather than
    refused, and the count of rows that needed it is reported rather than
    silently absorbed.
    """
    report = loader.load_jevbench(PUBLIC_FIXTURE)

    assert report.normalized_gold_rows == 6
    assert any("6" in note and "number" in note for note in report.notes)
    scored = next(case for case in report.cases if case.id == "hard-sol-a-trap-05")
    assert scored.gold_label == "0"
    assert scored.gold_label in scored.labels


def test_option_descriptions_come_from_the_criteria_that_describe_the_options() -> None:
    case = next(
        case
        for case in loader.load_jevbench(PUBLIC_FIXTURE).cases
        if case.id == "hard-opus-a-long_policy-01"
    )

    assert case.label_descriptions is not None
    assert set(case.label_descriptions) == set(case.labels)


def test_criteria_that_do_not_describe_the_options_are_dropped_and_said_so() -> None:
    """The yes/no rows describe the statement, not each option. Mapping their
    true/false keys onto yes/no would be a guess about someone else's file."""
    report = loader.load_jevbench(PUBLIC_FIXTURE)
    case = next(case for case in report.cases if set(case.labels) == {"yes", "no"})

    assert case.label_descriptions is None
    assert any("criteria" in note for note in report.notes)


def test_the_case_text_carries_the_question_and_the_state() -> None:
    case = next(
        case
        for case in loader.load_jevbench(PUBLIC_FIXTURE).cases
        if case.id == "hard-opus-a-long_policy-01"
    )

    assert "coverage reviewer" in case.text  # from the question
    assert "HARBORLINE MUTUAL" in case.text  # from the state


def test_a_structured_state_is_rendered_the_same_way_every_time() -> None:
    first = loader.load_jevbench(PUBLIC_FIXTURE).cases
    second = loader.load_jevbench(PUBLIC_FIXTURE).cases
    structured = next(case for case in first if case.id == "hard-opus-b-ambiguous-10")

    assert "agent_policy" in structured.text
    assert [case.text for case in first] == [case.text for case in second]


def test_the_report_counts_the_question_types_it_was_handed() -> None:
    report = loader.load_jevbench(PUBLIC_FIXTURE)

    assert report.question_types == {"choice": 67, "noul": 38, "score": 6}


def test_a_jevbench_row_whose_gold_is_not_an_option_is_refused_too(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "j.jsonl",
        [
            {
                "id": "made-up",
                "expected": "maybe",
                "labels": ["yes", "no"],
                "question": {"type": "noul", "instructions": "Well?", "criteria": {}},
                "state": "something happened",
            }
        ],
    )

    report = loader.load_jevbench(path)

    assert report.cases == ()
    assert "maybe" in report.refusals[0].reason


# What kind of question each row asks


def test_a_yes_no_row_is_carried_through_as_a_noul() -> None:
    """38 of these rows exist, and asking them as choices is a different question."""
    report = loader.load_jevbench(PUBLIC_FIXTURE)
    noul_cases = [case for case in report.cases if case.question_type == "noul"]

    assert len(noul_cases) == 38
    assert all(set(case.labels) == {"yes", "no"} for case in noul_cases)


def test_choice_rows_stay_choices() -> None:
    report = loader.load_jevbench(PUBLIC_FIXTURE)

    assert len([case for case in report.cases if case.question_type == "choice"]) == 67


def test_score_rows_are_loaded_as_scoreable_with_their_rubric() -> None:
    """A score row is scored by rank now, and its rubric describes its levels."""
    report = loader.load_jevbench(PUBLIC_FIXTURE)
    score_cases = [case for case in report.cases if case.question_type == "score"]

    assert len(score_cases) == 6
    assert report.row_count == 111
    assert all(case.is_scoreable for case in score_cases)
    roster = next(case for case in score_cases if case.id == "hard-opus-a-temporal_numeric-12")
    assert roster.label_descriptions == {
        "0": "No violations",
        "1": "Exactly one violation",
        "2": "Exactly two violations",
        "3": "Three or more violations",
    }


def test_the_notes_say_score_rows_are_read_by_rank_apart_from_the_choice_figures() -> None:
    report = loader.load_jevbench(PUBLIC_FIXTURE)

    assert len(report.scoreable) == 111
    assert report.unsupported_by_type == {}
    note = " ".join(report.notes)
    assert "6 rows ask for an ordinal score" in note and "by rank" in note


def test_a_score_row_whose_options_are_not_levels_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "s1",
                "text": "rate it",
                "labels": ["low", "high"],
                "gold_label": "low",
                "question_type": "score",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = loader.load_jsonl(path)

    assert not report.cases
    assert "levels" in str(report.refusals[0])


def test_our_own_jsonl_can_declare_the_question_type(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "d.jsonl",
        [a_row(labels=["no", "yes"], gold_label="yes", question_type="noul")],
    )

    report = loader.load_jsonl(path)

    assert report.cases[0].question_type == "noul"


def test_an_unknown_question_type_is_refused_rather_than_assumed_to_be_a_choice(
    tmp_path: Path,
) -> None:
    path = write_jsonl(tmp_path / "d.jsonl", [a_row(question_type="ranking")])

    report = loader.load_jsonl(path)

    assert report.cases == ()
    assert "question_type" in report.refusals[0].reason


def test_the_example_in_the_dataset_docs_loads_as_documented(tmp_path) -> None:
    """docs/datasets.md shows the format; the format it shows must be the one the loader reads."""
    import re
    from pathlib import Path

    doc = Path(__file__).resolve().parent.parent / "docs" / "datasets.md"
    block = re.search(r"```jsonl\n(.*?)```", doc.read_text(encoding="utf-8"), re.S)
    assert block is not None, "docs/datasets.md has no jsonl example"
    path = tmp_path / "example.jsonl"
    path.write_text(block.group(1), encoding="utf-8")

    report = loader.load_jsonl(path)

    assert not report.refusals
    kinds = sorted(case.question_type for case in report.cases)
    assert kinds == ["choice", "choice", "choice", "noul", "score"]
    assert len(report.scoreable) == 5  # the score row is scored by rank


# Validation the loaders were missing (#44, #45)


def jevbench_row(case_id: str) -> dict:
    return {
        "id": case_id,
        "expected": "yes",
        "labels": ["yes", "no"],
        "question": {"type": "noul", "instructions": "Well?", "criteria": {}},
        "state": f"something happened to {case_id}",
    }


def test_the_jevbench_loader_refuses_a_duplicate_id_like_the_jsonl_one(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "j.jsonl", [jevbench_row("dup"), jevbench_row("dup")])

    report = loader.load_jevbench(path)

    assert len(report.cases) == 1
    assert len(report.refusals) == 1 and "duplicate id" in report.refusals[0].reason


def test_a_byte_order_mark_is_read_as_utf8_rather_than_refused(tmp_path: Path) -> None:
    """Windows editors and spreadsheet exports write one; the first row must still load."""
    path = tmp_path / "bom.jsonl"
    path.write_text(json.dumps(a_row()) + "\n", encoding="utf-8-sig")

    report = loader.load_jsonl(path)

    assert len(report.cases) == 1 and not report.refusals


def test_a_file_that_is_not_utf8_names_the_file_and_the_line(tmp_path: Path) -> None:
    path = tmp_path / "latin1.jsonl"
    path.write_bytes(
        (json.dumps(a_row()) + "\n").encode("utf-8")
        # ensure_ascii=False keeps the é as a character, so latin-1 writes it as 0xE9.
        + (json.dumps(a_row(id="two", text="café"), ensure_ascii=False) + "\n").encode("latin-1")
    )

    with pytest.raises(DatasetError, match=r"latin1\.jsonl.*line 2"):
        loader.load_jsonl(path)


@pytest.mark.parametrize(
    ("labels", "gold"),
    [
        ([None, "a"], "a"),
        ([True, False], "True"),
        (["a", ""], "a"),
        (["a", "   "], "a"),
        ([1, 2], "1"),
    ],
    ids=["null", "bools", "empty", "blank", "numbers"],
)
def test_every_label_must_be_a_non_empty_string(
    tmp_path: Path, labels: list[object], gold: str
) -> None:
    """str() used to turn null into 'None' and true into 'True', and load the row."""
    path = write_jsonl(tmp_path / "d.jsonl", [a_row(labels=labels, gold_label=gold)])

    report = loader.load_jsonl(path)

    assert not report.cases
    assert "non-empty string" in report.refusals[0].reason


@pytest.mark.parametrize(
    "descriptions",
    [{"zzz": "not an option"}, {"billing": None}, {"billing": 3}],
    ids=["unknown-key", "null-value", "number-value"],
)
def test_label_descriptions_must_describe_the_options_in_words(
    tmp_path: Path, descriptions: dict[str, object]
) -> None:
    path = write_jsonl(tmp_path / "d.jsonl", [a_row(label_descriptions=descriptions)])

    report = loader.load_jsonl(path)

    assert not report.cases
    assert "label_descriptions" in report.refusals[0].reason
