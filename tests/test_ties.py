"""A tie for the top option, recorded and counted (issue #10).

Probabilities arrive on a two-decimal grid, so two options sharing the maximum
is ordinary. On such a row the vendor's tie-break chose the answer, not a
margin, and an auditor has to be able to see that.
"""

from __future__ import annotations

import json
from pathlib import Path

from plumbline.adapters.mock import MockAdapter
from plumbline.report import markdown
from plumbline.runner import execute
from plumbline.types import Case, Prediction


def a_prediction(distribution: dict[str, float], label: str) -> Prediction:
    return Prediction(
        label=label,
        prob_selected=distribution[label],
        distribution=distribution,
        confidence=None,
        latency_ms=1.0,
        cost_usd=None,
        input_tokens=None,
        output_tokens=None,
        model_reported=None,
    )


def test_the_options_tied_for_the_top_are_named() -> None:
    assert a_prediction({"a": 0.4, "b": 0.4, "c": 0.2}, "a").tied_for_top == ("a", "b")
    assert a_prediction({"a": 0.5, "b": 0.3, "c": 0.2}, "a").tied_for_top == ()
    no_distribution = a_prediction({"a": 1.0}, "a")
    assert no_distribution.tied_for_top == ()


class TiesOnOddRows(MockAdapter):
    """Picks "a" everywhere; on odd rows "a" and "b" tie at the top."""

    def classify(self, text: str, labels: list[str], **_kwargs: object) -> Prediction:
        tied = int(text[-1]) % 2 == 1
        spread = {"a": 0.4, "b": 0.4, "c": 0.2} if tied else {"a": 0.6, "b": 0.3, "c": 0.1}
        return a_prediction(spread, "a")


def test_a_tie_is_in_the_artifact_and_counted_in_the_report(tmp_path: Path) -> None:
    # Gold is "b" on every row, so each tied row went to the other tied option.
    cases = [
        Case(id=f"c{i}", text=f"t{i}", labels=("a", "b", "c"), gold_label="b") for i in range(10)
    ]
    result = execute.run(TiesOnOddRows(gold_by_text={}), cases)

    stored = json.loads(result.write(tmp_path).read_text(encoding="utf-8"))
    flagged = {row["case_id"]: row["tied_for_top"] for row in stored["records"]}
    assert flagged["c1"] == ["a", "b"] and flagged["c0"] == []

    document = markdown.render([result], options=markdown.ReportOptions(n_boot=50))
    ties = next(line for line in document.splitlines() if "**Ties**" in line)
    assert "5 of 10 scored rows" in ties
    assert "5 of them" in ties


def test_no_tie_line_when_nothing_tied() -> None:
    cases = [Case(id=f"c{i}", text=f"t{i}", labels=("a", "b"), gold_label="a") for i in range(6)]
    result = execute.run(MockAdapter(gold_by_text={case.text: "a" for case in cases}), cases)

    document = markdown.render([result], options=markdown.ReportOptions(n_boot=50))

    assert "**Ties**" not in document
