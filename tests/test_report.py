"""The markdown report: what it groups, what it refuses to put side by side.

Two rules the rest of this file exists to protect.

Nothing prints as a bare number. Every figure carries the row count it was
computed on and the null it is read against, because a calibration number
without its sample size cannot be acted on and an accuracy without its chance
baseline cannot either.

A restricted softmax is never placed next to a calibrated claim without the
label between them. The whole point of the tool is that those two numbers are
not the same kind of number, and a table that lists them together invites
exactly the comparison it should prevent.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import ClassVar

import pytest
from typesafe_sdk import NoulAnswer, SystemOneResponse, Usage

from plumbline.adapters.mock import MockAdapter
from plumbline.adapters.typesafe_wire import QUESTION_NAME, TypeSafeWireAdapter
from plumbline.datasets import loader
from plumbline.report import markdown
from plumbline.runner import execute
from plumbline.types import Case
from tests.helpers import gold_by_text, make_cases

LABELS = ("billing", "returns", "shipping", "other")
PUBLIC_FIXTURE = Path(__file__).resolve().parent.parent / "datasets/public/jevbench-hard.jsonl"
OPTIONS = markdown.ReportOptions(n_boot=200, today=date(2026, 9, 20))


def a_run(
    *,
    semantics: str = "calibrated_claim",
    name: str = "mock",
    n_cases: int = 120,
    cases: list[Case] | None = None,
    **config: object,
) -> execute.RunResult:
    cases = cases if cases is not None else make_cases(n_cases, labels=LABELS)
    adapter = MockAdapter(
        gold_by_text(cases),
        name=name,
        seed=5,
        accuracy=0.8,
        probability_semantics=semantics,  # type: ignore[arg-type]
        **config,  # type: ignore[arg-type]
    )
    return execute.run(adapter, cases, workers=1)


class FakeNoulClient:
    """The wire adapter's transport, answering every case with a bare P(yes)."""

    def __init__(self, noul: float = 0.7) -> None:
        self.response = SystemOneResponse(
            model="jev-1.2",
            usage=Usage(input_tokens=100, output_tokens=2),
            answers={QUESTION_NAME: NoulAnswer(noul=noul)},
        )

    def system_one(self, state: object, questions: object, **kwargs: object) -> SystemOneResponse:
        return self.response


def yes_no_cases(n_cases: int = 40) -> list[Case]:
    return [
        Case(
            id=f"yn-{index}",
            text=f"is this one true? {index}",
            labels=("no", "yes"),
            gold_label="yes" if index % 2 else "no",
            question_type="noul",
        )
        for index in range(n_cases)
    ]


def render(*results: execute.RunResult, **kwargs: object) -> str:
    return markdown.render(list(results), options=OPTIONS, **kwargs)  # type: ignore[arg-type]


# Grouping


def test_each_arm_is_filed_under_the_kind_of_number_it_reports() -> None:
    text = render(
        a_run(semantics="calibrated_claim", name="wire"),
        a_run(semantics="restricted_softmax", name="local"),
        a_run(semantics="none", name="generative"),
    )

    assert "## Calibrated claims" in text
    assert "## Restricted softmax" in text
    assert "## No probability reported" in text


def test_a_restricted_softmax_arm_is_never_adjacent_to_a_calibrated_one() -> None:
    """The group label always sits between them, whatever order they arrive in."""
    text = render(
        a_run(semantics="restricted_softmax", name="local"),
        a_run(semantics="calibrated_claim", name="wire"),
    )

    first = text.index("### local")
    second = text.index("### wire")
    between = text[min(first, second) : max(first, second)]

    assert "## " in between


def test_the_report_says_that_the_groups_are_not_comparable() -> None:
    text = render(a_run(semantics="calibrated_claim"), a_run(semantics="restricted_softmax"))

    assert "not comparable" in text.lower()


# No bare numbers


@pytest.mark.parametrize("figure", ["ACCURACY", "ECE"])
def test_every_headline_figure_states_the_rows_it_was_computed_on(figure: str) -> None:
    text = render(a_run())

    lines = [line for line in text.splitlines() if figure in line.upper()]
    assert lines
    assert all("rows" in line for line in lines)


def test_accuracy_is_read_against_chance_on_this_mix_of_options() -> None:
    text = render(a_run())

    line = next(line for line in text.splitlines() if "ccuracy" in line)
    assert "chance" in line
    assert "0.25" in line  # four options on every case


def test_calibration_is_read_against_its_floor() -> None:
    text = render(a_run())

    line = next(line for line in text.splitlines() if "ECE" in line)
    assert "floor" in line
    assert "INCONCLUSIVE" in line or "distinguishable from sampling noise" in line


def test_the_preamble_says_inconclusive_is_not_a_pass() -> None:
    """The report defines its own terms, because the report is what gets read."""
    text = render(a_run())

    assert '**"INCONCLUSIVE" is not a pass.**' in text
    assert "Nothing was established in either direction." in text


def test_the_maximum_error_is_a_diagnostic_and_not_a_headline() -> None:
    """MCE cannot see gross overconfidence at a few hundred rows. Demote it."""
    text = render(a_run())

    assert "Diagnostics" in text
    assert text.index("ECE") < text.index("Diagnostics")
    assert text.index("Diagnostics") < text.index("MCE")


# Absences, said out loud


def test_an_arm_with_no_probability_is_not_reported_rather_than_scored() -> None:
    text = render(a_run(semantics="none", name="generative"))

    section = text[text.index("### generative") :]
    assert "not reported" in section
    assert "ECE" not in section


def test_confidence_absent_by_construction_is_named_as_such() -> None:
    """A noul has no distribution to summarize, so the blank is a fact, not a gap."""
    cases = yes_no_cases()
    adapter = TypeSafeWireAdapter(client=FakeNoulClient())  # type: ignore[arg-type]
    text = render(execute.run(adapter, cases, workers=1))

    line = next(line for line in text.splitlines() if "onfidence" in line)
    assert "not reported" in line


def test_a_row_asked_as_something_other_than_what_it_is_says_so() -> None:
    text = render(a_run(cases=yes_no_cases(30), name="local"))

    assert "asked as" in text.lower()
    assert "30" in text


def test_rows_of_an_unsupported_type_are_excluded_and_counted() -> None:
    report = loader.load_jevbench(PUBLIC_FIXTURE)
    text = render(a_run(cases=list(report.scoreable)[:20]), load=report)

    assert "ordinal" in text
    assert "111 rows read" in text


# Cost, latency, provenance


def test_cost_says_which_kind_of_blank_it_is() -> None:
    text = render(a_run(report_tokens=False))

    section = text[text.index("Cost") :]
    assert "reports no token counts at all" in section


def test_the_report_carries_the_dataset_hash_and_the_row_count() -> None:
    result = a_run(n_cases=30)
    text = render(result)

    assert result.dataset_hash[:8] in text
    assert "30" in text


# Recalibration: verdicts, not temperatures with caveats


def skewed(result: execute.RunResult, temperature_for) -> execute.RunResult:
    """Re-skew a finished run per case, without moving any predicted label."""
    from dataclasses import replace

    from plumbline.types import apply_temperature, docs_confidence

    records = []
    for record in result.records:
        prediction = record.prediction
        if prediction is None or prediction.distribution is None:
            records.append(record)
            continue
        distribution = apply_temperature(prediction.distribution, temperature_for(prediction))
        records.append(
            replace(
                record,
                prediction=replace(
                    prediction,
                    distribution=distribution,
                    prob_selected=distribution[prediction.label],
                    confidence=docs_confidence(distribution),
                ),
            )
        )
    return replace(result, records=records)


def overconfident(temperature: float = 0.5):
    def temperature_for(prediction) -> float:
        return temperature

    return temperature_for


def per_label(labels: tuple[str, ...], temperature: float = 0.4):
    def temperature_for(prediction) -> float:
        return temperature if prediction.label in labels else 1.0

    return temperature_for


def section(text: str, heading: str) -> str:
    start = text.index(f"#### {heading}")
    rest = text[start + 1 :]
    end = rest.find("####")
    return rest if end == -1 else rest[:end]


def test_a_recommended_fit_prints_the_temperature_with_its_interval() -> None:
    result = skewed(a_run(n_cases=600), overconfident())

    body = section(render(result), "Recalibration")

    assert "Recommended" in body
    assert "Fitted T" in body
    assert "interval" in body


def test_a_fit_states_the_split_it_was_fitted_and_judged_on() -> None:
    body = section(render(skewed(a_run(n_cases=600), overconfident())), "Recalibration")

    assert "300 rows" in body  # fit half
    assert "300 held-out rows" in body


def test_a_refused_fit_emits_no_number_anywhere_except_the_split() -> None:
    """A number that ships is a number that gets hardcoded. Refused means none."""
    import re

    body = section(render(a_run(n_cases=600)), "Recalibration")
    without_split = "\n".join(line for line in body.splitlines() if "held-out rows" not in line)

    assert "Refused" in body
    assert "Fitted T" not in body
    assert not re.search(r"\d\.\d", without_split)


def test_a_refused_fit_still_says_what_it_was_fitted_on() -> None:
    body = section(render(a_run(n_cases=600)), "Recalibration")

    assert "300 rows" in body
    assert "300 held-out rows" in body


def test_a_partial_fit_prints_the_residual_beside_the_temperature() -> None:
    """Temperature was the wrong shape for part of it. Say what survived."""
    result = skewed(a_run(n_cases=600), per_label(("billing", "returns")))

    body = section(render(result), "Recalibration")

    assert "Partial" in body or "Refused" in body
    if "Partial" in body:
        assert "Fitted T" in body
        assert "floor" in body
        assert "after" in body


def test_recalibration_is_not_reported_for_an_arm_with_no_probability() -> None:
    body = section(render(a_run(semantics="none", name="generative")), "Recalibration")

    assert "not reported" in body.lower()
    assert "Fitted T" not in body


def test_recalibration_is_not_reported_when_the_held_out_half_is_too_small() -> None:
    body = section(render(a_run(n_cases=60)), "Recalibration")

    assert "not reported" in body.lower()
    assert "200" in body  # the rule it failed


# The cascade: one sentence, or none


def with_costs(escalation: float = 0.02, error: float = 1.0) -> markdown.ReportOptions:
    return markdown.ReportOptions(
        n_boot=200,
        today=date(2026, 9, 20),
        cost_escalation_usd=escalation,
        cost_error_usd=error,
    )


def test_the_cascade_ends_in_a_sentence_a_person_can_act_on() -> None:
    result = skewed(a_run(n_cases=600), overconfident())

    body = section(markdown.render([result], options=with_costs()), "Cascade")

    assert "stays on the cheap arm" in body
    assert "versus" in body
    assert "$" in body


def test_the_cascade_says_so_when_escalating_everything_is_cheapest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A threshold of infinity is not a number to paste into a config (#34)."""
    from plumbline.metrics import cascade

    def escalate_all(series, outcomes, escalation, error):  # type: ignore[no-untyped-def]
        # The sweep's last row is the one above every score.
        return cascade.cascade_sweep(series, outcomes, escalation, error)[-1]

    monkeypatch.setattr(markdown.cascade, "optimal_threshold", escalate_all)
    result = skewed(a_run(n_cases=600), overconfident())

    body = section(markdown.render([result], options=with_costs()), "Cascade")

    assert "Escalating every case is cheapest" in body
    assert "inf" not in body


def test_the_cascade_needs_the_two_numbers_no_benchmark_can_know() -> None:
    body = section(render(a_run(n_cases=600)), "Cascade")

    assert "not reported" in body.lower()
    assert "escalation" in body.lower()
    assert "stays on the cheap arm" not in body


def test_without_a_temperature_the_threshold_is_still_scored_on_held_out_rows() -> None:
    """A threshold chosen and scored on the same rows is in-sample, whatever it says (#30).

    A calibrated arm gets no temperature, so there is no recalibrated scale; the
    threshold must still be picked on one half and reported on the other, and
    the 200-row minimum counts the held-out half. 300 rows leave 150 held out.
    """
    body = section(markdown.render([a_run(n_cases=300)], options=with_costs()), "Cascade")

    assert "not reported" in body.lower()
    assert "held-out" in body
    assert "stays on the cheap arm" not in body


def test_the_cascade_says_where_its_threshold_was_chosen_and_where_it_was_scored() -> None:
    body = section(markdown.render([a_run(n_cases=600)], options=with_costs()), "Cascade")

    assert "stays on the cheap arm" in body
    assert "chosen on 300 rows" in body
    assert "300 held-out rows" in body


def test_a_threshold_is_refused_on_an_evaluation_set_too_small_to_support_one() -> None:
    body = section(markdown.render([a_run(n_cases=40)], options=with_costs()), "Cascade")

    assert "not reported" in body.lower()
    assert "stays on the cheap arm" not in body


def test_the_cascade_says_which_scale_the_threshold_is_on() -> None:
    result = skewed(a_run(n_cases=600), overconfident())

    body = section(markdown.render([result], options=with_costs()), "Cascade")

    assert "recalibrated" in body.lower()


def test_the_cascade_is_not_reported_for_an_arm_with_no_probability() -> None:
    body = section(
        markdown.render([a_run(semantics="none", name="generative")], options=with_costs()),
        "Cascade",
    )

    assert "not reported" in body.lower()


def test_the_artifact_and_the_report_name_a_non_default_endpoint() -> None:
    """Which server answered is part of what was measured (#40)."""
    cases = make_cases(40, labels=LABELS)
    adapter = MockAdapter(gold_by_text(cases), seed=5, accuracy=0.8)
    adapter.base_url = "http://self-hosted.example"  # type: ignore[attr-defined]
    result = execute.run(adapter, cases, workers=1)

    assert result.config["endpoint"] == "http://self-hosted.example"
    assert "endpoint `http://self-hosted.example`" in render(result)
    # The vendor's default endpoint is the ordinary case and says nothing.
    assert "endpoint `" not in render(a_run(n_cases=40))


class FailingAdapter:
    """Fails every case, with the same message or with one per case."""

    name = "failing"
    model_requested = "failing-1"
    revision = None
    probability_semantics = "calibrated_claim"
    reports_tokens = False
    call_params: ClassVar[dict[str, object]] = {}

    def __init__(self, same: bool) -> None:
        self.same = same

    def classify(self, text: str, labels: list[str], **asked: object) -> object:
        raise ValueError("the local extra is not installed" if self.same else f"broke on {text}")


@pytest.mark.parametrize("same", [True, False], ids=["one-reason", "many-reasons"])
def test_a_run_with_no_figures_names_a_shared_failure_reason(same: bool) -> None:
    """One reason is the thing the reader needs; many reasons are a log (#15)."""
    cases = make_cases(8, labels=LABELS)
    text = render(execute.run(FailingAdapter(same), cases, workers=1))  # type: ignore[arg-type]

    if same:
        assert "all 8 cases failed for the same reason" in text
        assert "the local extra is not installed" in text
    else:
        assert "8 different reasons, are in the artifact" in text
        assert "broke on" not in text
