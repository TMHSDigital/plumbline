"""The report states the grid probabilities arrived on (issue #9).

Hosted Jev rounds to two decimals. A reader has to know that to know what a
small difference is worth, and should not have to find it in METHODOLOGY.
"""

from __future__ import annotations

from plumbline.metrics.calibration import probability_grid
from plumbline.report import markdown
from plumbline.runner import execute
from plumbline.types import Case, Prediction
from tests.test_ties import a_prediction


def test_the_coarsest_grid_every_value_sits_on_is_found() -> None:
    assert probability_grid([0.25, 0.5, 0.75] * 10) == 0.05
    assert probability_grid([0.23, 0.41, 0.07] * 10) == 0.01
    assert probability_grid([0.231, 0.4, 0.07] * 10) == 0.001
    assert probability_grid([0.2317, 0.4] * 20) is None


def test_too_few_values_say_nothing_about_a_grid() -> None:
    assert probability_grid([0.5, 0.25]) is None


class OnTheGrid:
    """Answers on a two-decimal grid, as the hosted vendor does."""

    name = "grid"
    model_requested = "grid-1"
    revision = None
    probability_semantics = "calibrated_claim"
    reports_tokens = False

    @property
    def call_params(self) -> dict[str, object]:
        return {}

    def classify(self, text: str, labels: list[str], **_kwargs: object) -> Prediction:
        index = int(text[1:])
        top = 0.5 + (index % 40) / 100
        return a_prediction({"a": round(top, 2), "b": round(1 - top, 2)}, "a")


def test_the_report_names_the_grid_it_observed() -> None:
    cases = [Case(id=f"c{i}", text=f"t{i}", labels=("a", "b"), gold_label="a") for i in range(40)]
    result = execute.run(OnTheGrid(), cases)  # type: ignore[arg-type]

    document = markdown.render([result], options=markdown.ReportOptions(n_boot=50))
    line = next(line for line in document.splitlines() if "**Resolution**" in line)

    assert "multiples of 0.01" in line
    assert "80 probabilities" in line
