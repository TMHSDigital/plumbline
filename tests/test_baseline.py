"""Null bands for the figures whose floor is not a calibration floor.

Accuracy and AUROC are read against a null the same way ECE is: a number with no
sample size and no null beside it cannot be acted on. Accuracy's null is chance
on this exact mix of option counts, which is not 50% and not 25% but whatever
the dataset's own widths make it. AUROC's null is a score column carrying no
information about the outcome, which is 0.5 in expectation and further from it
than people expect at 100 rows.
"""

from __future__ import annotations

import pytest

from plumbline.metrics import baseline
from plumbline.types import ConfidenceSeries

N_BOOT = 400


def test_chance_on_four_option_cases_is_one_in_four() -> None:
    band = baseline.chance_band([4] * 200, n_boot=N_BOOT)

    assert band.mean == pytest.approx(0.25, abs=0.02)
    assert band.n == 200


def test_chance_follows_the_mix_of_option_counts_not_a_fixed_guess() -> None:
    """Half two-option rows and half five-option rows is not chance of one in four."""
    band = baseline.chance_band([2] * 100 + [5] * 100, n_boot=N_BOOT)

    assert band.mean == pytest.approx(0.35, abs=0.02)


def test_the_chance_band_widens_as_the_sample_shrinks() -> None:
    small = baseline.chance_band([4] * 40, n_boot=N_BOOT, seed=3)
    large = baseline.chance_band([4] * 4000, n_boot=N_BOOT, seed=3)

    assert small.p95 - small.mean > large.p95 - large.mean


def test_an_accuracy_figure_states_its_rows_and_its_null() -> None:
    figure = baseline.accuracy_figure([True] * 80 + [False] * 20, [4] * 100, n_boot=N_BOOT)

    assert figure.value == pytest.approx(0.8)
    assert figure.n == 100
    statement = figure.statement()
    assert "100 rows" in statement
    assert "0.8" in statement
    assert "chance" in statement


def test_an_accuracy_that_beats_chance_says_so() -> None:
    figure = baseline.accuracy_figure([True] * 80 + [False] * 20, [4] * 100, n_boot=N_BOOT)

    assert figure.is_distinguishable
    assert "better than chance" in figure.statement()


def test_an_accuracy_at_chance_is_not_dressed_up_as_a_result() -> None:
    figure = baseline.accuracy_figure([True] * 25 + [False] * 75, [4] * 100, n_boot=N_BOOT)

    assert not figure.is_distinguishable
    statement = figure.statement()
    assert "INCONCLUSIVE" in statement
    assert "cannot tell the two apart" in statement


def test_an_inconclusive_statement_does_not_read_as_a_pass() -> None:
    """The wording must not let a reader take "no difference found" as good news.

    The old wording was "not distinguishable from <null> at this sample size",
    which is what the arithmetic says and the opposite of what it means. A
    reader skimming a report saw their model compared against a null and no
    difference found, and read it as a clean result.
    """
    figure = baseline.accuracy_figure([True] * 25 + [False] * 75, [4] * 100, n_boot=N_BOOT)
    statement = figure.statement().lower()

    assert "not a result in either direction" in statement
    for pass_like in ("passes", "acceptable", "looks fine", "no issue", "is calibrated"):
        assert pass_like not in statement


def test_auroc_of_an_uninformative_score_is_not_distinguishable_from_the_null() -> None:
    scores = ConfidenceSeries(values=tuple((index % 10) / 10 for index in range(200)))
    outcomes = [index % 2 == 0 for index in range(200)]

    figure = baseline.auroc_figure(scores, outcomes, n_boot=N_BOOT)

    assert not figure.is_distinguishable
    assert figure.band.mean == pytest.approx(0.5, abs=0.02)


def test_auroc_of_a_perfect_score_column_is_distinguishable() -> None:
    values = tuple(0.9 if index < 100 else 0.1 for index in range(200))
    scores = ConfidenceSeries(values=values)
    outcomes = [index < 100 for index in range(200)]

    figure = baseline.auroc_figure(scores, outcomes, n_boot=N_BOOT)

    assert figure.value == pytest.approx(1.0)
    assert figure.is_distinguishable
    assert "200 rows" in figure.statement()


def test_a_figure_refuses_to_be_built_from_mismatched_columns() -> None:
    with pytest.raises(ValueError, match="must match"):
        baseline.accuracy_figure([True, False], [4], n_boot=N_BOOT)
