"""Rank-aware figures for ordinal score rows (issue #6).

Every choice metric is rank-blind: wrong by one level and wrong by three score
the same. These figures are not, and each is read against a null the way every
other figure is, or it is a number nobody can read.
"""

from __future__ import annotations

import numpy as np
import pytest

from plumbline.metrics import ordinal
from plumbline.metrics.ordinal import ScoreAnswer

LEVELS = (0, 1, 2, 3)


def point_mass(level: int, gold: int) -> ScoreAnswer:
    probabilities = tuple(1.0 if value == level else 0.0 for value in LEVELS)
    return ScoreAnswer(levels=LEVELS, probabilities=probabilities, expected=float(level), gold=gold)


def test_levels_are_read_as_ordered_integers() -> None:
    assert ordinal.levels_of(("2", "0", "1", "3")) == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="integer"):
        ordinal.levels_of(("low", "high"))


def test_an_answer_is_built_from_a_distribution_or_a_single_level() -> None:
    from_distribution = ordinal.answer_from(
        ("0", "1", "2", "3"), {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}, label="3", gold_label="1"
    )
    assert from_distribution.expected == pytest.approx(0.2 + 0.6 + 1.2)
    assert from_distribution.gold == 1

    reported = ordinal.answer_from(
        ("0", "1", "2", "3"), {"0": 0.5, "1": 0.5, "2": 0.0, "3": 0.0}, "1", "0", expected=0.5
    )
    assert reported.expected == 0.5  # the vendor's expected score is read as given

    level_only = ordinal.answer_from(("0", "1", "2"), None, label="2", gold_label="0")
    assert level_only.probabilities is None and level_only.expected == 2.0


def test_the_ranked_probability_score_charges_by_distance() -> None:
    exact = ordinal.ranked_probability_score([point_mass(0, gold=0)])
    near = ordinal.ranked_probability_score([point_mass(1, gold=0)])
    far = ordinal.ranked_probability_score([point_mass(3, gold=0)])

    assert exact == 0.0
    assert near == pytest.approx(1 / 3)
    assert far == pytest.approx(1.0)


def test_each_row_is_one_event_per_threshold() -> None:
    answers = [point_mass(1, gold=2), point_mass(2, gold=2)]

    predicted, observed = ordinal.cumulative_events(answers)

    assert len(predicted) == len(observed) == 2 * (len(LEVELS) - 1)
    # Row one: all mass on level 1, so P(level <= k) is 0, 1, 1; gold 2 gives 0, 0, 1.
    assert list(predicted[:3]) == [0.0, 1.0, 1.0]
    assert list(observed[:3]) == [0.0, 0.0, 1.0]


def draws(n: int, seed: int, *, sharpen: float = 1.0) -> list[ScoreAnswer]:
    """Answers whose gold is drawn from the true distribution; ``sharpen`` above 1
    makes the stated distribution overconfident relative to it."""
    rng = np.random.default_rng(seed)
    answers = []
    for _ in range(n):
        truth = rng.dirichlet(np.full(len(LEVELS), 0.8))
        gold = int(rng.choice(len(LEVELS), p=truth))
        stated = truth**sharpen
        stated = stated / stated.sum()
        expected = float(np.dot(stated, LEVELS))
        answers.append(
            ScoreAnswer(levels=LEVELS, probabilities=tuple(stated), expected=expected, gold=gold)
        )
    return answers


def test_a_calibrated_arm_clears_the_floors_about_one_time_in_twenty() -> None:
    # A floor's 95th percentile is exceeded by a calibrated arm one time in
    # twenty by design, so one seed proves nothing; the rate across seeds does.
    over = {"rps": 0, "cumulative ece": 0}
    for seed in range(20):
        answers = draws(1000, seed=seed)
        floors = ordinal.score_floor(answers, n_boot=300, seed=seed)
        over["rps"] += ordinal.ranked_probability_score(answers) > floors["rps"].p95
        over["cumulative ece"] += (
            ordinal.cumulative_calibration_error(answers) > floors["cumulative ece"].p95
        )

    assert over["rps"] <= 4, over
    assert over["cumulative ece"] <= 4, over


def test_an_overconfident_arm_clears_both_floors() -> None:
    answers = draws(2000, seed=1, sharpen=3.0)

    floors = ordinal.score_floor(answers, n_boot=400, seed=0)

    assert ordinal.ranked_probability_score(answers) > floors["rps"].p95
    assert ordinal.cumulative_calibration_error(answers) > floors["cumulative ece"].p95


def test_an_informative_arm_beats_the_permutation_null_and_a_constant_one_does_not() -> None:
    informative = draws(500, seed=2, sharpen=1.0)
    middle = [
        ScoreAnswer(levels=a.levels, probabilities=a.probabilities, expected=1.5, gold=a.gold)
        for a in informative
    ]

    beats = ordinal.mean_absolute_error_figure(informative, n_boot=400, seed=0)
    constant = ordinal.mean_absolute_error_figure(middle, n_boot=400, seed=0)

    assert beats.is_better_than_null
    assert "better than the permutation null" in beats.statement()
    assert not constant.is_better_than_null
    assert "INCONCLUSIVE" in constant.statement()


def test_the_statements_carry_the_rows_and_the_floor() -> None:
    lines = ordinal.score_statements(draws(300, seed=3), n_boot=200, seed=0)

    assert len(lines) == 3
    assert lines[0].startswith("Mean absolute error ")
    assert lines[1].startswith("Ranked probability score ")
    assert lines[2].startswith("Cumulative calibration error ")
    assert all("over 300 rows" in line for line in lines)
    assert "3 thresholds a row" in lines[2]


def test_an_arm_with_no_distributions_gets_the_error_only_and_says_why() -> None:
    levels_only = [
        ScoreAnswer(levels=LEVELS, probabilities=None, expected=float(a.gold), gold=a.gold)
        for a in draws(50, seed=4)
    ]

    lines = ordinal.score_statements(levels_only, n_boot=100, seed=0)

    assert lines[0].startswith("Mean absolute error ")
    assert len(lines) == 2 and "no distribution" in lines[1]


def test_the_report_reads_score_rows_by_rank_and_keeps_them_out_of_the_choice_figures(
    tmp_path,
) -> None:
    from pathlib import Path

    from typer.testing import CliRunner

    from plumbline import cli

    fixture = Path(__file__).resolve().parent.parent / "datasets/public/jevbench-hard.jsonl"
    done = CliRunner().invoke(
        cli.app,
        ["run", str(fixture), "--format", "jevbench", "--results", str(tmp_path), "--boot", "100"],
        catch_exceptions=False,
    )

    assert done.exit_code == 0, done.stderr
    report = done.stdout
    choice, score = report.split("#### Score questions", 1)
    assert "ECE 0.0740 over 105 rows" in choice  # the choice figures did not move
    assert "Mean absolute error" not in choice
    for figure in ("Mean absolute error", "Ranked probability score", "Cumulative calibration"):
        assert f"{figure}" in score and "over 6 rows" in score
    assert "not comparable with a choice figure" in score
