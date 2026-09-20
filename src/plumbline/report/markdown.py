"""The markdown report: figures with their nulls, grouped by what they mean.

Two rules shape every line of this file.

Nothing prints as a bare number. Every figure states the row count it was
computed on and the null it is read against, because a calibration number
without its sample size cannot be acted on, and an accuracy without the chance
baseline for this mix of option widths cannot either. A figure whose null cannot
be built is reported as not reported, never as a number on its own.

A restricted softmax is never placed beside a calibrated claim. The arms are
grouped by ``probability_semantics``, each group is separated by a rule and
introduced by a heading that says what kind of number is inside it, and the
report states in its own words that figures in different groups are not
comparable. A tool that printed one table of all of them would invite exactly
the comparison it exists to prevent.

Absences are printed too. "Not reported" with the reason attached is a result: a
generative arm has no probability, a noul has no confidence because there is no
distribution to summarize, and a local checkpoint has no cost because there are
no tokens to price. A blank cell says none of that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from plumbline.datasets.loader import LoadReport
from plumbline.metrics import baseline, calibration, cost, latency
from plumbline.metrics.calibration import Binning
from plumbline.metrics.cost import DEFAULT_PRICING_MAX_AGE_DAYS
from plumbline.runner.execute import CaseRecord, RunResult
from plumbline.types import (
    SUPPORTED_QUESTION_TYPES,
    ConfidenceSeries,
    NotCalibratableError,
    ProbabilitySeries,
)

GROUP_ORDER = ("calibrated_claim", "restricted_softmax", "none")

GROUP_TITLES = {
    "calibrated_claim": "Calibrated claims",
    "restricted_softmax": "Restricted softmax",
    "none": "No probability reported",
}

GROUP_NOTES = {
    "calibrated_claim": (
        "The vendor asserts these probabilities are calibrated. Whether that survives "
        "contact with this dataset is what the figures below answer."
    ),
    "restricted_softmax": (
        "A softmax over the declared options only, with no calibration claim attached. "
        "Conditional on the options supplied, so these numbers are a null hypothesis for "
        "the group above rather than a competitor to it."
    ),
    "none": (
        "No probability is reported at all. These arms are excluded from calibration "
        "entirely rather than scored as zero; their accuracy is the floor a "
        "probability-reporting system has to clear before its probabilities matter."
    ),
}


@dataclass(frozen=True)
class ReportOptions:
    """Everything the report needs that is not in the run itself."""

    n_boot: int = 2000
    seed: int = 0
    n_bins: int = calibration.DEFAULT_N_BINS
    binning: Binning = calibration.DEFAULT_BINNING
    today: date | None = None
    max_pricing_age_days: int = DEFAULT_PRICING_MAX_AGE_DAYS

    @property
    def clock(self) -> date:
        """The date prices are aged against. Injectable so a test is not seasonal."""
        return self.today or datetime.now(UTC).date()


def render(
    results: Sequence[RunResult],
    *,
    load: LoadReport | None = None,
    options: ReportOptions | None = None,
) -> str:
    """One markdown document covering every arm, grouped and labeled."""
    if not results:
        raise ValueError("no runs to report on")
    options = options or ReportOptions()

    lines = _header(results, load, options)
    lines.extend(_how_to_read())
    if load is not None:
        lines.extend(_dataset_section(load))

    for semantics in GROUP_ORDER:
        group = [result for result in results if result.probability_semantics == semantics]
        if not group:
            continue
        lines.extend(["", "---", "", f"## {GROUP_TITLES[semantics]}", "", GROUP_NOTES[semantics]])
        for result in group:
            lines.extend(_arm(result, options))

    return "\n".join(lines).rstrip() + "\n"


def _header(
    results: Sequence[RunResult], load: LoadReport | None, options: ReportOptions
) -> list[str]:
    first = results[0]
    rows = ", ".join(sorted({f"{result.dataset_rows} rows" for result in results}))
    return [
        "# plumbline report",
        "",
        f"Generated {options.clock.isoformat()} against dataset `{first.dataset_hash[:8]}`, "
        f"{rows}. {len(results)} arm(s).",
        "",
    ]


def _how_to_read() -> list[str]:
    return [
        "## How to read this",
        "",
        "- Every figure states the rows it was computed on and the null it is read "
        "against. A number on its own is not a finding.",
        '- "Not distinguishable" means the value sits inside what the null produces at '
        "this sample size. It does not mean the systems are the same; it means this "
        "dataset cannot tell them apart yet.",
        "- Arms are grouped by what kind of number they report. **Figures in different "
        "groups are not comparable** and are never placed side by side.",
        '- "Not reported" is a result with a reason attached, not a missing cell.',
        "",
    ]


def _dataset_section(load: LoadReport) -> list[str]:
    lines = ["## Dataset", "", f"- {load.statement()}"]
    for question_type, count in load.unsupported_by_type.items():
        lines.append(
            f"- {count} {question_type} rows are excluded from every figure below: "
            "plumbline v0.1 scores choice and yes/no questions only."
        )
    lines.append("")
    return lines


def _arm(result: RunResult, options: ReportOptions) -> list[str]:
    scoreable = [
        record for record in result.records if record.question_type in SUPPORTED_QUESTION_TYPES
    ]
    excluded = len(result.records) - len(scoreable)
    successes = [record for record in scoreable if record.prediction is not None]
    failures = [record for record in scoreable if record.prediction is None]

    lines = ["", f"### {result.adapter_name}", "", *_provenance(result, options)]
    if excluded:
        lines.append(
            f"- **Excluded** — {excluded} rows of an unsupported question type were not scored."
        )
    lines.extend(_asked_as(scoreable))

    if not successes:
        lines.append("- **No figures** — every case failed or was refused, so there is nothing")
        lines.append("  to measure. The failures are in the artifact.")
        return lines

    outcomes = [bool(record.correct) for record in successes]
    lines.append(
        "- "
        + baseline.accuracy_figure(
            outcomes,
            [len(record.labels) for record in successes],
            n_boot=options.n_boot,
            seed=options.seed,
        ).statement()
    )
    if failures:
        lines.append(
            f"- **Failures** — {len(failures)} of {len(scoreable)} cases produced no "
            "prediction and are excluded from accuracy rather than scored wrong."
        )

    probabilities = _probabilities(result, successes)
    lines.extend(_calibration_lines(probabilities, outcomes, options))
    lines.extend(_confidence_lines(successes, outcomes, options))
    lines.extend(_distribution_caveat(result, successes))
    lines.extend(_cost_lines(result, scoreable, options))
    lines.extend(_latency_lines(result))
    lines.extend(_diagnostics(result, probabilities, successes, outcomes, options))
    return lines


def _provenance(result: RunResult, options: ReportOptions) -> list[str]:
    reported = result.model_reported or "not reported"
    line = (
        f"- **Model** — requested `{result.model_requested}`, reported `{reported}`"
        + (f", revision `{result.revision}`" if result.revision else "")
        + "."
    )
    hits = result.cache_stats.get("hits", 0)
    if hits:
        line += f" {hits} of {len(result.records)} rows came from cache and cost nothing."
    return [line]


def _asked_as(records: Sequence[CaseRecord]) -> list[str]:
    """State how each kind of row was actually asked, and flag any mismatch."""
    pairs: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record.question_type, record.asked_as)
        pairs[key] = pairs.get(key, 0) + 1

    described = ", ".join(
        f"{count} {question_type} asked as {asked}"
        for (question_type, asked), count in sorted(pairs.items())
    )
    lines = [f"- **Asked** — {described}."]

    mismatched = {
        (question_type, asked): count
        for (question_type, asked), count in pairs.items()
        if question_type != asked and asked in {"choice", "noul"}
    }
    for (question_type, asked), count in sorted(mismatched.items()):
        lines.append(
            f"  - {count} {question_type} rows were asked as {asked} questions, which is a "
            "different question from the one the dataset states. Not comparable with an arm "
            f"that asked them as {question_type}."
        )
    return lines


def _probabilities(result: RunResult, successes: Sequence[CaseRecord]) -> ProbabilitySeries:
    return ProbabilitySeries(
        values=tuple(record.prediction.prob_selected for record in successes if record.prediction),
        semantics=result.probability_semantics,  # type: ignore[arg-type]
    )


def _calibration_lines(
    probabilities: ProbabilitySeries,
    outcomes: Sequence[bool],
    options: ReportOptions,
) -> list[str]:
    try:
        probabilities.require_reportable()
    except NotCalibratableError as absent:
        return [f"- **Calibration** — not reported. {absent}"]

    figure = calibration.ece_figure(
        probabilities,
        outcomes,
        options.n_bins,
        options.binning,
        n_boot=options.n_boot,
        seed=options.seed,
    )
    brier = calibration.brier_figure(
        probabilities,
        outcomes,
        options.n_bins,
        options.binning,
        n_boot=options.n_boot,
        seed=options.seed,
    )
    return [f"- {figure.statement()}", f"- {brier.statement()}"]


def _confidence_lines(
    successes: Sequence[CaseRecord],
    outcomes: Sequence[bool],
    options: ReportOptions,
) -> list[str]:
    values = tuple(record.prediction.confidence for record in successes if record.prediction)
    if all(value is None for value in values):
        return [
            "- **Confidence** — not reported. This arm reports no confidence statistic: a "
            "yes/no answer has no distribution to summarize, so the number does not exist "
            "rather than being missing."
        ]
    if any(value is None for value in values):
        missing = sum(1 for value in values if value is None)
        return [
            f"- **Confidence** — not reported. {missing} of {len(values)} rows carry no "
            "confidence, and dropping them silently would change which cases the figure "
            "covers."
        ]

    series = ConfidenceSeries(values=values)
    try:
        figure = baseline.auroc_figure(series, outcomes, n_boot=options.n_boot, seed=options.seed)
    except ValueError as undefined:
        return [f"- **Confidence** — not reported. {undefined}"]
    return [f"- **Confidence** — {figure.statement()}"]


def _distribution_caveat(result: RunResult, successes: Sequence[CaseRecord]) -> list[str]:
    """The one line the methodology requires wherever a top line arrives alone."""
    if result.probability_semantics == "none":
        return []
    without = sum(
        1 for record in successes if record.prediction and record.prediction.distribution is None
    )
    if not without:
        return []
    return [
        f"- **Distribution** — {without} of {len(successes)} rows reported a probability "
        "with no distribution behind it. Those rows are outside the multiclass Brier "
        "figure, and the temperature that can be fitted for them is the one-parameter "
        "approximation, which is weaker than the multiclass form even when the "
        "miscalibration is simple."
    ]


def _cost_lines(
    result: RunResult, scoreable: Sequence[CaseRecord], options: ReportOptions
) -> list[str]:
    summary = cost.summarize(
        [record.cost_usd for record in scoreable],
        [bool(record.correct) for record in scoreable],
        bases=[record.cost_basis for record in scoreable],
    )
    if summary.total_usd is None:
        return [f"- **Cost** — not reported. {summary.note}"]

    line = (
        f"- **Cost** — ${summary.total_usd:.4f} over {summary.priced_cases} priced rows, "
        f"${summary.per_case_usd:.6f} per case"
    )
    if summary.per_correct_usd is not None:
        line += f", ${summary.per_correct_usd:.6f} per correct answer"
    line += f". {summary.note}"
    pricing = result.pricing or {}
    if pricing.get("statement"):
        line += f" {pricing['statement']}"
    return [line]


def _latency_lines(result: RunResult) -> list[str]:
    live = [record.prediction.latency_ms for record in result.live_calls if record.prediction]
    if not live:
        return [
            "- **Latency** — not reported. No call went out, so every latency here would "
            "be a measurement of disk."
        ]
    summary = latency.summarize(live, excluded_cache_hits=len(result.records) - len(live))
    return [f"- **Latency** — {summary}"]


def _diagnostics(
    result: RunResult,
    probabilities: ProbabilitySeries,
    successes: Sequence[CaseRecord],
    outcomes: Sequence[bool],
    options: ReportOptions,
) -> list[str]:
    """Numbers that are real but must not sit beside the headline figures."""
    lines = [
        "",
        "#### Diagnostics",
        "",
        "Read these only after the figures above. MCE is a maximum over bins, decided by "
        "one bin, and at a few hundred rows it cannot detect overconfidence spread evenly "
        "across the range.",
        "",
    ]

    try:
        probabilities.require_reportable()
    except NotCalibratableError:
        lines.append("- **MCE** — not reported. This arm reports no probability.")
        return lines

    try:
        figure = calibration.mce_figure(
            probabilities,
            outcomes,
            options.n_bins,
            options.binning,
            n_boot=options.n_boot,
            seed=options.seed,
        )
        lines.append(f"- {figure.statement()}")
    except ValueError as undefined:
        lines.append(f"- **MCE** — not reported. {undefined}")

    distributions = [record.prediction.distribution for record in successes if record.prediction]
    if all(distribution is not None for distribution in distributions) and distributions:
        gold = [record.gold_label for record in successes]
        multiclass = calibration.multiclass_brier_figure(
            [distribution for distribution in distributions if distribution is not None],
            gold,
            n_boot=options.n_boot,
            seed=options.seed,
        )
        lines.append(f"- {multiclass.statement()}")
    else:
        lines.append(
            "- **Multiclass Brier** — not reported. This arm supplied no distribution, so "
            "the multiclass form does not exist for it. It is not zero."
        )
    return lines
