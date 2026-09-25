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

from plumbline.datasets.loader import LoadReport, LoadSummary
from plumbline.metrics import baseline, calibration, cascade, cost, latency, recalibration
from plumbline.metrics.calibration import Binning
from plumbline.metrics.cost import DEFAULT_PRICING_MAX_AGE_DAYS
from plumbline.runner.execute import CaseRecord, RunResult
from plumbline.types import (
    SUPPORTED_QUESTION_TYPES,
    ConfidenceSeries,
    InsufficientDataError,
    NotCalibratableError,
    Prediction,
    ProbabilitySeries,
    apply_temperature,
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

    fit_fraction: float = 0.5
    min_eval_rows: int = recalibration.DEFAULT_MIN_EVAL_ROWS

    cost_escalation_usd: float | None = None
    """What one escalation to the expensive arm costs. Yours to supply."""

    cost_error_usd: float | None = None
    """What one wrong answer costs. Yours to supply, and usually the larger one."""

    min_threshold_rows: int = 200
    """Below this, a threshold is fitted to noise and none is printed."""

    allow_mixed_datasets: bool = False
    """Put runs over different datasets in one document, each arm naming its own.

    Off by default: figures from different datasets are not comparable, and a
    document that lays them side by side invites exactly that comparison.
    """

    @property
    def has_costs(self) -> bool:
        return self.cost_escalation_usd is not None and self.cost_error_usd is not None

    @property
    def clock(self) -> date:
        """The date prices are aged against. Injectable so a test is not seasonal."""
        return self.today or datetime.now(UTC).date()


def render(
    results: Sequence[RunResult],
    *,
    load: LoadReport | LoadSummary | None = None,
    options: ReportOptions | None = None,
) -> str:
    """One markdown document covering every arm, grouped and labeled."""
    if not results:
        raise ValueError("no runs to report on")
    options = options or ReportOptions()
    datasets = sorted({result.dataset_hash for result in results})
    if len(datasets) > 1 and not options.allow_mixed_datasets:
        named = ", ".join(f"`{digest[:8]}`" for digest in datasets)
        raise ValueError(
            f"these runs are over {len(datasets)} different datasets ({named}), and figures "
            "from different datasets are not comparable. Report them separately, or allow "
            "mixed datasets (--allow-mixed) to put them in one document with each arm "
            "naming its dataset."
        )
    headings = _headings(results, mixed=len(datasets) > 1)

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
            lines.extend(_arm(result, options, headings[id(result)]))

    return "\n".join(lines).rstrip() + "\n"


def _header(
    results: Sequence[RunResult], load: LoadReport | LoadSummary | None, options: ReportOptions
) -> list[str]:
    first = results[0]
    datasets = sorted({result.dataset_hash for result in results})
    if len(datasets) > 1:
        named = ", ".join(f"`{digest[:8]}`" for digest in datasets)
        return [
            "# plumbline report",
            "",
            f"Generated {options.clock.isoformat()} against {len(datasets)} datasets "
            f"({named}). {len(results)} arm(s), each naming its own dataset and row "
            "count. Figures from different datasets are not comparable.",
            "",
        ]
    rows = ", ".join(sorted({f"{result.dataset_rows} rows" for result in results}))
    return [
        "# plumbline report",
        "",
        f"Generated {options.clock.isoformat()} against dataset `{first.dataset_hash[:8]}`, "
        f"{rows}. {len(results)} arm(s).",
        "",
    ]


def _headings(results: Sequence[RunResult], *, mixed: bool) -> dict[int, str]:
    """A heading per arm that tells it apart from every other.

    The adapter name alone, when it is unique; then the model; then the run's
    time. With mixed datasets every arm also names its dataset and row count.
    """

    def label(result: RunResult, depth: int) -> str:
        parts = [result.adapter_name]
        if depth >= 1:
            parts.append(result.model_requested)
        if depth >= 2:
            parts.append(result.timestamp)
        return ", ".join(parts)

    chosen: dict[int, str] = {}
    for result in results:
        for depth in range(3):
            candidate = label(result, depth)
            if sum(1 for other in results if label(other, depth) == candidate) == 1 or depth == 2:
                break
        if mixed:
            candidate += f", dataset `{result.dataset_hash[:8]}` ({result.dataset_rows} rows)"
        chosen[id(result)] = candidate
    return chosen


def _how_to_read() -> list[str]:
    return [
        "## How to read this",
        "",
        "- Every figure states the rows it was computed on and the null it is read "
        "against. A number on its own is not a finding.",
        '- **"INCONCLUSIVE" is not a pass.** It means the value sits inside what the '
        "null already produces at this sample size, so this dataset cannot tell the two "
        "apart. Nothing was established in either direction. A model that is genuinely "
        "well calibrated and one that is badly calibrated can both land here on too few "
        "rows, and the figure does not say which you have.",
        "- Arms are grouped by what kind of number they report. **Figures in different "
        "groups are not comparable** and are never placed side by side.",
        '- "Not reported" is a result with a reason attached, not a missing cell.',
        "",
    ]


def _dataset_section(load: LoadReport | LoadSummary) -> list[str]:
    lines = ["## Dataset", "", f"- {load.statement()}"]
    for question_type, count in load.unsupported_by_type.items():
        lines.append(
            f"- {count} {question_type} rows are excluded from every figure below: "
            "plumbline v0.1 scores choice and yes/no questions only."
        )
    lines.append("")
    return lines


def _arm(result: RunResult, options: ReportOptions, heading: str) -> list[str]:
    scoreable = [
        record for record in result.records if record.question_type in SUPPORTED_QUESTION_TYPES
    ]
    excluded = len(result.records) - len(scoreable)
    successes = [record for record in scoreable if record.prediction is not None]
    failures = [record for record in scoreable if record.prediction is None]

    lines = ["", f"### {heading}", "", *_provenance(result, options)]
    described = result.config.get("label_descriptions") or {}
    if described.get("rows") and not described.get("sent"):
        lines.append(
            f"- **Option descriptions**: {described['rows']} rows carried option descriptions, "
            "and this adapter does not send them, so they had no effect on its answers."
        )
    if excluded:
        lines.append(
            f"- **Excluded**: {excluded} rows of an unsupported question type were not scored."
        )
    lines.extend(_asked_as(scoreable))
    lines.extend(_resolution_lines(scoreable, options))

    if not successes:
        # One shared reason is almost always an install or setup step (a missing
        # extra, a bad key), and it is the one thing the reader needs, so it is
        # said here rather than left inside the artifact. Different reasons are
        # a log, and a report is not a log.
        reasons = {record.error for record in failures if record.error}
        if failures and len(reasons) == 1:
            lines.append(
                f"- **No figures**: all {len(failures)} cases failed for the same reason, "
                "so there is nothing to measure:"
            )
            lines.append(f"  {reasons.pop()}")
        else:
            lines.append("- **No figures**: every case failed or was refused, so there is nothing")
            lines.append(
                f"  to measure. The {len(failures)} failures, for {len(reasons)} different "
                "reasons, are in the artifact."
            )
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
            f"- **Failures**: {len(failures)} of {len(scoreable)} cases produced no "
            "prediction and are excluded from accuracy rather than scored wrong."
        )
    lines.extend(_tie_lines(successes))

    probabilities = _probabilities(result, successes)
    lines.extend(_calibration_lines(probabilities, outcomes, options))
    lines.extend(_confidence_lines(successes, outcomes, options))
    lines.extend(_distribution_caveat(result, successes))
    lines.extend(_cost_lines(result, scoreable, options))
    lines.extend(_latency_lines(result))

    fit = _fit(probabilities, successes, outcomes, options)
    lines.extend(_recalibration_section(fit, options))
    lines.extend(_cascade_section(fit, successes, outcomes, options, result.probability_semantics))
    lines.extend(_diagnostics(result, probabilities, successes, outcomes, options))
    return lines


@dataclass(frozen=True)
class _Fit:
    """What recalibration produced, or why it produced nothing."""

    result: recalibration.RecalibrationResult | None
    unavailable: str | None  # the reason, when there is no result at all


def _fit(
    probabilities: ProbabilitySeries,
    successes: Sequence[CaseRecord],
    outcomes: Sequence[bool],
    options: ReportOptions,
) -> _Fit:
    """Fit a temperature on one half and judge it on the other, or say why not."""
    try:
        probabilities.require_reportable()
    except NotCalibratableError as absent:
        return _Fit(None, str(absent))

    distributions = [record.prediction.distribution for record in successes if record.prediction]
    gold = [record.gold_label for record in successes]
    try:
        fitted = recalibration.recalibrate(
            probabilities,
            outcomes,
            distributions=distributions,
            gold_labels=gold,
            fit_fraction=options.fit_fraction,
            seed=options.seed,
            min_eval_rows=options.min_eval_rows,
            n_bins=options.n_bins,
            binning=options.binning,
            n_boot_floor=options.n_boot,
        )
    except (InsufficientDataError, ValueError) as refused:
        return _Fit(None, str(refused))
    return _Fit(fitted, None)


#: What a refusal says, with no number in it. A refused fit emits no
#: temperature and no metric: a number on the page is a number that gets
#: hardcoded, whatever the sentence around it says.
REFUSAL_REASONS = {
    "already_calibrated": (
        "Refused: this arm's calibration is already inside its calibrated-model floor, so "
        "there is nothing to correct. No temperature is emitted."
    ),
    "no_material_improvement": (
        "Refused: temperature scaling does not fit this arm's miscalibration. The change it "
        "produced is no larger than what this sample size produces by chance, so no "
        "temperature is emitted: this one is at least as likely to hurt as to help. The "
        "signature is consistent with a per-label bias, which one global parameter cannot "
        "reach, since flattening enough for the skewed labels over-flattens the honest ones."
    ),
    "interval_spans_one": (
        "Refused: the fitted interval spans 1.0, so this sample size does not establish that "
        "any correction is needed. No temperature is emitted. Collect more rows."
    ),
}


def _recalibration_section(fit: _Fit, options: ReportOptions) -> list[str]:
    lines = ["", "#### Recalibration", ""]
    if fit.result is None:
        return [*lines, f"- Not reported. {fit.unavailable}"]

    result = fit.result
    split = (
        f"- Fitted on {result.n_fit} rows and judged on {result.n_eval} held-out rows "
        f"(seed {result.seed}, {result.method} form)."
    )

    if result.recommendation == "refused":
        # Split sizes stay: they are provenance about the procedure, not a
        # measurement anybody can lift. Everything else goes.
        return [*lines, split, f"- {REFUSAL_REASONS[result.reason]}"]

    floor = result.floor["ece"]
    lines.extend(
        [
            split,
            f"- Fitted T = {result.temperature:.3f}, 95 percent interval "
            f"[{result.temperature_ci[0]:.3f}, {result.temperature_ci[1]:.3f}].",
            f"- ECE {result.before.ece:.4f} before, {result.after.ece:.4f} after, against a "
            f"calibrated-model floor of {floor.mean:.4f} (95th percentile "
            f"{floor.p95:.4f}) on the held-out rows.",
        ]
    )
    if result.recommendation == "partial":
        lines.append(
            f"- Partial: scaling removed {result.improvement:.4f} of ECE and "
            f"{result.after.ece:.4f} remains, which is {result.residual_ratio:.1f} times the "
            "floor. Temperature was the wrong shape of correction for part of this "
            "miscalibration. Use the temperature, and know that some of it survives."
        )
    else:
        lines.append(
            "- Recommended: the interval clears 1.0 and post-scaling ECE is inside the "
            "floor, so one temperature accounts for the miscalibration present."
        )
    if not result.had_distribution and result.semantics != "none":
        lines.append(f"- {recalibration.NO_DISTRIBUTION_NOTE}")
    return lines


def _cascade_section(
    fit: _Fit,
    successes: Sequence[CaseRecord],
    outcomes: Sequence[bool],
    options: ReportOptions,
    semantics: str,
) -> list[str]:
    """The sentence the whole tool exists to produce, or the reason there is none."""
    lines = ["", "#### Cascade", ""]

    if fit.result is None and fit.unavailable and "reports no probability" in fit.unavailable:
        return [*lines, f"- Not reported. {fit.unavailable}"]
    if not options.has_costs:
        return [
            *lines,
            "- Not reported. A threshold is decided by two numbers no benchmark can know: "
            "what one escalation to the expensive arm costs, and what one wrong answer "
            "costs. Supply both (`--escalation-cost`, `--error-cost`) and this section "
            "states where to set the threshold and what it buys.",
        ]

    scored, scale = _cascade_rows(fit, successes, outcomes, semantics, options)
    if scored is None:
        return [*lines, "- Not reported. This arm reports no probability to threshold on."]

    (fit_series, fit_outcomes), (held_series, held_outcomes) = scored
    chosen_on, rows = len(fit_outcomes), len(held_outcomes)
    if rows < options.min_threshold_rows:
        return [
            *lines,
            f"- Not reported. A threshold is chosen on one part of the rows and judged on "
            f"rows it has not seen, and this arm leaves {rows} held-out rows; "
            f"{options.min_threshold_rows} held-out rows are the minimum before one is "
            "worth acting on. No threshold is given.",
        ]

    escalation = options.cost_escalation_usd
    error = options.cost_error_usd
    assert escalation is not None and error is not None
    # Chosen on the fit rows, then scored on the held-out rows: a threshold
    # picked and reported on the same rows is fitted to them, and its cost
    # there is the best case rather than what it will do.
    chosen = cascade.optimal_threshold(fit_series, fit_outcomes, escalation, error)
    best = cascade.cascade_sweep(
        held_series, held_outcomes, escalation, error, thresholds=[chosen.threshold]
    )[0]
    all_escalated = rows * escalation
    where = f"chosen on {chosen_on} rows and scored on {rows} held-out rows"
    caveat = (
        "- Escalated traffic is assumed to answer correctly, so this is the optimistic "
        "bound: whatever you escalate to has its own error rate and this number does not "
        "know it. The cheap arm's own per-case cost is excluded, because it is paid at "
        "every threshold and cannot move the optimum."
    )
    if chosen.covered == 0:
        return [
            *lines,
            f"- Escalating every case is cheapest ({where}): at these costs, keeping rows "
            f"on the cheap arm costs more in errors than escalating them. Expected cost "
            f"${best.total_cost_usd:.2f} over the {rows} held-out rows "
            f"(${best.cost_per_case_usd:.4f} per case).",
            caveat,
        ]

    return [
        *lines,
        f"- At a threshold of {chosen.threshold:.3f} on the {scale} probability ({where}), "
        f"{best.coverage * 100:.0f} percent of traffic stays on the cheap arm and the "
        f"expected cost is ${best.total_cost_usd:.2f} over those {rows} rows "
        f"(${best.cost_per_case_usd:.4f} per case), versus ${all_escalated:.2f} if every "
        "case went to the expensive arm.",
        f"- {best.escalated} of {rows} held-out rows escalate. Of the {best.covered} covered "
        f"rows, {best.errors_covered} are wrong, so covered accuracy is "
        f"{(best.accuracy_covered or 0.0):.3f}.",
        caveat,
    ]


_Rows = tuple[ProbabilitySeries, list[bool]]


def _cascade_rows(
    fit: _Fit,
    successes: Sequence[CaseRecord],
    outcomes: Sequence[bool],
    semantics: str,
    options: ReportOptions,
) -> tuple[tuple[_Rows, _Rows] | None, str]:
    """The rows a threshold is chosen on, the rows it is scored on, and their scale.

    Always two disjoint halves: the recalibration split when there is one, or a
    split made the same way when recalibration produced nothing. A threshold
    set against a raw overconfident probability sits in the wrong place,
    because 0.9 from an overconfident model is not 0.9, so when a temperature
    was recommended both halves are on the recalibrated scale. The held-out
    half is then unseen by the temperature and the threshold alike.
    """
    predictions = [record.prediction for record in successes if record.prediction]
    result = fit.result
    split = (
        result.split
        if result is not None
        else recalibration.make_split(len(predictions), options.fit_fraction, options.seed)
    )

    if result is not None and result.temperature_to_use is not None:
        temperature = result.temperature_to_use

        def value(index: int) -> float:
            return _scaled(predictions[index], temperature)

        scale = "recalibrated"
    else:
        if any(prediction.prob_selected is None for prediction in predictions):
            return None, "raw"

        def value(index: int) -> float:
            probability = predictions[index].prob_selected
            assert probability is not None
            return probability

        scale = "raw"

    def half(indices: Sequence[int]) -> _Rows:
        return (
            ProbabilitySeries(
                values=tuple(value(index) for index in indices),
                semantics=semantics,  # type: ignore[arg-type]
            ),
            [bool(outcomes[index]) for index in indices],
        )

    return (half(split.fit_indices), half(split.eval_indices)), scale


def _scaled(prediction: Prediction, temperature: float) -> float:
    """One probability on the recalibrated scale, distribution-aware.

    Temperature scaling is monotone in log p, so the predicted label never
    moves and the outcomes carry over unchanged.
    """
    if prediction.distribution is not None:
        scaled = apply_temperature(prediction.distribution, temperature)
        return scaled[max(scaled, key=lambda label: scaled[label])]
    assert prediction.prob_selected is not None
    return recalibration.apply_temperature_binary(prediction.prob_selected, temperature)


def _provenance(result: RunResult, options: ReportOptions) -> list[str]:
    reported = result.model_reported or "not reported"
    endpoint = result.config.get("endpoint")
    line = (
        f"- **Model**: requested {_code(result.model_requested)}, reported {_code(reported)}"
        + (f", revision {_code(result.revision)}" if result.revision else "")
        + (f", endpoint {_code(str(endpoint))}" if endpoint else "")
        + "."
    )
    hits = result.cache_stats.get("hits", 0)
    if hits:
        line += f" {hits} of {len(result.records)} rows came from cache and cost nothing."
    lines = [line]
    if result.config.get("option_style") == "letter":
        lines.append(
            "- **Options**: asked as letters, A for the first option and so on, and read from "
            "the letter tokens. The softmax is over the letters, so it is still conditional on "
            "the options supplied, and a lettered question is not the question an arm that "
            "reads the option words is asked."
        )
    if result.config.get("semantics_set_by") == "operator":
        lines.append(
            f"- **Probability semantics**: {_code(result.probability_semantics)}, set by the "
            "operator with --semantics rather than declared by the adapter."
        )
    return lines


def _code(value: str) -> str:
    """A markdown code span that holds ``value`` whatever it contains.

    A model string is whatever the caller passed, and a backtick inside a single
    backtick span ends the span early. Markdown allows a longer fence: two
    backticks with a space inside each, which a single backtick cannot close.
    """
    if "`" not in value:
        return f"`{value}`"
    fence = "``" if "``" not in value else "```"
    return f"{fence} {value} {fence}"


def _asked_as(records: Sequence[CaseRecord]) -> list[str]:
    """State how each kind of row was actually asked, and flag any mismatch."""
    pairs: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record.question_type, record.asked_as)
        pairs[key] = pairs.get(key, 0) + 1

    def phrase(question_type: str, asked: str, count: int) -> str:
        # A row that failed or was refused was not asked as anything.
        if asked in {"failed", "refused"}:
            return f"{count} {question_type} rows {asked}"
        return f"{count} {question_type} rows asked as {asked}"

    described = ", ".join(
        phrase(question_type, asked, count)
        for (question_type, asked), count in sorted(pairs.items())
    )
    lines = [f"- **Asked**: {described}."]

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
        return [f"- **Calibration**: not reported. {absent}"]

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
        # A yes/no asked as one has a single probability and nothing to
        # summarize beside it. Any other arm without one simply returns none,
        # such as a local checkpoint, which has no vendor to report it.
        why = (
            "a yes/no answer has no distribution to summarize"
            if successes and all(record.asked_as == "noul" for record in successes)
            else "the adapter returns none beside its probabilities"
        )
        return [
            f"- **Confidence**: not reported. This arm reports no confidence statistic: {why}, "
            "so the number does not exist rather than being missing."
        ]
    if any(value is None for value in values):
        missing = sum(1 for value in values if value is None)
        return [
            f"- **Confidence**: not reported. {missing} of {len(values)} rows carry no "
            "confidence, and dropping them silently would change which cases the figure "
            "covers."
        ]

    series = ConfidenceSeries(values=values)
    try:
        figure = baseline.auroc_figure(series, outcomes, n_boot=options.n_boot, seed=options.seed)
    except ValueError as undefined:
        return [f"- **Confidence**: not reported. {undefined}"]
    return [f"- **Confidence**: {figure.statement()}"]


def _resolution_lines(records: Sequence[CaseRecord], options: ReportOptions) -> list[str]:
    """The grid the probabilities arrived on, when they arrived on one."""
    values: list[float] = []
    for record in records:
        prediction = record.prediction
        if prediction is None:
            continue
        if prediction.distribution:
            values.extend(prediction.distribution.values())
        elif prediction.prob_selected is not None:
            values.append(prediction.prob_selected)
    step = calibration.probability_grid(values)
    if step is None:
        return []
    line = (
        f"- **Resolution**: all {len(values)} probabilities this arm returned are multiples "
        f"of {step:g}, so no bin or threshold finer than {step:g} can mean anything, and a "
        "tie for the top option is ordinary rather than rare."
    )
    if options.n_bins * step > 1:
        line += (
            f" At {options.n_bins} bins each bin is narrower than the grid, so the binning "
            "measures the rounding rather than the model."
        )
    return [line]


def _tie_lines(successes: Sequence[CaseRecord]) -> list[str]:
    """How many answers the vendor's tie-break chose, and how many went against gold."""
    tied = [record for record in successes if record.prediction and record.prediction.tied_for_top]
    if not tied:
        return []
    against = sum(
        1
        for record in tied
        if record.prediction
        and record.gold_label in record.prediction.tied_for_top
        and not record.correct
    )
    line = (
        f"- **Ties**: {len(tied)} of {len(successes)} scored rows had two or more options tied "
        "for the highest probability, so the vendor's tie-break chose the answer, not a margin."
    )
    if against:
        line += (
            f" On {against} of them the gold label was one of the tied options and was not the "
            "one chosen, so those rows count as wrong by a tie-break."
        )
    return [line]


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
        f"- **Distribution**: {without} of {len(successes)} rows reported a probability "
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
        return [f"- **Cost**: not reported. {summary.note}"]

    line = (
        f"- **Cost**: ${summary.total_usd:.4f} over {summary.priced_cases} priced rows, "
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
            "- **Latency**: not reported. No call went out, so every latency here would "
            "be a measurement of disk."
        ]
    # Only answers served from cache are cache hits. A refused or failed case
    # made no timed call either, and counting it here named every local arm's
    # untokenizable cases as cache hits on a run with no cache at all.
    hits = sum(1 for record in result.records if record.ok and record.from_cache)
    summary = latency.summarize(live, excluded_cache_hits=hits)
    return [f"- **Latency**: {summary}"]


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
        lines.append("- **MCE**: not reported. This arm reports no probability.")
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
        lines.append(f"- **MCE**: not reported. {undefined}")

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
            "- **Multiclass Brier**: not reported. This arm supplied no distribution, so "
            "the multiclass form does not exist for it. It is not zero."
        )
    return lines
