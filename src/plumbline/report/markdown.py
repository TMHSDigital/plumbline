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
            f"- **Excluded**: {excluded} rows of an unsupported question type were not scored."
        )
    lines.extend(_asked_as(scoreable))

    if not successes:
        lines.append("- **No figures**: every case failed or was refused, so there is nothing")
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
            f"- **Failures**: {len(failures)} of {len(scoreable)} cases produced no "
            "prediction and are excluded from accuracy rather than scored wrong."
        )

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

    scored, scale = _cascade_rows(fit, successes, outcomes, semantics)
    if scored is None:
        return [*lines, "- Not reported. This arm reports no probability to threshold on."]

    series, chosen_outcomes = scored
    rows = len(chosen_outcomes)
    if rows < options.min_threshold_rows:
        return [
            *lines,
            f"- Not reported. A threshold picked on {rows} rows is fitted to noise; "
            f"{options.min_threshold_rows} held-out rows are the minimum before one is "
            "worth acting on. No threshold is given.",
        ]

    escalation = options.cost_escalation_usd
    error = options.cost_error_usd
    assert escalation is not None and error is not None
    best = cascade.optimal_threshold(series, chosen_outcomes, escalation, error)
    all_escalated = rows * escalation
    caveat = (
        "- Escalated traffic is assumed to answer correctly, so this is the optimistic "
        "bound: whatever you escalate to has its own error rate and this number does not "
        "know it. The cheap arm's own per-case cost is excluded, because it is paid at "
        "every threshold and cannot move the optimum."
    )
    if best.covered == 0:
        return [
            *lines,
            f"- Escalating every case is cheapest: at these costs, keeping any of the {rows} "
            f"rows on the cheap arm costs more in errors than escalating it. Expected cost "
            f"${best.total_cost_usd:.2f} over {rows} rows "
            f"(${best.cost_per_case_usd:.4f} per case).",
            caveat,
        ]

    return [
        *lines,
        f"- At a threshold of {best.threshold:.3f} on the {scale} probability, "
        f"{best.coverage * 100:.0f} percent of traffic stays on the cheap arm and the "
        f"expected cost is ${best.total_cost_usd:.2f} over {rows} rows "
        f"(${best.cost_per_case_usd:.4f} per case), versus ${all_escalated:.2f} if every "
        "case went to the expensive arm.",
        f"- {best.escalated} of {rows} rows escalate. Of the {best.covered} covered rows, "
        f"{best.errors_covered} are wrong, so covered accuracy is "
        f"{(best.accuracy_covered or 0.0):.3f}.",
        caveat,
    ]


def _cascade_rows(
    fit: _Fit,
    successes: Sequence[CaseRecord],
    outcomes: Sequence[bool],
    semantics: str,
) -> tuple[tuple[ProbabilitySeries, list[bool]] | None, str]:
    """The rows a threshold is chosen on, and what scale they are on.

    A threshold set against a raw overconfident probability sits in the wrong
    place, because 0.9 from an overconfident model is not 0.9. So when a
    temperature was recommended, the threshold is chosen on the held-out rows
    with that temperature applied: the same rows the temperature was judged
    on, never the rows it was fitted on.
    """
    predictions = [record.prediction for record in successes if record.prediction]
    result = fit.result

    if result is not None and result.temperature_to_use is not None:
        temperature = result.temperature_to_use
        indices = result.split.eval_indices
        values = tuple(_scaled(predictions[index], temperature) for index in indices)
        chosen = [bool(outcomes[index]) for index in indices]
        return (
            ProbabilitySeries(values=values, semantics=semantics),  # type: ignore[arg-type]
            chosen,
        ), "recalibrated"

    raw = [prediction.prob_selected for prediction in predictions]
    if any(value is None for value in raw):
        return None, "raw"
    return (
        ProbabilitySeries(
            values=tuple(value for value in raw if value is not None),
            semantics=semantics,  # type: ignore[arg-type]
        ),
        [bool(outcome) for outcome in outcomes],
    ), "raw"


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
    line = (
        f"- **Model**: requested `{result.model_requested}`, reported `{reported}`"
        + (f", revision `{result.revision}`" if result.revision else "")
        + (f", endpoint `{endpoint}`" if (endpoint := result.config.get("endpoint")) else "")
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
        return [
            "- **Confidence**: not reported. This arm reports no confidence statistic: a "
            "yes/no answer has no distribution to summarize, so the number does not exist "
            "rather than being missing."
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
    summary = latency.summarize(live, excluded_cache_hits=len(result.records) - len(live))
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
