"""Cost and latency: what gets reported, and what gets reported as absent.

Cost is None when tokens are missing, never zero, because free and unknown are
different claims. Latency has no mean, because a mean hides the tail that
decides whether a model can sit in a hot path.
"""

from __future__ import annotations

import pytest

from plumbline.metrics import cost, latency

PRICING = cost.Pricing(input_usd_per_million=1.0, output_usd_per_million=5.0)


# Cost


def test_cost_is_computed_from_tokens_and_a_pricing_table() -> None:
    # One million input tokens at $1.00, plus 200,000 output at $5.00 per million.
    assert cost.cost_of(1_000_000, 200_000, PRICING) == pytest.approx(1.0 + 1.0)


@pytest.mark.parametrize(("input_tokens", "output_tokens"), [(None, 10), (10, None), (None, None)])
def test_a_missing_token_count_makes_cost_none_not_zero(input_tokens, output_tokens) -> None:
    """Half a count is not a count. The SDK types each field independently."""
    assert cost.cost_of(input_tokens, output_tokens, PRICING) is None


def test_an_unpriced_model_makes_cost_none() -> None:
    assert cost.cost_of(1000, 10, None) is None


def test_pricing_prefers_the_model_the_api_said_it_used() -> None:
    """Billing follows what actually answered, not what the config asked for."""
    table = {"jev-1.13": PRICING, "jev-latest": cost.Pricing(2.0, 9.0)}
    pricing, key = cost.pricing_for(table, "jev-1.13", "jev-latest")
    assert key == "jev-1.13"
    assert pricing is PRICING


def test_pricing_falls_back_to_the_requested_model() -> None:
    table = {"jev-latest": PRICING}
    _, key = cost.pricing_for(table, "jev-1.13", "jev-latest")
    assert key == "jev-latest"


def test_an_unknown_model_yields_no_pricing_and_no_key() -> None:
    assert cost.pricing_for({}, "a", "b") == (None, None)


def test_a_summary_reports_total_per_case_and_per_correct() -> None:
    summary = cost.summarize([0.10, 0.10, 0.10, 0.10], [True, True, True, False])
    assert summary.total_usd == pytest.approx(0.40)
    assert summary.per_case_usd == pytest.approx(0.10)
    assert summary.per_correct_usd == pytest.approx(0.40 / 3)
    assert summary.is_complete


def test_a_summary_with_no_priced_rows_reports_none_and_says_why() -> None:
    summary = cost.summarize([None, None], [True, False])
    assert summary.total_usd is None
    assert summary.per_case_usd is None
    assert "not reported rather than being shown as zero" in summary.note


def test_a_partially_priced_summary_says_the_total_is_an_undercount() -> None:
    summary = cost.summarize([0.10, None, 0.10], [True, True, False])
    assert summary.total_usd == pytest.approx(0.20)
    assert not summary.is_complete
    assert summary.unpriced_cases == 1
    assert "undercount" in summary.note


def test_a_local_model_carries_a_note_instead_of_a_number() -> None:
    summary = cost.summarize(
        [None] * 3,
        [True, True, False],
        local_note="Local checkpoint on one RTX 3090, 42s wall-clock. No token cost applies.",
    )
    assert summary.total_usd is None
    assert "RTX 3090" in summary.note


def test_per_correct_cost_is_none_when_nothing_was_correct() -> None:
    summary = cost.summarize([0.10, 0.10], [False, False])
    assert summary.per_correct_usd is None


def test_negative_prices_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        cost.Pricing(input_usd_per_million=-1.0, output_usd_per_million=1.0)


def test_mismatched_column_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="must match"):
        cost.summarize([0.1, 0.2], [True])


def test_an_estimate_scales_with_the_case_text() -> None:
    short = cost.estimate_case_cost("hi", ["a", "b"], PRICING)
    long = cost.estimate_case_cost("hi" * 5000, ["a", "b"], PRICING)
    assert long > short > 0


def test_an_estimate_is_none_without_pricing() -> None:
    assert cost.estimate_case_cost("hi", ["a", "b"], None) is None


# Latency


def test_percentiles_come_back_in_order() -> None:
    summary = latency.summarize([float(value) for value in range(1, 101)])
    assert summary.p50 < summary.p95 < summary.p99
    assert summary.n == 100


def test_the_tail_is_not_hidden_by_the_bulk() -> None:
    """Two percent of calls take 2 seconds. The mean would report 79ms and hide it."""
    values = [40.0] * 980 + [2000.0] * 20
    summary = latency.summarize(values)

    assert summary.p50 == pytest.approx(40.0)
    assert summary.p95 == pytest.approx(40.0)
    assert summary.p99 == pytest.approx(2000.0)
    assert sum(values) / len(values) == pytest.approx(79.2)


def test_every_reported_percentile_is_a_latency_that_actually_happened() -> None:
    """Nearest rank, not interpolation. An invented p99 describes no real request."""
    values = [10.0] * 50 + [1000.0] * 50
    summary = latency.summarize(values)
    assert summary.p50 in values
    assert summary.p95 in values
    assert summary.p99 in values


def test_a_small_sample_cannot_see_far_into_the_tail_and_says_so_by_its_value() -> None:
    """One slow call in a hundred is the maximum, not the p99, and p99 reports that."""
    summary = latency.summarize([40.0] * 99 + [2000.0])
    assert summary.p99 == pytest.approx(40.0)
    assert max([40.0] * 99 + [2000.0]) == 2000.0


def test_there_is_no_mean_to_report() -> None:
    assert not hasattr(latency.summarize([1.0, 2.0]), "mean")
    assert not hasattr(latency, "mean")


def test_cache_hits_are_named_in_the_summary() -> None:
    summary = latency.summarize([40.0, 50.0], excluded_cache_hits=98)
    assert summary.excluded_cache_hits == 98
    assert "excluding 98 cache hits" in str(summary)


def test_an_empty_input_raises_rather_than_reporting_zero() -> None:
    """Zero latency is a claim. Nothing ran is not."""
    with pytest.raises(ValueError, match="no live calls"):
        latency.summarize([])


def test_negative_latency_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        latency.summarize([1.0, -1.0])
