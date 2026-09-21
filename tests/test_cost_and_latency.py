"""Cost and latency: what gets reported, and what gets reported as absent.

Cost is None when tokens are missing, never zero, because free and unknown are
different claims. Latency has no mean, because a mean hides the tail that
decides whether a model can sit in a hot path.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from plumbline import config
from plumbline.metrics import cost, latency

READ_ON = date(2026, 1, 1)

PRICING = cost.Pricing(
    input_usd_per_million=1.0,
    output_usd_per_million=5.0,
    source="test fixture",
    as_of=READ_ON,
)


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
    table = {"jev-1.13": PRICING, "jev-latest": a_price(2.0, 9.0)}
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
        a_price(-1.0, 1.0)


# Pricing provenance


def a_price(
    input_usd: float | None,
    output_usd: float | None,
    *,
    source: str = "test fixture",
    as_of: date = READ_ON,
) -> cost.Pricing:
    return cost.Pricing(
        input_usd_per_million=input_usd,
        output_usd_per_million=output_usd,
        source=source,
        as_of=as_of,
    )


def test_a_pricing_entry_cannot_be_built_without_saying_where_it_came_from() -> None:
    """A price with no source and no read date is a rumour, not a measurement."""
    with pytest.raises(TypeError):
        cost.Pricing(input_usd_per_million=1.0, output_usd_per_million=5.0)  # type: ignore[call-arg]


def test_a_pricing_entry_with_an_empty_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="source"):
        cost.Pricing(
            input_usd_per_million=1.0,
            output_usd_per_million=5.0,
            source="  ",
            as_of=READ_ON,
        )


def test_an_unpublished_price_yields_no_cost_rather_than_a_guess() -> None:
    """Free is a claim. Unpublished is an absence. They are not the same number."""
    entry = a_price(None, 0.0)
    assert cost.cost_of(1_000_000, 200_000, entry) is None


def test_an_entry_knows_how_old_it_is() -> None:
    assert a_price(1.0, 0.0).age_days(date(2026, 1, 31)) == 30


def test_a_recent_entry_is_not_stale_and_says_when_it_was_read() -> None:
    entry = a_price(1.0, 0.0)
    assert not entry.is_stale(date(2026, 1, 31))
    assert "2026-01-01" in entry.statement(date(2026, 1, 31))


def test_an_entry_older_than_the_threshold_is_stale_and_the_statement_says_so() -> None:
    """A current-state claim that nobody has re-read is not a current price."""
    entry = a_price(1.0, 0.0)
    later = date(2026, 1, 1) + timedelta(days=cost.DEFAULT_PRICING_MAX_AGE_DAYS + 1)
    assert entry.is_stale(later)
    statement = entry.statement(later)
    assert "may be out of date" in statement
    assert "2026-01-01" in statement


def test_provenance_is_json_shaped_so_the_artifact_can_carry_it() -> None:
    provenance = a_price(None, 0.0).provenance(date(2026, 1, 31))
    assert provenance["as_of"] == "2026-01-01"
    assert provenance["source"] == "test fixture"
    assert provenance["age_days"] == 30
    assert provenance["stale"] is False
    assert provenance["input_usd_per_million"] is None
    assert provenance["output_usd_per_million"] == 0.0


# The shipped pricing table


def test_the_shipped_jev_entry_names_its_source_but_ships_no_figures() -> None:
    """The vendor's terms make its prices confidential, so plumbline restates none.

    None on both sides, not 0.0. A zero would be plumbline asserting those tokens
    are free, which is a claim about someone else's prices that this file does
    not make. The source names the page so a reader can go and read it.
    """
    entry = config.DEFAULT_PRICING_TABLE["jev-1.13.0"]
    assert entry.input_usd_per_million is None
    assert entry.output_usd_per_million is None
    assert not entry.is_priced
    assert "docs.typesafe.ai/models" in entry.source
    assert cost.cost_of(1_000_000, 12, entry) is None


def test_no_shipped_entry_states_a_figure_for_the_confidential_vendor() -> None:
    """A guard against a price creeping back into a published file.

    This is the concrete thing the vendor's confidentiality clause reaches, and
    a number reintroduced here would be published to everyone who clones the
    repository.
    """
    for key in ("jev-1.13.0", "jev-latest", "jev-preview"):
        entry = config.DEFAULT_PRICING_TABLE[key]
        assert entry.input_usd_per_million is None, key
        assert entry.output_usd_per_million is None, key
        assert not any(char.isdigit() for char in entry.source.replace("1.13.0", "")), key


def test_the_version_that_answers_is_keyed_not_only_the_alias() -> None:
    """``pricing_for`` prefers what answered, so the reported version must be a key.

    A live call requesting ``jev-latest`` reports ``jev-1.13.0``. Keying the table
    on aliases alone made every real row price as ``model_not_priced``, which is
    how the shipped table was wrong before the first live call.
    """
    entry, key = cost.pricing_for(config.DEFAULT_PRICING_TABLE, "jev-1.13.0", "jev-latest")
    assert key == "jev-1.13.0"
    assert entry is not None


def test_an_unknown_version_does_not_match_a_known_one_by_prefix() -> None:
    """No prefix match: a future release must not inherit a superseded rate."""
    assert "jev-1.14.0" not in config.DEFAULT_PRICING_TABLE
    entry, key = cost.pricing_for(config.DEFAULT_PRICING_TABLE, "jev-1.14.0", "some-other-model")
    assert (entry, key) == (None, None)


def test_every_shipped_entry_carries_provenance() -> None:
    for name, entry in config.DEFAULT_PRICING_TABLE.items():
        assert entry.source.strip(), name
        assert entry.as_of <= config.PRICING_READ_ON, name


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


# Why a cost is blank


def test_a_summary_keeps_the_two_reasons_for_a_blank_cost_apart() -> None:
    """An adapter that never reports tokens is a different finding from an API
    that reported none on this run, and a bare count of blanks loses that."""
    summary = cost.summarize(
        [0.10, None, None],
        [True, True, False],
        bases=["priced", "adapter_reports_no_tokens", "tokens_not_reported"],
    )
    assert summary.unpriced_by_basis == {
        "adapter_reports_no_tokens": 1,
        "tokens_not_reported": 1,
    }
    assert "reports no token counts at all" in summary.note
    assert "returned none on this call" in summary.note


def test_a_summary_without_the_reasons_still_counts_the_blanks() -> None:
    summary = cost.summarize([0.10, None], [True, False])
    assert summary.unpriced_cases == 1
    assert summary.unpriced_by_basis == {}


def test_the_reasons_must_match_the_costs_they_explain() -> None:
    with pytest.raises(ValueError, match="must match"):
        cost.summarize([0.10, None], [True, False], bases=["priced"])
