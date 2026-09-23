"""A pricing table the operator supplies, and what it refuses to accept.

plumbline ships no figures for a vendor whose terms make its prices
confidential. That makes the supplied table the only route to a cost column for
those vendors, so it has to hold the same line the shipped table does: every
entry names where its price was read and the date it was read, and None is never
quietly turned into zero.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from plumbline import config
from plumbline.metrics import cost

ENTRY = {
    "input_usd_per_million": 1.5,
    "output_usd_per_million": 6.0,
    "source": "Vendor price page, https://example.invalid/pricing, read by me",
    "as_of": "2026-09-21",
}


def write(tmp_path: Path, payload: object, name: str = "pricing.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_supplied_entry_prices_a_model_the_shipped_table_leaves_unpriced(
    tmp_path: Path,
) -> None:
    supplied = config.load_pricing_file(write(tmp_path, {"jev-1.13.0": ENTRY}))
    table = config.merged_pricing_table(supplied)

    entry = table["jev-1.13.0"]
    assert entry.is_priced
    assert cost.cost_of(1_000_000, 1_000_000, entry) == pytest.approx(7.5)
    assert entry.as_of == date(2026, 9, 21)


def test_merging_keeps_the_shipped_entries_it_was_not_asked_about(tmp_path: Path) -> None:
    supplied = config.load_pricing_file(write(tmp_path, {"jev-1.13.0": ENTRY}))
    table = config.merged_pricing_table(supplied)

    assert table["claude-opus-5"] is config.DEFAULT_PRICING_TABLE["claude-opus-5"]
    assert table["jev-1.13.0"] is not config.DEFAULT_PRICING_TABLE["jev-1.13.0"]


def test_no_file_supplied_leaves_the_shipped_table_exactly_as_it_is() -> None:
    assert config.merged_pricing_table(None) is config.DEFAULT_PRICING_TABLE


def test_a_null_price_stays_none_rather_than_becoming_zero(tmp_path: Path) -> None:
    """None is the absence of a claim; zero is the claim that it is free."""
    payload = {"m": {**ENTRY, "input_usd_per_million": None}}
    entry = config.load_pricing_file(write(tmp_path, payload))["m"]

    assert entry.input_usd_per_million is None
    assert not entry.is_priced
    assert cost.cost_of(100, 100, entry) is None


def test_an_underscore_key_is_a_comment_and_not_a_model(tmp_path: Path) -> None:
    payload = {"_comment": ["JSON has no comments"], "m": ENTRY}
    table = config.load_pricing_file(write(tmp_path, payload))

    assert set(table) == {"m"}


EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "pricing.example.json"


def test_the_example_file_is_refused_until_it_is_filled_in() -> None:
    """An unedited copy must not price anything, least of all at $0 (#28)."""
    with pytest.raises(config.PricingConfigError, match="placeholder"):
        config.load_pricing_file(EXAMPLE)


def test_the_example_file_loads_once_it_is_filled_in(tmp_path: Path) -> None:
    """The file the README points operators at must parse once they have done their part."""
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    for name, entry in raw.items():
        if not name.startswith("_"):
            entry.update(input_usd_per_million=1.0, output_usd_per_million=4.0, as_of="2026-09-01")
    table = config.load_pricing_file(write(tmp_path, raw))

    assert "jev-1.13.0" in table
    assert "_comment" not in table
    assert table["jev-1.13.0"].is_priced


def test_both_prices_at_zero_are_refused_unless_the_entry_says_it_is_free(
    tmp_path: Path,
) -> None:
    """0 and 0 is the claim that every token is free, which a placeholder also looks like."""
    free = {**ENTRY, "input_usd_per_million": 0, "output_usd_per_million": 0.0}
    with pytest.raises(config.PricingConfigError, match="free"):
        config.load_pricing_file(write(tmp_path, {"m": free}))

    entry = config.load_pricing_file(write(tmp_path, {"m": {**free, "free": True}}))["m"]
    assert entry.is_priced
    assert cost.cost_of(1_000, 1_000, entry) == 0.0


def test_one_price_at_zero_is_an_ordinary_price(tmp_path: Path) -> None:
    payload = {"m": {**ENTRY, "input_usd_per_million": 0}}
    entry = config.load_pricing_file(write(tmp_path, payload))["m"]
    assert cost.cost_of(1_000_000, 1_000_000, entry) == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"m": {k: v for k, v in ENTRY.items() if k != "source"}}, "missing 'source'"),
        ({"m": {k: v for k, v in ENTRY.items() if k != "as_of"}}, "missing 'as_of'"),
        ({"m": {**ENTRY, "as_of": "last tuesday"}}, "not an ISO date"),
        ({"m": {**ENTRY, "input_usd_per_million": "cheap"}}, "not a number"),
        ({"m": {**ENTRY, "inpt_usd_per_million": 1.0}}, "unrecognized field"),
        ({"m": ["not", "an", "object"]}, "expected an object"),
        ({}, "holds no entries"),
        ([ENTRY], "expected an object keyed by model string"),
    ],
)
def test_a_table_that_cannot_be_trusted_is_refused_whole(
    tmp_path: Path, payload: object, expected: str
) -> None:
    with pytest.raises(config.PricingConfigError, match=expected):
        config.load_pricing_file(write(tmp_path, payload))


def test_a_misspelled_price_field_is_named_rather_than_ignored(tmp_path: Path) -> None:
    """Silently ignoring it would price the entry at nothing and look deliberate."""
    payload = {"m": {**ENTRY, "inpt_usd_per_million": 1.0}}
    with pytest.raises(config.PricingConfigError, match="inpt_usd_per_million"):
        config.load_pricing_file(write(tmp_path, payload))


def test_a_missing_file_says_so_rather_than_falling_back(tmp_path: Path) -> None:
    with pytest.raises(config.PricingConfigError, match="no pricing file at"):
        config.load_pricing_file(tmp_path / "absent.json")


def test_broken_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "pricing.json"
    path.write_text('{"m": {,}}', encoding="utf-8")
    with pytest.raises(config.PricingConfigError, match="not valid JSON"):
        config.load_pricing_file(path)


def test_a_bool_is_not_accepted_as_a_price(tmp_path: Path) -> None:
    """``True`` is an int in Python and would silently price at one per million."""
    payload = {"m": {**ENTRY, "output_usd_per_million": True}}
    with pytest.raises(config.PricingConfigError, match="not a number"):
        config.load_pricing_file(write(tmp_path, payload))
