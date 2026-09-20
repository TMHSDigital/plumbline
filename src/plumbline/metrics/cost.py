"""Cost, computed from reported tokens against a pricing table held in config.

Pricing never lives in an adapter. An adapter that hardcoded a price would go
stale silently and there would be nothing in the results artifact saying which
price was used. The table is config, the table is recorded in the artifact, and
cost is derived.

When an adapter reports no tokens, cost is None. Not zero. A zero in a cost
column reads as free, and free is a different claim from unknown. Local models
are the clearest case: they report no tokens and their real cost is hardware and
wall-clock, which is recorded as a note instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    """What one model costs, per million tokens, in USD."""

    input_usd_per_million: float
    output_usd_per_million: float

    def __post_init__(self) -> None:
        if self.input_usd_per_million < 0 or self.output_usd_per_million < 0:
            raise ValueError("prices must be non-negative")


PricingTable = Mapping[str, Pricing]


@dataclass(frozen=True)
class CostSummary:
    """Total, per case, and per correct answer, with the unpriced rows named.

    ``total_usd`` covers the priced rows only. When any row is unpriced the
    total is an undercount of the real bill, so ``is_complete`` is False and
    ``note`` says how many rows were left out and why.
    """

    total_usd: float | None
    per_case_usd: float | None
    per_correct_usd: float | None
    priced_cases: int
    unpriced_cases: int
    correct_cases: int
    note: str

    @property
    def is_complete(self) -> bool:
        return self.unpriced_cases == 0 and self.priced_cases > 0


def cost_of(
    input_tokens: int | None,
    output_tokens: int | None,
    pricing: Pricing | None,
) -> float | None:
    """Cost for one case, or None when tokens or pricing are missing.

    Both token counts must be present. The TypeSafe SDK types either field as
    ``int | None`` independently, so half a count is not a count.
    """
    if pricing is None or input_tokens is None or output_tokens is None:
        return None
    return (
        input_tokens * pricing.input_usd_per_million
        + output_tokens * pricing.output_usd_per_million
    ) / 1_000_000


def pricing_for(
    table: PricingTable,
    model_reported: str | None,
    model_requested: str,
) -> tuple[Pricing | None, str | None]:
    """Look up pricing, preferring what the API said it used.

    A model alias can resolve somewhere other than where the config pointed, and
    billing follows what actually answered. The matched key is returned so the
    artifact can record which row of the table was applied.
    """
    for candidate in (model_reported, model_requested):
        if candidate is not None and candidate in table:
            return table[candidate], candidate
    return None, None


def summarize(
    costs: Sequence[float | None],
    correct: Sequence[bool],
    *,
    local_note: str | None = None,
) -> CostSummary:
    """Aggregate per-case costs, saying plainly what could not be priced."""
    if len(costs) != len(correct):
        raise ValueError(f"{len(costs)} costs against {len(correct)} outcomes; these must match")

    priced = [cost for cost in costs if cost is not None]
    unpriced = len(costs) - len(priced)
    correct_count = sum(1 for outcome in correct if outcome)

    if not priced:
        note = local_note or (
            f"No cost available. None of the {len(costs)} cases reported token counts, "
            "so cost is not reported rather than being shown as zero."
        )
        return CostSummary(
            total_usd=None,
            per_case_usd=None,
            per_correct_usd=None,
            priced_cases=0,
            unpriced_cases=unpriced,
            correct_cases=correct_count,
            note=note,
        )

    total = sum(priced)
    if unpriced:
        note = (
            f"{unpriced} of {len(costs)} cases reported no tokens and are excluded, so "
            "this total is an undercount of the real bill."
        )
    else:
        note = f"All {len(costs)} cases priced."

    return CostSummary(
        total_usd=total,
        per_case_usd=total / len(priced),
        per_correct_usd=total / correct_count if correct_count else None,
        priced_cases=len(priced),
        unpriced_cases=unpriced,
        correct_cases=correct_count,
        note=note,
    )


def estimate_case_cost(
    text: str,
    labels: Sequence[str],
    pricing: Pricing | None,
    *,
    chars_per_token: float = 4.0,
    output_tokens: int = 8,
    overhead_tokens: int = 32,
) -> float | None:
    """A rough pre-run estimate, for the cost guard to abort against.

    Deliberately crude and deliberately not conservative in the user's favour:
    it counts the case text, the option names, and a fixed overhead for whatever
    scaffolding the adapter adds. Real tokenization differs by model. The guard
    exists to stop a runaway run, not to quote a price.
    """
    if pricing is None:
        return None
    characters = len(text) + sum(len(label) for label in labels)
    input_tokens = int(characters / chars_per_token) + overhead_tokens
    return cost_of(input_tokens, output_tokens, pricing)
