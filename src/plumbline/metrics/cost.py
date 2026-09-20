"""Cost, computed from reported tokens against a pricing table held in config.

Pricing never lives in an adapter. An adapter that hardcoded a price would go
stale silently and there would be nothing in the results artifact saying which
price was used. The table is config, the table is recorded in the artifact, and
cost is derived.

Every entry carries where its price was read and the date it was read, because a
price is a current-state claim rather than a property of a model. Jev's output
tokens are the clearest case: the wire schema says they are "currently stated at https://docs.typesafe.ai/models", which is true on the day someone read it and says nothing about the day
the report is printed. An entry older than ``DEFAULT_PRICING_MAX_AGE_DAYS`` is
still used, and the report says plainly that it may be out of date rather than
presenting its numbers as current.

When an adapter reports no tokens, cost is None. Not zero. A zero in a cost
column reads as free, and free is a different claim from unknown. Local models
are the clearest case: they report no tokens and their real cost is hardware and
wall-clock, which is recorded as a note instead. ``CostBasis`` names which of
those reasons applied, so a blank column can be read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

#: How old a pricing entry may be before the report stops presenting its numbers
#: as current.
DEFAULT_PRICING_MAX_AGE_DAYS = 90


@dataclass(frozen=True, kw_only=True)
class Pricing:
    """What one model costs per million tokens, and where that number came from.

    Every field is keyword-only so that no entry can be written without saying
    where the price was read and on what date. A price with no provenance cannot
    be checked and cannot be aged, and it reads in a report exactly like one that
    was verified this morning.

    A price of None means the vendor publishes none. That is different from zero:
    zero is the claim that those tokens are free, None is the absence of a claim,
    and a cost derived from a None price is not reported rather than counted as
    nothing. ``as_of`` is the date the ``source`` was read, not the date the
    vendor set the price, because the read is the only event plumbline witnessed.
    """

    input_usd_per_million: float | None
    output_usd_per_million: float | None
    source: str
    as_of: date
    note: str = ""

    def __post_init__(self) -> None:
        for name in ("input_usd_per_million", "output_usd_per_million"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError("prices must be non-negative")
        if not self.source.strip():
            raise ValueError(
                "a pricing entry must name its source: where the price was read, so a "
                "reader can go and check whether it still says that"
            )

    @property
    def is_priced(self) -> bool:
        """Whether both sides are published, which is what costing a case needs."""
        return self.input_usd_per_million is not None and self.output_usd_per_million is not None

    def age_days(self, today: date) -> int:
        return (today - self.as_of).days

    def is_stale(self, today: date, max_age_days: int = DEFAULT_PRICING_MAX_AGE_DAYS) -> bool:
        return self.age_days(today) > max_age_days

    def statement(self, today: date, max_age_days: int = DEFAULT_PRICING_MAX_AGE_DAYS) -> str:
        """One line for the report, naming the source, the date, and the doubt."""
        age = self.age_days(today)
        parts = [f"Priced from {self.source}, read {self.as_of.isoformat()} ({age} days ago)."]
        if not self.is_priced:
            unpublished = [
                side
                for side, value in (
                    ("input", self.input_usd_per_million),
                    ("output", self.output_usd_per_million),
                )
                if value is None
            ]
            parts.append(
                f"No {' or '.join(unpublished)} price is published, so cost is not reported "
                "for this model rather than being estimated."
            )
        if self.is_stale(today, max_age_days):
            parts.append(
                f"This entry is older than {max_age_days} days and may be out of date, "
                "because a price is a current-state claim. Re-read the source before "
                "treating this cost as current."
            )
        if self.note:
            parts.append(self.note)
        return " ".join(parts)

    def provenance(
        self, today: date, max_age_days: int = DEFAULT_PRICING_MAX_AGE_DAYS
    ) -> dict[str, Any]:
        """The JSON block a run artifact carries, so an old result stays readable.

        A result re-read a year later must not be silently re-scored against the
        prices of the day it is read. The artifact records the entry that was
        applied and the date that entry was read; the report compares that date
        against its own clock and says so when the gap is large.
        """
        return {
            "source": self.source,
            "as_of": self.as_of.isoformat(),
            "age_days": self.age_days(today),
            "stale": self.is_stale(today, max_age_days),
            "max_age_days": max_age_days,
            "input_usd_per_million": self.input_usd_per_million,
            "output_usd_per_million": self.output_usd_per_million,
            "note": self.note,
            "statement": self.statement(today, max_age_days),
        }


PricingTable = Mapping[str, Pricing]


CostBasis = Literal[
    "priced",
    "cache_hit",
    "adapter_reports_no_tokens",
    "tokens_not_reported",
    "model_not_priced",
    "no_prediction",
]
"""Why a case costs what it costs, or why it costs nothing that can be reported.

A blank cost column has more than one cause, and they are different facts about
the system under test.

``adapter_reports_no_tokens``
    This adapter cannot report cost at all. A local checkpoint is the clear
    case: its real cost is hardware and wall-clock, and there is no token count
    to price. Every row from such an adapter is blank, and that says nothing
    about this particular run.

``tokens_not_reported``
    This adapter does report tokens and the API returned none on this call. The
    Jev wire schema marks both counts required while the SDK types them
    optional, so this is a fact about the run, and how often it happens is worth
    knowing before anyone quotes a bill from it.

``model_not_priced``
    Tokens arrived, but the model that answered has no usable entry in the
    pricing table.

``cache_hit``
    No call went out, so there is nothing to charge.
"""

#: Human-readable form of each basis, for the report's cost column. Keyed by
#: plain strings, because a basis read back out of a stored artifact is a
#: string and should still find its sentence.
COST_BASIS_NOTES: Mapping[str, str] = {
    "priced": "priced from reported tokens",
    "cache_hit": "cache hit, no call was made",
    "adapter_reports_no_tokens": (
        "this adapter reports no token counts at all, so cost cannot be derived for any "
        "of its rows; its real cost is hardware and wall-clock"
    ),
    "tokens_not_reported": (
        "this adapter reports token counts, and the API returned none on this call"
    ),
    "model_not_priced": "tokens were reported, but the model that answered is not priced",
    "no_prediction": "the case produced no prediction",
}


@dataclass(frozen=True)
class CostSummary:
    """Total, per case, and per correct answer, with the unpriced rows named.

    ``total_usd`` covers the priced rows only. When any row is unpriced the
    total is an undercount of the real bill, so ``is_complete`` is False and
    ``note`` says how many rows were left out and why. ``unpriced_by_basis``
    keeps the reasons apart: an adapter that cannot report tokens and an API
    that returned none on this run both produce a blank column and are not the
    same finding.
    """

    total_usd: float | None
    per_case_usd: float | None
    per_correct_usd: float | None
    priced_cases: int
    unpriced_cases: int
    correct_cases: int
    note: str
    unpriced_by_basis: Mapping[str, int] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.unpriced_cases == 0 and self.priced_cases > 0


def cost_of(
    input_tokens: int | None,
    output_tokens: int | None,
    pricing: Pricing | None,
) -> float | None:
    """Cost for one case, or None when tokens or a published price are missing.

    Both token counts must be present. The TypeSafe SDK types either field as
    ``int | None`` independently, so half a count is not a count. Both sides of
    the price must be published for the same reason: an entry with an
    unpublished input price cannot produce a total, and charging half a price as
    though it were the whole one would put a number in the report that nobody
    ever quoted.
    """
    if pricing is None or not pricing.is_priced:
        return None
    if input_tokens is None or output_tokens is None:
        return None
    input_price = pricing.input_usd_per_million
    output_price = pricing.output_usd_per_million
    assert input_price is not None and output_price is not None  # is_priced narrows both
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


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
    bases: Sequence[str] | None = None,
) -> CostSummary:
    """Aggregate per-case costs, saying plainly what could not be priced and why.

    ``bases`` is the per-case ``CostBasis``, when the caller has it. Without it
    the summary can still count the blanks; with it the summary can say which
    kind of blank they are, which is the difference between "this adapter never
    reports cost" and "the API stopped reporting tokens on this run".
    """
    if len(costs) != len(correct):
        raise ValueError(f"{len(costs)} costs against {len(correct)} outcomes; these must match")
    if bases is not None and len(bases) != len(costs):
        raise ValueError(f"{len(bases)} bases against {len(costs)} costs; these must match")

    priced = [cost for cost in costs if cost is not None]
    unpriced = len(costs) - len(priced)
    correct_count = sum(1 for outcome in correct if outcome)
    by_basis = _count_unpriced_bases(costs, bases)

    if not priced:
        note = local_note or (
            f"No cost available. None of the {len(costs)} cases could be priced, "
            "so cost is not reported rather than being shown as zero."
        )
        return CostSummary(
            total_usd=None,
            per_case_usd=None,
            per_correct_usd=None,
            priced_cases=0,
            unpriced_cases=unpriced,
            correct_cases=correct_count,
            note=" ".join([note, *_basis_sentences(by_basis)]).strip(),
            unpriced_by_basis=by_basis,
        )

    total = sum(priced)
    if unpriced:
        note = (
            f"{unpriced} of {len(costs)} cases could not be priced and are excluded, so "
            "this total is an undercount of the real bill."
        )
        note = " ".join([note, *_basis_sentences(by_basis)]).strip()
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
        unpriced_by_basis=by_basis,
    )


def _count_unpriced_bases(
    costs: Sequence[float | None], bases: Sequence[str] | None
) -> dict[str, int]:
    """Count the blank rows by the reason they are blank."""
    if bases is None:
        return {}
    counts: dict[str, int] = {}
    for cost, basis in zip(costs, bases, strict=True):
        if cost is None:
            counts[basis] = counts.get(basis, 0) + 1
    return dict(sorted(counts.items()))


def _basis_sentences(by_basis: Mapping[str, int]) -> list[str]:
    """One sentence per reason, so the reasons are never merged into a count."""
    return [
        f"{count} because {COST_BASIS_NOTES.get(basis, basis)}."
        for basis, count in by_basis.items()
    ]


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
    if pricing is None or not pricing.is_priced:
        return None
    characters = len(text) + sum(len(label) for label in labels)
    input_tokens = int(characters / chars_per_token) + overhead_tokens
    return cost_of(input_tokens, output_tokens, pricing)
