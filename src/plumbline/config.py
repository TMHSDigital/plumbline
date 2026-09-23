"""Shipped configuration: the pricing table, with provenance on every entry.

Pricing is config rather than code in an adapter, and this module is the default
config plumbline ships with. Nothing here is authoritative. Each entry records
the source it was read from and the date it was read, so a reader can check
whether that source still says what it said, and so a result produced today is
never silently re-scored against next year's prices.

A caller with better information passes its own table to ``runner.run``; this one
is a starting point, not a quote.

Some vendors treat their prices as confidential, so plumbline ships no figures
for them at all: the entry names the page to read and prices nothing until the
operator supplies the numbers. ``load_pricing_file`` reads that supplied table
and ``merged_pricing_table`` lays it over the shipped defaults, which is what
``--pricing`` does. A user-supplied entry must carry its own source and date for
the same reason a shipped one must.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from plumbline.metrics.cost import Pricing, PricingTable
from plumbline.types import PlumblineError


class PricingConfigError(PlumblineError):
    """A supplied pricing file could not be read as a pricing table.

    Separate from a dataset error because the remedy is different: a bad row in
    a dataset is refused and the run continues over the rest, while a pricing
    file that cannot be read would silently change every cost figure in the
    report. It is refused whole.
    """


#: The date the sources below were read. Every shipped entry carries it, and the
#: report ages entries against the clock of the machine printing the report.
PRICING_READ_ON = date(2026, 9, 21)

#: Where to read the Jev tariff. The URL, deliberately without the figures on it.
#:
#: The vendor's customer agreement makes its pricing information confidential and
#: overrides the usual public-knowledge carve-out, so plumbline does not restate
#: the numbers in a file it publishes: not in a price field, not in a quotation,
#: and not in a worked example whose token counts would let a reader divide one
#: out. The page is public and the reader can go and read it. Supplying the
#: figures is the operator's act, not plumbline's.
JEV_SOURCE = "TypeSafe published tariff, https://docs.typesafe.ai/models.md (read it and supply)"

#: Why the shipped entry prices nothing, and how to make it price something.
JEV_NOTE = (
    "plumbline ships no figures for this vendor. Read the tariff at the source above "
    "and pass your own table with --pricing to get a cost column; see "
    "docs/pricing.example.json. The keys here are the model strings that need an "
    "entry: an alias resolves server-side and the response reports a version, and "
    "pricing_for prefers what answered, so a table keyed only on the alias prices "
    "every live row as model_not_priced. There is no prefix matching anywhere, so a "
    "later version never inherits a superseded rate."
)

_JEV = Pricing(
    # Both sides are None on purpose, and None is not zero. None is the absence of
    # a claim; a 0.0 here would be plumbline asserting those tokens are free, which
    # is a statement about the vendor's prices that this file does not make.
    input_usd_per_million=None,
    output_usd_per_million=None,
    source=JEV_SOURCE,
    as_of=PRICING_READ_ON,
    note=JEV_NOTE,
)

#: Where the generative arm's prices come from, and the date that table was
#: compiled. It is a cached copy of a list price rather than an invoice, which
#: is exactly why the date travels with it.
ANTHROPIC_SOURCE = (
    "Anthropic list pricing, as quoted by the bundled claude-api skill model table "
    "(cached 2026-06-24)"
)

#: The date ANTHROPIC_SOURCE was compiled, not the date it was copied in here.
#: Ageing from the copy date would restart a clock that was already running.
ANTHROPIC_READ_ON = date(2026, 6, 24)


def _anthropic(input_usd: float, output_usd: float) -> Pricing:
    return Pricing(
        input_usd_per_million=input_usd,
        output_usd_per_million=output_usd,
        source=ANTHROPIC_SOURCE,
        as_of=ANTHROPIC_READ_ON,
    )


#: Default table, keyed by the model strings plumbline can actually be pointed
#: at, and by the version strings the API reports back. ``pricing_for`` prefers
#: whatever the API said it used, so a response reporting a narrower version
#: string than the alias that was asked for finds an entry only if that version
#: is listed here; otherwise it is reported as unpriced rather than guessed at.
#: There is deliberately no prefix or fuzzy match: ``jev-1.14.0`` is not
#: ``jev-1.13.0`` and must not inherit its price.
DEFAULT_PRICING_TABLE: PricingTable = {
    "jev-1.13.0": _JEV,
    "jev-latest": _JEV,
    "jev-preview": _JEV,
    "claude-opus-5": _anthropic(5.0, 25.0),
    "claude-sonnet-5": _anthropic(2.0, 10.0),
    "claude-haiku-4-5": _anthropic(1.0, 5.0),
}

#: Fields a supplied entry may carry. Anything else is a typo worth naming.
_PRICING_FIELDS = frozenset(
    {"input_usd_per_million", "output_usd_per_million", "source", "as_of", "note", "free"}
)

#: The ``as_of`` that ``docs/pricing.example.json`` ships with. A copy that still
#: carries it has not been filled in, so it is refused rather than priced.
TEMPLATE_AS_OF = "YYYY-MM-DD"

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def load_pricing_file(path: Path | str) -> PricingTable:
    """Read a pricing table the operator wrote, requiring provenance on every entry.

    The file is a JSON object keyed by model string. Each value carries
    ``input_usd_per_million``, ``output_usd_per_million``, ``source``, ``as_of``
    (ISO date), and an optional ``note``. A price may be ``null``, meaning the
    vendor publishes none, which is different from ``0``.

    Nothing is defaulted. A missing ``source`` or ``as_of`` is an error rather
    than a blank, because an undated price with no source reads in a report
    exactly like one that was checked this morning, and the whole point of this
    file is that the operator is the one making the claim.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise PricingConfigError(
            f"no pricing file at {path}. plumbline never guesses a pricing file "
            "location, because a relative default resolves against whatever directory "
            "the process started in."
        ) from None
    except json.JSONDecodeError as broken:
        raise PricingConfigError(
            f"{path} is not valid JSON: {broken.msg} (line {broken.lineno})"
        ) from broken

    if not isinstance(raw, dict):
        raise PricingConfigError(
            f"{path} holds a {type(raw).__name__}, expected an object keyed by model string."
        )
    # JSON has no comments, and a pricing file is exactly the kind of file that
    # wants them. A key starting with an underscore is a note to the next reader,
    # not a model, and is skipped. No real model string begins with one.
    entries = {name: value for name, value in raw.items() if not name.startswith("_")}
    if not entries:
        raise PricingConfigError(
            f"{path} holds no entries. An empty pricing file is refused rather than "
            "treated as 'use the defaults', because those are different intentions."
        )

    return {name: _entry(name, value, path) for name, value in entries.items()}


def merged_pricing_table(supplied: PricingTable | None) -> PricingTable:
    """The shipped table with the operator's entries laid over it.

    An operator's entry wins on its key. That is what makes it possible to price
    a vendor plumbline ships no figures for without editing the package, and to
    correct a shipped price that has gone stale without waiting for a release.
    """
    if not supplied:
        return DEFAULT_PRICING_TABLE
    return {**DEFAULT_PRICING_TABLE, **supplied}


def _entry(name: str, value: Any, path: Path) -> Pricing:
    """One supplied entry, or an error naming the model it belongs to."""
    if not isinstance(value, Mapping):
        raise PricingConfigError(
            f"entry {name!r} in {path} is a {type(value).__name__}, expected an object."
        )

    unexpected = sorted(set(value) - _PRICING_FIELDS)
    if unexpected:
        raise PricingConfigError(
            f"entry {name!r} in {path} has unrecognized field(s) {unexpected!r}. "
            f"Allowed: {sorted(_PRICING_FIELDS)!r}. A misspelled price field would be "
            "silently ignored and the entry would price nothing, so it is refused."
        )
    for required in ("source", "as_of"):
        if required not in value:
            raise PricingConfigError(
                f"entry {name!r} in {path} is missing {required!r}. Every price carries "
                "where it was read and the date it was read, so a reader can check "
                "whether that source still says it and a report can age it."
            )

    as_of = _as_of(name, value["as_of"], path)
    input_price = _price(name, value, "input_usd_per_million", path)
    output_price = _price(name, value, "output_usd_per_million", path)
    free = value.get("free", False)
    if not isinstance(free, bool):
        raise PricingConfigError(
            f"entry {name!r} in {path} has free={free!r}; it is true or absent."
        )
    # Both prices at exactly zero is what an unedited template looks like, and
    # it would let any run past --max-cost-usd. A model that really is free says
    # so, and one that says so must not also carry a price.
    both_zero = input_price == 0 and output_price == 0
    if both_zero and not free:
        raise PricingConfigError(
            f"entry {name!r} in {path} prices input and output at 0, which claims every "
            "token is free and would let any run past --max-cost-usd. If that is true, "
            'add "free": true to the entry; if the price is unknown, use null.'
        )
    if free and not both_zero:
        raise PricingConfigError(
            f'entry {name!r} in {path} says "free": true but carries a price. '
            "Set both prices to 0, or remove free."
        )

    return Pricing(
        input_usd_per_million=input_price,
        output_usd_per_million=output_price,
        source=str(value["source"]),
        as_of=as_of,
        note=str(value.get("note", "")),
    )


def _price(name: str, value: Mapping[str, Any], field: str, path: Path) -> float | None:
    """A price, or None where the vendor publishes none. Never a silent zero."""
    if field not in value or value[field] is None:
        return None
    number = value[field]
    if isinstance(number, bool) or not isinstance(number, int | float):
        raise PricingConfigError(
            f"entry {name!r} in {path} has {field}={number!r}, which is not a number "
            "or null. Use null where the vendor publishes no price; null and 0 are "
            "different claims and plumbline keeps them apart."
        )
    if not math.isfinite(number):
        raise PricingConfigError(
            f"entry {name!r} in {path} has {field}={number!r}; a price must be finite, "
            "or every row it prices becomes NaN or infinite."
        )
    if number < 0:
        raise PricingConfigError(
            f"entry {name!r} in {path} has {field}={number!r}; a price cannot be negative."
        )
    return float(number)


def _as_of(name: str, value: Any, path: Path) -> date:
    if value == TEMPLATE_AS_OF:
        raise PricingConfigError(
            f"entry {name!r} in {path} still has the template's placeholder "
            f"as_of={TEMPLATE_AS_OF!r}. Fill in the prices you read and the date you read "
            "them; an unedited copy of docs/pricing.example.json is refused rather than "
            "used to price a run."
        )
    # fromisoformat also accepts 20260901 and week dates such as 2026-W36-1,
    # which nobody writes meaning a date, so the shape is checked first.
    if not (isinstance(value, str) and _ISO_DATE.fullmatch(value)):
        raise PricingConfigError(
            f"entry {name!r} in {path} has as_of={value!r}, which is not an ISO date "
            "written YYYY-MM-DD. The date a price was read is what lets the report say the price "
            "may be out of date, so it is not optional and not free-form."
        )
    try:
        read_on = date.fromisoformat(value)
    except ValueError:
        raise PricingConfigError(
            f"entry {name!r} in {path} has as_of={value!r}, which is not an ISO date "
            "(YYYY-MM-DD). The date a price was read is what lets the report say the "
            "price may be out of date, so it is not optional and not free-form."
        ) from None
    if read_on > date.today():
        raise PricingConfigError(
            f"entry {name!r} in {path} has as_of={value!r}, which is in the future. A "
            "price cannot have been read on a day that has not happened, and a future date "
            "would keep the report from ever calling it out of date."
        )
    return read_on
