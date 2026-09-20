"""Shipped configuration: the pricing table, with provenance on every entry.

Pricing is config rather than code in an adapter, and this module is the default
config plumbline ships with. Nothing here is authoritative. Each entry records
the source it was read from and the date it was read, so a reader can check
whether that source still says what it said, and so a result produced today is
never silently re-scored against next year's prices.

A caller with better information passes its own table to ``runner.run``; this one
is a starting point, not a quote.
"""

from __future__ import annotations

from datetime import date

from plumbline.metrics.cost import Pricing, PricingTable

#: The date the sources below were read. Every shipped entry carries it, and the
#: report ages entries against the clock of the machine printing the report.
PRICING_READ_ON = date(2026, 9, 20)

#: Where the Jev output price comes from. It is a field description in the wire
#: schema rather than a price list, and it is phrased as a current-state claim.
JEV_OUTPUT_SOURCE = (
    "typesafe-sdk wire schema (typesafe_sdk/_schemas/models.py), Usage.output_tokens: "
    '"Number of output tokens used to answer the questions. '
    'Output tokens are currently stated at https://docs.typesafe.ai/models."'
)

#: Why no input price is shipped. The SDK publishes none, and neither the client
#: nor the schema quotes one.
JEV_INPUT_NOTE = (
    "No input price is published by the SDK or its schema, so plumbline reports no cost "
    "for Jev rather than inventing one. Add a jev entry with a real input price and the "
    "date you read it, and cost becomes available without any code change."
)

_JEV = Pricing(
    input_usd_per_million=None,
    output_usd_per_million=None,
    source=JEV_OUTPUT_SOURCE,
    as_of=PRICING_READ_ON,
    note=JEV_INPUT_NOTE,
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
#: at. ``pricing_for`` prefers whatever the API said it used, so a response
#: reporting a narrower version string than the alias that was asked for simply
#: finds no entry, which is reported as unpriced rather than guessed at.
DEFAULT_PRICING_TABLE: PricingTable = {
    "jev-1": _JEV,
    "jev-latest": _JEV,
    "claude-opus-5": _anthropic(5.0, 25.0),
    "claude-sonnet-5": _anthropic(2.0, 10.0),
    "claude-haiku-4-5": _anthropic(1.0, 5.0),
}
