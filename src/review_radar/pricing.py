"""Token prices and cost estimates.

Prices are USD per 1M tokens (input, output) for prompts up to 100K tokens; every batch is far
below that. Estimates only count input and output tokens, as the model reports them. Not review
data: nothing here touches text.
"""

from decimal import Decimal

from review_radar.models import Usage

PRICES_PER_MTOK: dict[str, tuple[Decimal, Decimal]] = {
    "claude-haiku-5-5": (Decimal("0.10"), Decimal("0.50")),
    "claude-sonnet-5-5": (Decimal("2.00"), Decimal("10.00")),
}

_MILLION = Decimal(1_000_000)


def estimate_cost(model: str, usage: Usage) -> Decimal | None:
    """Return the cost in USD of `usage` on `model`, or None when the model has no known price."""
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    price_in, price_out = prices
    return Decimal(usage.input_tokens) * price_in / _MILLION + (
        Decimal(usage.output_tokens) * price_out / _MILLION
    )
