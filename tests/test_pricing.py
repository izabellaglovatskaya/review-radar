from decimal import Decimal

import pytest

from review_radar.models import Usage
from review_radar.pricing import PRICES_PER_MTOK, estimate_cost


def test_price_table_matches_the_plan() -> None:
    assert {
        "claude-haiku-5-5": (Decimal("0.10"), Decimal("0.50")),
        "claude-sonnet-5-5": (Decimal("2.00"), Decimal("10.00")),
    } == PRICES_PER_MTOK


def test_haiku_cost_of_a_300_review_run_is_exact() -> None:
    usage = Usage(input_tokens=39_000, output_tokens=9_300)

    assert estimate_cost("claude-haiku-5-5", usage) == Decimal("0.00855")


def test_sonnet_cost_is_exact() -> None:
    usage = Usage(input_tokens=39_000, output_tokens=9_300)

    assert estimate_cost("claude-sonnet-5-5", usage) == Decimal("0.171")


def test_zero_usage_costs_zero() -> None:
    assert estimate_cost("claude-haiku-5-5", Usage(input_tokens=0, output_tokens=0)) == 0


def test_one_million_tokens_each_way_costs_the_list_price() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)

    assert estimate_cost("claude-haiku-5-5", usage) == Decimal("0.60")


@pytest.mark.parametrize("model", ["demo", "", "claude-haiku-5-5-preview", "CLAUDE-HAIKU-5-5"])
def test_unknown_model_has_no_estimate(model: str) -> None:
    assert estimate_cost(model, Usage(input_tokens=1, output_tokens=1)) is None


def test_result_is_a_decimal_not_a_float() -> None:
    cost = estimate_cost("claude-haiku-5-5", Usage(input_tokens=1, output_tokens=1))

    assert isinstance(cost, Decimal)
