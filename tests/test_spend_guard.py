"""`SpendGuard` and `GuardedClient`: a hard spending limit in front of the Anthropic client.

Everything runs against the scripted `FakeClient`: no network, no real key, fictional texts.
"""

import contextlib
import datetime as dt
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import anthropic
import pytest
from pydantic import SecretStr

from anthropic_fakes import (
    FAKE_KEY,
    FAKE_TEXT,
    FakeClient,
    RecordingSleep,
    make_message,
    result,
    status_error,
)
from review_radar.analysis import analyze_pending
from review_radar.anthropic_classifier import PROMPT_VERSION, AnthropicClassifier
from review_radar.classifier import ClassifyItem, FatalClassifierError
from review_radar.evaluation import EvalError, GuardedClient, SpendGuard
from review_radar.models import Review, Usage
from review_radar.pricing import estimate_cost
from review_radar.storage import connect, load_analyses, save_reviews

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
LIMIT_CALL_TWO = Decimal("0.0040")  # fits one default-size request, not a second one after spending
HEAVY = {"input_tokens": 10_000, "output_tokens": 3_000}  # costs 0.0025 on haiku


def ok(n: int = 1, **usage: int):
    return make_message([result(f"r{i}") for i in range(1, n + 1)], **usage)


def request(*, chars: int = 1000, max_tokens: int = 1000) -> dict[str, Any]:
    return {
        "model": HAIKU,
        "max_tokens": max_tokens,
        "system": "s" * chars,
        "messages": [{"role": "user", "content": "u" * chars}],
    }


def guarded(
    client: FakeClient, limit: str | Decimal, *, model: str = HAIKU, spent: str | Decimal = 0
) -> tuple[GuardedClient, SpendGuard]:
    guard = SpendGuard(Decimal(limit), Decimal(spent))
    return GuardedClient(client, guard, model), guard


class TestConstruction:
    def test_a_negative_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="limit"):
            SpendGuard(Decimal("-0.01"))

    def test_a_zero_limit_is_allowed(self) -> None:
        assert SpendGuard(Decimal(0)).spent == 0

    def test_spent_starts_at_zero_or_at_the_given_amount(self) -> None:
        assert SpendGuard(Decimal(1)).spent == Decimal(0)
        assert SpendGuard(Decimal(1), Decimal("0.25")).spent == Decimal("0.25")

    def test_an_unknown_model_raises_before_any_call(self) -> None:
        fake = FakeClient(ok())

        with pytest.raises(EvalError):
            client = GuardedClient(fake, SpendGuard(Decimal(10)), "claude-unknown-9")
            client.messages.create(**request())

        assert fake.calls == []


class TestForwarding:
    def test_a_call_within_the_limit_is_forwarded_unchanged(self) -> None:
        message = ok()
        fake = FakeClient(message)
        client, _ = guarded(fake, "1.00")

        returned = client.messages.create(**request())

        assert returned is message
        assert fake.calls == [request()]

    def test_the_real_usage_is_added_to_spent(self) -> None:
        fake = FakeClient(
            ok(input_tokens=100, output_tokens=20), ok(input_tokens=50, output_tokens=0)
        )
        client, guard = guarded(fake, "1.00")

        client.messages.create(**request())
        assert guard.spent == estimate_cost(HAIKU, Usage(input_tokens=100, output_tokens=20))
        client.messages.create(**request())

        assert guard.spent == estimate_cost(HAIKU, Usage(input_tokens=150, output_tokens=20))

    def test_spent_is_priced_for_the_guarded_model(self) -> None:
        client, guard = guarded(
            FakeClient(ok(input_tokens=100, output_tokens=20)), "5", model=SONNET
        )

        client.messages.create(**request())

        assert guard.spent == Decimal("0.0004")  # 100 * 2 / 1M + 20 * 10 / 1M

    def test_an_exception_passes_through_unchanged_and_adds_nothing(self) -> None:
        error = status_error(500)
        fake = FakeClient(error)
        client, guard = guarded(fake, "1.00")

        with pytest.raises(anthropic.InternalServerError) as caught:
            client.messages.create(**request())

        assert caught.value is error
        assert guard.spent == 0
        assert len(fake.calls) == 1


class TestRefusal:
    def test_a_call_whose_worst_case_exceeds_the_limit_is_refused_without_calling(self) -> None:
        fake = FakeClient(ok())
        client, guard = guarded(fake, "0.0001")

        with pytest.raises(FatalClassifierError) as caught:
            client.messages.create(**request())

        assert caught.value.reason == "budget_exceeded"
        assert fake.calls == []
        assert fake.remaining == 1
        assert guard.spent == 0

    def test_the_refusal_message_holds_no_text_and_no_key(self) -> None:
        sent = request()
        sent["messages"] = [{"role": "user", "content": f"{FAKE_TEXT} {FAKE_KEY}" * 100}]
        client, _ = guarded(FakeClient(ok()), "0.0001")

        with pytest.raises(FatalClassifierError) as caught:
            client.messages.create(**sent)

        assert FAKE_TEXT not in str(caught.value)
        assert FAKE_KEY not in str(caught.value)

    def test_the_output_is_priced_at_max_tokens(self) -> None:
        # 4096 output tokens on haiku cost 0.002048 on their own.
        refused = FakeClient(ok())
        client, _ = guarded(refused, "0.0020")
        with pytest.raises(FatalClassifierError):
            client.messages.create(**request(chars=10, max_tokens=4096))
        assert refused.calls == []

        accepted = FakeClient(ok())
        client, _ = guarded(accepted, "0.0020")
        client.messages.create(**request(chars=10, max_tokens=1000))
        assert len(accepted.calls) == 1

    def test_the_input_is_counted_at_least_one_token_per_character(self) -> None:
        # Output alone (1000 tokens) is 0.0005; 100 more input tokens leave far too little room
        # for 2000 characters of system prompt and message.
        fake = FakeClient(ok())
        client, _ = guarded(fake, Decimal("0.0005") + Decimal("0.00001"))

        with pytest.raises(FatalClassifierError):
            client.messages.create(**request(chars=1000, max_tokens=1000))

        assert fake.calls == []

    def test_the_worst_case_still_fits_a_reasonable_limit(self) -> None:
        fake = FakeClient(ok())
        client, _ = guarded(fake, "0.0020")

        client.messages.create(**request(chars=1000, max_tokens=1000))

        assert len(fake.calls) == 1

    def test_message_text_in_content_blocks_is_counted(self) -> None:
        sent = request(chars=1, max_tokens=1000)
        sent["system"] = ""
        sent["messages"] = [
            {"role": "user", "content": [{"type": "text", "text": "u" * 2000}]},
        ]
        fake = FakeClient(ok())
        client, _ = guarded(fake, Decimal("0.0005") + Decimal("0.00001"))

        with pytest.raises(FatalClassifierError):
            client.messages.create(**sent)

        assert fake.calls == []

    def test_the_price_of_the_guarded_model_is_used(self) -> None:
        # 1000 output tokens: 0.0005 on haiku, 0.01 on sonnet.
        haiku_fake, sonnet_fake = FakeClient(ok()), FakeClient(ok())
        haiku, _ = guarded(haiku_fake, "0.005", model=HAIKU)
        sonnet, _ = guarded(sonnet_fake, "0.005", model=SONNET)

        haiku.messages.create(**request(chars=10))
        with pytest.raises(FatalClassifierError):
            sonnet.messages.create(**request(chars=10))

        assert len(haiku_fake.calls) == 1
        assert sonnet_fake.calls == []

    def test_money_already_spent_counts_against_the_limit(self) -> None:
        fake = FakeClient(ok())
        client, _ = guarded(fake, "1.00", spent="1.00")

        with pytest.raises(FatalClassifierError):
            client.messages.create(**request(chars=1, max_tokens=1))

        assert fake.calls == []

    def test_spending_during_the_run_leaves_less_room(self) -> None:
        fake = FakeClient(ok(**HEAVY), ok())
        client, guard = guarded(fake, LIMIT_CALL_TWO)
        client.messages.create(**request(chars=10, max_tokens=1000))  # worst case 0.0005
        assert guard.spent == Decimal("0.0025")

        with pytest.raises(FatalClassifierError):
            client.messages.create(**request(chars=10, max_tokens=4096))  # 0.0025 + 0.002048

        assert len(fake.calls) == 1

    def test_a_zero_limit_refuses_every_call(self) -> None:
        fake = FakeClient(ok())
        client, _ = guarded(fake, "0")

        with pytest.raises(FatalClassifierError):
            client.messages.create(**request(chars=1, max_tokens=1))

        assert fake.calls == []


def classifier_for(client: GuardedClient, sleep: Any = None) -> AnthropicClassifier:
    return AnthropicClassifier(
        api_key=SecretStr(FAKE_KEY),
        model=HAIKU,
        client=client,
        sleep=sleep or RecordingSleep(),
    )


def items(n: int) -> list[ClassifyItem]:
    return [
        ClassifyItem(item_id=f"r{i}", text=f"fictional review {i}", rating=3)
        for i in range(1, n + 1)
    ]


class TestWithTheClassifier:
    def test_the_run_stops_before_the_call_that_could_exceed_the_limit(
        self, tmp_path: Path
    ) -> None:
        fake = FakeClient(ok(1, **HEAVY), ok(1, **HEAVY), ok(1, **HEAVY))
        client, guard = guarded(fake, LIMIT_CALL_TWO)
        with contextlib.closing(connect(tmp_path / "guard.db")) as conn:
            save_reviews(
                conn,
                [
                    Review(
                        external_id=f"g{i}",
                        review_date=dt.date(2026, 11, 10),
                        rating=3,
                        text=f"fictional review {i}",
                        source="web",
                    )
                    for i in range(3)
                ],
            )

            report = analyze_pending(conn, classifier_for(client), batch_size=1)

            saved = load_analyses(
                conn, provider="anthropic", model=HAIKU, prompt_version=PROMPT_VERSION
            )

        assert len(fake.calls) == 1  # the second call was never made
        assert fake.remaining == 2
        assert report.fatal_reason == "budget_exceeded"
        assert report.analyzed == 1  # the valid analysis of the first call survives
        assert len(saved) == 1
        assert report.failed == 2
        assert report.usage == Usage(**HEAVY)
        assert guard.spent == Decimal("0.0025")

    def test_a_refused_first_call_makes_no_call_and_raises_fatal(self) -> None:
        fake = FakeClient(ok(1))
        client, _ = guarded(fake, "0.0001")

        with pytest.raises(FatalClassifierError) as caught:
            classifier_for(client).classify(items(1))

        assert caught.value.reason == "budget_exceeded"
        assert fake.calls == []

    def test_every_retry_attempt_is_checked(self) -> None:
        fake = FakeClient(status_error(429), ok(1))
        client, guard = guarded(fake, "1.00")

        def other_spender(_: float) -> None:
            guard.spent = guard.limit  # money is spent elsewhere while the classifier waits

        with pytest.raises(FatalClassifierError) as caught:
            classifier_for(client, other_spender).classify(items(1))

        assert caught.value.reason == "budget_exceeded"
        assert len(fake.calls) == 1  # the retry never reached the client
        assert fake.remaining == 1

    def test_a_retry_within_the_limit_goes_through_and_costs_only_the_success(self) -> None:
        fake = FakeClient(status_error(429), ok(1, input_tokens=100, output_tokens=20))
        client, guard = guarded(fake, "1.00")

        outcome = classifier_for(client).classify(items(1))

        assert set(outcome.analyses) == {"r1"}
        assert len(fake.calls) == 2
        assert guard.spent == Decimal("0.00002")

    def test_every_batch_split_is_checked(self) -> None:
        # The first response is unusable (paid for), so the batch splits into two halves.
        fake = FakeClient(make_message("not json", **HEAVY), ok(1), ok(1))
        client, guard = guarded(fake, LIMIT_CALL_TWO)

        with pytest.raises(FatalClassifierError) as caught:
            classifier_for(client).classify(items(2))

        assert caught.value.reason == "budget_exceeded"
        assert len(fake.calls) == 1  # the first half was refused
        assert fake.remaining == 2
        assert caught.value.usage == Usage(**HEAVY)  # the paid call is still reported
        assert guard.spent == Decimal("0.0025")

    def test_a_split_within_the_limit_makes_all_three_calls(self) -> None:
        second_half = make_message([result("r2")])
        fake = FakeClient(make_message("not json", **HEAVY), ok(1), second_half)
        client, guard = guarded(fake, "1.00")

        outcome = classifier_for(client).classify(items(2))

        assert set(outcome.analyses) == {"r1", "r2"}
        assert len(fake.calls) == 3
        assert guard.spent >= Decimal("0.0025")

    def test_what_the_classifier_sends_is_what_the_guard_prices(self) -> None:
        fake = FakeClient(ok(1))
        client, _ = guarded(fake, "1.00")

        classifier_for(client).classify(items(1))

        sent = fake.calls[0]
        assert json.loads(sent["messages"][0]["content"])[0]["id"] == "r1"
        assert sent["max_tokens"] > 0
