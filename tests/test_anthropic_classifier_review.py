"""Review round: spend-limit and credit detection, SDK error handling, status codes,
`x-should-retry`, partial results on failure and the per-run call budget."""

import json
import logging
from typing import Any

import pytest
from pydantic import SecretStr

from anthropic_fakes import (
    FAKE_KEY,
    FAKE_TEXT,
    FakeClient,
    RecordingSleep,
    connection_error,
    generic_api_error,
    make_message,
    response_validation_error,
    result,
    status_error,
)
from review_radar import anthropic_classifier as module
from review_radar.anthropic_classifier import AnthropicClassifier
from review_radar.classifier import ClassifierError, ClassifyItem, FatalClassifierError
from review_radar.models import Usage

ITEMS = [
    ClassifyItem("r1", "Great app, setup took two minutes.", 5),
    ClassifyItem("r2", "Parcel arrived ten days late and the box was crushed.", 1),
    ClassifyItem("r3", "Support never answered and the price went up again.", None),
]
ONE = ITEMS[:1]
SPEND = (
    "The Anthropic workspace reached its monthly spend limit; "
    "raise the limit in the Console or wait until it resets."
)
BILLING = "The Anthropic account has no available credit; add credit in the Console."
BAD_REQUEST = (
    "The Anthropic API rejected the request (HTTP 400). "
    "Check the model name and the request settings."
)


def build(client: FakeClient, sleep: RecordingSleep | None = None, **kwargs: Any):
    return AnthropicClassifier(
        api_key=SecretStr(FAKE_KEY), client=client, sleep=sleep or RecordingSleep(), **kwargs
    )


def ids(call: dict[str, Any]) -> list[str]:
    return [e["id"] for e in json.loads(call["messages"][0]["content"])]


class TestSpendLimitAndCredit:
    def spend_cap(self, **kwargs: Any):
        return status_error(
            429, error_type="rate_limit_error", error_code="enforced_spend_limit_reached", **kwargs
        )

    def test_a_spend_cap_429_is_fatal_after_exactly_one_call_and_no_sleep(self) -> None:
        client = FakeClient(self.spend_cap(), make_message([result("r1")]))
        sleep = RecordingSleep()

        with pytest.raises(FatalClassifierError) as excinfo:
            build(client, sleep).classify(ITEMS)

        assert excinfo.value.reason == "spend_limit_reached"
        assert str(excinfo.value).startswith(SPEND)
        assert "HTTP 429" in str(excinfo.value)
        assert len(client.calls) == 1
        assert sleep.delays == []

    def test_other_429_codes_stay_transient(self) -> None:
        errors = [status_error(429, error_code="some_other_code") for _ in range(4)]
        client = FakeClient(*errors)

        with pytest.raises(ClassifierError) as excinfo:
            build(client).classify(ONE)

        assert not isinstance(excinfo.value, FatalClassifierError)
        assert excinfo.value.reason == "rate_limited"
        assert len(client.calls) == 4

    @pytest.mark.parametrize(
        "body",
        [None, "text", [], {"error": None}, {"error": "x"}, {"error": {"details": "x"}}],
        ids=["none", "str", "list", "error-none", "error-str", "details-str"],
    )
    def test_an_odd_429_body_is_read_defensively_and_stays_transient(self, body: Any) -> None:
        client = FakeClient(status_error(429, body=body), make_message([result("r1")]))

        outcome = build(client).classify(ONE)

        assert "r1" in outcome.analyses
        assert len(client.calls) == 2

    def test_the_spend_cap_wins_over_x_should_retry_true(self) -> None:
        client = FakeClient(self.spend_cap(headers={"x-should-retry": "true"}))

        with pytest.raises(FatalClassifierError) as excinfo:
            build(client).classify(ONE)

        assert excinfo.value.reason == "spend_limit_reached"
        assert len(client.calls) == 1

    @pytest.mark.parametrize(
        ("message", "code"),
        [
            ("Your credit balance is too low to access the Anthropic API.", None),
            ("YOUR CREDIT BALANCE IS TOO LOW", None),
            ("Something", "insufficient_credit"),
            ("Something", "Billing_hold"),
            ("Something", "CREDITS_EXHAUSTED"),
        ],
    )
    def test_a_400_showing_credit_exhaustion_is_billing_error(
        self, message: str, code: str | None
    ) -> None:
        client = FakeClient(
            status_error(400, error_type="invalid_request_error", message=message, error_code=code)
        )

        with pytest.raises(FatalClassifierError) as excinfo:
            build(client).classify(ONE)

        assert excinfo.value.reason == "billing_error"
        assert str(excinfo.value).startswith(BILLING)
        assert len(client.calls) == 1

    def test_a_plain_400_is_bad_request_with_the_new_wording(self) -> None:
        client = FakeClient(status_error(400, error_type="invalid_request_error"))

        with pytest.raises(FatalClassifierError) as excinfo:
            build(client).classify(ONE)

        assert excinfo.value.reason == "bad_request"
        assert str(excinfo.value).startswith(BAD_REQUEST)
        assert "malformed" not in str(excinfo.value)

    def test_a_402_is_still_billing_error(self) -> None:
        with pytest.raises(FatalClassifierError) as excinfo:
            build(FakeClient(status_error(402))).classify(ONE)

        assert excinfo.value.reason == "billing_error"

    @pytest.mark.parametrize(
        "make_error",
        [
            lambda: status_error(
                400, message=f"credit balance too low {FAKE_TEXT} {FAKE_KEY}", request_id="req_x"
            ),
            lambda: status_error(
                429,
                message=f"cap {FAKE_TEXT} {FAKE_KEY}",
                error_code="enforced_spend_limit_reached",
            ),
            lambda: status_error(400, message=f"{FAKE_TEXT} {FAKE_KEY}"),
        ],
        ids=["credit-400", "spend-cap", "plain-400"],
    )
    def test_the_sdk_text_is_never_logged_printed_or_stored(
        self, make_error: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG), pytest.raises(FatalClassifierError) as excinfo:
            build(FakeClient(make_error())).classify(ONE)

        everything = caplog.text + str(excinfo.value) + repr(excinfo.value)
        for forbidden in (FAKE_TEXT, FAKE_KEY, "ZEBRA", "credit balance", "SDK says"):
            assert forbidden not in everything
        assert excinfo.value.__cause__ is None


class TestSdkErrorsOutsideTheStatusFamily:
    def test_a_response_validation_error_takes_the_split_path(self) -> None:
        client = FakeClient(
            response_validation_error(),
            make_message([result("r1")]),
            make_message([result("r2"), result("r3")]),
        )

        outcome = build(client).classify(ITEMS)

        assert [ids(c) for c in client.calls] == [["r1", "r2", "r3"], ["r1"], ["r2", "r3"]]
        assert set(outcome.analyses) == {"r1", "r2", "r3"}
        assert outcome.usage == Usage(input_tokens=200, output_tokens=40)

    def test_a_single_item_response_validation_error_fails_as_invalid_output(self) -> None:
        outcome = build(FakeClient(response_validation_error())).classify(ONE)

        assert outcome.failures == {"r1": "invalid_output"}
        assert outcome.usage is None

    def test_any_other_api_error_is_fatal_unexpected_error_with_a_fixed_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG), pytest.raises(FatalClassifierError) as excinfo:
            build(FakeClient(generic_api_error())).classify(ONE)

        assert excinfo.value.reason == "unexpected_error"
        assert str(excinfo.value) == "The Anthropic client reported an unexpected error."
        assert FAKE_TEXT not in caplog.text + str(excinfo.value)
        assert FAKE_KEY not in caplog.text + str(excinfo.value)

    @pytest.mark.parametrize("bogus", [object(), {"content": []}, None, "text"])
    def test_a_response_that_is_not_a_message_is_invalid_output(self, bogus: Any) -> None:
        client = FakeClient()
        client.messages._script.append(bogus)

        outcome = build(client).classify(ONE)

        assert outcome.failures == {"r1": "invalid_output"}

    def test_an_escaping_exception_carries_the_partial_result(self) -> None:
        client = FakeClient(
            make_message("bad", input_tokens=100, output_tokens=10),
            make_message([result("r1", topics=["weather"])], input_tokens=30, output_tokens=3),
            ValueError("boom"),
        )

        with pytest.raises(ValueError, match="boom") as excinfo:
            build(client).classify(ITEMS)

        partial = excinfo.value.review_radar_partial  # type: ignore[attr-defined]
        assert set(partial.analyses) == {"r1"}
        assert partial.usage == Usage(input_tokens=130, output_tokens=13)
        assert partial.unknown_topics == 1

    def test_no_partial_is_attached_when_nothing_was_spent(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            build(FakeClient(ValueError("boom"))).classify(ONE)

        assert getattr(excinfo.value, "review_radar_partial", None) is None

    def test_a_keyboard_interrupt_carries_the_partial_result_too(self) -> None:
        def interrupt(_seconds: float) -> None:
            raise KeyboardInterrupt

        client = FakeClient(make_message("bad"), make_message([result("r1")]), status_error(429))

        with pytest.raises(KeyboardInterrupt) as excinfo:
            AnthropicClassifier(
                api_key=SecretStr(FAKE_KEY), client=client, sleep=interrupt
            ).classify(ITEMS)

        partial = excinfo.value.review_radar_partial  # type: ignore[attr-defined]
        assert set(partial.analyses) == {"r1"}


class TestPartialResultOnFatal:
    def test_fatal_in_a_split_carries_analyses_failures_counters_and_usage(self) -> None:
        client = FakeClient(
            make_message("bad", input_tokens=100, output_tokens=10),
            make_message("bad", input_tokens=20, output_tokens=2),
            status_error(401),
        )

        with pytest.raises(FatalClassifierError) as excinfo:
            build(client).classify(ITEMS)

        partial = excinfo.value.partial
        assert partial is not None
        assert partial.analyses == {}
        assert partial.failures == {"r1": "invalid_output"}
        assert partial.usage == Usage(input_tokens=120, output_tokens=12)
        assert excinfo.value.usage == partial.usage

    def test_partial_counts_topics_and_unexpected_ids(self) -> None:
        client = FakeClient(
            make_message("bad"),
            make_message([result("r1", topics=["weather"]), result("zz")]),
            status_error(402),
        )

        with pytest.raises(FatalClassifierError) as excinfo:
            build(client).classify(ITEMS)

        partial = excinfo.value.partial
        assert partial is not None
        assert set(partial.analyses) == {"r1"}
        assert (partial.unknown_topics, partial.unexpected_results) == (1, 1)

    def test_a_first_call_fatal_error_has_no_partial(self) -> None:
        with pytest.raises(FatalClassifierError) as excinfo:
            build(FakeClient(status_error(401))).classify(ONE)

        assert excinfo.value.partial is None


class TestStatusCodes:
    @pytest.mark.parametrize(("status", "reason"), [(408, "network_error"), (409, "server_error")])
    def test_408_and_409_are_transient(self, status: int, reason: str) -> None:
        client = FakeClient(*[status_error(status) for _ in range(4)])
        sleep = RecordingSleep()

        with pytest.raises(ClassifierError) as excinfo:
            build(client, sleep).classify(ONE)

        assert not isinstance(excinfo.value, FatalClassifierError)
        assert excinfo.value.reason == reason
        assert sleep.delays == [1, 2, 4]

    @pytest.mark.parametrize("value", ["true", "TRUE", " True "])
    def test_x_should_retry_true_makes_a_fatal_status_retryable(self, value: str) -> None:
        client = FakeClient(
            status_error(400, headers={"x-should-retry": value}), make_message([result("r1")])
        )
        sleep = RecordingSleep()

        outcome = build(client, sleep).classify(ONE)

        assert "r1" in outcome.analyses
        assert sleep.delays == [1]

    @pytest.mark.parametrize(
        ("status", "reason"), [(429, "rate_limited"), (503, "server_error"), (408, "network_error")]
    )
    def test_x_should_retry_false_gives_up_at_once_without_a_fatal_stop(
        self, status: int, reason: str
    ) -> None:
        client = FakeClient(status_error(status, headers={"x-should-retry": "false"}))
        sleep = RecordingSleep()

        with pytest.raises(ClassifierError) as excinfo:
            build(client, sleep).classify(ONE)

        assert not isinstance(excinfo.value, FatalClassifierError)
        assert excinfo.value.reason == reason
        assert len(client.calls) == 1
        assert sleep.delays == []

    def test_x_should_retry_false_on_a_fatal_status_stays_fatal(self) -> None:
        client = FakeClient(status_error(401, headers={"x-should-retry": "false"}))

        with pytest.raises(FatalClassifierError) as excinfo:
            build(client).classify(ONE)

        assert excinfo.value.reason == "authentication_failed"

    def test_an_unreadable_x_should_retry_is_ignored(self) -> None:
        client = FakeClient(
            status_error(429, headers={"x-should-retry": "maybe"}), make_message([result("r1")])
        )

        assert "r1" in build(client).classify(ONE).analyses

    def test_413_splits_the_batch(self) -> None:
        client = FakeClient(
            status_error(413),
            make_message([result("r1")]),
            make_message([result("r2"), result("r3")]),
        )
        sleep = RecordingSleep()

        outcome = build(client, sleep).classify(ITEMS)

        assert [ids(c) for c in client.calls] == [["r1", "r2", "r3"], ["r1"], ["r2", "r3"]]
        assert set(outcome.analyses) == {"r1", "r2", "r3"}
        assert sleep.delays == []

    def test_a_single_item_413_fails_as_request_too_large(self) -> None:
        client = FakeClient(status_error(413))

        outcome = build(client).classify(ONE)

        assert outcome.failures == {"r1": "request_too_large"}
        assert len(client.calls) == 1

    def test_the_413_message_is_not_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG):
            build(FakeClient(status_error(413))).classify(ONE)

        assert FAKE_TEXT not in caplog.text
        assert FAKE_KEY not in caplog.text


class TestCallBudget:
    def test_every_attempt_counts_as_a_call(self) -> None:
        client = FakeClient(status_error(429), connection_error(), make_message([result("r1")]))
        classifier = build(client)

        classifier.classify(ONE)

        assert classifier.api_calls == 3
        assert classifier.call_budget_exhausted is False

    def test_no_budget_means_no_limit(self) -> None:
        classifier = build(FakeClient(*[make_message("bad")] * 5))

        classifier.classify(ITEMS)

        assert classifier.api_calls == 5
        assert classifier.call_budget_exhausted is False

    def test_a_refused_attempt_ends_the_batch_with_call_budget_exceeded(self) -> None:
        client = FakeClient(status_error(429), status_error(429), make_message([result("r1")]))
        classifier = build(client)
        classifier.set_call_budget(2)

        with pytest.raises(ClassifierError) as excinfo:
            classifier.classify(ONE)

        assert excinfo.value.reason == "call_budget_exceeded"
        assert not isinstance(excinfo.value, FatalClassifierError)
        assert len(client.calls) == 2
        assert classifier.call_budget_exhausted is True

    def test_budget_exhausted_inside_a_split_fails_only_the_unclassified_items(self) -> None:
        client = FakeClient(make_message("bad"), make_message([result("r1")]), make_message("bad"))
        classifier = build(client)
        classifier.set_call_budget(3)

        outcome = classifier.classify(ITEMS)

        assert len(client.calls) == 3
        assert set(outcome.analyses) == {"r1"}
        assert outcome.failures == {"r2": "call_budget_exceeded", "r3": "call_budget_exceeded"}
        assert classifier.call_budget_exhausted is True

    def test_using_the_whole_budget_is_not_exhaustion(self) -> None:
        classifier = build(FakeClient(make_message([result("r1")])))
        classifier.set_call_budget(1)

        classifier.classify(ONE)

        assert classifier.call_budget_exhausted is False
        assert classifier.api_calls == 1

    def test_setting_a_budget_resets_the_counters(self) -> None:
        classifier = build(FakeClient(make_message([result("r1")]), make_message([result("r1")])))
        classifier.set_call_budget(1)
        classifier.classify(ONE)

        classifier.set_call_budget(1)

        assert classifier.api_calls == 0
        classifier.classify(ONE)
        assert classifier.api_calls == 1

    def test_a_20_item_batch_that_is_always_invalid_stops_at_the_budget(self) -> None:
        items = [ClassifyItem(f"r{n}", f"review {n}", 3) for n in range(1, 21)]
        client = FakeClient(*[make_message("not json") for _ in range(13)])
        classifier = build(client)
        classifier.set_call_budget(13)

        outcome = classifier.classify(items)

        assert len(client.calls) == 13
        assert client.remaining == 0
        assert classifier.api_calls == 13
        assert outcome.analyses == {}
        assert set(outcome.failures) == {i.item_id for i in items}
        assert "call_budget_exceeded" in set(outcome.failures.values())
        assert set(outcome.failures.values()) <= {"invalid_output", "call_budget_exceeded"}
        assert outcome.usage == Usage(input_tokens=1300, output_tokens=260)


class TestClientProtocol:
    def test_the_fake_client_satisfies_the_protocol(self) -> None:
        assert isinstance(FakeClient(), module.MessagesClient)

    def test_something_without_messages_does_not(self) -> None:
        assert not isinstance(object(), module.MessagesClient)
