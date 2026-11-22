"""AnthropicClassifier: request shape, parsing, split-and-retry, retries, fatal errors, privacy.

Every test drives the classifier through a scripted fake client; nothing touches the network.
"""

import hashlib
import json
import logging
import re
from typing import Any

import anthropic
import pytest
from pydantic import SecretStr

from anthropic_fakes import (
    FAKE_KEY,
    FAKE_TEXT,
    FakeClient,
    RecordingSleep,
    connection_error,
    fixture_message,
    make_message,
    result,
    status_error,
    timeout_error,
)
from review_radar import anthropic_classifier as module
from review_radar.anthropic_classifier import (
    PROMPT_VERSION,
    SCHEMA,
    SYSTEM_PROMPT,
    AnthropicClassifier,
    compute_prompt_version,
)
from review_radar.classifier import (
    ClassifierError,
    ClassifyItem,
    ClassifyResult,
    FatalClassifierError,
)
from review_radar.models import Sentiment, Topic, Usage

ITEMS = [
    ClassifyItem("r1", "Great app, setup took two minutes.", 5),
    ClassifyItem("r2", "Parcel arrived ten days late and the box was crushed.", 1),
    ClassifyItem("r3", "Support never answered and the price went up again.", None),
]
ONE = ITEMS[:1]


def build(client: FakeClient, sleep: RecordingSleep | None = None, **kwargs: Any):
    return AnthropicClassifier(
        api_key=SecretStr(FAKE_KEY), client=client, sleep=sleep or RecordingSleep(), **kwargs
    )


def ids(call: dict[str, Any]) -> list[str]:
    return [entry["id"] for entry in json.loads(call["messages"][0]["content"])]


class TestIdentity:
    def test_provider_and_default_model(self) -> None:
        classifier = build(FakeClient())

        assert classifier.provider == "anthropic"
        assert classifier.model == "claude-haiku-5-5"

    def test_model_is_taken_as_given(self) -> None:
        assert build(FakeClient(), model="claude-sonnet-5-5").model == "claude-sonnet-5-5"

    def test_prompt_version_is_anthropic_1_plus_eight_hex(self) -> None:
        assert re.fullmatch(r"anthropic-1\+[0-9a-f]{8}", build(FakeClient()).prompt_version)

    def test_prompt_version_hashes_the_prompt_and_the_canonical_schema(self) -> None:
        canonical = json.dumps(SCHEMA, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256((SYSTEM_PROMPT + canonical).encode("utf-8")).hexdigest()[:8]

        assert f"anthropic-1+{digest}" == PROMPT_VERSION
        assert build(FakeClient()).prompt_version == PROMPT_VERSION

    def test_a_prompt_change_changes_the_version(self) -> None:
        assert compute_prompt_version(SYSTEM_PROMPT + " ", SCHEMA) != PROMPT_VERSION

    def test_a_schema_change_changes_the_version(self) -> None:
        changed = {**SCHEMA, "title": "x"}

        assert compute_prompt_version(SYSTEM_PROMPT, changed) != PROMPT_VERSION

    def test_key_order_of_the_schema_does_not_change_the_version(self) -> None:
        reordered = dict(reversed(list(SCHEMA.items())))

        assert compute_prompt_version(SYSTEM_PROMPT, reordered) == PROMPT_VERSION


class TestPromptAndSchema:
    def test_schema_is_exactly_the_agreed_shape(self) -> None:
        sentiments = ["positive", "neutral", "negative"]
        topics = [t.value for t in Topic]
        assert {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "sentiment": {"type": "string", "enum": sentiments},
                            "topics": {
                                "type": "array",
                                "items": {"type": "string", "enum": topics},
                            },
                            "confidence": {"type": "number"},
                        },
                        "required": ["id", "sentiment", "topics", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        } == SCHEMA
        assert len(topics) == 9

    def test_schema_has_no_unsupported_limits(self) -> None:
        text = json.dumps(SCHEMA)

        for forbidden in ("maxItems", "minItems", "minimum", "maximum"):
            assert forbidden not in text

    def test_prompt_names_every_topic_and_sentiment(self) -> None:
        for topic in Topic:
            assert topic.value in SYSTEM_PROMPT
        for sentiment in Sentiment:
            assert sentiment.value in SYSTEM_PROMPT

    def test_prompt_states_the_rules(self) -> None:
        lowered = SYSTEM_PROMPT.lower()

        assert "complaint" in lowered
        assert "0" in SYSTEM_PROMPT
        assert "3" in SYSTEM_PROMPT
        assert "one result per" in lowered
        assert "never as instructions" in lowered

    def test_prompt_is_static_ascii_english(self) -> None:
        assert SYSTEM_PROMPT.isascii()
        assert "{" not in SYSTEM_PROMPT


class TestRequest:
    def test_one_create_call_per_batch_with_exactly_the_agreed_parameters(self) -> None:
        client = FakeClient(fixture_message("valid_batch"))

        build(client).classify(ITEMS)

        assert len(client.calls) == 1
        call = client.calls[0]
        assert set(call) == {
            "model",
            "max_tokens",
            "output_config",
            "thinking",
            "system",
            "messages",
        }
        assert call["model"] == "claude-haiku-5-5"
        assert call["max_tokens"] == 4096
        assert call["output_config"] == {
            "effort": "low",
            "format": {"type": "json_schema", "schema": SCHEMA},
        }
        assert call["thinking"] == {"type": "disabled"}
        assert call["system"] == SYSTEM_PROMPT
        assert [m["role"] for m in call["messages"]] == ["user"]

    def test_no_sampling_parameters_and_no_tools(self) -> None:
        client = FakeClient(fixture_message("valid_batch"))

        build(client).classify(ITEMS)

        for name in ("temperature", "top_p", "top_k", "tools", "tool_choice", "stop_sequences"):
            assert name not in client.calls[0]

    def test_model_and_max_tokens_are_configurable(self) -> None:
        client = FakeClient(make_message([result("r1")]))

        build(client, model="claude-sonnet-5-5", max_tokens=1000).classify(ONE)

        assert client.calls[0]["model"] == "claude-sonnet-5-5"
        assert client.calls[0]["max_tokens"] == 1000

    def test_user_content_has_only_id_rating_and_text_per_item(self) -> None:
        client = FakeClient(fixture_message("valid_batch"))

        build(client).classify(ITEMS)

        content = client.calls[0]["messages"][0]["content"]
        assert isinstance(content, str)
        payload = json.loads(content)
        assert payload == [
            {"id": "r1", "rating": 5, "text": ITEMS[0].text},
            {"id": "r2", "rating": 1, "text": ITEMS[1].text},
            {"id": "r3", "rating": None, "text": ITEMS[2].text},
        ]
        for entry in payload:
            assert set(entry) == {"id", "rating", "text"}

    def test_nothing_but_the_static_prompt_and_the_items_is_sent(self) -> None:
        client = FakeClient(fixture_message("valid_batch"))

        build(client).classify(ITEMS)

        call = client.calls[0]
        sent = json.dumps([call["system"], call["messages"]])
        assert FAKE_KEY not in sent
        assert call["system"] == SYSTEM_PROMPT
        assert len(call["messages"]) == 1
        assert set(call["messages"][0]) == {"role", "content"}

    def test_unicode_and_special_characters_survive_the_round_trip(self) -> None:
        text = 'Cafe "bad" \\ ☃ emoji \U0001f600 отзыв \'; DROP TABLE x;'
        client = FakeClient(make_message([result("r1")]))

        build(client).classify([ClassifyItem("r1", text, 3)])

        assert json.loads(client.calls[0]["messages"][0]["content"])[0]["text"] == text

    def test_empty_batch_makes_no_call(self) -> None:
        client = FakeClient()

        outcome = build(client).classify([])

        assert client.calls == []
        assert outcome.analyses == {}
        assert outcome.usage is None

    def test_max_attempts_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            build(FakeClient(), max_attempts=0)


class TestValidResponse:
    def test_recorded_valid_batch_becomes_analyses(self) -> None:
        outcome = build(FakeClient(fixture_message("valid_batch"))).classify(ITEMS)

        assert set(outcome.analyses) == {"r1", "r2", "r3"}
        r1, r2, r3 = (outcome.analyses[i] for i in ("r1", "r2", "r3"))
        assert (r1.sentiment, r1.topics, r1.confidence) == (Sentiment.POSITIVE, (), 0.95)
        assert r2.topics == (Topic.DELIVERY_DELAY, Topic.PACKAGING)
        assert r3.topics == (Topic.CUSTOMER_SUPPORT, Topic.PRICING)
        assert outcome.failures == {}
        assert outcome.usage == Usage(input_tokens=410, output_tokens=96)
        assert (outcome.unknown_topics, outcome.truncated_topics, outcome.fallback) == (0, 0, 0)

    def test_result_type(self) -> None:
        assert isinstance(
            build(FakeClient(fixture_message("valid_batch"))).classify(ITEMS), ClassifyResult
        )

    @pytest.mark.parametrize("confidence", [0, 1, 0.0, 1.0, 0.5])
    def test_confidence_on_the_boundaries_is_accepted(self, confidence: float) -> None:
        client = FakeClient(make_message([result("r1", conf=confidence)]))

        outcome = build(client).classify(ONE)

        assert outcome.analyses["r1"].confidence == confidence

    def test_unknown_and_surplus_topics_are_normalized_and_counted(self) -> None:
        outcome = build(FakeClient(fixture_message("unknown_topic"))).classify(ITEMS)

        assert outcome.analyses["r2"].topics == (Topic.DELIVERY_DELAY, Topic.OTHER)
        assert len(outcome.analyses["r3"].topics) == 3
        assert outcome.unknown_topics == 1
        assert outcome.truncated_topics == 1
        assert outcome.failures == {}

    def test_empty_topics_are_valid(self) -> None:
        client = FakeClient(make_message([result("r1", "neutral", [])]))

        assert build(client).classify(ONE).analyses["r1"].topics == ()

    def test_a_thinking_block_before_the_text_is_ignored(self) -> None:
        blocks = [
            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
            {"type": "text", "text": json.dumps({"results": [result("r1")]})},
        ]
        client = FakeClient(make_message(None, blocks=blocks))

        assert "r1" in build(client).classify(ONE).analyses

    def test_result_order_does_not_matter(self) -> None:
        client = FakeClient(make_message([result("r3"), result("r1"), result("r2")]))

        assert set(build(client).classify(ITEMS).analyses) == {"r1", "r2", "r3"}

    def test_extra_ids_are_dropped_and_counted(self) -> None:
        client = FakeClient(fixture_message("extra_id"))

        outcome = build(client).classify(ITEMS)

        assert set(outcome.analyses) == {"r1", "r2", "r3"}
        assert outcome.unexpected_results == 1
        assert outcome.failures == {}
        assert len(client.calls) == 1

    def test_unexpected_topics_of_a_dropped_id_are_not_counted(self) -> None:
        client = FakeClient(make_message([result("r1"), result("zz", topics=["weather"])]))

        outcome = build(client).classify(ONE)

        assert (outcome.unexpected_results, outcome.unknown_topics) == (1, 0)


class TestMissingIds:
    def test_missing_id_is_retried_once_as_a_smaller_batch(self) -> None:
        client = FakeClient(
            fixture_message("missing_id"), make_message([result("r2", "negative", ["packaging"])])
        )

        outcome = build(client).classify(ITEMS)

        assert [ids(c) for c in client.calls] == [["r1", "r2", "r3"], ["r2"]]
        assert set(outcome.analyses) == {"r1", "r2", "r3"}
        assert outcome.analyses["r2"].topics == (Topic.PACKAGING,)
        assert outcome.failures == {}
        assert outcome.usage == Usage(input_tokens=410 + 100, output_tokens=64 + 20)

    def test_still_missing_after_the_retry_fails_as_missing_result(self) -> None:
        client = FakeClient(fixture_message("missing_id"), make_message([]))

        outcome = build(client).classify(ITEMS)

        assert len(client.calls) == 2
        assert set(outcome.analyses) == {"r1", "r3"}
        assert outcome.failures == {"r2": "missing_result"}

    def test_the_retry_is_not_retried_again_when_it_is_missing_too(self) -> None:
        client = FakeClient(make_message([result("r1")]), make_message([result("r2")]))

        outcome = build(client).classify(ITEMS)

        assert [ids(c) for c in client.calls] == [["r1", "r2", "r3"], ["r2", "r3"]]
        assert set(outcome.analyses) == {"r1", "r2"}
        assert outcome.failures == {"r3": "missing_result"}

    def test_an_invalid_retry_is_split_down_to_single_items(self) -> None:
        client = FakeClient(
            make_message([result("r1")]),
            make_message("not json"),
            make_message([result("r2")]),
            make_message("not json"),
        )

        outcome = build(client).classify(ITEMS)

        assert [ids(c) for c in client.calls] == [
            ["r1", "r2", "r3"],
            ["r2", "r3"],
            ["r2"],
            ["r3"],
        ]
        assert set(outcome.analyses) == {"r1", "r2"}
        assert outcome.failures == {"r3": "invalid_output"}

    def test_an_empty_results_array_retries_everything_once_then_fails(self) -> None:
        client = FakeClient(make_message([]), make_message([]))

        outcome = build(client).classify(ONE)

        assert len(client.calls) == 2
        assert outcome.failures == {"r1": "missing_result"}


class TestSplitAndRetry:
    def test_invalid_sentiment_splits_the_batch_in_half(self) -> None:
        client = FakeClient(
            fixture_message("invalid_sentiment"),
            make_message([result("r1")]),
            make_message([result("r2", "negative"), result("r3", "negative")]),
        )

        outcome = build(client).classify(ITEMS)

        assert [ids(c) for c in client.calls] == [["r1", "r2", "r3"], ["r1"], ["r2", "r3"]]
        assert set(outcome.analyses) == {"r1", "r2", "r3"}
        assert outcome.failures == {}

    def test_usage_of_invalid_calls_is_summed_too(self) -> None:
        client = FakeClient(
            fixture_message("invalid_sentiment"),
            make_message([result("r1")], input_tokens=50, output_tokens=5),
            make_message([result("r2"), result("r3")], input_tokens=70, output_tokens=9),
        )

        outcome = build(client).classify(ITEMS)

        assert outcome.usage == Usage(input_tokens=410 + 50 + 70, output_tokens=90 + 5 + 9)

    def test_a_single_invalid_item_is_failed_with_its_reason(self) -> None:
        client = FakeClient(make_message("not json"))

        outcome = build(client).classify(ONE)

        assert len(client.calls) == 1
        assert outcome.analyses == {}
        assert outcome.failures == {"r1": "invalid_output"}
        assert outcome.usage == Usage(input_tokens=100, output_tokens=20)

    def test_only_the_bad_item_fails_when_the_rest_are_fine(self) -> None:
        client = FakeClient(
            make_message("bad"),
            make_message([result("r1")]),
            make_message("bad"),
            make_message([result("r2")]),
            make_message("bad"),
        )

        outcome = build(client).classify(ITEMS)

        assert [ids(c) for c in client.calls] == [
            ["r1", "r2", "r3"],
            ["r1"],
            ["r2", "r3"],
            ["r2"],
            ["r3"],
        ]
        assert set(outcome.analyses) == {"r1", "r2"}
        assert outcome.failures == {"r3": "invalid_output"}

    def test_two_items_split_into_two_singles(self) -> None:
        client = FakeClient(
            make_message("bad"), make_message([result("r1")]), make_message([result("r2")])
        )

        outcome = build(client).classify(ITEMS[:2])

        assert [ids(c) for c in client.calls] == [["r1", "r2"], ["r1"], ["r2"]]
        assert set(outcome.analyses) == {"r1", "r2"}

    def test_everything_invalid_costs_five_calls_for_three_items(self) -> None:
        script = [make_message("bad")] * 5
        client = FakeClient(*script)

        outcome = build(client).classify(ITEMS)

        assert len(client.calls) == 5
        assert client.remaining == 0
        assert outcome.analyses == {}
        assert outcome.failures == {
            "r1": "invalid_output",
            "r2": "invalid_output",
            "r3": "invalid_output",
        }
        assert outcome.usage == Usage(input_tokens=500, output_tokens=100)

    def test_split_keeps_the_original_ids(self) -> None:
        client = FakeClient(
            make_message("bad"),
            make_message([result("r1")]),
            make_message([result("r2"), result("r3")]),
        )

        build(client).classify(ITEMS)

        assert ids(client.calls[2]) == ["r2", "r3"]
        sent = json.loads(client.calls[2]["messages"][0]["content"])
        assert sent[0]["text"] == ITEMS[1].text

    def test_invalid_output_is_never_cached_as_an_analysis(self) -> None:
        client = FakeClient(
            fixture_message("invalid_sentiment"),
            make_message("bad"),
            make_message("bad"),
            make_message("bad"),
            make_message("bad"),
        )

        outcome = build(client).classify(ITEMS)

        assert outcome.analyses == {}

    def test_truncated_output_splits_then_succeeds(self) -> None:
        client = FakeClient(
            fixture_message("max_tokens"),
            make_message([result("r1")]),
            make_message([result("r2"), result("r3")]),
        )

        outcome = build(client).classify(ITEMS)

        assert len(client.calls) == 3
        assert set(outcome.analyses) == {"r1", "r2", "r3"}
        assert outcome.usage is not None
        assert outcome.usage.output_tokens == 4096 + 20 + 20

    def test_a_single_truncated_item_fails_as_truncated_output(self) -> None:
        outcome = build(FakeClient(fixture_message("max_tokens"))).classify(ONE)

        assert outcome.failures == {"r1": "truncated_output"}

    def test_a_context_window_stop_counts_as_truncated(self) -> None:
        client = FakeClient(make_message("{}", stop_reason="model_context_window_exceeded"))

        assert build(client).classify(ONE).failures == {"r1": "truncated_output"}

    def test_refusal_splits_and_a_single_refused_item_fails_as_refusal(self) -> None:
        client = FakeClient(
            fixture_message("refusal"),
            make_message([result("r1")]),
            fixture_message("refusal"),
            make_message([result("r2")]),
            fixture_message("refusal"),
        )

        outcome = build(client).classify(ITEMS)

        assert [ids(c) for c in client.calls] == [
            ["r1", "r2", "r3"],
            ["r1"],
            ["r2", "r3"],
            ["r2"],
            ["r3"],
        ]
        assert set(outcome.analyses) == {"r1", "r2"}
        assert outcome.failures == {"r3": "refusal"}

    def test_refusal_is_checked_before_the_content(self) -> None:
        valid_looking = make_message([result("r1")], stop_reason="refusal")

        assert build(FakeClient(valid_looking)).classify(ONE).failures == {"r1": "refusal"}

    def test_max_tokens_is_checked_before_the_content(self) -> None:
        valid_looking = make_message([result("r1")], stop_reason="max_tokens")

        assert build(FakeClient(valid_looking)).classify(ONE).failures == {"r1": "truncated_output"}

    def test_a_failure_in_one_half_keeps_the_other_half(self) -> None:
        client = FakeClient(
            make_message("bad"),
            make_message([result("r1")]),
            fixture_message("refusal"),
            fixture_message("refusal"),
            fixture_message("refusal"),
        )

        outcome = build(client).classify(ITEMS)

        assert set(outcome.analyses) == {"r1"}
        assert outcome.failures == {"r2": "refusal", "r3": "refusal"}


RAW_CONFIDENCE = (
    '{"results": [{"id": "r1", "sentiment": "positive", "topics": [], "confidence": %s}]}'
)

INVALID_TEXTS = {
    "not json": "this is not json",
    "empty text": "",
    "json array": "[]",
    "json null": "null",
    "results not a list": json.dumps({"results": "r1"}),
    "results missing": json.dumps({}),
    "extra top-level key": json.dumps({"results": [result("r1")], "note": "hi"}),
    "extra item key": json.dumps({"results": [{**result("r1"), "reason": "x"}]}),
    "item missing sentiment": json.dumps(
        {"results": [{"id": "r1", "topics": [], "confidence": 1}]}
    ),
    "item missing confidence": json.dumps(
        {"results": [{"id": "r1", "sentiment": "positive", "topics": []}]}
    ),
    "unknown sentiment": json.dumps({"results": [result("r1", "furious")]}),
    "sentiment case": json.dumps({"results": [result("r1", "Positive")]}),
    "sentiment not a string": json.dumps({"results": [result("r1", 5)]}),
    "id not a string": json.dumps({"results": [{**result("r1"), "id": 1}]}),
    "id null": json.dumps({"results": [{**result("r1"), "id": None}]}),
    "topics is a string": json.dumps({"results": [{**result("r1"), "topics": "pricing"}]}),
    "topics has a non-string": json.dumps({"results": [{**result("r1"), "topics": [5]}]}),
    "topics null": json.dumps({"results": [{**result("r1"), "topics": None}]}),
    "confidence above one": json.dumps({"results": [result("r1", conf=1.01)]}),
    "confidence below zero": json.dumps({"results": [result("r1", conf=-0.01)]}),
    "confidence a string": json.dumps({"results": [result("r1", conf="0.5")]}),
    "confidence null": json.dumps({"results": [result("r1", conf=None)]}),
    "confidence bool": json.dumps({"results": [result("r1", conf=True)]}),
    "confidence NaN": RAW_CONFIDENCE % "NaN",
    "confidence Infinity": RAW_CONFIDENCE % "Infinity",
    "duplicate ids": json.dumps({"results": [result("r1"), result("r1")]}),
    "item is not an object": json.dumps({"results": ["r1"]}),
    "text echoes the review": f"{FAKE_TEXT} {FAKE_KEY}",
}


class TestInvalidOutputForASingleItem:
    @pytest.mark.parametrize("text", list(INVALID_TEXTS.values()), ids=list(INVALID_TEXTS))
    def test_is_failed_as_invalid_output_and_never_analyzed(self, text: str) -> None:
        client = FakeClient(make_message(text))

        outcome = build(client).classify(ONE)

        assert len(client.calls) == 1
        assert outcome.analyses == {}
        assert outcome.failures == {"r1": "invalid_output"}

    def test_no_text_block_is_invalid(self) -> None:
        blocks = [{"type": "thinking", "thinking": "hmm", "signature": "s"}]

        outcome = build(FakeClient(make_message(None, blocks=blocks))).classify(ONE)

        assert outcome.failures == {"r1": "invalid_output"}

    def test_empty_content_without_a_refusal_is_invalid(self) -> None:
        outcome = build(FakeClient(make_message(None, blocks=[]))).classify(ONE)

        assert outcome.failures == {"r1": "invalid_output"}

    def test_duplicate_ids_in_a_batch_split_it(self) -> None:
        client = FakeClient(
            make_message([result("r1"), result("r1"), result("r2"), result("r3")]),
            make_message([result("r1")]),
            make_message([result("r2"), result("r3")]),
        )

        outcome = build(client).classify(ITEMS)

        assert len(client.calls) == 3
        assert set(outcome.analyses) == {"r1", "r2", "r3"}


class TestRetriesOfTransientErrors:
    @pytest.mark.parametrize(
        ("error", "reason"),
        [
            (lambda: status_error(429), "rate_limited"),
            (lambda: status_error(500), "server_error"),
            (lambda: status_error(502), "server_error"),
            (lambda: status_error(503), "server_error"),
            (lambda: status_error(529, error_type="overloaded_error"), "server_error"),
            (connection_error, "network_error"),
            (timeout_error, "network_error"),
        ],
        ids=["429", "500", "502", "503", "529", "connection", "timeout"],
    )
    def test_gives_up_after_four_attempts_with_backoff_1_2_4(self, error: Any, reason: str) -> None:
        client = FakeClient(error(), error(), error(), error())
        sleep = RecordingSleep()

        with pytest.raises(ClassifierError) as excinfo:
            build(client, sleep).classify(ITEMS)

        assert not isinstance(excinfo.value, FatalClassifierError)
        assert excinfo.value.reason == reason
        assert len(client.calls) == 4
        assert sleep.delays == [1, 2, 4]

    def test_recovers_when_a_later_attempt_succeeds(self) -> None:
        client = FakeClient(status_error(429), status_error(500), fixture_message("valid_batch"))
        sleep = RecordingSleep()

        outcome = build(client, sleep).classify(ITEMS)

        assert set(outcome.analyses) == {"r1", "r2", "r3"}
        assert sleep.delays == [1, 2]
        assert len(client.calls) == 3
        assert outcome.usage == Usage(input_tokens=410, output_tokens=96)

    def test_the_same_request_is_repeated_on_retry(self) -> None:
        client = FakeClient(status_error(429), fixture_message("valid_batch"))

        build(client).classify(ITEMS)

        assert client.calls[0] == client.calls[1]

    def test_the_backoff_is_capped_at_thirty_seconds(self) -> None:
        errors = [status_error(500) for _ in range(10)]
        client = FakeClient(*errors)
        sleep = RecordingSleep()

        with pytest.raises(ClassifierError):
            build(client, sleep, max_attempts=10).classify(ONE)

        assert sleep.delays == [1, 2, 4, 8, 16, 30, 30, 30, 30]

    def test_one_attempt_means_no_sleep(self) -> None:
        client = FakeClient(status_error(429))
        sleep = RecordingSleep()

        with pytest.raises(ClassifierError):
            build(client, sleep, max_attempts=1).classify(ONE)

        assert sleep.delays == []

    @pytest.mark.parametrize(
        ("header", "expected"),
        [("7", 7), ("7.5", 7.5), ("100", 30), ("0", 0), ("30", 30)],
    )
    def test_a_numeric_retry_after_is_honoured_and_capped(
        self, header: str, expected: float
    ) -> None:
        client = FakeClient(
            status_error(429, headers={"retry-after": header}), make_message([result("r1")])
        )
        sleep = RecordingSleep()

        build(client, sleep).classify(ONE)

        assert sleep.delays == [expected]

    @pytest.mark.parametrize(
        "header", ["Wed, 21 Oct 2026 07:28:00 GMT", "abc", "-5", "nan", "inf", ""]
    )
    def test_an_unusable_retry_after_falls_back_to_the_backoff(self, header: str) -> None:
        client = FakeClient(
            status_error(429, headers={"retry-after": header}),
            status_error(429, headers={"retry-after": header}),
            make_message([result("r1")]),
        )
        sleep = RecordingSleep()

        build(client, sleep).classify(ONE)

        assert sleep.delays == [1, 2]

    def test_retry_after_is_read_on_server_errors_too(self) -> None:
        client = FakeClient(
            status_error(503, headers={"retry-after": "3"}), make_message([result("r1")])
        )
        sleep = RecordingSleep()

        build(client, sleep).classify(ONE)

        assert sleep.delays == [3]

    def test_the_retry_counter_restarts_for_every_api_call(self) -> None:
        client = FakeClient(
            status_error(429),
            status_error(429),
            make_message("bad"),
            status_error(429),
            make_message([result("r1")]),
            make_message([result("r2")]),
        )
        sleep = RecordingSleep()

        outcome = build(client, sleep).classify(ITEMS[:2])

        assert set(outcome.analyses) == {"r1", "r2"}
        assert sleep.delays == [1, 2, 1]

    def test_a_transient_failure_inside_a_split_fails_only_those_items(self) -> None:
        client = FakeClient(
            make_message("bad", input_tokens=100, output_tokens=10),
            make_message([result("r1")], input_tokens=30, output_tokens=3),
            *[status_error(429) for _ in range(4)],
        )
        sleep = RecordingSleep()

        outcome = build(client, sleep).classify(ITEMS)

        assert set(outcome.analyses) == {"r1"}
        assert outcome.failures == {"r2": "rate_limited", "r3": "rate_limited"}
        assert outcome.usage == Usage(input_tokens=130, output_tokens=13)
        assert sleep.delays == [1, 2, 4]

    def test_unrelated_exceptions_are_not_swallowed(self) -> None:
        client = FakeClient(ValueError("boom"))

        with pytest.raises(ValueError, match="boom"):
            build(client).classify(ONE)

    def test_default_sleep_is_time_sleep(self) -> None:
        classifier = AnthropicClassifier(api_key=SecretStr(FAKE_KEY), client=FakeClient())

        assert classifier._sleep is module.time.sleep


class TestFatalErrors:
    @pytest.mark.parametrize(
        ("status", "error_type", "reason", "message"),
        [
            (
                401,
                "authentication_error",
                "authentication_failed",
                "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY.",
            ),
            (
                403,
                "permission_error",
                "permission_denied",
                "The Anthropic API key is not allowed to make this request.",
            ),
            (
                402,
                None,
                "billing_error",
                "The Anthropic account has no available credit; add credit in the Console.",
            ),
            (
                400,
                "billing_error",
                "billing_error",
                "The Anthropic account has no available credit; add credit in the Console.",
            ),
            (
                404,
                "not_found_error",
                "model_not_found",
                "The Anthropic model 'claude-haiku-5-5' was not found. Check ANTHROPIC_MODEL.",
            ),
            (
                400,
                "invalid_request_error",
                "bad_request",
                "The Anthropic API rejected the request (HTTP 400). "
                "Check the model name and the request settings.",
            ),
            (
                422,
                None,
                "unexpected_status",
                "The Anthropic API returned an unexpected client error.",
            ),
        ],
    )
    def test_raises_fatal_at_once_without_retry(
        self, status: int, error_type: str | None, reason: str, message: str
    ) -> None:
        client = FakeClient(status_error(status, error_type=error_type))
        sleep = RecordingSleep()

        with pytest.raises(FatalClassifierError) as excinfo:
            build(client, sleep).classify(ITEMS)

        assert excinfo.value.reason == reason
        assert str(excinfo.value).startswith(message)
        assert f"HTTP {status}" in str(excinfo.value)
        assert "req_test_1" in str(excinfo.value)
        assert len(client.calls) == 1
        assert sleep.delays == []

    def test_the_message_names_the_configured_model_for_404(self) -> None:
        client = FakeClient(status_error(404, error_type="not_found_error"))

        with pytest.raises(FatalClassifierError) as excinfo:
            build(client, model="claude-nonesuch-9").classify(ONE)

        assert "claude-nonesuch-9" in str(excinfo.value)

    def test_the_request_id_is_optional(self) -> None:
        client = FakeClient(status_error(401, request_id=None))

        with pytest.raises(FatalClassifierError) as excinfo:
            build(client).classify(ONE)

        assert "HTTP 401" in str(excinfo.value)
        assert "request id" not in str(excinfo.value)

    def test_an_odd_request_id_is_left_out(self) -> None:
        client = FakeClient(status_error(401, request_id="req x<script>"))

        with pytest.raises(FatalClassifierError) as excinfo:
            build(client).classify(ONE)

        assert "script" not in str(excinfo.value)

    @pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
    def test_messages_never_echo_the_sdk_message_review_text_or_the_key(self, status: int) -> None:
        client = FakeClient(status_error(status))

        with pytest.raises(FatalClassifierError) as excinfo:
            build(client).classify(ONE)

        text = f"{excinfo.value!s} {excinfo.value!r} {excinfo.value.args}"
        assert FAKE_TEXT not in text
        assert FAKE_KEY not in text
        assert "SDK says" not in text

    def test_fatal_in_the_middle_of_a_split_carries_the_usage_already_spent(self) -> None:
        client = FakeClient(
            make_message("bad", input_tokens=100, output_tokens=10),
            make_message([result("r1")], input_tokens=30, output_tokens=3),
            status_error(402),
        )

        with pytest.raises(FatalClassifierError) as excinfo:
            build(client).classify(ITEMS)

        assert excinfo.value.reason == "billing_error"
        assert excinfo.value.usage == Usage(input_tokens=130, output_tokens=13)

    def test_a_first_call_fatal_error_carries_no_usage(self) -> None:
        with pytest.raises(FatalClassifierError) as excinfo:
            build(FakeClient(status_error(401))).classify(ONE)

        assert excinfo.value.usage is None


class TestSecretsNeverLeak:
    def test_repr_and_str_of_the_classifier_hide_the_key(self) -> None:
        classifier = build(FakeClient())

        assert FAKE_KEY not in repr(classifier)
        assert FAKE_KEY not in str(classifier)
        assert "AnthropicClassifier" in repr(classifier)
        assert "claude-haiku-5-5" in repr(classifier)

    def test_no_attribute_holds_the_key(self) -> None:
        classifier = build(FakeClient())

        assert FAKE_KEY not in repr(vars(classifier))

    def test_the_key_is_unwrapped_only_to_build_the_sdk_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        class SpyAnthropic:
            def __init__(self, **kwargs: Any) -> None:
                seen.update(kwargs)

        monkeypatch.setattr(anthropic, "Anthropic", SpyAnthropic)

        classifier = AnthropicClassifier(api_key=SecretStr(FAKE_KEY))

        assert seen == {"api_key": FAKE_KEY, "max_retries": 0, "timeout": 60.0}
        assert FAKE_KEY not in repr(classifier)

    def test_an_injected_client_means_no_sdk_client_is_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(**_: Any) -> None:
            raise AssertionError("must not be built")

        monkeypatch.setattr(anthropic, "Anthropic", explode)

        build(FakeClient())

    @pytest.mark.parametrize(
        "script",
        [
            [status_error(429)] * 4,
            [status_error(500)] * 4,
            [connection_error()] * 4,
            [timeout_error()] * 4,
            [status_error(401)],
            [status_error(402)],
            [status_error(404)],
            [status_error(400)],
        ],
        ids=["429", "500", "connection", "timeout", "401", "402", "404", "400"],
    )
    def test_no_log_record_or_exception_contains_the_key_or_review_text(
        self, script: list[Exception], caplog: pytest.LogCaptureFixture
    ) -> None:
        items = [ClassifyItem("r1", FAKE_TEXT, 1)]
        caught: list[BaseException] = []

        with caplog.at_level(logging.DEBUG):
            try:
                build(FakeClient(*script)).classify(items)
            except ClassifierError as exc:
                caught.append(exc)

        assert len(caught) == 1
        everything = caplog.text + str(caught[0]) + repr(caught[0])
        assert FAKE_KEY not in everything
        assert FAKE_TEXT not in everything
        assert "ZEBRA" not in everything
        assert "SDK says" not in everything

    def test_invalid_outputs_that_echo_the_text_are_not_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        items = [ClassifyItem("r1", FAKE_TEXT, 1)]
        script = [
            make_message(f"{FAKE_TEXT} {FAKE_KEY}"),
            make_message(json.dumps({"results": [result(FAKE_TEXT, "bogus " + FAKE_TEXT)]})),
        ]

        with caplog.at_level(logging.DEBUG):
            build(FakeClient(*script)).classify(items)
            build(FakeClient(script[1])).classify(items)

        assert "ZEBRA" not in caplog.text
        assert FAKE_KEY not in caplog.text

    def test_topics_that_echo_the_text_are_not_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeClient(make_message([result("r1", topics=[FAKE_TEXT, "pricing"])]))

        with caplog.at_level(logging.DEBUG):
            outcome = build(client).classify(ONE)

        assert outcome.unknown_topics == 1
        assert "ZEBRA" not in caplog.text


class TestLogging:
    def test_retries_log_attempt_status_and_request_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeClient(status_error(429), make_message([result("r1")]))

        with caplog.at_level(logging.WARNING):
            build(client).classify(ONE)

        retry_lines = [r.getMessage() for r in caplog.records if "429" in r.getMessage()]
        assert len(retry_lines) == 1
        assert "attempt 1/4" in retry_lines[0]
        assert "req_test_1" in retry_lines[0]

    def test_an_invalid_response_is_logged_with_its_reason_and_batch_size(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = FakeClient(
            fixture_message("invalid_sentiment"),
            make_message([result("r1")]),
            make_message([result("r2"), result("r3")]),
        )

        with caplog.at_level(logging.WARNING):
            build(client).classify(ITEMS)

        lines = [r.getMessage() for r in caplog.records]
        assert any("invalid_output" in line and "3 item" in line for line in lines)

    def test_a_clean_run_logs_no_warnings(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            build(FakeClient(fixture_message("valid_batch"))).classify(ITEMS)

        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_a_fatal_error_is_logged_with_the_status(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING), pytest.raises(FatalClassifierError):
            build(FakeClient(status_error(401))).classify(ONE)

        assert any(
            "401" in r.getMessage() and "req_test_1" in r.getMessage() for r in caplog.records
        )
