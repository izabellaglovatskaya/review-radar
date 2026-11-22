import datetime as dt
import logging

import pytest
from pydantic import ValidationError

from review_radar.models import (
    MAX_EXTERNAL_ID_LENGTH,
    MAX_SOURCE_LENGTH,
    Analysis,
    Review,
    Sentiment,
    Topic,
    Usage,
    normalize_topics,
    parse_topic,
)

D = dt.date(2025, 1, 15)


def make_review(**overrides: object) -> Review:
    data: dict[str, object] = {
        "external_id": None,
        "review_date": D,
        "rating": 3,
        "text": "ok",
        "source": None,
    }
    data.update(overrides)
    return Review.model_validate(data)


def make_analysis(**overrides: object) -> Analysis:
    data: dict[str, object] = {"sentiment": "neutral", "topics": (), "confidence": None}
    data.update(overrides)
    return Analysis.model_validate(data)


class TestParseTopic:
    @pytest.mark.parametrize("raw", ["pricing", "PRICING", "  Pricing \n"])
    def test_known_value_is_case_and_whitespace_insensitive(self, raw: str) -> None:
        assert parse_topic(raw) is Topic.PRICING

    def test_topic_instance_passes_through(self) -> None:
        assert parse_topic(Topic.BILLING) is Topic.BILLING

    @pytest.mark.parametrize("raw", ["weather", "", "   ", "delivery delay"])
    def test_unknown_maps_to_other_with_warning(
        self, raw: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="review_radar.models"):
            assert parse_topic(raw) is Topic.OTHER
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    @pytest.mark.parametrize(
        "raw", ["ZEBRA-QUOKKA-9917 secret review text", b"ZEBRA-QUOKKA-9917", 5]
    )
    def test_the_raw_value_is_never_logged(
        self, raw: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="review_radar.models"):
            parse_topic(raw)

        assert "ZEBRA" not in caplog.text
        assert "QUOKKA" not in caplog.text
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_the_warning_names_the_type_and_length(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="review_radar.models"):
            parse_topic("weather")

        assert "str" in caplog.text
        assert "length 7" in caplog.text

    def test_known_value_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="review_radar.models"):
            parse_topic("packaging")
        assert caplog.records == []

    def test_non_string_maps_to_other(self) -> None:
        assert parse_topic(None) is Topic.OTHER
        assert parse_topic(5) is Topic.OTHER


class TestReview:
    def test_valid(self) -> None:
        r = make_review(external_id="x1", rating=5, text="  Great!  ", source="web")
        assert r.text == "Great!"
        assert r.rating == 5

    def test_optional_fields(self) -> None:
        assert make_review(rating=None).rating is None

    def test_is_frozen(self) -> None:
        r = make_review()
        with pytest.raises(ValidationError):
            r.text = "other"  # type: ignore[misc]

    @pytest.mark.parametrize("text", ["", "   ", "\n\t "])
    def test_blank_text_rejected(self, text: str) -> None:
        with pytest.raises(ValidationError):
            make_review(text=text)

    @pytest.mark.parametrize("rating", [0, 6, -1])
    def test_rating_out_of_range(self, rating: int) -> None:
        with pytest.raises(ValidationError):
            make_review(rating=rating)

    @pytest.mark.parametrize("rating", [1, 5])
    def test_rating_boundaries(self, rating: int) -> None:
        assert make_review(rating=rating).rating == rating

    @pytest.mark.parametrize("rating", [True, False, "4", 4.0])
    def test_rating_is_strict(self, rating: object) -> None:
        with pytest.raises(ValidationError):
            make_review(rating=rating)

    @pytest.mark.parametrize("value", ["", "   ", chr(9) + chr(10)])
    def test_blank_external_id_and_source_become_none(self, value: str) -> None:
        r = make_review(external_id=value, source=value)
        assert r.external_id is None
        assert r.source is None

    def test_external_id_and_source_are_stripped(self) -> None:
        r = make_review(external_id="  r1 ", source=" web ")
        assert r.external_id == "r1"
        assert r.source == "web"

    def test_length_limits_are_the_documented_values(self) -> None:
        assert (MAX_EXTERNAL_ID_LENGTH, MAX_SOURCE_LENGTH) == (128, 64)

    def test_external_id_and_source_at_the_limit_are_accepted(self) -> None:
        r = make_review(external_id="i" * MAX_EXTERNAL_ID_LENGTH, source="s" * MAX_SOURCE_LENGTH)
        assert len(r.external_id or "") == MAX_EXTERNAL_ID_LENGTH
        assert len(r.source or "") == MAX_SOURCE_LENGTH

    def test_padded_value_at_the_limit_is_accepted_after_strip(self) -> None:
        r = make_review(external_id=" " + "i" * MAX_EXTERNAL_ID_LENGTH + " ")
        assert r.external_id == "i" * MAX_EXTERNAL_ID_LENGTH

    @pytest.mark.parametrize(
        ("field", "limit"),
        [("external_id", MAX_EXTERNAL_ID_LENGTH), ("source", MAX_SOURCE_LENGTH)],
    )
    def test_over_limit_is_rejected(self, field: str, limit: int) -> None:
        with pytest.raises(ValidationError):
            make_review(**{field: "x" * (limit + 1)})

    def test_bad_date_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_review(review_date="not a date")

    def test_unicode_text(self) -> None:
        assert make_review(text="Ужасно 😡").text == "Ужасно 😡"


class TestAnalysis:
    def test_valid(self) -> None:
        a = make_analysis(sentiment="negative", topics=["pricing"], confidence=0.5)
        assert a.sentiment is Sentiment.NEGATIVE
        assert a.topics == (Topic.PRICING,)

    def test_invalid_sentiment_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_analysis(sentiment="angry")

    def test_unknown_topic_becomes_other(self) -> None:
        assert make_analysis(topics=("Weather",)).topics == (Topic.OTHER,)

    def test_normalizes_case(self) -> None:
        assert make_analysis(topics=(" BILLING ",)).topics == (Topic.BILLING,)

    def test_dedup_preserves_order(self) -> None:
        a = make_analysis(topics=("pricing", "billing", "PRICING", "x", "y"))
        assert a.topics == (Topic.PRICING, Topic.BILLING, Topic.OTHER)

    def test_truncates_to_three_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="review_radar.models"):
            a = make_analysis(topics=("pricing", "billing", "packaging", "refund_process"))
        assert a.topics == (Topic.PRICING, Topic.BILLING, Topic.PACKAGING)
        assert any("truncat" in r.getMessage().lower() for r in caplog.records)

    def test_exactly_three_kept(self) -> None:
        assert len(make_analysis(topics=("pricing", "billing", "packaging")).topics) == 3

    @pytest.mark.parametrize("bad", ["pricing", 5, None])
    def test_non_sequence_topics_rejected(self, bad: object) -> None:
        with pytest.raises(ValidationError):
            make_analysis(topics=bad)

    def test_truncation_keeps_known_topics_before_other(self) -> None:
        a = make_analysis(topics=["x", "pricing", "billing", "packaging"])
        assert a.topics == (Topic.PRICING, Topic.BILLING, Topic.PACKAGING)

    def test_truncation_keeps_other_when_slot_left(self) -> None:
        a = make_analysis(topics=["x", "pricing", "billing", "packaging", "refund_process"])
        assert a.topics == (Topic.PRICING, Topic.BILLING, Topic.PACKAGING)
        b = make_analysis(topics=["pricing", "x"])
        assert b.topics == (Topic.PRICING, Topic.OTHER)

    def test_no_truncation_keeps_original_order(self) -> None:
        assert make_analysis(topics=["x", "pricing"]).topics == (Topic.OTHER, Topic.PRICING)
        a = make_analysis(topics=["x", "pricing", "billing"])
        assert a.topics == (Topic.OTHER, Topic.PRICING, Topic.BILLING)

    def test_empty_topics_allowed(self) -> None:
        assert make_analysis().topics == ()

    @pytest.mark.parametrize("c", [-0.01, 1.01])
    def test_confidence_out_of_range(self, c: float) -> None:
        with pytest.raises(ValidationError):
            make_analysis(confidence=c)

    @pytest.mark.parametrize("c", [0.0, 1.0, None])
    def test_confidence_boundaries(self, c: float | None) -> None:
        assert make_analysis(confidence=c).confidence == c

    def test_frozen(self) -> None:
        with pytest.raises(ValidationError):
            make_analysis().confidence = 0.1  # type: ignore[misc]


class TestUsage:
    def test_add(self) -> None:
        total = Usage(input_tokens=1, output_tokens=2) + Usage(input_tokens=10, output_tokens=20)
        assert total == Usage(input_tokens=11, output_tokens=22)

    def test_add_non_usage_not_supported(self) -> None:
        with pytest.raises(TypeError):
            Usage(input_tokens=1, output_tokens=2) + 1  # type: ignore[operator]  # noqa: B018

    @pytest.mark.parametrize(
        "kwargs",
        [{"input_tokens": -1, "output_tokens": 0}, {"input_tokens": 0, "output_tokens": -1}],
    )
    def test_negative_rejected(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValidationError):
            Usage(**kwargs)

    def test_zero_ok_and_frozen(self) -> None:
        u = Usage(input_tokens=0, output_tokens=0)
        with pytest.raises(ValidationError):
            u.input_tokens = 1  # type: ignore[misc]


class TestNormalizeTopics:
    def test_known_topics_pass_through_in_order(self) -> None:
        result = normalize_topics(["billing", "pricing"])

        assert result.topics == (Topic.BILLING, Topic.PRICING)
        assert (result.unknown, result.truncated) == (0, 0)

    def test_empty(self) -> None:
        result = normalize_topics([])

        assert (result.topics, result.unknown, result.truncated) == ((), 0, 0)

    def test_unknown_values_become_other_and_are_counted(self) -> None:
        result = normalize_topics(["weather", None, 5])

        assert result.topics == (Topic.OTHER,)
        assert result.unknown == 3
        assert result.truncated == 0

    def test_a_valid_other_is_not_unknown(self) -> None:
        result = normalize_topics(["other", "pricing"])

        assert result.topics == (Topic.OTHER, Topic.PRICING)
        assert result.unknown == 0

    def test_loose_case_and_whitespace_are_known_not_unknown(self) -> None:
        result = normalize_topics([" Pricing "])

        assert result.topics == (Topic.PRICING,)
        assert result.unknown == 0

    def test_duplicates_are_removed_but_unknown_repeats_are_all_counted(self) -> None:
        result = normalize_topics(["pricing", "pricing", "weather", "weather"])

        assert result.topics == (Topic.PRICING, Topic.OTHER)
        assert result.unknown == 2
        assert result.truncated == 0

    def test_more_than_three_is_truncated_and_counted(self) -> None:
        result = normalize_topics(["delivery_delay", "pricing", "billing", "packaging", "other"])

        assert result.topics == (Topic.DELIVERY_DELAY, Topic.PRICING, Topic.BILLING)
        assert result.unknown == 0
        assert result.truncated == 2

    def test_mixed_unknown_and_truncation(self) -> None:
        raw = ["pricing", "weather", "billing", "packaging", "delivery_delay"]

        result = normalize_topics(raw)

        assert result.topics == (Topic.PRICING, Topic.BILLING, Topic.PACKAGING)
        assert result.unknown == 1
        assert result.truncated == 2

    def test_accepts_any_iterable(self) -> None:
        result = normalize_topics(t for t in ("pricing", "billing"))

        assert result.topics == (Topic.PRICING, Topic.BILLING)

    def test_analysis_uses_the_same_rules(self) -> None:
        raw = ["pricing", "weather", "billing", "packaging", "delivery_delay"]

        analysis = Analysis(sentiment=Sentiment.NEGATIVE, topics=raw, confidence=None)

        assert analysis.topics == normalize_topics(raw).topics

    def test_warnings_never_contain_the_raw_values(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="review_radar.models"):
            normalize_topics(["ZEBRA-QUOKKA-9917", "a", "b", "c", "d"])

        assert "ZEBRA" not in caplog.text
