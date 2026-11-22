import datetime as dt
import logging

import pytest
from pydantic import ValidationError

from review_radar.models import Analysis, Review, Sentiment, Topic, Usage, parse_topic

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
