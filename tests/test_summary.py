"""Daily summary: aggregation and the rendered Telegram message."""

import contextlib
import datetime as dt
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from review_radar.hashing import text_hash
from review_radar.models import Analysis, Review
from review_radar.storage import PendingText, StorageError, connect, save_analyses, save_reviews
from review_radar.summary import (
    MAX_MESSAGE_LENGTH,
    MessageTooLong,
    Period,
    Summary,
    TopicCount,
    build_summary,
    check_length,
    percentages,
    render_message,
)

KEY = {"provider": "spy", "model": "spy-1", "prompt_version": "p1"}
DAY = dt.date(2026, 11, 10)
NEXT_DAY = dt.date(2026, 11, 11)
MARKER = "SECRET-REVIEW-TEXT"


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    with contextlib.closing(connect(tmp_path / "summary.db")) as connection:
        yield connection


def add(
    conn: sqlite3.Connection,
    review_id: str,
    day: dt.date = DAY,
    sentiment: str | None = None,
    topics: tuple[str, ...] = (),
    key: dict[str, str] | None = None,
) -> None:
    text = f"{MARKER} <a href='https://bad.example'>{review_id}</a>"
    review = Review(external_id=review_id, review_date=day, rating=3, text=text, source="web")
    save_reviews(conn, [review])
    if sentiment is not None:
        pending = PendingText(text_hash(review.text), review.text, 3, 1)
        analysis = Analysis(sentiment=sentiment, topics=list(topics), confidence=0.9)
        save_analyses(conn, [(pending, analysis)], **(key or KEY))


def summary_of(conn: sqlite3.Connection, period: Period | None = None) -> Summary:
    return build_summary(conn, period or Period.day(DAY), **KEY)


class TestPeriod:
    def test_single_day_label(self) -> None:
        assert Period.day(DAY).label == "2026-11-10"

    def test_range_label(self) -> None:
        assert Period(DAY, NEXT_DAY).label == "2026-11-10 to 2026-11-11"

    def test_reversed_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="start"):
            Period(NEXT_DAY, DAY)


class TestBuildSummary:
    def test_counts_sentiments_for_the_day_only(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="positive")
        add(conn, "r2", sentiment="neutral")
        add(conn, "r3", sentiment="negative", topics=("billing",))
        add(conn, "r4", NEXT_DAY, sentiment="negative", topics=("pricing",))

        result = summary_of(conn)

        assert (result.total, result.analyzed) == (3, 3)
        assert (result.positive, result.neutral, result.negative) == (1, 1, 1)
        assert result.top_topics == (TopicCount("Billing", 1),)

    def test_period_includes_both_ends(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", DAY, sentiment="positive")
        add(conn, "r2", NEXT_DAY, sentiment="positive")
        add(conn, "r3", NEXT_DAY + dt.timedelta(days=1), sentiment="positive")

        assert summary_of(conn, Period(DAY, NEXT_DAY)).total == 2

    def test_topics_count_only_negative_reviews(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="positive", topics=("packaging",))
        add(conn, "r2", sentiment="neutral", topics=("packaging",))
        add(conn, "r3", sentiment="negative", topics=("billing",))

        assert summary_of(conn).top_topics == (TopicCount("Billing", 1),)

    def test_top_three_by_count_with_a_stable_tie_break(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="negative", topics=("pricing", "packaging"))
        add(conn, "r2", sentiment="negative", topics=("pricing", "billing"))
        add(conn, "r3", sentiment="negative", topics=("delivery_delay",))

        assert summary_of(conn).top_topics == (
            TopicCount("Pricing", 2),
            TopicCount("Billing", 1),
            TopicCount("Delivery delay", 1),
        )

    def test_unanalyzed_reviews_are_reported(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="positive")
        add(conn, "r2")

        result = summary_of(conn)

        assert (result.total, result.analyzed, result.not_analyzed) == (2, 1, 1)

    def test_analyses_of_another_key_do_not_count(self, conn: sqlite3.Connection) -> None:
        other = {**KEY, "model": "other"}
        add(conn, "r1", sentiment="negative", topics=("billing",), key=other)

        result = summary_of(conn)

        assert (result.total, result.analyzed, result.negative) == (1, 0, 0)

    def test_reviews_sharing_a_text_each_count(self, conn: sqlite3.Connection) -> None:
        text = "Same text"
        reviews = [
            Review(external_id=i, review_date=DAY, rating=1, text=text, source="web")
            for i in ("a", "b")
        ]
        save_reviews(conn, reviews)
        pending = PendingText(text_hash(text), text, 1, 2)
        analysis = Analysis(sentiment="negative", topics=["billing"], confidence=0.9)
        save_analyses(conn, [(pending, analysis)], **KEY)

        result = summary_of(conn)

        assert (result.total, result.negative) == (2, 2)
        assert result.top_topics == (TopicCount("Billing", 2),)

    def test_empty_period(self, conn: sqlite3.Connection) -> None:
        result = summary_of(conn)

        assert (result.total, result.analyzed, result.top_topics) == (0, 0, ())

    def test_corrupt_stored_topics_are_a_clean_error(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="negative", topics=("billing",))
        conn.execute("UPDATE analyses SET topics = 'not json'")
        conn.commit()

        with pytest.raises(StorageError) as caught:
            summary_of(conn)

        assert MARKER not in str(caught.value)


class TestPercentages:
    @pytest.mark.parametrize(
        ("counts", "expected"),
        [
            ((1, 1, 1), (34, 33, 33)),
            ((2, 1, 0), (67, 33, 0)),
            ((1, 0, 0), (100, 0, 0)),
            ((0, 0, 0), (0, 0, 0)),
            ((135, 59, 106), (45, 20, 35)),
        ],
    )
    def test_shares_are_whole_and_sum_to_100(
        self, counts: tuple[int, int, int], expected: tuple[int, int, int]
    ) -> None:
        result = percentages(counts)

        assert result == expected
        assert sum(result) in (0, 100)


def sample(**changes: object) -> Summary:
    values: dict[str, object] = {
        "period": Period.day(DAY),
        "total": 4,
        "analyzed": 4,
        "positive": 2,
        "neutral": 1,
        "negative": 1,
        "top_topics": (TopicCount("Billing", 1),),
    }
    values.update(changes)
    return Summary(**values)  # type: ignore[arg-type]


class TestRenderMessage:
    def test_shows_counts_shares_and_topics(self) -> None:
        message = render_message(sample())

        assert "<b>Review Radar: 2026-11-10</b>" in message
        assert "Reviews: 4" in message
        assert "Positive: 2 (50%)" in message
        assert "Neutral: 1 (25%)" in message
        assert "Negative: 1 (25%)" in message
        assert "1. Billing: 1" in message

    def test_no_complaints_is_said_plainly(self) -> None:
        assert "No complaint topics" in render_message(sample(top_topics=()))

    def test_unanalyzed_reviews_are_visible(self) -> None:
        message = render_message(sample(total=6))

        assert "Not analyzed: 2" in message

    def test_fully_analyzed_has_no_warning_line(self) -> None:
        assert "Not analyzed" not in render_message(sample())

    def test_labels_are_html_escaped(self) -> None:
        message = render_message(sample(top_topics=(TopicCount("a<b>&c", 1),)))

        assert "a&lt;b&gt;&amp;c" in message
        assert "<b>&c" not in message

    def test_a_demo_message_is_labelled(self) -> None:
        message = render_message(sample(), demo=True)

        assert "<b>[DEMO] Review Radar: 2026-11-10</b>" in message

    def test_review_text_never_reaches_the_message(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="negative", topics=("billing",))

        message = render_message(summary_of(conn))

        assert MARKER not in message
        assert "bad.example" not in message


class TestLength:
    def test_a_message_at_the_limit_passes(self) -> None:
        check_length("x" * MAX_MESSAGE_LENGTH)

    def test_a_longer_message_is_rejected(self) -> None:
        with pytest.raises(MessageTooLong):
            check_length("x" * (MAX_MESSAGE_LENGTH + 1))

    def test_length_counts_utf16_units(self) -> None:
        with pytest.raises(MessageTooLong):
            check_length("\U0001f600" * (MAX_MESSAGE_LENGTH // 2 + 1))

    def test_the_limit_is_telegrams(self) -> None:
        assert MAX_MESSAGE_LENGTH == 4096
