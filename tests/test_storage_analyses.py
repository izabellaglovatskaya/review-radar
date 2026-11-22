import contextlib
import datetime as dt
import sqlite3
from pathlib import Path

import pytest

from review_radar.hashing import cache_key, text_hash
from review_radar.models import Analysis, Review, Sentiment, Topic
from review_radar.storage import (
    PendingText,
    StorageError,
    connect,
    count_analyzed_reviews,
    count_reviews,
    load_analyses,
    pending_texts,
    save_analyses,
    save_reviews,
)

NOW = dt.datetime(2026, 11, 11, 12, 0, tzinfo=dt.UTC)
KEY = {"provider": "fixture", "model": "demo", "prompt_version": "v1"}


def review(text: str, *, day: int = 10, rating: int | None = 4, source: str = "web") -> Review:
    return Review.model_validate(
        {
            "external_id": None,
            "review_date": dt.date(2026, 11, day),
            "rating": rating,
            "text": text,
            "source": source,
        }
    )


def analysis(
    sentiment: Sentiment = Sentiment.NEGATIVE,
    topics: tuple[Topic, ...] = (Topic.PRICING,),
    confidence: float | None = None,
) -> Analysis:
    return Analysis(sentiment=sentiment, topics=topics, confidence=confidence)


@pytest.fixture
def conn(tmp_path: Path):
    with contextlib.closing(connect(tmp_path / "radar.db")) as connection:
        yield connection


def pending_of(text: str, count: int = 1, rating: int | None = 4) -> PendingText:
    return PendingText(text_hash=text_hash(text), text=text, rating=rating, review_count=count)


class TestPendingTexts:
    def test_empty_database(self, conn: sqlite3.Connection) -> None:
        assert pending_texts(conn, **KEY) == []

    def test_one_entry_per_distinct_hash_with_earliest_review(
        self, conn: sqlite3.Connection
    ) -> None:
        save_reviews(
            conn,
            [
                review("Too  EXPENSIVE", rating=2),
                review("other", rating=5),
                review("too expensive", day=11, rating=4),
                review("TOO expensive", day=12, source="app", rating=None),
            ],
        )

        result = pending_texts(conn, **KEY)

        assert result == [
            PendingText(text_hash("too expensive"), "Too  EXPENSIVE", 2, 3),
            PendingText(text_hash("other"), "other", 5, 1),
        ]

    def test_ordered_by_first_insertion_not_by_date(self, conn: sqlite3.Connection) -> None:
        save_reviews(conn, [review("b", day=12), review("a", day=1), review("c", day=5)])

        assert [p.text for p in pending_texts(conn, **KEY)] == ["b", "a", "c"]

    def test_rating_can_be_none(self, conn: sqlite3.Connection) -> None:
        save_reviews(conn, [review("x", rating=None)])

        assert pending_texts(conn, **KEY)[0].rating is None

    def test_analyzed_texts_are_excluded_for_the_same_key_only(
        self, conn: sqlite3.Connection
    ) -> None:
        save_reviews(conn, [review("a"), review("b")])
        save_analyses(conn, [(pending_of("a"), analysis())], **KEY, now=NOW)

        assert [p.text for p in pending_texts(conn, **KEY)] == ["b"]
        assert len(pending_texts(conn, **{**KEY, "prompt_version": "v2"})) == 2
        assert len(pending_texts(conn, **{**KEY, "model": "other"})) == 2
        assert len(pending_texts(conn, **{**KEY, "provider": "other"})) == 2

    def test_sql_metacharacters_in_parameters_are_data(self, conn: sqlite3.Connection) -> None:
        save_reviews(conn, [review("a")])

        result = pending_texts(conn, provider="x' OR '1'='1", model="m", prompt_version="v")

        assert len(result) == 1

    def test_closed_connection_is_a_storage_error(self, conn: sqlite3.Connection) -> None:
        conn.close()

        with pytest.raises(StorageError, match="Cannot read"):
            pending_texts(conn, **KEY)


class TestCountAnalyzedReviews:
    def test_counts_reviews_not_texts(self, conn: sqlite3.Connection) -> None:
        save_reviews(conn, [review("a"), review("A", day=11), review("b")])
        assert count_analyzed_reviews(conn, **KEY) == 0

        save_analyses(conn, [(pending_of("a", 2), analysis())], **KEY, now=NOW)

        assert count_analyzed_reviews(conn, **KEY) == 2
        assert count_analyzed_reviews(conn, **{**KEY, "model": "other"}) == 0

    def test_closed_connection_is_a_storage_error(self, conn: sqlite3.Connection) -> None:
        conn.close()

        with pytest.raises(StorageError, match="Cannot read"):
            count_analyzed_reviews(conn, **KEY)


class TestSaveAnalyses:
    def test_round_trip(self, conn: sqlite3.Connection) -> None:
        first = analysis(Sentiment.NEGATIVE, (Topic.PRICING, Topic.BILLING), 0.5)
        second = analysis(Sentiment.POSITIVE, (), None)

        saved = save_analyses(
            conn, [(pending_of("a"), first), (pending_of("b"), second)], **KEY, now=NOW
        )

        assert saved == 2
        assert load_analyses(conn, **KEY) == {text_hash("a"): first, text_hash("b"): second}

    def test_stored_columns(self, conn: sqlite3.Connection) -> None:
        save_analyses(
            conn,
            [(pending_of("Hello"), analysis(topics=(Topic.PRICING, Topic.OTHER)))],
            **KEY,
            now=NOW,
        )

        row = conn.execute(
            "SELECT cache_key, text_hash, provider, model, prompt_version, sentiment, topics,"
            " confidence, analyzed_at FROM analyses"
        ).fetchone()

        assert row == (
            cache_key("Hello", **KEY),
            text_hash("Hello"),
            "fixture",
            "demo",
            "v1",
            "negative",
            '["pricing", "other"]',
            None,
            "2026-11-11T12:00:00+00:00",
        )

    def test_saving_again_inserts_nothing_and_keeps_the_first(
        self, conn: sqlite3.Connection
    ) -> None:
        save_analyses(conn, [(pending_of("a"), analysis(Sentiment.NEGATIVE))], **KEY, now=NOW)

        saved = save_analyses(
            conn, [(pending_of("a"), analysis(Sentiment.POSITIVE, ()))], **KEY, now=NOW
        )

        assert saved == 0
        assert load_analyses(conn, **KEY)[text_hash("a")].sentiment is Sentiment.NEGATIVE

    def test_different_key_parts_are_separate_rows(self, conn: sqlite3.Connection) -> None:
        pair = [(pending_of("a"), analysis())]
        save_analyses(conn, pair, **KEY, now=NOW)

        assert save_analyses(conn, pair, **{**KEY, "prompt_version": "v2"}, now=NOW) == 1
        assert save_analyses(conn, pair, **{**KEY, "model": "m2"}, now=NOW) == 1
        assert len(load_analyses(conn, **KEY)) == 1

    def test_empty_input(self, conn: sqlite3.Connection) -> None:
        assert save_analyses(conn, [], **KEY, now=NOW) == 0

    def test_accepts_a_generator(self, conn: sqlite3.Connection) -> None:
        rows = ((pending_of(t), analysis()) for t in ("a", "b"))

        assert save_analyses(conn, rows, **KEY, now=NOW) == 2

    def test_is_one_transaction(self, conn: sqlite3.Connection) -> None:
        def rows():
            yield pending_of("a"), analysis()
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            save_analyses(conn, rows(), **KEY, now=NOW)

        assert load_analyses(conn, **KEY) == {}

    def test_naive_now_is_rejected(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="timezone"):
            save_analyses(
                conn, [(pending_of("a"), analysis())], **KEY, now=dt.datetime(2026, 11, 11)
            )
        assert load_analyses(conn, **KEY) == {}

    def test_default_now_is_utc(self, conn: sqlite3.Connection) -> None:
        save_analyses(conn, [(pending_of("a"), analysis())], **KEY)

        stamp = conn.execute("SELECT analyzed_at FROM analyses").fetchone()[0]

        assert dt.datetime.fromisoformat(stamp).utcoffset() == dt.timedelta(0)

    def test_blank_key_part_is_a_value_error(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="model"):
            save_analyses(conn, [(pending_of("a"), analysis())], **{**KEY, "model": " "}, now=NOW)

    def test_database_error_is_a_storage_error_without_text(self, conn: sqlite3.Connection) -> None:
        bad = Analysis.model_construct(sentiment="happy", topics=(), confidence=None)

        with pytest.raises(StorageError, match="Cannot save analyses") as excinfo:
            save_analyses(conn, [(pending_of("secret text"), bad)], **KEY, now=NOW)

        assert "secret text" not in str(excinfo.value)


class TestLoadAnalyses:
    def test_empty(self, conn: sqlite3.Connection) -> None:
        assert load_analyses(conn, **KEY) == {}

    def test_only_the_requested_key(self, conn: sqlite3.Connection) -> None:
        save_analyses(conn, [(pending_of("a"), analysis())], **KEY, now=NOW)
        save_analyses(conn, [(pending_of("b"), analysis())], **{**KEY, "model": "m2"}, now=NOW)

        assert set(load_analyses(conn, **KEY)) == {text_hash("a")}

    def test_closed_connection_is_a_storage_error(self, conn: sqlite3.Connection) -> None:
        conn.close()

        with pytest.raises(StorageError, match="Cannot read"):
            load_analyses(conn, **KEY)

    def test_corrupt_topics_are_a_storage_error(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO analyses VALUES ('k', 'h', 'fixture', 'demo', 'v1', 'negative',"
            " 'not json', NULL, 't')"
        )

        with pytest.raises(StorageError, match="Cannot read"):
            load_analyses(conn, **KEY)

    def test_review_counts_unaffected(self, conn: sqlite3.Connection) -> None:
        save_reviews(conn, [review("a")])
        save_analyses(conn, [(pending_of("a"), analysis())], **KEY, now=NOW)

        assert count_reviews(conn) == 1
