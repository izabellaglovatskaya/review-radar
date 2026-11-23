"""`load_daily_analyses`: every review with its day and its analysis (or None) for one key."""

import contextlib
import datetime as dt
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from review_radar.hashing import text_hash
from review_radar.models import Analysis, Review
from review_radar.storage import (
    PendingText,
    StorageError,
    connect,
    load_daily_analyses,
    save_analyses,
    save_reviews,
)

KEY = {"provider": "spy", "model": "spy-1", "prompt_version": "p1"}
OTHER_KEY = {**KEY, "model": "other"}
DAY = dt.date(2026, 11, 10)
NEXT_DAY = dt.date(2026, 11, 11)
MARKER = "SECRET-REVIEW-TEXT"


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    with contextlib.closing(connect(tmp_path / "daily.db")) as connection:
        yield connection


def add(
    conn: sqlite3.Connection,
    review_id: str,
    day: dt.date = DAY,
    sentiment: str | None = None,
    topics: tuple[str, ...] = (),
    key: dict[str, str] | None = None,
) -> None:
    text = f"{MARKER} {review_id}"
    review = Review(external_id=review_id, review_date=day, rating=3, text=text, source="web")
    save_reviews(conn, [review])
    if sentiment is not None:
        pending = PendingText(text_hash(text), text, 3, 1)
        analysis = Analysis(sentiment=sentiment, topics=list(topics), confidence=0.9)
        save_analyses(conn, [(pending, analysis)], **(key or KEY))


def load(conn: sqlite3.Connection) -> list[tuple[dt.date, Analysis | None]]:
    return load_daily_analyses(conn, **KEY)


def shape(rows: list[tuple[dt.date, Analysis | None]]) -> list[tuple[dt.date, str | None]]:
    return [(day, None if item is None else item.sentiment.value) for day, item in rows]


class TestLoadDailyAnalyses:
    def test_an_empty_database_gives_an_empty_list(self, conn: sqlite3.Connection) -> None:
        assert load(conn) == []

    def test_returns_one_entry_per_review_with_its_analysis(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="negative", topics=("billing", "pricing"))
        add(conn, "r2", sentiment="positive")

        rows = load(conn)

        assert [day for day, _ in rows] == [DAY, DAY]
        first, second = rows[0][1], rows[1][1]
        assert first is not None
        assert first.sentiment.value == "negative"
        assert [topic.value for topic in first.topics] == ["billing", "pricing"]
        assert second is not None
        assert second.sentiment.value == "positive"

    def test_a_review_without_an_analysis_is_none(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="positive")
        add(conn, "r2")

        assert shape(load(conn)) == [(DAY, "positive"), (DAY, None)]

    def test_ordered_by_date_then_insertion(self, conn: sqlite3.Connection) -> None:
        add(conn, "late-1", NEXT_DAY, sentiment="neutral")
        add(conn, "early-1", DAY, sentiment="positive")
        add(conn, "late-2", NEXT_DAY, sentiment="negative", topics=("billing",))
        add(conn, "early-2", DAY, sentiment="negative", topics=("billing",))

        assert shape(load(conn)) == [
            (DAY, "positive"),
            (DAY, "negative"),
            (NEXT_DAY, "neutral"),
            (NEXT_DAY, "negative"),
        ]

    def test_analyses_of_another_key_are_none(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="negative", topics=("billing",), key=OTHER_KEY)

        assert shape(load(conn)) == [(DAY, None)]
        assert shape(load_daily_analyses(conn, **OTHER_KEY)) == [(DAY, "negative")]

    @pytest.mark.parametrize("field", ["provider", "model", "prompt_version"])
    def test_every_part_of_the_key_matters(self, conn: sqlite3.Connection, field: str) -> None:
        add(conn, "r1", sentiment="positive")

        assert shape(load_daily_analyses(conn, **{**KEY, field: "different"})) == [(DAY, None)]

    def test_reviews_sharing_a_text_each_get_the_analysis(self, conn: sqlite3.Connection) -> None:
        text = "Same text"
        reviews = [
            Review(external_id=i, review_date=DAY, rating=1, text=text, source="web")
            for i in ("a", "b")
        ]
        save_reviews(conn, reviews)
        analysis = Analysis(sentiment="negative", topics=["billing"], confidence=0.9)
        save_analyses(conn, [(PendingText(text_hash(text), text, 1, 2), analysis)], **KEY)

        assert shape(load(conn)) == [(DAY, "negative"), (DAY, "negative")]

    def test_review_text_is_never_selected(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="negative", topics=("billing",))
        statements: list[str] = []
        conn.set_trace_callback(statements.append)

        rows = load(conn)
        conn.set_trace_callback(None)

        assert rows
        selects = [s for s in statements if s.lstrip().upper().startswith(("SELECT", "WITH"))]
        assert selects
        for statement in selects:
            # `text_hash` is allowed; the bare `text` column is not.
            assert not re.search(r"\btext\b", statement, re.IGNORECASE), statement
        assert MARKER not in repr(rows)

    def test_corrupt_stored_topics_are_a_clean_error(self, conn: sqlite3.Connection) -> None:
        add(conn, "r1", sentiment="negative", topics=("billing",))
        conn.execute("UPDATE analyses SET topics = 'not json'")
        conn.commit()

        with pytest.raises(StorageError) as caught:
            load(conn)

        assert MARKER not in str(caught.value)
