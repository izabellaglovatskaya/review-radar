import contextlib
import datetime as dt
import logging
import os
import re
import sqlite3
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from review_radar import storage
from review_radar.hashing import text_hash
from review_radar.models import UNKNOWN_SOURCE, Review
from review_radar.storage import (
    APPLICATION_ID,
    MIN_UPGRADABLE_VERSION,
    SCHEMA_VERSION,
    SaveResult,
    StorageError,
    connect,
    count_reviews,
    dedup_key,
    list_reviews,
    save_reviews,
)

D = dt.date(2026, 11, 10)
NOW = dt.datetime(2026, 11, 11, 12, 0, tzinfo=dt.UTC)


def make_review(**overrides: object) -> Review:
    data: dict[str, object] = {
        "external_id": None,
        "review_date": D,
        "rating": 4,
        "text": "Fine product",
        "source": "web",
    }
    data.update(overrides)
    return Review.model_validate(data)


def db_path(tmp_path: Path) -> Path:
    return tmp_path / "radar.db"


def save(path: Path, reviews: list[Review]) -> SaveResult:
    with contextlib.closing(connect(path)) as conn:
        return save_reviews(conn, reviews, now=NOW)


def count(path: Path) -> int:
    with contextlib.closing(connect(path)) as conn:
        return count_reviews(conn)


class TestConnect:
    def test_fresh_db_has_version_and_table(self, tmp_path: Path) -> None:
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            tables = [
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            ]

        assert version == SCHEMA_VERSION == 6
        assert MIN_UPGRADABLE_VERSION == 2
        assert {"reviews", "analyses", "runs"} <= set(tables)

    def test_stage_3_database_is_rejected_with_a_recreate_hint(self, tmp_path: Path) -> None:
        path = db_path(tmp_path)
        with contextlib.closing(sqlite3.connect(path)) as raw:
            raw.execute("CREATE TABLE reviews (x INTEGER)")
            raw.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            raw.execute("PRAGMA user_version = 1")
            raw.commit()

        with pytest.raises(StorageError, match=r"version 1, expected 6.*run ingest again"):
            connect(path)

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "radar.db"

        with contextlib.closing(connect(path)):
            pass

        assert path.is_file()

    def test_reconnect_keeps_data(self, tmp_path: Path) -> None:
        path = db_path(tmp_path)
        save(path, [make_review(text="one")])

        with contextlib.closing(connect(path)) as conn:
            assert count_reviews(conn) == 1
            assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    def test_unknown_future_version_is_rejected(self, tmp_path: Path) -> None:
        path = db_path(tmp_path)
        with contextlib.closing(sqlite3.connect(path)) as raw:
            raw.execute("PRAGMA user_version = 7")

        with pytest.raises(StorageError, match=r"7, expected 6"):
            connect(path)

    def test_foreign_database_is_rejected(self, tmp_path: Path) -> None:
        path = db_path(tmp_path)
        with contextlib.closing(sqlite3.connect(path)) as raw:
            raw.execute("CREATE TABLE other (x INTEGER)")
            raw.commit()

        with pytest.raises(StorageError, match="not a Review Radar database"):
            connect(path)

    def test_non_sqlite_file_is_rejected_and_not_locked(self, tmp_path: Path) -> None:
        path = db_path(tmp_path)
        path.write_bytes(os.urandom(2048))

        with pytest.raises(StorageError) as excinfo:
            connect(path)

        assert "Traceback" not in str(excinfo.value)
        path.unlink()  # fails on Windows if the connection was left open

    def test_unopenable_path_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(StorageError):
            connect(tmp_path)  # a directory

    def test_parent_that_is_a_file_is_rejected(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("x")

        with pytest.raises(StorageError):
            connect(blocker / "radar.db")


class TestSchemaInit:
    def test_application_id_literal_matches_constant(self, tmp_path: Path) -> None:
        match = re.search(r"PRAGMA application_id = (\d+);", storage._SCHEMA)
        assert match is not None
        assert int(match.group(1)) == APPLICATION_ID == 0x52765264
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            assert conn.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID

    def test_failure_mid_init_leaves_an_empty_database(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        good = storage._SCHEMA
        broken = good.replace("PRAGMA user_version = 6;", "SELECT * FROM no_such_table;")
        assert broken != good
        monkeypatch.setattr(storage, "_SCHEMA", broken)

        with pytest.raises(StorageError):
            connect(db_path(tmp_path))

        with contextlib.closing(sqlite3.connect(db_path(tmp_path))) as raw:
            assert raw.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] == 0
            assert raw.execute("PRAGMA user_version").fetchone()[0] == 0
        monkeypatch.setattr(storage, "_SCHEMA", good)
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            assert count_reviews(conn) == 0

    def test_initialized_database_is_not_written_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with contextlib.closing(connect(db_path(tmp_path))):
            pass
        monkeypatch.setattr(storage, "_SCHEMA", "SELECT * FROM no_such_table;")

        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            assert count_reviews(conn) == 0

    def test_foreign_database_with_matching_version_is_rejected(self, tmp_path: Path) -> None:
        path = db_path(tmp_path)
        with contextlib.closing(sqlite3.connect(path)) as raw:
            raw.execute("CREATE TABLE reviews (x INTEGER)")
            raw.execute("PRAGMA user_version = 3")
            raw.commit()

        with pytest.raises(StorageError, match="not a Review Radar database"):
            connect(path)

    def test_empty_database_with_foreign_application_id_is_rejected(self, tmp_path: Path) -> None:
        path = db_path(tmp_path)
        with contextlib.closing(sqlite3.connect(path)) as raw:
            raw.execute("PRAGMA application_id = 12345")

        with pytest.raises(StorageError, match="not a Review Radar database"):
            connect(path)

    def test_zero_byte_file_is_initialized(self, tmp_path: Path) -> None:
        path = db_path(tmp_path)
        path.write_bytes(b"")

        with contextlib.closing(connect(path)) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert count_reviews(conn) == 0


SCHEMA_V2 = """
BEGIN;
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY,
    dedup_key TEXT NOT NULL UNIQUE,
    external_id TEXT,
    review_date TEXT NOT NULL,
    rating INTEGER CHECK (rating IS NULL OR rating BETWEEN 1 AND 5),
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    source TEXT,
    imported_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_review_date ON reviews (review_date);
PRAGMA application_id = 1383486052;
PRAGMA user_version = 2;
COMMIT;
"""


def make_v2_database(path: Path) -> None:
    """Build a database exactly as stage 3 did, holding one review."""
    with contextlib.closing(sqlite3.connect(path)) as raw:
        raw.executescript(SCHEMA_V2)
        raw.execute(
            "INSERT INTO reviews (dedup_key, external_id, review_date, rating, text, text_hash,"
            " source, imported_at) VALUES ('id:[\"web\",\"r1\"]', 'r1', '2026-11-10', 4,"
            " 'Old review', ?, 'web', '2026-11-11T12:00:00+00:00')",
            (text_hash("Old review"),),
        )
        raw.commit()


def table_names(path: Path) -> set[str]:
    with contextlib.closing(sqlite3.connect(path)) as raw:
        return {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def user_version(path: Path) -> int:
    with contextlib.closing(sqlite3.connect(path)) as raw:
        return int(raw.execute("PRAGMA user_version").fetchone()[0])


class TestUpgrade:
    def test_v2_database_is_upgraded_and_keeps_its_reviews(self, tmp_path: Path) -> None:
        path = db_path(tmp_path)
        make_v2_database(path)
        assert "analyses" not in table_names(path)

        with contextlib.closing(connect(path)) as conn:
            assert [r.text for r in list_reviews(conn)] == ["Old review"]
            assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert conn.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID

        assert "analyses" in table_names(path)

    def test_upgrade_is_atomic(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = db_path(tmp_path)
        make_v2_database(path)
        good = storage._SCHEMA
        broken = good.replace("COMMIT;", "SELECT * FROM no_such_table;\nCOMMIT;")
        assert broken != good
        monkeypatch.setattr(storage, "_SCHEMA", broken)

        with pytest.raises(StorageError):
            connect(path)

        assert user_version(path) == 2
        assert "analyses" not in table_names(path)
        monkeypatch.setattr(storage, "_SCHEMA", good)
        with contextlib.closing(connect(path)) as conn:
            assert count_reviews(conn) == 1
        assert user_version(path) == SCHEMA_VERSION

    def test_upgrade_with_foreign_application_id_is_rejected(self, tmp_path: Path) -> None:
        path = db_path(tmp_path)
        with contextlib.closing(sqlite3.connect(path)) as raw:
            raw.execute("CREATE TABLE reviews (x INTEGER)")
            raw.execute("PRAGMA user_version = 2")
            raw.commit()

        with pytest.raises(StorageError, match="not a Review Radar database"):
            connect(path)

        assert user_version(path) == 2
        assert "analyses" not in table_names(path)

    def test_current_database_is_not_rewritten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with contextlib.closing(connect(db_path(tmp_path))):
            pass
        monkeypatch.setattr(storage, "_SCHEMA", "SELECT * FROM no_such_table;")

        with contextlib.closing(connect(db_path(tmp_path))):
            pass

    def test_script_is_additive_only(self) -> None:
        statements = [
            line.strip().rstrip(";")
            for line in storage._SCHEMA.splitlines()
            if line.strip() and not line.startswith((" ", ")"))
        ]
        allowed = ("BEGIN", "COMMIT", "PRAGMA", "CREATE TABLE IF NOT EXISTS", "CREATE INDEX IF")
        assert all(s.startswith(allowed) for s in statements), statements


def index_names(path: Path, table: str) -> set[str]:
    with contextlib.closing(sqlite3.connect(path)) as raw:
        return {r[1] for r in raw.execute(f"PRAGMA index_list({table})")}


class TestTextHashIndex:
    def test_fresh_database_has_the_reviews_text_hash_index(self, tmp_path: Path) -> None:
        with contextlib.closing(connect(db_path(tmp_path))):
            pass

        assert "idx_reviews_text_hash" in index_names(db_path(tmp_path), "reviews")

    def test_upgraded_database_gets_the_index(self, tmp_path: Path) -> None:
        make_v2_database(db_path(tmp_path))
        assert "idx_reviews_text_hash" not in index_names(db_path(tmp_path), "reviews")

        with contextlib.closing(connect(db_path(tmp_path))):
            pass

        assert "idx_reviews_text_hash" in index_names(db_path(tmp_path), "reviews")

    def test_index_covers_text_hash(self, tmp_path: Path) -> None:
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            columns = [r[2] for r in conn.execute("PRAGMA index_info(idx_reviews_text_hash)")]

        assert columns == ["text_hash"]


class TestAnalysesTable:
    def test_columns_and_index(self, tmp_path: Path) -> None:
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            columns = [r[1] for r in conn.execute("PRAGMA table_info(analyses)")]
            index_columns = [
                [c[2] for c in conn.execute(f"PRAGMA index_info({idx[1]})")]
                for idx in conn.execute("PRAGMA index_list(analyses)")
            ]

        assert columns == [
            "cache_key",
            "text_hash",
            "provider",
            "model",
            "prompt_version",
            "sentiment",
            "topics",
            "confidence",
            "analyzed_at",
        ]
        assert ["text_hash", "provider", "model", "prompt_version"] in index_columns

    @pytest.mark.parametrize(
        ("sentiment", "confidence"), [("happy", None), ("positive", 1.5), ("positive", -0.1)]
    )
    def test_check_constraints(self, tmp_path: Path, sentiment: str, confidence: float) -> None:
        with (
            contextlib.closing(connect(db_path(tmp_path))) as conn,
            pytest.raises(sqlite3.IntegrityError),
        ):
            conn.execute(
                "INSERT INTO analyses VALUES ('k', 'h', 'p', 'm', 'v', ?, '[]', ?, 't')",
                (sentiment, confidence),
            )


class TestReadErrors:
    def test_count_reviews_wraps_sqlite_errors(self, tmp_path: Path) -> None:
        conn = connect(db_path(tmp_path))
        conn.close()

        with pytest.raises(StorageError, match="Cannot read reviews"):
            count_reviews(conn)

    def test_list_reviews_wraps_sqlite_errors(self, tmp_path: Path) -> None:
        conn = connect(db_path(tmp_path))
        conn.close()

        with pytest.raises(StorageError, match="Cannot read reviews"):
            list_reviews(conn)

    def test_list_reviews_wraps_invalid_rows_without_review_text(self, tmp_path: Path) -> None:
        fake_text = "ZEBRA-QUOKKA-9917 distinctive"
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            conn.execute(
                "INSERT INTO reviews (dedup_key, external_id, review_date, rating, text,"
                " text_hash, source, imported_at) VALUES ('k', ?, '2026-11-10', 4, ?, 'h', 'web',"
                " 't')",
                ("x" * 500, fake_text),
            )
            with pytest.raises(StorageError) as excinfo:
                list_reviews(conn)

        assert fake_text not in str(excinfo.value)
        assert "x" * 50 not in str(excinfo.value)

    def test_list_reviews_wraps_a_bad_date(self, tmp_path: Path) -> None:
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            conn.execute(
                "INSERT INTO reviews (dedup_key, external_id, review_date, rating, text,"
                " text_hash, source, imported_at) VALUES ('k', NULL, 'not-a-date', 4, 'T', 'h',"
                " NULL, 't')"
            )
            with pytest.raises(StorageError, match="Cannot read reviews"):
                list_reviews(conn)


class TestNaiveNow:
    def test_save_reviews_rejects_a_naive_now(self, tmp_path: Path) -> None:
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            with pytest.raises(ValueError, match="timezone"):
                save_reviews(conn, [make_review()], now=dt.datetime(2026, 11, 11, 12, 0))
            assert count_reviews(conn) == 0

    def test_an_aware_now_is_stored_in_utc(self, tmp_path: Path) -> None:
        plus_three = dt.timezone(dt.timedelta(hours=3))
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            save_reviews(
                conn, [make_review()], now=dt.datetime(2026, 11, 11, 15, 0, tzinfo=plus_three)
            )
            stamp = conn.execute("SELECT imported_at FROM reviews").fetchone()[0]

        assert stamp == "2026-11-11T12:00:00+00:00"


class TestSaveReviews:
    def test_returns_new_count(self, tmp_path: Path) -> None:
        reviews = [make_review(text="a"), make_review(text="b"), make_review(external_id="r1")]

        result = save(db_path(tmp_path), reviews)

        assert result == SaveResult(new=3, existing=0)
        assert count(db_path(tmp_path)) == 3

    def test_saving_again_adds_nothing(self, tmp_path: Path) -> None:
        reviews = [make_review(text="a"), make_review(external_id="r1", text="b")]
        save(db_path(tmp_path), reviews)

        result = save(db_path(tmp_path), reviews)

        assert result == SaveResult(new=0, existing=2)
        assert count(db_path(tmp_path)) == 2

    def test_empty_batch(self, tmp_path: Path) -> None:
        assert save(db_path(tmp_path), []) == SaveResult(new=0, existing=0)

    def test_accepts_a_generator(self, tmp_path: Path) -> None:
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            result = save_reviews(conn, (make_review(text=f"t{i}") for i in range(3)))

        assert result == SaveResult(new=3, existing=0)

    def test_duplicates_inside_one_batch(self, tmp_path: Path) -> None:
        result = save(db_path(tmp_path), [make_review(), make_review()])

        assert result == SaveResult(new=1, existing=1)

    def test_idless_same_content_is_one_row(self, tmp_path: Path) -> None:
        save(db_path(tmp_path), [make_review()])

        result = save(db_path(tmp_path), [make_review()])

        assert result == SaveResult(new=0, existing=1)

    def test_idless_case_and_whitespace_variants_are_the_same_review(self, tmp_path: Path) -> None:
        save(db_path(tmp_path), [make_review(text="Fine product")])

        result = save(db_path(tmp_path), [make_review(text="  FINE   product ")])

        assert result == SaveResult(new=0, existing=1)

    def test_idless_different_source_or_date_are_separate(self, tmp_path: Path) -> None:
        save(db_path(tmp_path), [make_review()])

        result = save(
            db_path(tmp_path),
            [make_review(source="app"), make_review(review_date=dt.date(2026, 11, 11))],
        )

        assert result == SaveResult(new=2, existing=0)
        assert count(db_path(tmp_path)) == 3

    def test_same_id_different_text_keeps_the_first(self, tmp_path: Path) -> None:
        save(db_path(tmp_path), [make_review(external_id="r1", text="original", rating=5)])

        result = save(
            db_path(tmp_path),
            [make_review(external_id="r1", text="changed", rating=1, source="WEB")],
        )

        assert result == SaveResult(new=0, existing=1)
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            (stored,) = list_reviews(conn)
        assert (stored.text, stored.rating, stored.source) == ("original", 5, "web")

    def test_same_id_from_another_source_is_stored_separately(self, tmp_path: Path) -> None:
        save(db_path(tmp_path), [make_review(external_id="r1", source="web")])

        result = save(db_path(tmp_path), [make_review(external_id="r1", source="app")])

        assert result == SaveResult(new=1, existing=0)

    def test_optional_fields_round_trip(self, tmp_path: Path) -> None:
        review = make_review(rating=None, source=None)
        save(db_path(tmp_path), [review])

        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            assert list_reviews(conn) == [review]

    def test_full_review_round_trip(self, tmp_path: Path) -> None:
        review = make_review(external_id="r9", rating=2, text="Отличный товар \U0001f60d")
        save(db_path(tmp_path), [review])

        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            assert list_reviews(conn) == [review]

    def test_stored_columns(self, tmp_path: Path) -> None:
        review = make_review(external_id="r1", text="Hello")
        save(db_path(tmp_path), [review])

        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            row = conn.execute(
                "SELECT dedup_key, external_id, review_date, text_hash, imported_at FROM reviews"
            ).fetchone()

        assert row == (
            'id:["web","r1"]',
            "r1",
            "2026-11-10",
            text_hash("Hello"),
            "2026-11-11T12:00:00+00:00",
        )

    def test_imported_at_defaults_to_utc_now(self, tmp_path: Path) -> None:
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            save_reviews(conn, [make_review()])
            stamp = conn.execute("SELECT imported_at FROM reviews").fetchone()[0]

        assert dt.datetime.fromisoformat(stamp).utcoffset() == dt.timedelta(0)

    def test_list_is_ordered_by_date_then_id(self, tmp_path: Path) -> None:
        late = make_review(review_date=dt.date(2026, 11, 12), text="late")
        early_a = make_review(review_date=dt.date(2026, 11, 1), text="early a")
        early_b = make_review(review_date=dt.date(2026, 11, 1), text="early b")
        save(db_path(tmp_path), [late, early_a, early_b])

        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            assert [r.text for r in list_reviews(conn)] == ["early a", "early b", "late"]

    def test_failure_mid_batch_rolls_everything_back(self, tmp_path: Path) -> None:
        good = make_review(text="good")
        bad = Review.model_construct(
            external_id=None, review_date=D, rating=9, text="bad", source=None
        )

        with pytest.raises(StorageError) as excinfo:
            save(db_path(tmp_path), [good, bad])

        assert "bad" not in str(excinfo.value)
        assert count(db_path(tmp_path)) == 0

    def test_failure_does_not_touch_earlier_data(self, tmp_path: Path) -> None:
        save(db_path(tmp_path), [make_review(text="kept")])
        bad = Review.model_construct(
            external_id=None, review_date=D, rating=9, text="bad", source=None
        )

        with pytest.raises(StorageError):
            save(db_path(tmp_path), [make_review(text="new"), bad])

        assert count(db_path(tmp_path)) == 1

    def test_sql_metacharacters_are_data(self, tmp_path: Path) -> None:
        nasty = make_review(
            external_id="x'; DROP TABLE reviews;--", text="'); DELETE FROM reviews;"
        )
        save(db_path(tmp_path), [nasty])

        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            assert list_reviews(conn) == [nasty]


class TestDedupKey:
    def test_id_form_is_scoped_by_source(self) -> None:
        assert dedup_key(make_review(external_id="r1")) == 'id:["web","r1"]'

    def test_id_form_ignores_text_date_and_rating(self) -> None:
        a = make_review(external_id="r1", text="a", rating=1)
        b = make_review(external_id="r1", text="b", rating=5, review_date=dt.date(2026, 1, 1))

        assert dedup_key(a) == dedup_key(b)

    def test_same_id_from_another_source_is_a_different_review(self) -> None:
        assert dedup_key(make_review(external_id="r1", source="web")) != dedup_key(
            make_review(external_id="r1", source="app")
        )

    def test_source_case_does_not_matter_for_the_id_key(self) -> None:
        assert dedup_key(make_review(external_id="r1", source="Web")) == dedup_key(
            make_review(external_id="r1", source="web")
        )

    def test_missing_source_uses_unknown(self) -> None:
        assert UNKNOWN_SOURCE == "unknown"
        assert dedup_key(make_review(external_id="r1", source=None)) == 'id:["unknown","r1"]'
        assert dedup_key(make_review(external_id="r1", source=None)) == dedup_key(
            make_review(external_id="r1", source="unknown")
        )

    def test_id_and_source_boundary_cannot_collide(self) -> None:
        a = make_review(external_id='b"]', source="a")
        b = make_review(external_id="b", source='a"]')

        assert dedup_key(a) != dedup_key(b)

    def test_hash_form(self) -> None:
        key = dedup_key(make_review())

        assert key.startswith("hash:")
        assert len(key) == len("hash:") + 64

    def test_stable_across_calls(self) -> None:
        assert dedup_key(make_review()) == dedup_key(make_review())

    def test_normalized_text(self) -> None:
        assert dedup_key(make_review(text="Hello  World")) == dedup_key(
            make_review(text="hello world")
        )

    def test_none_source_differs_from_text_source_none_string(self) -> None:
        assert dedup_key(make_review(source=None)) != dedup_key(make_review(source="None"))

    def test_text_source_boundary_cannot_collide(self) -> None:
        a = make_review(text="a|b", source="c")
        b = make_review(text="a", source="b|c")

        assert dedup_key(a) != dedup_key(b)

    def test_date_matters(self) -> None:
        assert dedup_key(make_review()) != dedup_key(make_review(review_date=dt.date(2026, 1, 1)))

    def test_id_key_cannot_equal_hash_key(self) -> None:
        hash_key = dedup_key(make_review())

        assert dedup_key(make_review(external_id=hash_key)) != hash_key


class TestNewDatabaseFilePermissions:
    """A new database holds review text, so it is created owner-only (0o600)."""

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_new_file_is_owner_only_even_with_a_permissive_umask(self, tmp_path: Path) -> None:
        old_umask = os.umask(0o022)
        try:
            path = db_path(tmp_path)
            connect(path).close()
        finally:
            os.umask(old_umask)

        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_requests_mode_0o600_for_a_new_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = db_path(tmp_path)
        calls: list[tuple[int, int]] = []
        real_open = os.open

        def recording_open(
            file: str | os.PathLike[str], flags: int, mode: int = 0o777, **kw: Any
        ) -> int:
            if Path(file) == path:
                calls.append((flags, mode))
            return real_open(file, flags, mode, **kw)

        monkeypatch.setattr(storage.os, "open", recording_open)

        connect(path).close()

        assert len(calls) == 1
        flags, mode = calls[0]
        assert mode == 0o600
        assert flags & os.O_CREAT
        assert flags & os.O_EXCL

    def test_an_existing_database_keeps_its_data(self, tmp_path: Path) -> None:
        path = db_path(tmp_path)
        with contextlib.closing(connect(path)) as conn:
            save_reviews(conn, [make_review()])

        with contextlib.closing(connect(path)) as conn:
            assert count_reviews(conn) == 1

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_an_existing_file_keeps_its_mode(self, tmp_path: Path) -> None:
        path = db_path(tmp_path)
        connect(path).close()
        path.chmod(0o640)

        connect(path).close()

        assert stat.S_IMODE(path.stat().st_mode) == 0o640

    def test_a_new_folder_is_requested_owner_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "new-folder" / "test.db"
        modes: list[int] = []
        real_mkdir = Path.mkdir

        def recording_mkdir(self: Path, mode: int = 0o777, **kw: Any) -> None:
            if self == path.parent:
                modes.append(mode)
            real_mkdir(self, mode, **kw)

        monkeypatch.setattr(Path, "mkdir", recording_mkdir)

        connect(path).close()

        assert modes == [0o700]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_a_new_folder_is_owner_only(self, tmp_path: Path) -> None:
        path = tmp_path / "new-folder" / "test.db"
        old_umask = os.umask(0o022)
        try:
            connect(path).close()
        finally:
            os.umask(old_umask)

        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    @pytest.mark.parametrize(
        ("mode", "shared"), [(0o600, False), (0o400, False), (0o640, True), (0o604, True)]
    )
    def test_group_or_other_bits_count_as_shared(self, mode: int, shared: bool) -> None:
        assert storage.readable_by_others(mode) is shared

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    def test_warns_when_an_existing_database_is_readable_by_others(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = db_path(tmp_path)
        connect(path).close()
        path.chmod(0o644)

        with caplog.at_level(logging.WARNING, logger="review_radar"):
            connect(path).close()

        assert f"chmod 600 {path.name}" in caplog.text
        assert stat.S_IMODE(path.stat().st_mode) == 0o644  # warned, never changed

    def test_no_warning_for_a_private_database(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = db_path(tmp_path)
        connect(path).close()

        with caplog.at_level(logging.WARNING, logger="review_radar"):
            connect(path).close()

        assert caplog.records == []

    def test_an_os_error_while_creating_is_reported_without_the_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing_open(*_args: object, **_kwargs: object) -> int:
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(storage.os, "open", failing_open)

        with pytest.raises(StorageError, match="Cannot open the database: Permission denied"):
            connect(db_path(tmp_path))
