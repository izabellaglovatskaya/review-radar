import contextlib
import datetime as dt
import os
import re
import sqlite3
from pathlib import Path

import pytest

from review_radar import storage
from review_radar.hashing import text_hash
from review_radar.models import Review
from review_radar.storage import (
    APPLICATION_ID,
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

        assert version == SCHEMA_VERSION == 1
        assert "reviews" in tables

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "radar.db"

        with contextlib.closing(connect(path)):
            pass

        assert path.is_file()

    def test_foreign_keys_enabled(self, tmp_path: Path) -> None:
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

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

        with pytest.raises(StorageError, match=r"7.*1"):
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
        broken = good.replace("PRAGMA user_version = 1;", "SELECT * FROM no_such_table;")
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
            raw.execute("PRAGMA user_version = 1")
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
            [make_review(external_id="r1", text="changed", rating=1, source="app")],
        )

        assert result == SaveResult(new=0, existing=1)
        with contextlib.closing(connect(db_path(tmp_path))) as conn:
            (stored,) = list_reviews(conn)
        assert (stored.text, stored.rating, stored.source) == ("original", 5, "web")

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
            "id:r1",
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
    def test_id_form(self) -> None:
        assert dedup_key(make_review(external_id="r1")) == "id:r1"

    def test_id_form_ignores_other_fields(self) -> None:
        a = make_review(external_id="r1", text="a")
        b = make_review(external_id="r1", text="b", source="app")

        assert dedup_key(a) == dedup_key(b)

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
