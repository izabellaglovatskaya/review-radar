"""Schema v4: the `runs` table (one row per analyze run) and the v2/v3 upgrades."""

import contextlib
import datetime as dt
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from review_radar import storage
from review_radar.storage import RunRecord, StorageError, connect, list_runs, save_run
from test_storage import make_v2_database

START = dt.datetime(2026, 11, 22, 10, 0, tzinfo=dt.UTC)
END = dt.datetime(2026, 11, 22, 10, 0, 5, tzinfo=dt.UTC)

SCHEMA_V3 = """
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
CREATE TABLE IF NOT EXISTS analyses (
    cache_key TEXT PRIMARY KEY,
    text_hash TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    topics TEXT NOT NULL,
    confidence REAL,
    analyzed_at TEXT NOT NULL
);
PRAGMA application_id = 1383486052;
PRAGMA user_version = 3;
COMMIT;
"""


def record(**overrides: object) -> RunRecord:
    values: dict[str, object] = {
        "started_at": START,
        "finished_at": END,
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "prompt_version": "anthropic-1+abcd1234",
        "reviews_total": 20,
        "analyzed": 18,
        "cached": 0,
        "failed": 2,
        "input_tokens": 2600,
        "output_tokens": 620,
        "cost_usd": Decimal("0.00057"),
        "fatal_reason": None,
        "not_analyzed_limit": 0,
        "outcome": "completed",
        "failures": {"refusal": 2},
        "api_calls": 3,
    }
    values.update(overrides)
    return RunRecord(**values)  # type: ignore[arg-type]


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "radar.db"


def table_names(path: Path) -> set[str]:
    with contextlib.closing(sqlite3.connect(path)) as raw:
        return {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


class TestRunsTable:
    def test_fresh_database_has_the_runs_table_with_the_agreed_columns(self, path: Path) -> None:
        with contextlib.closing(connect(path)) as conn:
            columns = [r[1] for r in conn.execute("PRAGMA table_info(runs)")]

        assert columns == [
            "id",
            "started_at",
            "finished_at",
            "provider",
            "model",
            "prompt_version",
            "reviews_total",
            "analyzed",
            "cached",
            "failed",
            "input_tokens",
            "output_tokens",
            "cost_usd",
            "fatal_reason",
            "not_analyzed_limit",
            "outcome",
            "failures",
            "api_calls",
        ]

    def test_save_and_list_round_trip(self, path: Path) -> None:
        with contextlib.closing(connect(path)) as conn:
            first = save_run(conn, record())
            second = save_run(
                conn, record(cost_usd=None, fatal_reason="authentication_failed", failed=20)
            )

            runs = list_runs(conn)

        assert (first, second) == (1, 2)
        assert runs[0] == record()
        assert runs[1].cost_usd is None
        assert runs[1].fatal_reason == "authentication_failed"

    def test_the_new_columns_round_trip(self, path: Path) -> None:
        with contextlib.closing(connect(path)) as conn:
            save_run(
                conn,
                record(
                    not_analyzed_limit=7,
                    outcome="stopped_transient",
                    failures={"rate_limited": 4, "skipped_after_transient": 6},
                    api_calls=9,
                ),
            )
            (run,) = list_runs(conn)

        assert run.not_analyzed_limit == 7
        assert run.outcome == "stopped_transient"
        assert run.failures == {"rate_limited": 4, "skipped_after_transient": 6}
        assert run.api_calls == 9

    def test_failures_are_stored_as_a_json_object_of_counts(self, path: Path) -> None:
        with contextlib.closing(connect(path)) as conn:
            save_run(conn, record(failures={"b": 2, "a": 1}))
            raw = conn.execute("SELECT failures FROM runs").fetchone()[0]

        assert json.loads(raw) == {"a": 1, "b": 2}

    def test_defaults_for_the_new_fields(self) -> None:
        values = {k: v for k, v in record().__dict__.items()}
        for name in ("not_analyzed_limit", "outcome", "failures", "api_calls"):
            del values[name]

        run = RunRecord(**values)

        assert (run.not_analyzed_limit, run.outcome, run.failures, run.api_calls) == (
            0,
            "completed",
            {},
            0,
        )

    @pytest.mark.parametrize(
        "outcome",
        [
            "completed",
            "fatal",
            "aborted",
            "interrupted",
            "stopped_transient",
            "call_budget_exceeded",
        ],
    )
    def test_every_outcome_is_accepted(self, path: Path, outcome: str) -> None:
        with contextlib.closing(connect(path)) as conn:
            save_run(conn, record(outcome=outcome))

            assert list_runs(conn)[0].outcome == outcome

    def test_an_unknown_outcome_is_rejected_by_the_database(self, path: Path) -> None:
        with contextlib.closing(connect(path)) as conn, pytest.raises(StorageError):
            save_run(conn, record(outcome="exploded"))

    def test_the_new_columns_are_not_null_with_defaults_where_agreed(self, path: Path) -> None:
        with contextlib.closing(connect(path)) as conn:
            info = {r[1]: r for r in conn.execute("PRAGMA table_info(runs)")}

        assert info["not_analyzed_limit"][3] == 1 and info["not_analyzed_limit"][4] == "0"
        assert info["api_calls"][3] == 1 and info["api_calls"][4] == "0"
        assert info["outcome"][3] == 1
        assert info["failures"][3] == 1

    def test_cost_keeps_exact_decimal_text(self, path: Path) -> None:
        with contextlib.closing(connect(path)) as conn:
            save_run(conn, record(cost_usd=Decimal("0.00855")))
            stored = conn.execute("SELECT cost_usd, typeof(cost_usd) FROM runs").fetchone()

        assert stored == ("0.00855", "text")

    def test_timestamps_are_stored_as_utc_iso(self, path: Path) -> None:
        local = dt.timezone(dt.timedelta(hours=3))
        with contextlib.closing(connect(path)) as conn:
            save_run(conn, record(started_at=START.astimezone(local)))
            started = conn.execute("SELECT started_at FROM runs").fetchone()[0]

        assert started == "2026-11-22T10:00:00+00:00"

    def test_naive_timestamps_are_rejected(self, path: Path) -> None:
        with contextlib.closing(connect(path)) as conn, pytest.raises(ValueError, match="timezone"):
            save_run(conn, record(started_at=dt.datetime(2026, 11, 22, 10, 0)))

    def test_sql_characters_in_values_are_data(self, path: Path) -> None:
        nasty = "x'); DROP TABLE runs; --"
        with contextlib.closing(connect(path)) as conn:
            save_run(conn, record(model=nasty, fatal_reason=nasty))

            runs = list_runs(conn)

        assert runs[0].model == nasty
        assert runs[0].fatal_reason == nasty

    def test_write_error_is_a_clean_storage_error(self, path: Path) -> None:
        with contextlib.closing(connect(path)) as conn:
            conn.execute("DROP TABLE runs")

            with pytest.raises(StorageError, match="Cannot save the run"):
                save_run(conn, record())

    def test_read_error_is_a_clean_storage_error(self, path: Path) -> None:
        with contextlib.closing(connect(path)) as conn:
            conn.execute("DROP TABLE runs")

            with pytest.raises(StorageError, match="Cannot read runs"):
                list_runs(conn)

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("cost_usd", "not-a-number"),
            ("started_at", "yesterday"),
            ("failures", "not json"),
            ("failures", "[1]"),
        ],
    )
    def test_a_corrupt_stored_row_is_a_clean_error(
        self, path: Path, column: str, value: str
    ) -> None:
        with contextlib.closing(connect(path)) as conn:
            save_run(conn, record())
            conn.execute(f"UPDATE runs SET {column} = ?", (value,))  # noqa: S608 (test column)

            with pytest.raises(StorageError, match="stored data is invalid"):
                list_runs(conn)

    def test_empty_table_lists_nothing(self, path: Path) -> None:
        with contextlib.closing(connect(path)) as conn:
            assert list_runs(conn) == []


class TestUpgradeToV4:
    def test_v3_database_gets_runs_and_keeps_its_data(self, path: Path) -> None:
        with contextlib.closing(sqlite3.connect(path)) as raw:
            raw.executescript(SCHEMA_V3)
            raw.execute(
                "INSERT INTO reviews (dedup_key, review_date, text, text_hash, imported_at)"
                " VALUES ('k', '2026-11-10', 'Old review', 'h', '2026-11-11T12:00:00+00:00')"
            )
            raw.commit()
        assert "runs" not in table_names(path)

        with contextlib.closing(connect(path)) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == storage.SCHEMA_VERSION
            assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1
            save_run(conn, record())

        assert "runs" in table_names(path)

    def test_v2_database_is_upgraded_straight_to_current(self, path: Path) -> None:
        make_v2_database(path)

        with contextlib.closing(connect(path)) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == storage.SCHEMA_VERSION
            assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1
            assert list_runs(conn) == []

        assert {"reviews", "analyses", "runs"} <= table_names(path)

    def test_min_upgradable_version_stays_2(self) -> None:
        assert storage.MIN_UPGRADABLE_VERSION == 2
        assert storage.SCHEMA_VERSION == 5
