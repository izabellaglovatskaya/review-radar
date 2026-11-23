"""Schema v5: the `summaries` table that makes sending idempotent."""

import contextlib
import datetime as dt
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from review_radar import storage
from review_radar.storage import (
    SummaryClaim,
    claim_summary,
    connect,
    finish_summary,
    list_summaries,
    summary_blocker,
)

START = dt.date(2026, 11, 10)
END = dt.date(2026, 11, 10)
NOW = dt.datetime(2026, 11, 11, 8, 0, tzinfo=dt.UTC)

KEY = {"provider": "anthropic", "model": "claude-haiku-5-5", "prompt_version": "p1"}
SCHEMA_V4_MARKER = "PRAGMA user_version = 4"


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    with contextlib.closing(connect(tmp_path / "s.db")) as connection:
        yield connection


def claim(conn: sqlite3.Connection, *, force: bool = False, end: dt.date = END) -> SummaryClaim:
    return claim_summary(conn, START, end, **KEY, reviews_total=3, force=force, now=NOW)


class TestClaim:
    def test_first_claim_gets_a_row_in_sending_state(self, conn: sqlite3.Connection) -> None:
        result = claim(conn)

        assert result.summary_id is not None
        assert result.blocked_by is None
        rows = list_summaries(conn)
        assert [(r.status, r.forced, r.reviews_total) for r in rows] == [("sending", False, 3)]

    def test_a_sent_period_blocks_a_second_claim(self, conn: sqlite3.Connection) -> None:
        first = claim(conn)
        finish_summary(conn, first.summary_id, "sent", message_id=42, now=NOW)

        second = claim(conn)

        assert (second.summary_id, second.blocked_by) == (None, "sent")
        assert len(list_summaries(conn)) == 1

    @pytest.mark.parametrize("state", ["sending", "unknown"])
    def test_an_uncertain_delivery_blocks_a_claim(
        self, conn: sqlite3.Connection, state: str
    ) -> None:
        first = claim(conn)
        if state != "sending":
            finish_summary(conn, first.summary_id, state, reason="network_error", now=NOW)

        assert claim(conn).blocked_by == state

    def test_a_failed_send_can_be_claimed_again(self, conn: sqlite3.Connection) -> None:
        first = claim(conn)
        finish_summary(
            conn, first.summary_id, "failed", reason="unauthorized", status_code=401, now=NOW
        )

        second = claim(conn)

        assert second.summary_id not in (None, first.summary_id)
        assert [r.status for r in list_summaries(conn)] == ["failed", "sending"]

    def test_force_claims_past_a_sent_period(self, conn: sqlite3.Connection) -> None:
        first = claim(conn)
        finish_summary(conn, first.summary_id, "sent", message_id=1, now=NOW)

        forced = claim(conn, force=True)

        assert forced.summary_id is not None
        assert list_summaries(conn)[-1].forced is True

    def test_a_forced_send_does_not_block_nor_get_blocked(self, conn: sqlite3.Connection) -> None:
        one = claim(conn, force=True)
        finish_summary(conn, one.summary_id, "sent", message_id=1, now=NOW)
        two = claim(conn, force=True)

        assert two.summary_id is not None

    def test_a_forced_sent_period_blocks_a_plain_claim(self, conn: sqlite3.Connection) -> None:
        forced = claim(conn, force=True)
        finish_summary(conn, forced.summary_id, "sent", message_id=1, now=NOW)

        assert claim(conn).blocked_by == "sent"

    def test_a_different_period_is_independent(self, conn: sqlite3.Connection) -> None:
        first = claim(conn)
        finish_summary(conn, first.summary_id, "sent", message_id=1, now=NOW)

        assert claim(conn, end=END + dt.timedelta(days=1)).summary_id is not None

    def test_naive_time_is_rejected(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="timezone"):
            claim_summary(
                conn,
                START,
                END,
                **KEY,
                reviews_total=1,
                force=False,
                now=dt.datetime(2026, 11, 11),
            )


class TestBlocker:
    def test_a_free_period_has_no_blocker(self, conn: sqlite3.Connection) -> None:
        assert summary_blocker(conn, START, END) is None

    def test_a_failed_attempt_is_not_a_blocker(self, conn: sqlite3.Connection) -> None:
        first = claim(conn)
        finish_summary(conn, first.summary_id, "failed", reason="x", now=NOW)

        assert summary_blocker(conn, START, END) is None

    def test_sent_outranks_an_older_unknown(self, conn: sqlite3.Connection) -> None:
        first = claim(conn)
        finish_summary(conn, first.summary_id, "unknown", reason="x", now=NOW)
        forced = claim(conn, force=True)
        finish_summary(conn, forced.summary_id, "sent", message_id=1, now=NOW)

        assert summary_blocker(conn, START, END) == "sent"

    def test_the_row_stores_the_classifier_key(self, conn: sqlite3.Connection) -> None:
        claim(conn)

        row = list_summaries(conn)[0]

        assert (row.provider, row.model, row.prompt_version) == tuple(KEY.values())


class TestFinish:
    def test_records_the_outcome(self, conn: sqlite3.Connection) -> None:
        first = claim(conn)

        finish_summary(
            conn, first.summary_id, "failed", reason="rate_limited", status_code=429, now=NOW
        )

        row = list_summaries(conn)[0]
        assert (row.status, row.reason, row.status_code) == ("failed", "rate_limited", 429)
        assert row.finished_at == NOW

    def test_a_row_cannot_be_finished_twice(self, conn: sqlite3.Connection) -> None:
        first = claim(conn)
        finish_summary(conn, first.summary_id, "sent", message_id=1, now=NOW)

        with pytest.raises(storage.StorageError):
            finish_summary(conn, first.summary_id, "failed", reason="x", now=NOW)

    def test_an_invalid_status_is_a_clean_error(self, conn: sqlite3.Connection) -> None:
        first = claim(conn)

        with pytest.raises(storage.StorageError):
            finish_summary(conn, first.summary_id, "nonsense", now=NOW)


class TestSchemaV5:
    def test_version_constants(self) -> None:
        assert storage.SCHEMA_VERSION == 5
        assert storage.MIN_UPGRADABLE_VERSION == 2

    def test_v4_database_is_upgraded_and_keeps_its_data(self, tmp_path: Path) -> None:
        path = tmp_path / "v4.db"
        v4 = storage._SCHEMA.split("CREATE TABLE IF NOT EXISTS summaries")[0]
        script = f"{v4}PRAGMA application_id = 1383486052;\n{SCHEMA_V4_MARKER};\nCOMMIT;"
        with contextlib.closing(sqlite3.connect(path)) as raw:
            raw.executescript(script)
            raw.execute(
                "INSERT INTO reviews (dedup_key, review_date, text, text_hash, imported_at)"
                " VALUES ('k', '2026-11-10', 'Old', 'h', '2026-11-11T12:00:00+00:00')"
            )
            raw.commit()
            assert raw.execute("PRAGMA user_version").fetchone()[0] == 4

        with contextlib.closing(connect(path)) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 5
            assert conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1
            assert claim(conn).summary_id is not None
