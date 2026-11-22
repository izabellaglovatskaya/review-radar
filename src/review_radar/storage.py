"""SQLite storage for reviews.

Privacy rule: `StorageError` messages never contain review text. Every statement uses `?`
placeholders; SQL is never built from data. The schema version lives in `PRAGMA
user_version`; there are no migrations yet, so a database with another version is rejected.

Re-importing is idempotent: every review has a `dedup_key` (its external id scoped by source,
or a hash of date + normalized text + source) and the first stored copy of a key is never
modified.

Schema versions: 1 (stage 3) keyed reviews by external id alone; 2 scopes the id by source.
A version 1 database cannot be upgraded in place and must be recreated from the CSV.
"""

import datetime as dt
import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from review_radar.hashing import text_hash
from review_radar.models import Review, dedup_key

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
APPLICATION_ID = 0x52765264  # ASCII "RvRd"

_SCHEMA = """
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

_INSERT = """
INSERT INTO reviews
    (dedup_key, external_id, review_date, rating, text, text_hash, source, imported_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(dedup_key) DO NOTHING
"""

_SELECT = """
SELECT external_id, review_date, rating, text, source FROM reviews ORDER BY review_date, id
"""


class StorageError(Exception):
    """The database cannot be opened, validated or written. Never carries review text."""


@dataclass(frozen=True)
class SaveResult:
    """How many reviews of a batch were stored for the first time and how many already were."""

    new: int
    existing: int


def connect(path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the database at `path` and make sure the schema is current.

    The caller owns the connection and must close it (e.g. with `contextlib.closing`).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
    except (sqlite3.Error, OSError) as exc:
        raise StorageError(f"Cannot open the database: {_reason(exc)}.") from None
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        _ensure_schema(conn, path)
    except sqlite3.DatabaseError as exc:
        conn.close()
        raise StorageError(f"Cannot use {path.name} as a database: {_reason(exc)}.") from None
    except BaseException:
        conn.close()
        raise
    return conn


def _reason(exc: Exception) -> str:
    return (getattr(exc, "strerror", None) or str(exc) or type(exc).__name__).rstrip(".")


def _ensure_schema(conn: sqlite3.Connection, path: Path) -> None:
    """Initialize an empty database; accept an initialized one; reject everything else."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    application_id = conn.execute("PRAGMA application_id").fetchone()[0]
    if version == 0:
        has_tables = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table'").fetchone()
        if has_tables is not None or application_id != 0:
            raise StorageError(f"{path.name} is not a Review Radar database.")
        conn.executescript(_SCHEMA)  # one transaction: all or nothing
    elif version != SCHEMA_VERSION:
        hint = " Delete it and run ingest again." if version < SCHEMA_VERSION else ""
        raise StorageError(
            f"{path.name} has schema version {version}, expected {SCHEMA_VERSION}.{hint}"
        )
    elif application_id != APPLICATION_ID:
        raise StorageError(f"{path.name} is not a Review Radar database.")


def save_reviews(
    conn: sqlite3.Connection, reviews: Iterable[Review], *, now: dt.datetime | None = None
) -> SaveResult:
    """Store `reviews` in one transaction; rows whose dedup key exists are left untouched."""
    stamp = (now or dt.datetime.now(dt.UTC)).isoformat()
    total = 0
    new = 0
    try:
        with conn:
            for review in reviews:
                total += 1
                cursor = conn.execute(_INSERT, _row(review, stamp))
                new += cursor.rowcount
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot save reviews: {_reason(exc)}.") from None
    logger.info("Saved reviews: %d new, %d already stored", new, total - new)
    return SaveResult(new=new, existing=total - new)


def _row(review: Review, imported_at: str) -> tuple[object, ...]:
    return (
        dedup_key(review),
        review.external_id,
        review.review_date.isoformat(),
        review.rating,
        review.text,
        text_hash(review.text),
        review.source,
        imported_at,
    )


def count_reviews(conn: sqlite3.Connection) -> int:
    """Return the number of stored reviews."""
    return int(conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0])


def list_reviews(conn: sqlite3.Connection) -> list[Review]:
    """Return all stored reviews ordered by date, then insertion order."""
    return [
        Review(
            external_id=external_id,
            review_date=dt.date.fromisoformat(review_date),
            rating=rating,
            text=text,
            source=source,
        )
        for external_id, review_date, rating, text, source in conn.execute(_SELECT)
    ]
