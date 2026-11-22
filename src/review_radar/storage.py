"""SQLite storage for reviews and their analyses.

Privacy rule: `StorageError` messages never contain review text. Every statement uses `?`
placeholders; SQL is never built from data.

Schema versions live in `PRAGMA user_version`. Policy: there is ONE idempotent schema script
(`_SCHEMA`: only `CREATE TABLE/INDEX IF NOT EXISTS` and the two PRAGMAs, wrapped in
`BEGIN; ... COMMIT;`). A version bump may only ADD tables or indexes. On connect:

- version 0 (empty database, application_id 0, no tables): run the script;
- `MIN_UPGRADABLE_VERSION <= version < SCHEMA_VERSION` with our application_id: run the same
  script (an atomic upgrade);
- `version == SCHEMA_VERSION`: nothing to do;
- below `MIN_UPGRADABLE_VERSION`: rejected with a hint to delete the file and ingest again;
- above `SCHEMA_VERSION`: rejected.

A future change that is not additive (changing or dropping a column, a key format) must raise
`MIN_UPGRADABLE_VERSION`, so older databases are rejected instead of silently misread.

Re-importing is idempotent: every review has a `dedup_key` (its external id scoped by source,
or a hash of date + normalized text + source) and the first stored copy of a key is never
modified.

Schema versions: 1 (stage 3) keyed reviews by external id alone; 2 scopes the id by source;
3 adds the `analyses` cache table. A version 1 database cannot be upgraded in place and must be
recreated from the CSV; version 2 is upgraded.
"""

import datetime as dt
import json
import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from review_radar.hashing import cache_key, text_hash
from review_radar.models import Analysis, Review, dedup_key

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3
MIN_UPGRADABLE_VERSION = 2
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
CREATE INDEX IF NOT EXISTS idx_reviews_text_hash ON reviews (text_hash);
CREATE TABLE IF NOT EXISTS analyses (
    cache_key TEXT PRIMARY KEY,
    text_hash TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    sentiment TEXT NOT NULL CHECK (sentiment IN ('positive', 'neutral', 'negative')),
    topics TEXT NOT NULL,
    confidence REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    analyzed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_lookup
    ON analyses (text_hash, provider, model, prompt_version);
PRAGMA application_id = 1383486052;
PRAGMA user_version = 3;
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

_INSERT_ANALYSIS = """
INSERT INTO analyses
    (cache_key, text_hash, provider, model, prompt_version, sentiment, topics, confidence,
     analyzed_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(cache_key) DO NOTHING
"""

_PENDING = """
SELECT r.text_hash, r.text, r.rating, g.review_count
FROM reviews r
JOIN (
    SELECT MIN(id) AS first_id, COUNT(*) AS review_count FROM reviews GROUP BY text_hash
) g ON r.id = g.first_id
WHERE NOT EXISTS (
    SELECT 1 FROM analyses a
    WHERE a.text_hash = r.text_hash AND a.provider = ? AND a.model = ? AND a.prompt_version = ?
)
ORDER BY r.id
"""

_COUNT_ANALYZED = """
SELECT COUNT(*) FROM reviews r
WHERE EXISTS (
    SELECT 1 FROM analyses a
    WHERE a.text_hash = r.text_hash AND a.provider = ? AND a.model = ? AND a.prompt_version = ?
)
"""

_SELECT_ANALYSES = """
SELECT text_hash, sentiment, topics, confidence FROM analyses
WHERE provider = ? AND model = ? AND prompt_version = ?
"""


class StorageError(Exception):
    """The database cannot be opened, validated or written. Never carries review text."""


@dataclass(frozen=True)
class SaveResult:
    """How many reviews of a batch were stored for the first time and how many already were."""

    new: int
    existing: int


@dataclass(frozen=True)
class PendingText:
    """One distinct text to analyze.

    `text` and `rating` come from the earliest stored review with that text hash;
    `review_count` is how many stored reviews share the hash.
    """

    text_hash: str
    text: str
    rating: int | None
    review_count: int


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
    """Initialize an empty database, upgrade an older one, accept a current one, reject the rest."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    application_id = conn.execute("PRAGMA application_id").fetchone()[0]
    if version == 0:
        has_tables = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table'").fetchone()
        if has_tables is not None or application_id != 0:
            raise StorageError(f"{path.name} is not a Review Radar database.")
        conn.executescript(_SCHEMA)  # one transaction: all or nothing
    elif not MIN_UPGRADABLE_VERSION <= version <= SCHEMA_VERSION:
        hint = " Delete it and run ingest again." if version < MIN_UPGRADABLE_VERSION else ""
        raise StorageError(
            f"{path.name} has schema version {version}, expected {SCHEMA_VERSION}.{hint}"
        )
    elif application_id != APPLICATION_ID:
        raise StorageError(f"{path.name} is not a Review Radar database.")
    elif version < SCHEMA_VERSION:
        logger.info("Upgrading %s from schema version %d to %d", path.name, version, SCHEMA_VERSION)
        conn.executescript(_SCHEMA)  # additive and atomic


def utc_stamp(now: dt.datetime | None) -> str:
    """Return `now` (default: the current time) as a UTC ISO 8601 string.

    Raises ValueError for a naive datetime, which would be stored without a timezone.
    """
    if now is None:
        return dt.datetime.now(dt.UTC).isoformat()
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(dt.UTC).isoformat()


def save_reviews(
    conn: sqlite3.Connection, reviews: Iterable[Review], *, now: dt.datetime | None = None
) -> SaveResult:
    """Store `reviews` in one transaction; rows whose dedup key exists are left untouched."""
    stamp = utc_stamp(now)
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


def _unreadable(what: str) -> StorageError:
    return StorageError(f"Cannot read {what}: the stored data is invalid.")


def count_reviews(conn: sqlite3.Connection) -> int:
    """Return the number of stored reviews."""
    try:
        return int(conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0])
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot read reviews: {_reason(exc)}.") from None


def list_reviews(conn: sqlite3.Connection) -> list[Review]:
    """Return all stored reviews ordered by date, then insertion order."""
    try:
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
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot read reviews: {_reason(exc)}.") from None
    except ValueError:  # includes pydantic.ValidationError, whose message can quote review text
        raise _unreadable("reviews") from None


def pending_texts(
    conn: sqlite3.Connection, *, provider: str, model: str, prompt_version: str
) -> list[PendingText]:
    """Return one entry per distinct text hash that has no analysis for this key.

    Ordered by the id of the earliest review with that hash, so the order is deterministic.
    """
    try:
        rows = conn.execute(_PENDING, (provider, model, prompt_version)).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot read pending texts: {_reason(exc)}.") from None
    return [PendingText(*row) for row in rows]


def count_analyzed_reviews(
    conn: sqlite3.Connection, *, provider: str, model: str, prompt_version: str
) -> int:
    """Return how many reviews already have an analysis for this key."""
    try:
        return int(conn.execute(_COUNT_ANALYZED, (provider, model, prompt_version)).fetchone()[0])
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot read analyses: {_reason(exc)}.") from None


def save_analyses(
    conn: sqlite3.Connection,
    rows: Iterable[tuple[PendingText, Analysis]],
    *,
    provider: str,
    model: str,
    prompt_version: str,
    now: dt.datetime | None = None,
) -> int:
    """Store analyses in one transaction; existing cache keys are kept. Returns rows inserted."""
    stamp = utc_stamp(now)
    inserted = 0
    try:
        with conn:
            for pending, analysis in rows:
                key = cache_key(
                    pending.text, provider=provider, model=model, prompt_version=prompt_version
                )
                topics = json.dumps([str(t) for t in analysis.topics])
                params = (
                    key,
                    pending.text_hash,
                    provider,
                    model,
                    prompt_version,
                    str(analysis.sentiment),
                    topics,
                    analysis.confidence,
                    stamp,
                )
                inserted += conn.execute(_INSERT_ANALYSIS, params).rowcount
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot save analyses: {_reason(exc)}.") from None
    logger.info("Saved analyses: %d new", inserted)
    return inserted


def load_analyses(
    conn: sqlite3.Connection, *, provider: str, model: str, prompt_version: str
) -> dict[str, Analysis]:
    """Return the stored analyses for this key, keyed by text hash."""
    try:
        rows = conn.execute(_SELECT_ANALYSES, (provider, model, prompt_version)).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot read analyses: {_reason(exc)}.") from None
    try:
        return {
            hash_: Analysis(sentiment=sentiment, topics=json.loads(topics), confidence=confidence)
            for hash_, sentiment, topics, confidence in rows
        }
    except ValueError:  # includes pydantic.ValidationError
        raise _unreadable("analyses") from None
