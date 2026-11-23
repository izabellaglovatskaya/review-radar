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
3 adds the `analyses` cache table; 4 adds the `runs` table (one row per analyze run, with token
usage and estimated cost); 5 adds the `summaries` table (one row per Telegram send attempt). A
version 1 database cannot be upgraded in place and must be recreated from the CSV; versions 2
to 4 are upgraded.
"""

import datetime as dt
import json
import logging
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

from review_radar.hashing import cache_key, text_hash
from review_radar.models import Analysis, Review, dedup_key

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 5
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
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    reviews_total INTEGER NOT NULL,
    analyzed INTEGER NOT NULL,
    cached INTEGER NOT NULL,
    failed INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd TEXT,
    fatal_reason TEXT,
    not_analyzed_limit INTEGER NOT NULL DEFAULT 0,
    outcome TEXT NOT NULL CHECK (outcome IN (
        'completed', 'fatal', 'aborted', 'interrupted', 'stopped_transient',
        'call_budget_exceeded'
    )),
    failures TEXT NOT NULL,
    api_calls INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('sending', 'sent', 'failed', 'unknown')),
    forced INTEGER NOT NULL CHECK (forced IN (0, 1)),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    reviews_total INTEGER NOT NULL,
    reason TEXT,
    status_code INTEGER,
    message_id INTEGER,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_summaries_period ON summaries (period_start, period_end);
PRAGMA application_id = 1383486052;
PRAGMA user_version = 5;
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

_INSERT_RUN = """
INSERT INTO runs
    (started_at, finished_at, provider, model, prompt_version, reviews_total, analyzed, cached,
     failed, input_tokens, output_tokens, cost_usd, fatal_reason, not_analyzed_limit, outcome,
     failures, api_calls)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_RUNS = """
SELECT started_at, finished_at, provider, model, prompt_version, reviews_total, analyzed, cached,
       failed, input_tokens, output_tokens, cost_usd, fatal_reason, not_analyzed_limit, outcome,
       failures, api_calls
FROM runs ORDER BY id
"""

_SELECT_PERIOD_ANALYSES = """
SELECT a.sentiment, a.topics
FROM reviews r
LEFT JOIN analyses a
    ON a.text_hash = r.text_hash AND a.provider = ? AND a.model = ? AND a.prompt_version = ?
WHERE r.review_date BETWEEN ? AND ?
ORDER BY r.id
"""

_SELECT_SUMMARY_BLOCKER = """
SELECT status FROM summaries
WHERE period_start = ? AND period_end = ? AND status IN ('sending', 'sent', 'unknown')
ORDER BY CASE status WHEN 'sent' THEN 0 ELSE 1 END, id
LIMIT 1
"""

_INSERT_SUMMARY = """
INSERT INTO summaries
    (period_start, period_end, status, forced, provider, model, prompt_version, reviews_total,
     started_at)
VALUES (?, ?, 'sending', ?, ?, ?, ?, ?, ?)
"""

_FINISH_SUMMARY = """
UPDATE summaries
SET status = ?, reason = ?, status_code = ?, message_id = ?, finished_at = ?
WHERE id = ? AND status = 'sending'
"""

_SELECT_SUMMARIES = """
SELECT period_start, period_end, status, forced, provider, model, prompt_version, reviews_total,
       reason, status_code, message_id, started_at, finished_at
FROM summaries ORDER BY id
"""

_FINAL_SUMMARY_STATUSES = frozenset({"sent", "failed", "unknown"})


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


@dataclass(frozen=True)
class RunRecord:
    """One analyze run. Counts are in reviews; `cost_usd` is None when the model has no price.

    `fatal_reason` is set when the run was stopped by a fatal classifier error. `outcome` is one
    of completed, fatal, aborted, interrupted, stopped_transient, call_budget_exceeded.
    `failures` maps a reason code to a number of reviews (counts only, never text).
    """

    started_at: dt.datetime
    finished_at: dt.datetime
    provider: str
    model: str
    prompt_version: str
    reviews_total: int
    analyzed: int
    cached: int
    failed: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None
    fatal_reason: str | None
    not_analyzed_limit: int = 0
    outcome: str = "completed"
    failures: Mapping[str, int] = field(default_factory=dict)
    api_calls: int = 0


@dataclass(frozen=True)
class SummaryClaim:
    """The right to send one summary: a row id, or the status that blocks the send."""

    summary_id: int | None
    blocked_by: str | None


@dataclass(frozen=True)
class SummaryRecord:
    """One Telegram send attempt (aggregates are not stored, only the outcome)."""

    period_start: dt.date
    period_end: dt.date
    status: str
    forced: bool
    provider: str
    model: str
    prompt_version: str
    reviews_total: int
    reason: str | None
    status_code: int | None
    message_id: int | None
    started_at: dt.datetime
    finished_at: dt.datetime | None


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


def save_run(conn: sqlite3.Connection, record: RunRecord) -> int:
    """Store one run row in its own transaction and return its id.

    Raises ValueError for a naive timestamp.
    """
    params = (
        utc_stamp(record.started_at),
        utc_stamp(record.finished_at),
        record.provider,
        record.model,
        record.prompt_version,
        record.reviews_total,
        record.analyzed,
        record.cached,
        record.failed,
        record.input_tokens,
        record.output_tokens,
        None if record.cost_usd is None else str(record.cost_usd),
        record.fatal_reason,
        record.not_analyzed_limit,
        record.outcome,
        json.dumps(dict(record.failures), sort_keys=True),
        record.api_calls,
    )
    try:
        with conn:
            cursor = conn.execute(_INSERT_RUN, params)
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot save the run: {_reason(exc)}.") from None
    logger.info("Saved run %s", cursor.lastrowid)
    return int(cursor.lastrowid or 0)


def list_runs(conn: sqlite3.Connection) -> list[RunRecord]:
    """Return all recorded runs, oldest first."""
    try:
        rows = conn.execute(_SELECT_RUNS).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot read runs: {_reason(exc)}.") from None
    try:
        return [
            RunRecord(
                started_at=dt.datetime.fromisoformat(started),
                finished_at=dt.datetime.fromisoformat(finished),
                provider=provider,
                model=model,
                prompt_version=prompt_version,
                reviews_total=total,
                analyzed=analyzed,
                cached=cached,
                failed=failed,
                input_tokens=tokens_in,
                output_tokens=tokens_out,
                cost_usd=None if cost is None else Decimal(cost),
                fatal_reason=fatal,
                not_analyzed_limit=not_analyzed,
                outcome=outcome,
                failures=_failures(failures),
                api_calls=api_calls,
            )
            for (
                started,
                finished,
                provider,
                model,
                prompt_version,
                total,
                analyzed,
                cached,
                failed,
                tokens_in,
                tokens_out,
                cost,
                fatal,
                not_analyzed,
                outcome,
                failures,
                api_calls,
            ) in rows
        ]
    except (ValueError, TypeError, InvalidOperation):
        raise _unreadable("runs") from None


def load_period_analyses(
    conn: sqlite3.Connection,
    start: dt.date,
    end: dt.date,
    *,
    provider: str,
    model: str,
    prompt_version: str,
) -> list[Analysis | None]:
    """Return one entry per review dated in `[start, end]`; None when it has no analysis.

    Review text is deliberately not selected.
    """
    params = (provider, model, prompt_version, start.isoformat(), end.isoformat())
    try:
        rows = conn.execute(_SELECT_PERIOD_ANALYSES, params).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot read analyses: {_reason(exc)}.") from None
    try:
        return [
            None
            if sentiment is None
            else Analysis(sentiment=sentiment, topics=json.loads(topics), confidence=None)
            for sentiment, topics in rows
        ]
    except (ValueError, TypeError):  # includes pydantic.ValidationError
        raise _unreadable("analyses") from None


def summary_blocker(conn: sqlite3.Connection, start: dt.date, end: dt.date) -> str | None:
    """Return the status that blocks a plain send for this exact period, or None."""
    try:
        row = conn.execute(_SELECT_SUMMARY_BLOCKER, (start.isoformat(), end.isoformat())).fetchone()
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot read summaries: {_reason(exc)}.") from None
    return None if row is None else str(row[0])


def claim_summary(
    conn: sqlite3.Connection,
    start: dt.date,
    end: dt.date,
    *,
    provider: str,
    model: str,
    prompt_version: str,
    reviews_total: int,
    force: bool,
    now: dt.datetime | None = None,
) -> SummaryClaim:
    """Record the intent to send before sending, so a crash can never lead to a silent resend.

    Blocked when the period was sent, is being sent, or ended with an unknown delivery, unless
    `force` is set. A failed attempt does not block. Raises ValueError for a naive `now`.
    """
    stamp = utc_stamp(now)
    period = (start.isoformat(), end.isoformat())
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            if not force:
                blocker = conn.execute(_SELECT_SUMMARY_BLOCKER, period).fetchone()
                if blocker is not None:
                    return SummaryClaim(summary_id=None, blocked_by=str(blocker[0]))
            cursor = conn.execute(
                _INSERT_SUMMARY,
                (*period, int(force), provider, model, prompt_version, reviews_total, stamp),
            )
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot record the summary: {_reason(exc)}.") from None
    return SummaryClaim(summary_id=int(cursor.lastrowid or 0), blocked_by=None)


def finish_summary(
    conn: sqlite3.Connection,
    summary_id: int | None,
    status: str,
    *,
    reason: str | None = None,
    status_code: int | None = None,
    message_id: int | None = None,
    now: dt.datetime | None = None,
) -> None:
    """Move a claimed row from `sending` to its final status (exactly once)."""
    if status not in _FINAL_SUMMARY_STATUSES:
        raise StorageError("Cannot record the summary: invalid status.")
    params = (status, reason, status_code, message_id, utc_stamp(now), summary_id)
    try:
        with conn:
            updated = conn.execute(_FINISH_SUMMARY, params).rowcount
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot record the summary: {_reason(exc)}.") from None
    if updated != 1:
        raise StorageError("Cannot record the summary: no open send attempt.")


def list_summaries(conn: sqlite3.Connection) -> list[SummaryRecord]:
    """Return all recorded send attempts, oldest first."""
    try:
        rows = conn.execute(_SELECT_SUMMARIES).fetchall()
    except sqlite3.Error as exc:
        raise StorageError(f"Cannot read summaries: {_reason(exc)}.") from None
    try:
        return [
            SummaryRecord(
                period_start=dt.date.fromisoformat(start),
                period_end=dt.date.fromisoformat(end),
                status=status,
                forced=bool(forced),
                provider=provider,
                model=model,
                prompt_version=prompt_version,
                reviews_total=total,
                reason=reason,
                status_code=status_code,
                message_id=message_id,
                started_at=dt.datetime.fromisoformat(started),
                finished_at=None if finished is None else dt.datetime.fromisoformat(finished),
            )
            for (
                start,
                end,
                status,
                forced,
                provider,
                model,
                prompt_version,
                total,
                reason,
                status_code,
                message_id,
                started,
                finished,
            ) in rows
        ]
    except (ValueError, TypeError):
        raise _unreadable("summaries") from None


def _failures(raw: str) -> dict[str, int]:
    """Parse the stored failures JSON (a reason -> count object); ValueError if it is not."""
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("failures must be a JSON object")
    return {str(reason): int(count) for reason, count in parsed.items()}
