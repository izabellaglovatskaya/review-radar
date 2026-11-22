"""Read reviews from a CSV file, collecting per-row problems instead of failing.

Privacy rule: review text never appears in `RowError`, `IngestError` messages or logs.
Only line numbers, reasons and counts do. Ignored column names come from the header (never
from data rows) and are only reported once the header has been accepted as valid.

Malformed CSV records: `csv.reader(strict=True)` can keep reading after a "',' expected after
'\"'" error; it drops the rest of the offending physical line and resumes on the next one.
That is only safe when the dropped tail cannot have opened a quoted field that continues on
the following lines (the reader would then parse the continuation as a bogus record). We
therefore resume only when the offending physical line has an even number of quote
characters, and record a `malformed_row`. Otherwise, and for every other csv error
(unterminated quote at the end of the file, field above the size limit), the whole file is
rejected with `IngestError`.
"""

import csv
import datetime as dt
import io
import logging
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from review_radar.models import Review

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 2000
REQUIRED_COLUMNS = ("review_date", "text")
OPTIONAL_COLUMNS = ("review_id", "rating", "source")
_DETECTED_DELIMITERS = (",", ";")
_ALLOWED_DELIMITERS = (",", ";", "\t", "|")
_MAX_IGNORED_NAME_LENGTH = 40
_BLANK_COLUMN = "(blank)"
_RECOVERABLE_CSV_ERROR = "',' expected after"

_ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_RATING = re.compile(r"[1-5]")
_BARE_CR = re.compile(r"\r(?!\n)")
_LINE_BREAK = re.compile(r"\r\n|\n")


class IngestError(Exception):
    """The file as a whole cannot be read (as opposed to a single bad row)."""


class RowErrorReason(StrEnum):
    MISSING_TEXT = "missing_text"
    TEXT_TOO_LONG = "text_too_long"
    INVALID_DATE = "invalid_date"
    INVALID_RATING = "invalid_rating"
    WRONG_FIELD_COUNT = "wrong_field_count"
    MALFORMED_ROW = "malformed_row"
    DUPLICATE_REVIEW_ID = "duplicate_review_id"


@dataclass(frozen=True)
class RowError:
    """A skipped row: where it starts in the file and why. Never carries row content."""

    line: int
    reason: RowErrorReason


@dataclass(frozen=True)
class IngestResult:
    """Outcome of reading a file; `rows_read == len(reviews) + len(errors)`."""

    reviews: tuple[Review, ...]
    errors: tuple[RowError, ...]
    rows_read: int
    ignored_columns: tuple[str, ...] = ()

    def skipped_by_reason(self) -> dict[RowErrorReason, int]:
        counts = Counter(e.reason for e in self.errors)
        return {reason: counts[reason] for reason in RowErrorReason if reason in counts}


def read_reviews(
    path: Path, *, delimiter: str | None = None, encoding: str = "utf-8-sig"
) -> IngestResult:
    """Read reviews from `path`.

    Raises IngestError for file-level problems (unreadable, wrong encoding, empty, no data
    rows, missing required columns, bare CR line breaks, unrecoverable malformed CSV).
    Row-level problems are returned in `IngestResult.errors`. When a review_id occurs more
    than once, the first valid occurrence is imported and later ones are skipped.
    """
    if delimiter is not None and delimiter not in _ALLOWED_DELIMITERS:
        raise IngestError("The delimiter must be one of: comma, semicolon, tab, pipe.")

    text = _read_text(path, encoding)
    _reject_nul(text)
    _reject_bare_cr(text)

    header_line = next((ln for ln in text.splitlines() if ln.strip()), None)
    if header_line is None:
        raise IngestError("The file is empty.")
    sep = delimiter or _detect_delimiter(header_line)
    physical_lines = text.splitlines(keepends=True)

    reviews: list[Review] = []
    errors: list[RowError] = []
    seen_ids: set[str] = set()
    rows_read = 0
    columns: dict[str, int] | None = None
    ignored: tuple[str, ...] = ()
    header_width = 0

    reader = csv.reader(io.StringIO(text, newline=""), delimiter=sep, strict=True)
    last_line = 0
    while True:
        start_line = last_line + 1
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error as exc:
            last_line = reader.line_num
            if columns is None or not _can_resume(exc, physical_lines, last_line):
                raise IngestError(f"Malformed CSV at line {start_line}: {exc}.") from None
            rows_read += 1
            errors.append(RowError(start_line, RowErrorReason.MALFORMED_ROW))
            continue
        last_line = reader.line_num

        if not row:
            continue  # physically empty line
        if columns is None and len(row) == 1 and not row[0].strip():
            continue  # whitespace-only line before the header

        if columns is None:
            columns, ignored = _parse_header(row, start_line)
            header_width = len(row)
            continue

        rows_read += 1
        if len(row) != header_width:
            errors.append(RowError(start_line, RowErrorReason.WRONG_FIELD_COUNT))
            continue
        outcome = _parse_row(row, columns)
        if isinstance(outcome, RowErrorReason):
            errors.append(RowError(start_line, outcome))
        elif outcome.external_id is not None and outcome.external_id in seen_ids:
            errors.append(RowError(start_line, RowErrorReason.DUPLICATE_REVIEW_ID))
        else:
            if outcome.external_id is not None:
                seen_ids.add(outcome.external_id)
            reviews.append(outcome)

    if rows_read == 0:
        raise IngestError("The file has no data rows.")

    result = IngestResult(tuple(reviews), tuple(errors), rows_read, ignored)
    logger.info(
        "Ingest finished: %d rows read, %d imported, %d skipped",
        result.rows_read,
        len(result.reviews),
        len(result.errors),
    )
    return result


def _can_resume(exc: csv.Error, physical_lines: list[str], line_num: int) -> bool:
    """True if reading can safely continue after `exc` (see the module docstring)."""
    if not str(exc).startswith(_RECOVERABLE_CSV_ERROR):
        return False
    return 0 < line_num <= len(physical_lines) and physical_lines[line_num - 1].count('"') % 2 == 0


def _read_text(path: Path, encoding: str) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise IngestError(f"Cannot read the file: {exc.strerror or type(exc).__name__}.") from None
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError:
        raise IngestError(
            f"The file is not valid {encoding}. Pass another encoding, e.g. cp1251."
        ) from None
    except LookupError:
        raise IngestError(f"Unknown encoding {encoding!r}.") from None


def _reject_nul(text: str) -> None:
    index = text.find("\x00")
    if index != -1:
        line = text.count("\n", 0, index) + 1
        raise IngestError(f"The file contains a NUL byte at line {line}; it is not a text CSV.")


def _reject_bare_cr(text: str) -> None:
    match = _BARE_CR.search(text)
    if match:
        line = len(_LINE_BREAK.findall(text, 0, match.start())) + 1
        raise IngestError(
            f"The file contains bare CR line breaks (first at line {line}); "
            "save it with LF or CRLF line endings."
        )


def _detect_delimiter(header_line: str) -> str:
    """Pick the more frequent of `,` and `;` in the header; ties go to `,`."""
    return max(_DETECTED_DELIMITERS, key=lambda d: (header_line.count(d), d == ","))


def _parse_header(row: list[str], line: int) -> tuple[dict[str, int], tuple[str, ...]]:
    names = [cell.strip().casefold() for cell in row]
    duplicates = sorted({n for n in names if n and names.count(n) > 1})
    if duplicates:
        raise IngestError(f"Duplicate column(s) in the header at line {line}: {duplicates}.")
    missing = [c for c in REQUIRED_COLUMNS if c not in names]
    if missing:
        raise IngestError(
            f"Missing required column(s) in the header at line {line}: {', '.join(missing)}."
        )
    known = {*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS}
    ignored = tuple(
        _display_name(cell) for cell, name in zip(row, names, strict=True) if name not in known
    )
    return {name: i for i, name in enumerate(names) if name}, ignored


def _display_name(cell: str) -> str:
    name = " ".join(cell.split())
    return name[:_MAX_IGNORED_NAME_LENGTH] if name else _BLANK_COLUMN


def _parse_row(row: list[str], columns: dict[str, int]) -> Review | RowErrorReason:
    def cell(name: str) -> str:
        index = columns.get(name)
        return row[index].strip() if index is not None else ""

    text = cell("text")
    if not text:
        return RowErrorReason.MISSING_TEXT
    if len(text) > MAX_TEXT_LENGTH:
        return RowErrorReason.TEXT_TOO_LONG
    date = _parse_date(cell("review_date"))
    if date is None:
        return RowErrorReason.INVALID_DATE
    rating_raw = cell("rating")
    if rating_raw and not _RATING.fullmatch(rating_raw):
        return RowErrorReason.INVALID_RATING

    return Review(
        external_id=cell("review_id"),
        review_date=date,
        rating=int(rating_raw) if rating_raw else None,
        text=text,
        source=cell("source"),
    )


def _parse_date(value: str) -> dt.date | None:
    """Parse strict ISO YYYY-MM-DD (date.fromisoformat alone also accepts other forms)."""
    if not _ISO_DATE.fullmatch(value):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None
