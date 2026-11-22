"""Analysis service: classify the texts that have no cached analysis yet.

The cache is per normalized text (`text_hash`) and per (provider, model, prompt_version), so
reviews sharing a text share one analysis. Each batch is saved in its own transaction right
after it is classified, so an interrupted run keeps the finished batches.

How a run can end (`AnalysisReport.stop_reason`, `outcome`):

- `ClassifierError` fails only its batch (its message, text-free by contract, is logged with the
  batch number); a classifier can also fail single items (`ClassifyResult.failures`).
- Circuit breaker: after `MAX_CONSECUTIVE_TRANSIENT` batches in a row that failed only with
  transient reasons (rate limit, server error, network), the run stops (`stopped_transient`)
  and the remaining batches are counted as `skipped_after_transient`.
- Call budget (`max_calls`, for classifiers that have `set_call_budget`): when it is used up the
  run stops (`call_budget_exceeded`); unclassified reviews fail with `call_budget_exceeded`.
- `FatalClassifierError` (bad key, no credit, spend cap) stops the run at once (`fatal`): the
  valid analyses it carries are saved, the rest of that batch fails with the fatal reason, the
  batches after it with `skipped_after_fatal`; the report carries `fatal_reason` / `fatal_message`.
- Any other exception stops the run (`aborted`): it is logged (type name only) and re-raised as
  `AnalysisAborted`, which carries the counts so far.
- Ctrl+C (`KeyboardInterrupt`) stops the run (`interrupted`) and is re-raised as
  `AnalysisInterrupted`, a `KeyboardInterrupt` subclass (so unaware callers still see Ctrl+C)
  that carries the report.

For the last two, and for fatal errors, whatever the classifier had already produced and paid
for (attached to the exception, see `classifier.attach_partial`) is saved and its usage counted.
A usage is added to the totals before the analyses are saved, so a failed save never hides the
spend; a failed save makes the batch `save_failed` and the later batches `not_run`.

`limit` caps how many pending texts are classified in one run; the rest are reported as
`not_analyzed_limit` (not failures) and picked up by a later run.

Privacy rule: logs and reports contain counts, reason codes and exception type names only,
never review text.
"""

import datetime as dt
import logging
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from review_radar.classifier import (
    TRANSIENT_REASONS,
    Classifier,
    ClassifierError,
    ClassifyItem,
    ClassifyResult,
    FatalClassifierError,
    get_partial,
    safe_reason,
)
from review_radar.models import Analysis, Usage
from review_radar.storage import (
    PendingText,
    StorageError,
    count_analyzed_reviews,
    count_reviews,
    pending_texts,
    save_analyses,
    utc_stamp,
)

logger = logging.getLogger(__name__)

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 100
MAX_CONSECUTIVE_TRANSIENT = 2
AUTO: Literal["auto"] = "auto"

MISSING_RESULT = "missing_result"
NOT_RUN = "not_run"
SAVE_FAILED = "save_failed"
INTERRUPTED = "interrupted"
SKIPPED_AFTER_FATAL = "skipped_after_fatal"
SKIPPED_AFTER_TRANSIENT = "skipped_after_transient"
CALL_BUDGET_EXCEEDED = "call_budget_exceeded"

STOP_FATAL = "fatal"
STOP_ABORTED = "aborted"
STOP_INTERRUPTED = "interrupted"
STOP_TRANSIENT = "stopped_transient"
STOP_BUDGET = "call_budget_exceeded"


def default_max_calls(batches: int) -> int:
    """The default call budget: three calls per batch (room for splits) plus ten."""
    return 3 * batches + 10


@dataclass(frozen=True)
class AnalysisReport:
    """Outcome of one run, counted in reviews.

    `reviews_total == analyzed + cached + failed + not_analyzed_limit`.
    """

    reviews_total: int
    analyzed: int
    cached: int
    failed: int
    failures: Mapping[str, int] = field(default_factory=dict)  # reason -> number of reviews
    usage: Usage | None = None
    unexpected_results: int = 0  # ids the classifier returned that were not asked for
    fallback: int = 0  # answers produced by built-in rules, not by the real source
    unknown_topics: int = 0  # topics mapped to `other`
    truncated_topics: int = 0  # topics dropped to keep at most three per review
    not_analyzed_limit: int = 0  # reviews left for a later run by `limit` (not failures)
    fatal_reason: str | None = None  # set when a fatal classifier error stopped the run
    fatal_message: str | None = None  # its message (text-free by contract)
    stop_reason: str | None = None  # why the run ended early; None when it completed
    api_calls: int | None = None  # API calls made, for classifiers that count them
    max_calls: int | None = None  # the call budget that applied, if any

    @property
    def outcome(self) -> str:
        """`completed`, or the `stop_reason` (fatal, aborted, interrupted, ...)."""
        return self.stop_reason or "completed"


class AnalysisAborted(Exception):
    """The run stopped on an unexpected exception; `report` holds the counts so far.

    The original exception is available as `__cause__`.
    """

    def __init__(self, report: AnalysisReport) -> None:
        super().__init__("Analysis stopped.")
        self.report = report


class AnalysisInterrupted(KeyboardInterrupt):
    """The user pressed Ctrl+C; `report` holds the counts so far.

    A `KeyboardInterrupt` subclass on purpose: code that does not know about it still treats it
    as Ctrl+C. The original interrupt is available as `__cause__`.
    """

    def __init__(self, report: AnalysisReport) -> None:
        super().__init__("Analysis interrupted.")
        self.report = report


@dataclass
class _Totals:
    """Counts for the whole run (mutable accumulator)."""

    analyzed: int = 0
    unexpected: int = 0
    fallback: int = 0
    unknown_topics: int = 0
    truncated_topics: int = 0
    failures: Counter[str] = field(default_factory=Counter)
    usages: list[Usage] = field(default_factory=list)

    @property
    def done(self) -> int:
        """Reviews accounted for so far (analyzed or failed)."""
        return self.analyzed + sum(self.failures.values())

    def add_result(self, result: ClassifyResult, asked: set[str]) -> None:
        self.unexpected += len(set(result.analyses) - asked) + result.unexpected_results
        self.fallback += result.fallback
        self.unknown_topics += result.unknown_topics
        self.truncated_topics += result.truncated_topics
        if result.usage is not None:
            self.usages.append(result.usage)


@dataclass
class _Ending:
    """How the run ended, filled in by the batch loop."""

    stop_reason: str | None = None
    fatal: FatalClassifierError | None = None
    max_calls: int | None = None
    api_calls: int | None = None


def analyze_pending(
    conn: sqlite3.Connection,
    classifier: Classifier,
    *,
    batch_size: int = 20,
    limit: int | None = None,
    max_calls: int | Literal["auto"] | None = None,
    now: dt.datetime | None = None,
) -> AnalysisReport:
    """Classify every text without an analysis for this classifier and store the results.

    `max_calls` is a budget of API calls (`"auto"` is `default_max_calls(batches)`); it applies
    only to classifiers with `set_call_budget`, others ignore it.

    Raises ValueError for a bad `batch_size`, `limit` or `max_calls` or a naive `now`;
    `AnalysisAborted` when an exception other than `ClassifierError` interrupts the run, and
    `AnalysisInterrupted` on Ctrl+C (finished batches stay saved either way). A
    `FatalClassifierError` does not raise: it ends the run and is in the report.
    """
    if not MIN_BATCH_SIZE <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if max_calls is not None and max_calls != AUTO and max_calls < 1:
        raise ValueError("max_calls must be at least 1")
    utc_stamp(now)  # reject a naive `now` before any work is done
    key = {
        "provider": classifier.provider,
        "model": classifier.model,
        "prompt_version": classifier.prompt_version,
    }
    total = count_reviews(conn)
    cached = count_analyzed_reviews(conn, **key)
    pending = pending_texts(conn, **key)
    deferred = 0
    if limit is not None:
        deferred = _review_count(pending[limit:])
        pending = pending[:limit]
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    ending = _Ending(max_calls=_set_budget(classifier, max_calls, len(batches)))
    calls_before = getattr(classifier, "api_calls", None)
    totals = _Totals()
    grand_total = _review_count(pending)

    def report(stop_reason: str | None = None) -> AnalysisReport:
        ending.stop_reason = stop_reason or ending.stop_reason
        calls_after = getattr(classifier, "api_calls", None)
        if calls_before is not None and calls_after is not None:
            ending.api_calls = calls_after - calls_before
        return _build_report(total, cached, totals, deferred, ending)

    try:
        _run_batches(conn, classifier, batches, key, now, totals, ending, grand_total)
    except KeyboardInterrupt as exc:
        _skip(totals, grand_total, INTERRUPTED)
        interrupted = report(STOP_INTERRUPTED)
        logger.warning(
            "Analysis interrupted: analyzed=%d failed=%d", interrupted.analyzed, interrupted.failed
        )
        raise AnalysisInterrupted(interrupted) from exc
    except Exception as exc:
        _skip(totals, grand_total, NOT_RUN)
        aborted = report(STOP_ABORTED)
        logger.warning(
            "Analysis stopped by %s: analyzed=%d cached=%d failed=%d",
            type(exc).__name__,
            aborted.analyzed,
            aborted.cached,
            aborted.failed,
        )
        raise AnalysisAborted(aborted) from exc
    final = report()
    _log(final)
    return final


def _set_budget(
    classifier: Classifier, max_calls: int | Literal["auto"] | None, batches: int
) -> int | None:
    """Give the classifier its call budget (if it supports one); return the budget applied."""
    setter = getattr(classifier, "set_call_budget", None)
    if max_calls is None or setter is None:
        return None
    resolved = default_max_calls(batches) if max_calls == AUTO else max_calls
    setter(resolved)
    return resolved


def _run_batches(
    conn: sqlite3.Connection,
    classifier: Classifier,
    batches: Sequence[Sequence[PendingText]],
    key: Mapping[str, str],
    now: dt.datetime | None,
    totals: _Totals,
    ending: _Ending,
    grand_total: int,
) -> None:
    """Run the batches in order until done or a stop condition; fill `ending`."""
    consecutive = 0
    for index, batch in enumerate(batches):
        try:
            transient = _run_batch(conn, classifier, batch, key, now, totals, number=index + 1)
        except FatalClassifierError as exc:
            ending.fatal = exc
            ending.stop_reason = STOP_FATAL
            logger.warning("Batch %d failed fatally (%s): %s", index + 1, exc.reason, exc)
            _skip(totals, grand_total, SKIPPED_AFTER_FATAL)
            return
        consecutive = consecutive + 1 if transient else 0
        if getattr(classifier, "call_budget_exhausted", False):
            ending.stop_reason = STOP_BUDGET
            logger.warning("Call budget used up after batch %d; stopping", index + 1)
            _skip(totals, grand_total, CALL_BUDGET_EXCEEDED)
            return
        if consecutive >= MAX_CONSECUTIVE_TRANSIENT and index + 1 < len(batches):
            ending.stop_reason = STOP_TRANSIENT
            logger.warning("%d batches in a row failed transiently; stopping", consecutive)
            _skip(totals, grand_total, SKIPPED_AFTER_TRANSIENT)
            return


def _skip(totals: _Totals, grand_total: int, reason: str) -> None:
    """Count every review not accounted for yet as failed with `reason`."""
    remaining = grand_total - totals.done
    if remaining > 0:
        totals.failures[reason] += remaining


def _review_count(batch: Sequence[PendingText]) -> int:
    return sum(p.review_count for p in batch)


def _run_batch(
    conn: sqlite3.Connection,
    classifier: Classifier,
    batch: Sequence[PendingText],
    key: Mapping[str, str],
    now: dt.datetime | None,
    totals: _Totals,
    *,
    number: int,
) -> bool:
    """Classify one batch, save what came back and count it; True if it failed only transiently.

    What a failed `classify` call had already produced (attached to the exception) is saved too.
    A `FatalClassifierError` or any other exception propagates after that.
    """
    items = [
        ClassifyItem(item_id=f"r{n}", text=p.text, rating=p.rating)
        for n, p in enumerate(batch, start=1)
    ]
    try:
        result = classifier.classify(items)
    except FatalClassifierError as exc:
        partial = get_partial(exc) or ClassifyResult(analyses={}, usage=exc.usage)
        _absorb_quietly(conn, batch, items, partial, key, now, totals, leftover=exc.reason)
        raise
    except ClassifierError as exc:
        logger.warning("Batch %d failed (%s): %s", number, exc.reason, exc)
        totals.failures[exc.reason] += _review_count(batch)
        return exc.reason in TRANSIENT_REASONS
    except BaseException as exc:
        partial = get_partial(exc)
        if partial is not None:
            _absorb_quietly(conn, batch, items, partial, key, now, totals)
        raise
    return _absorb(conn, batch, items, result, key, now, totals, leftover=MISSING_RESULT)


def _absorb_quietly(
    conn: sqlite3.Connection,
    batch: Sequence[PendingText],
    items: Sequence[ClassifyItem],
    partial: ClassifyResult,
    key: Mapping[str, str],
    now: dt.datetime | None,
    totals: _Totals,
    *,
    leftover: str | None = None,
) -> None:
    """Save a partial result while another exception is propagating; never raise from here."""
    try:
        _absorb(conn, batch, items, partial, key, now, totals, leftover=leftover)
    except StorageError as exc:
        logger.warning("Could not save partial results: %s", exc)


def _absorb(
    conn: sqlite3.Connection,
    batch: Sequence[PendingText],
    items: Sequence[ClassifyItem],
    result: ClassifyResult,
    key: Mapping[str, str],
    now: dt.datetime | None,
    totals: _Totals,
    *,
    leftover: str | None,
) -> bool:
    """Count and save one classifier result; items it says nothing about fail with `leftover`.

    With `leftover=None` they are left uncounted (the caller counts them). The usage is added
    first, so a failed save cannot hide the spend. Returns True if the batch had no analysis
    and every failure was transient.
    """
    totals.add_result(result, {item.item_id for item in items})
    reasons = {item_id: safe_reason(reason) for item_id, reason in result.failures.items()}
    saved: list[tuple[PendingText, Analysis]] = []
    batch_failures: Counter[str] = Counter()
    for item, pending in zip(items, batch, strict=True):
        analysis = result.analyses.get(item.item_id)
        if analysis is not None:
            saved.append((pending, analysis))
            continue
        reason = reasons.get(item.item_id, leftover)
        if reason is not None:
            batch_failures[reason] += pending.review_count
    totals.failures.update(batch_failures)
    saved_reviews = _review_count([p for p, _ in saved])
    try:
        if saved:
            save_analyses(conn, saved, **key, now=now)
    except StorageError:
        totals.failures[SAVE_FAILED] += saved_reviews
        raise
    totals.analyzed += saved_reviews
    return (
        not saved
        and bool(batch_failures)
        and all(reason in TRANSIENT_REASONS for reason in batch_failures)
    )


def _build_report(
    total: int, cached: int, totals: _Totals, deferred: int, ending: _Ending
) -> AnalysisReport:
    usage: Usage | None = None
    for part in totals.usages:
        usage = part if usage is None else usage + part
    fatal = ending.fatal
    return AnalysisReport(
        reviews_total=total,
        analyzed=totals.analyzed,
        cached=cached,
        failed=sum(totals.failures.values()),
        failures=dict(totals.failures),
        usage=usage,
        unexpected_results=totals.unexpected,
        fallback=totals.fallback,
        unknown_topics=totals.unknown_topics,
        truncated_topics=totals.truncated_topics,
        not_analyzed_limit=deferred,
        fatal_reason=None if fatal is None else fatal.reason,
        fatal_message=None if fatal is None else str(fatal),
        stop_reason=ending.stop_reason,
        api_calls=ending.api_calls,
        max_calls=ending.max_calls,
    )


def _log(report: AnalysisReport) -> None:
    if report.unexpected_results:
        logger.warning(
            "Classifier returned %d unexpected result(s); ignored", report.unexpected_results
        )
    for reason, count in report.failures.items():
        logger.warning("Analysis failed for %d review(s): %s", count, reason)
    logger.info(
        "Analysis done: total=%d analyzed=%d cached=%d failed=%d",
        report.reviews_total,
        report.analyzed,
        report.cached,
        report.failed,
    )
