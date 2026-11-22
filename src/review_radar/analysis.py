"""Analysis service: classify the texts that have no cached analysis yet.

The cache is per normalized text (`text_hash`) and per (provider, model, prompt_version), so
reviews sharing a text share one analysis. Each batch is saved in its own transaction right
after it is classified, so an interrupted run keeps the finished batches.

`ClassifierError` fails only its batch. Any other exception stops the run: it is logged
(type name only) and re-raised as `AnalysisAborted`, which carries the counts so far; batches
that did not run are counted as failed with the reason `not_run`.

Privacy rule: logs and reports contain counts, reason codes and exception type names only,
never review text.
"""

import datetime as dt
import logging
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from review_radar.classifier import Classifier, ClassifierError, ClassifyItem
from review_radar.models import Analysis, Usage
from review_radar.storage import (
    PendingText,
    count_analyzed_reviews,
    count_reviews,
    pending_texts,
    save_analyses,
    utc_stamp,
)

logger = logging.getLogger(__name__)

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 100
MISSING_RESULT = "missing_result"
NOT_RUN = "not_run"


@dataclass(frozen=True)
class AnalysisReport:
    """Outcome of one run, counted in reviews: `reviews_total == analyzed + cached + failed`."""

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


class AnalysisAborted(Exception):
    """The run stopped on an unexpected exception; `report` holds the counts so far.

    The original exception is available as `__cause__`.
    """

    def __init__(self, report: AnalysisReport) -> None:
        super().__init__("Analysis stopped.")
        self.report = report


@dataclass
class _Totals:
    """Counts for one batch or for the whole run (mutable accumulator)."""

    analyzed: int = 0
    unexpected: int = 0
    fallback: int = 0
    unknown_topics: int = 0
    truncated_topics: int = 0
    failures: Counter[str] = field(default_factory=Counter)
    usages: list[Usage] = field(default_factory=list)

    def merge(self, other: "_Totals") -> None:
        self.analyzed += other.analyzed
        self.unexpected += other.unexpected
        self.fallback += other.fallback
        self.unknown_topics += other.unknown_topics
        self.truncated_topics += other.truncated_topics
        self.failures.update(other.failures)
        self.usages.extend(other.usages)


def analyze_pending(
    conn: sqlite3.Connection,
    classifier: Classifier,
    *,
    batch_size: int = 20,
    now: dt.datetime | None = None,
) -> AnalysisReport:
    """Classify every text without an analysis for this classifier and store the results.

    Raises ValueError for a bad `batch_size` or a naive `now`, and `AnalysisAborted` when an
    exception other than `ClassifierError` interrupts the run (finished batches stay saved).
    """
    if not MIN_BATCH_SIZE <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}")
    utc_stamp(now)  # reject a naive `now` before any work is done
    key = {
        "provider": classifier.provider,
        "model": classifier.model,
        "prompt_version": classifier.prompt_version,
    }
    total = count_reviews(conn)
    cached = count_analyzed_reviews(conn, **key)
    pending = pending_texts(conn, **key)
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    totals = _Totals()
    index = 0
    try:
        for index, batch in enumerate(batches):  # noqa: B007 (index is used after a failure)
            totals.merge(_run_batch(conn, classifier, batch, key, now))
    except Exception as exc:
        totals.failures[NOT_RUN] += sum(_review_count(b) for b in batches[index:])
        report = _build_report(total, cached, totals)
        logger.warning(
            "Analysis stopped by %s: analyzed=%d cached=%d failed=%d",
            type(exc).__name__,
            report.analyzed,
            report.cached,
            report.failed,
        )
        raise AnalysisAborted(report) from exc
    report = _build_report(total, cached, totals)
    _log(report)
    return report


def _review_count(batch: Sequence[PendingText]) -> int:
    return sum(p.review_count for p in batch)


def _run_batch(
    conn: sqlite3.Connection,
    classifier: Classifier,
    batch: Sequence[PendingText],
    key: Mapping[str, str],
    now: dt.datetime | None,
) -> _Totals:
    """Classify one batch and save what came back; return its counts (nothing on a raise)."""
    items = [
        ClassifyItem(item_id=f"r{n}", text=p.text, rating=p.rating)
        for n, p in enumerate(batch, start=1)
    ]
    outcome = _Totals()
    try:
        result = classifier.classify(items)
    except ClassifierError as exc:
        outcome.failures[exc.reason] += _review_count(batch)
        return outcome
    outcome.unexpected = len(set(result.analyses) - {item.item_id for item in items})
    saved: list[tuple[PendingText, Analysis]] = []
    for item, pending in zip(items, batch, strict=True):
        analysis = result.analyses.get(item.item_id)
        if analysis is None:
            outcome.failures[MISSING_RESULT] += pending.review_count
        else:
            saved.append((pending, analysis))
    save_analyses(conn, saved, **key, now=now)
    outcome.analyzed = _review_count([p for p, _ in saved])
    outcome.fallback = result.fallback
    outcome.unknown_topics = result.unknown_topics
    outcome.truncated_topics = result.truncated_topics
    if result.usage is not None:
        outcome.usages.append(result.usage)
    return outcome


def _build_report(total: int, cached: int, totals: _Totals) -> AnalysisReport:
    usages = totals.usages
    return AnalysisReport(
        reviews_total=total,
        analyzed=totals.analyzed,
        cached=cached,
        failed=sum(totals.failures.values()),
        failures=dict(totals.failures),
        usage=sum(usages[1:], usages[0]) if usages else None,
        unexpected_results=totals.unexpected,
        fallback=totals.fallback,
        unknown_topics=totals.unknown_topics,
        truncated_topics=totals.truncated_topics,
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
