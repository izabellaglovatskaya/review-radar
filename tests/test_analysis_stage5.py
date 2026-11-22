"""Analysis service: per-item failures, fatal stop, batch-index warnings and `limit`."""

import contextlib
import datetime as dt
import logging
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from review_radar.analysis import AnalysisReport, analyze_pending
from review_radar.classifier import (
    ClassifierError,
    ClassifyItem,
    ClassifyResult,
    FatalClassifierError,
)
from review_radar.models import Analysis, Review, Sentiment, Usage
from review_radar.storage import connect, load_analyses, save_reviews

FAKE_TEXT = "ZEBRA-QUOKKA-9917 distinctive fake review"
POSITIVE = Analysis(sentiment=Sentiment.POSITIVE, topics=(), confidence=None)
KEY = {"provider": "spy", "model": "spy-1", "prompt_version": "p1"}


class ScriptedClassifier:
    """Per call: a ClassifyResult-building rule, or an exception to raise."""

    provider = "spy"
    model = "spy-1"
    prompt_version = "p1"

    def __init__(
        self,
        *,
        failures_by_call: dict[int, dict[str, str]] | None = None,
        raise_on: dict[int, Exception] | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.calls: list[list[ClassifyItem]] = []
        self._failures_by_call = failures_by_call or {}
        self._raise_on = raise_on or {}
        self._usage = usage

    def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
        self.calls.append(list(items))
        number = len(self.calls)
        if number in self._raise_on:
            raise self._raise_on[number]
        failures = self._failures_by_call.get(number, {})
        analyses = {i.item_id: POSITIVE for i in items if i.item_id not in failures}
        return ClassifyResult(analyses=analyses, usage=self._usage, failures=failures)


def review(text: str, day: int = 10) -> Review:
    return Review.model_validate(
        {
            "external_id": None,
            "review_date": dt.date(2026, 11, day),
            "rating": 4,
            "text": text,
            "source": "web",
        }
    )


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    with contextlib.closing(connect(tmp_path / "radar.db")) as connection:
        yield connection


def seed(conn: sqlite3.Connection, n: int) -> None:
    save_reviews(conn, [review(f"review number {i}") for i in range(n)])


class TestPerItemFailures:
    def test_failed_items_are_counted_by_reason_and_not_saved(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 4)
        spy = ScriptedClassifier(failures_by_call={1: {"r2": "refusal", "r4": "invalid_output"}})

        report = analyze_pending(conn, spy)

        assert (report.analyzed, report.failed) == (2, 2)
        assert report.failures == {"refusal": 1, "invalid_output": 1}
        assert len(load_analyses(conn, **KEY)) == 2
        assert report.reviews_total == report.analyzed + report.cached + report.failed

    def test_failures_count_every_review_sharing_the_text(self, conn: sqlite3.Connection) -> None:
        save_reviews(conn, [review("dup", 1), review("dup", 2), review("dup", 3)])
        spy = ScriptedClassifier(failures_by_call={1: {"r1": "truncated_output"}})

        report = analyze_pending(conn, spy)

        assert report.failures == {"truncated_output": 3}
        assert report.failed == 3

    def test_an_item_without_analysis_or_failure_is_still_missing_result(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 2)

        class Dropping(ScriptedClassifier):
            def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
                return ClassifyResult(analyses={items[0].item_id: POSITIVE})

        report = analyze_pending(conn, Dropping())

        assert report.failures == {"missing_result": 1}

    def test_an_analysis_wins_over_a_failure_for_the_same_id(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 1)

        class Both(ScriptedClassifier):
            def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
                return ClassifyResult(analyses={"r1": POSITIVE}, failures={"r1": "invalid_output"})

        report = analyze_pending(conn, Both())

        assert (report.analyzed, report.failed) == (1, 0)

    def test_failure_reasons_from_a_classifier_are_sanitized(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 1)
        spy = ScriptedClassifier(failures_by_call={1: {"r1": f"Bad reason {FAKE_TEXT}"}})

        report = analyze_pending(conn, spy)

        assert report.failures == {"unknown_error": 1}

    def test_failures_for_ids_that_were_not_asked_for_are_ignored(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 1)

        class Extra(ScriptedClassifier):
            def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
                return ClassifyResult(analyses={"r1": POSITIVE}, failures={"r77": "invalid_output"})

        report = analyze_pending(conn, Extra())

        assert report.failed == 0


class TestWarningWithBatchIndex:
    def test_classifier_error_message_is_logged_with_the_batch_index(
        self, conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        seed(conn, 4)
        spy = ScriptedClassifier(
            raise_on={2: ClassifierError("rate_limited", "The API rate limit was hit.")}
        )

        with caplog.at_level(logging.WARNING):
            analyze_pending(conn, spy, batch_size=2)

        lines = [r.getMessage() for r in caplog.records if "batch" in r.getMessage().lower()]
        assert len(lines) == 1
        assert "2" in lines[0]
        assert "rate_limited" in lines[0]
        assert "The API rate limit was hit." in lines[0]

    def test_the_first_batch_is_numbered_one(
        self, conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        seed(conn, 2)
        spy = ScriptedClassifier(raise_on={1: ClassifierError("timeout", "Timed out.")})

        with caplog.at_level(logging.WARNING):
            analyze_pending(conn, spy)

        assert any("batch 1 " in r.getMessage().lower() for r in caplog.records)


class TestFatalStop:
    def fatal(self) -> FatalClassifierError:
        return FatalClassifierError("authentication_failed", "The API key was rejected.")

    def test_stops_at_once_and_counts_the_rest_as_skipped(self, conn: sqlite3.Connection) -> None:
        seed(conn, 7)
        spy = ScriptedClassifier(raise_on={2: self.fatal()})

        report = analyze_pending(conn, spy, batch_size=2)

        assert len(spy.calls) == 2
        assert (report.reviews_total, report.analyzed, report.cached, report.failed) == (7, 2, 0, 5)
        assert report.failures == {"authentication_failed": 2, "skipped_after_fatal": 3}
        assert report.reviews_total == report.analyzed + report.cached + report.failed

    def test_report_carries_the_fatal_reason_and_message(self, conn: sqlite3.Connection) -> None:
        seed(conn, 3)

        report = analyze_pending(conn, ScriptedClassifier(raise_on={1: self.fatal()}))

        assert report.fatal_reason == "authentication_failed"
        assert report.fatal_message == "The API key was rejected."

    def test_normal_report_has_no_fatal_fields(self, conn: sqlite3.Connection) -> None:
        seed(conn, 1)

        report = analyze_pending(conn, ScriptedClassifier())

        assert (report.fatal_reason, report.fatal_message) == (None, None)

    def test_finished_batches_stay_saved(self, conn: sqlite3.Connection) -> None:
        seed(conn, 6)

        analyze_pending(conn, ScriptedClassifier(raise_on={2: self.fatal()}), batch_size=2)

        assert len(load_analyses(conn, **KEY)) == 2

    def test_usage_spent_before_the_fatal_error_is_kept(self, conn: sqlite3.Connection) -> None:
        seed(conn, 4)
        spent = Usage(input_tokens=7, output_tokens=3)
        fatal = FatalClassifierError("billing_error", "No credit.", usage=spent)
        spy = ScriptedClassifier(raise_on={2: fatal}, usage=Usage(input_tokens=10, output_tokens=2))

        report = analyze_pending(conn, spy, batch_size=2)

        assert report.usage == Usage(input_tokens=17, output_tokens=5)

    def test_fatal_on_the_only_batch_fails_everything(self, conn: sqlite3.Connection) -> None:
        seed(conn, 2)

        report = analyze_pending(conn, ScriptedClassifier(raise_on={1: self.fatal()}))

        assert report.failures == {"authentication_failed": 2}
        assert (report.analyzed, report.failed) == (0, 2)

    def test_fatal_is_logged_as_a_warning_with_the_batch_index(
        self, conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        seed(conn, 4)

        with caplog.at_level(logging.WARNING):
            analyze_pending(conn, ScriptedClassifier(raise_on={2: self.fatal()}), batch_size=2)

        assert any("The API key was rejected." in r.getMessage() for r in caplog.records)
        assert FAKE_TEXT not in caplog.text

    def test_a_transient_error_does_not_stop_the_run(self, conn: sqlite3.Connection) -> None:
        seed(conn, 4)
        spy = ScriptedClassifier(raise_on={1: ClassifierError("rate_limited")})

        report = analyze_pending(conn, spy, batch_size=2)

        assert len(spy.calls) == 2
        assert report.fatal_reason is None


class TestLimit:
    def test_only_the_first_n_pending_texts_are_classified(self, conn: sqlite3.Connection) -> None:
        seed(conn, 5)
        spy = ScriptedClassifier()

        report = analyze_pending(conn, spy, limit=2)

        assert [i.text for c in spy.calls for i in c] == ["review number 0", "review number 1"]
        assert (report.analyzed, report.failed, report.not_analyzed_limit) == (2, 0, 3)

    def test_the_limit_applies_before_batching(self, conn: sqlite3.Connection) -> None:
        seed(conn, 10)
        spy = ScriptedClassifier()

        analyze_pending(conn, spy, batch_size=4, limit=6)

        assert [len(c) for c in spy.calls] == [4, 2]

    def test_deferred_reviews_are_counted_per_review_not_per_text(
        self, conn: sqlite3.Connection
    ) -> None:
        save_reviews(
            conn,
            [review("first"), review("second", 1), review("second", 2), review("second", 3)],
        )

        report = analyze_pending(conn, ScriptedClassifier(), limit=1)

        assert report.not_analyzed_limit == 3

    def test_deferred_reviews_are_not_failures(self, conn: sqlite3.Connection) -> None:
        seed(conn, 5)

        report = analyze_pending(conn, ScriptedClassifier(), limit=1)

        assert report.failed == 0
        assert report.failures == {}
        assert report.reviews_total == 5
        assert report.reviews_total == (
            report.analyzed + report.cached + report.failed + report.not_analyzed_limit
        )

    def test_a_limit_above_the_pending_count_defers_nothing(self, conn: sqlite3.Connection) -> None:
        seed(conn, 3)

        report = analyze_pending(conn, ScriptedClassifier(), limit=50)

        assert (report.analyzed, report.not_analyzed_limit) == (3, 0)

    def test_none_means_no_limit(self, conn: sqlite3.Connection) -> None:
        seed(conn, 3)

        report = analyze_pending(conn, ScriptedClassifier(), limit=None)

        assert report.analyzed == 3

    def test_a_later_run_continues_with_the_rest(self, conn: sqlite3.Connection) -> None:
        seed(conn, 5)
        analyze_pending(conn, ScriptedClassifier(), limit=2)
        spy = ScriptedClassifier()

        report = analyze_pending(conn, spy, limit=2)

        assert [i.text for c in spy.calls for i in c] == ["review number 2", "review number 3"]
        assert (report.analyzed, report.cached, report.not_analyzed_limit) == (2, 2, 1)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_limit_below_one_is_rejected_before_any_call(
        self, conn: sqlite3.Connection, bad: int
    ) -> None:
        seed(conn, 2)
        spy = ScriptedClassifier()

        with pytest.raises(ValueError, match="limit"):
            analyze_pending(conn, spy, limit=bad)

        assert spy.calls == []

    def test_a_default_report_has_nothing_deferred(self) -> None:
        report = AnalysisReport(reviews_total=1, analyzed=1, cached=0, failed=0)

        assert report.not_analyzed_limit == 0
        assert report.fatal_reason is None
