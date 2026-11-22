import contextlib
import datetime as dt
import logging
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from review_radar import analysis as analysis_module
from review_radar.analysis import AnalysisAborted, AnalysisReport, analyze_pending
from review_radar.classifier import ClassifierError, ClassifyItem, ClassifyResult
from review_radar.hashing import text_hash
from review_radar.models import Analysis, Review, Sentiment, Usage, normalize_topics
from review_radar.storage import StorageError, connect, load_analyses, save_reviews

FAKE_TEXT = "ZEBRA-QUOKKA-9917 distinctive fake review"
POSITIVE = Analysis(sentiment=Sentiment.POSITIVE, topics=(), confidence=None)


class SpyClassifier:
    """Counts calls; can fail, drop ids or add unexpected ones on chosen (1-based) calls."""

    def __init__(
        self,
        *,
        model: str = "spy-1",
        prompt_version: str = "p1",
        fail_calls: dict[int, Exception] | None = None,
        drop_ids: frozenset[str] = frozenset(),
        extra_ids: tuple[str, ...] = (),
        usage: Usage | None = None,
        fallback: int = 0,
        unknown_topics: int = 0,
        truncated_topics: int = 0,
    ) -> None:
        self.provider = "spy"
        self.model = model
        self.prompt_version = prompt_version
        self.calls: list[list[ClassifyItem]] = []
        self._fail_calls = fail_calls or {}
        self._drop_ids = drop_ids
        self._extra_ids = extra_ids
        self._usage = usage
        self._counters = {
            "fallback": fallback,
            "unknown_topics": unknown_topics,
            "truncated_topics": truncated_topics,
        }

    def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
        self.calls.append(list(items))
        failure = self._fail_calls.get(len(self.calls))
        if failure is not None:
            raise failure
        analyses = {i.item_id: POSITIVE for i in items if i.item_id not in self._drop_ids}
        analyses.update({extra: POSITIVE for extra in self._extra_ids})
        return ClassifyResult(analyses=analyses, usage=self._usage, **self._counters)

    @property
    def called_texts(self) -> list[str]:
        return [i.text for call in self.calls for i in call]


def review(text: str, day: int = 10, rating: int | None = 4, source: str = "web") -> Review:
    return Review.model_validate(
        {
            "external_id": None,
            "review_date": dt.date(2026, 11, day),
            "rating": rating,
            "text": text,
            "source": source,
        }
    )


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    with contextlib.closing(connect(tmp_path / "radar.db")) as connection:
        yield connection


def seed(conn: sqlite3.Connection, n: int) -> list[str]:
    texts = [f"review number {i}" for i in range(n)]
    save_reviews(conn, [review(t) for t in texts])
    return texts


class TestFirstAndSecondRun:
    def test_first_run_analyzes_everything(self, conn: sqlite3.Connection) -> None:
        seed(conn, 5)
        spy = SpyClassifier()

        report = analyze_pending(conn, spy)

        assert report == AnalysisReport(
            reviews_total=5, analyzed=5, cached=0, failed=0, failures={}, usage=None
        )
        assert len(load_analyses(conn, provider="spy", model="spy-1", prompt_version="p1")) == 5

    def test_second_run_makes_zero_calls_and_reports_everything_cached(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 5)
        analyze_pending(conn, SpyClassifier())
        spy = SpyClassifier()

        report = analyze_pending(conn, spy)

        assert spy.calls == []
        assert (report.analyzed, report.cached, report.failed) == (0, 5, 0)
        assert report.reviews_total == 5
        assert report.usage is None

    def test_new_reviews_classify_only_the_new_texts(self, conn: sqlite3.Connection) -> None:
        seed(conn, 3)
        analyze_pending(conn, SpyClassifier())
        save_reviews(conn, [review("brand new one"), review("brand new two")])
        spy = SpyClassifier()

        report = analyze_pending(conn, spy)

        assert spy.called_texts == ["brand new one", "brand new two"]
        assert (report.reviews_total, report.analyzed, report.cached) == (5, 2, 3)

    def test_a_new_review_with_a_known_text_needs_no_call(self, conn: sqlite3.Connection) -> None:
        seed(conn, 2)
        analyze_pending(conn, SpyClassifier())
        save_reviews(conn, [review("REVIEW  number 0", day=15)])
        spy = SpyClassifier()

        report = analyze_pending(conn, spy)

        assert spy.calls == []
        assert (report.reviews_total, report.analyzed, report.cached) == (3, 0, 3)

    def test_empty_database(self, conn: sqlite3.Connection) -> None:
        spy = SpyClassifier()

        report = analyze_pending(conn, spy)

        assert spy.calls == []
        assert report == AnalysisReport(
            reviews_total=0, analyzed=0, cached=0, failed=0, failures={}, usage=None
        )


class TestDuplicatesAndCacheKey:
    def test_duplicate_texts_are_classified_once_and_counted_per_review(
        self, conn: sqlite3.Connection
    ) -> None:
        save_reviews(
            conn,
            [
                review("same words", day=1),
                review("Same  Words", day=2),
                review("SAME WORDS", day=3, source="app"),
                review("other"),
            ],
        )
        spy = SpyClassifier()

        report = analyze_pending(conn, spy)

        assert spy.called_texts == ["same words", "other"]
        assert (report.reviews_total, report.analyzed, report.cached) == (4, 4, 0)

    @pytest.mark.parametrize(
        "other",
        [{"prompt_version": "p2"}, {"model": "spy-2"}],
    )
    def test_different_prompt_version_or_model_is_a_cache_miss(
        self, conn: sqlite3.Connection, other: dict[str, str]
    ) -> None:
        seed(conn, 3)
        analyze_pending(conn, SpyClassifier())
        spy = SpyClassifier(**other)

        report = analyze_pending(conn, spy)

        assert len(spy.called_texts) == 3
        assert (report.analyzed, report.cached) == (3, 0)

    def test_rating_comes_from_the_earliest_stored_review(self, conn: sqlite3.Connection) -> None:
        save_reviews(conn, [review("same", day=5, rating=2), review("same", day=1, rating=5)])
        spy = SpyClassifier()

        analyze_pending(conn, spy)

        assert [i.rating for i in spy.calls[0]] == [2]


class TestBatching:
    def test_batches_use_short_ids_restarting_per_batch(self, conn: sqlite3.Connection) -> None:
        seed(conn, 5)
        spy = SpyClassifier()

        analyze_pending(conn, spy, batch_size=2)

        assert [[i.item_id for i in call] for call in spy.calls] == [
            ["r1", "r2"],
            ["r1", "r2"],
            ["r1"],
        ]

    def test_default_batch_size_is_twenty(self, conn: sqlite3.Connection) -> None:
        seed(conn, 45)
        spy = SpyClassifier()

        analyze_pending(conn, spy)

        assert [len(c) for c in spy.calls] == [20, 20, 5]

    @pytest.mark.parametrize("size", [0, -1, 101])
    def test_invalid_batch_size(self, conn: sqlite3.Connection, size: int) -> None:
        seed(conn, 1)
        spy = SpyClassifier()

        with pytest.raises(ValueError, match="batch_size"):
            analyze_pending(conn, spy, batch_size=size)

        assert spy.calls == []

    @pytest.mark.parametrize("size", [1, 100])
    def test_boundary_batch_sizes_are_accepted(self, conn: sqlite3.Connection, size: int) -> None:
        seed(conn, 2)

        assert analyze_pending(conn, SpyClassifier(), batch_size=size).analyzed == 2

    def test_naive_now_is_rejected_before_any_call(self, conn: sqlite3.Connection) -> None:
        seed(conn, 1)
        spy = SpyClassifier()

        with pytest.raises(ValueError, match="timezone"):
            analyze_pending(conn, spy, now=dt.datetime(2026, 11, 11))

        assert spy.calls == []


class TestFailures:
    def test_classifier_error_fails_only_its_batch(self, conn: sqlite3.Connection) -> None:
        seed(conn, 6)
        spy = SpyClassifier(fail_calls={2: ClassifierError("rate_limited")})

        report = analyze_pending(conn, spy, batch_size=2)

        assert (report.reviews_total, report.analyzed, report.failed) == (6, 4, 2)
        assert report.failures == {"rate_limited": 2}
        saved = load_analyses(conn, provider="spy", model="spy-1", prompt_version="p1")
        assert set(saved) == {text_hash(f"review number {i}") for i in (0, 1, 4, 5)}

    def test_failures_are_counted_in_reviews(self, conn: sqlite3.Connection) -> None:
        save_reviews(conn, [review("dup", day=1), review("dup", day=2), review("dup", day=3)])
        spy = SpyClassifier(fail_calls={1: ClassifierError("bad_output")})

        report = analyze_pending(conn, spy)

        assert report.failed == 3
        assert report.failures == {"bad_output": 3}

    def test_invariant_holds_with_cache_failures_and_successes(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 2)
        analyze_pending(conn, SpyClassifier())
        save_reviews(conn, [review(f"fresh {i}", day=11) for i in range(5)])
        save_reviews(conn, [review("review number 0", day=12)])
        spy = SpyClassifier(fail_calls={2: ClassifierError("timeout")}, drop_ids=frozenset({"r2"}))

        report = analyze_pending(conn, spy, batch_size=2)

        assert report.reviews_total == 8
        assert report.reviews_total == report.analyzed + report.cached + report.failed
        assert report.cached == 3
        assert report.failures == {"timeout": 2, "missing_result": 1}

    def test_missing_ids_fail_as_missing_result(self, conn: sqlite3.Connection) -> None:
        seed(conn, 3)
        spy = SpyClassifier(drop_ids=frozenset({"r2"}))

        report = analyze_pending(conn, spy)

        assert (report.analyzed, report.failed) == (2, 1)
        assert report.failures == {"missing_result": 1}

    def test_unexpected_ids_are_ignored_and_counted(
        self, conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        seed(conn, 2)
        spy = SpyClassifier(extra_ids=("r99", "zzz"))

        with caplog.at_level(logging.WARNING):
            report = analyze_pending(conn, spy)

        assert report.unexpected_results == 2
        assert (report.analyzed, report.failed) == (2, 0)
        assert len(load_analyses(conn, provider="spy", model="spy-1", prompt_version="p1")) == 2
        assert "2 unexpected" in caplog.text
        assert "zzz" not in caplog.text

    def test_rerun_after_a_failure_retries_only_the_failed_texts(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 6)
        analyze_pending(
            conn,
            SpyClassifier(fail_calls={2: ClassifierError("rate_limited")}),
            batch_size=2,
        )
        spy = SpyClassifier()

        report = analyze_pending(conn, spy, batch_size=2)

        assert spy.called_texts == ["review number 2", "review number 3"]
        assert (report.reviews_total, report.analyzed, report.cached, report.failed) == (
            6,
            2,
            4,
            0,
        )

    def test_other_exceptions_abort_and_earlier_batches_stay_saved(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 6)
        spy = SpyClassifier(fail_calls={2: RuntimeError("boom")})

        with pytest.raises(AnalysisAborted) as excinfo:
            analyze_pending(conn, spy, batch_size=2)

        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert len(spy.calls) == 2
        saved = load_analyses(conn, provider="spy", model="spy-1", prompt_version="p1")
        assert set(saved) == {text_hash("review number 0"), text_hash("review number 1")}


class TestUsage:
    def test_usage_is_summed_over_batches(self, conn: sqlite3.Connection) -> None:
        seed(conn, 5)
        spy = SpyClassifier(usage=Usage(input_tokens=10, output_tokens=2))

        report = analyze_pending(conn, spy, batch_size=2)

        assert report.usage == Usage(input_tokens=30, output_tokens=6)

    def test_usage_is_none_when_no_batch_reports_it(self, conn: sqlite3.Connection) -> None:
        seed(conn, 3)

        assert analyze_pending(conn, SpyClassifier(), batch_size=2).usage is None

    def test_usage_of_a_failed_batch_is_not_counted(self, conn: sqlite3.Connection) -> None:
        seed(conn, 4)
        spy = SpyClassifier(
            fail_calls={1: ClassifierError("timeout")}, usage=Usage(input_tokens=5, output_tokens=1)
        )

        report = analyze_pending(conn, spy, batch_size=2)

        assert report.usage == Usage(input_tokens=5, output_tokens=1)


class TestLogging:
    def test_summary_is_logged_and_failures_warned_without_text(
        self, conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        save_reviews(conn, [review(FAKE_TEXT), review(f"{FAKE_TEXT} two")])
        spy = SpyClassifier(fail_calls={1: ClassifierError("rate_limited", "Slow down.")})

        with caplog.at_level(logging.DEBUG):
            analyze_pending(conn, spy, batch_size=1)

        info = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and r.name == "review_radar.analysis"
        ]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(info) == 1
        assert "analyzed=1" in info[0].getMessage()
        assert "failed=1" in info[0].getMessage()
        assert len(warnings) == 2  # the batch warning and the per-reason summary
        assert all("rate_limited" in w.getMessage() for w in warnings)
        assert FAKE_TEXT not in caplog.text
        assert "ZEBRA" not in caplog.text

    def test_one_warning_per_failure_reason(
        self, conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        seed(conn, 4)
        spy = SpyClassifier(
            fail_calls={1: ClassifierError("a_reason"), 2: ClassifierError("b_reason")}
        )

        with caplog.at_level(logging.WARNING):
            analyze_pending(conn, spy, batch_size=2)

        messages = [r.getMessage() for r in caplog.records]
        summaries = [m for m in messages if "Analysis failed for" in m]
        assert len([m for m in summaries if "a_reason" in m]) == 1
        assert len([m for m in summaries if "b_reason" in m]) == 1


class TestCounters:
    def test_fallback_and_topic_counters_are_summed_over_batches(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 5)
        spy = SpyClassifier(fallback=2, unknown_topics=1, truncated_topics=3)

        report = analyze_pending(conn, spy, batch_size=2)

        assert (report.fallback, report.unknown_topics, report.truncated_topics) == (6, 3, 9)

    def test_counters_are_zero_by_default(self, conn: sqlite3.Connection) -> None:
        seed(conn, 2)

        report = analyze_pending(conn, SpyClassifier())

        assert (report.fallback, report.unknown_topics, report.truncated_topics) == (0, 0, 0)

    def test_counters_of_cached_runs_are_zero(self, conn: sqlite3.Connection) -> None:
        seed(conn, 2)
        analyze_pending(conn, SpyClassifier(fallback=2))

        report = analyze_pending(conn, SpyClassifier(fallback=2))

        assert report.fallback == 0

    def test_counters_of_a_failed_batch_are_not_counted(self, conn: sqlite3.Connection) -> None:
        seed(conn, 4)
        spy = SpyClassifier(fallback=1, fail_calls={1: ClassifierError("timeout")})

        report = analyze_pending(conn, spy, batch_size=2)

        assert report.fallback == 1


class FailingSave:
    """Wraps the real save_analyses and raises StorageError on the chosen call."""

    def __init__(self, fail_on: int) -> None:
        self.calls = 0
        self.fail_on = fail_on
        self.real = analysis_module.save_analyses

    def __call__(self, *args: object, **kwargs: object) -> int:
        self.calls += 1
        if self.calls == self.fail_on:
            raise StorageError("Cannot save analyses: disk full.")
        return self.real(*args, **kwargs)  # type: ignore[arg-type]


class TestAborted:
    def test_storage_error_on_batch_two_keeps_batch_one_and_reports_the_rest(
        self, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed(conn, 6)
        monkeypatch.setattr(analysis_module, "save_analyses", FailingSave(fail_on=2))

        with pytest.raises(AnalysisAborted) as excinfo:
            analyze_pending(conn, SpyClassifier(), batch_size=2)

        report = excinfo.value.report
        assert isinstance(excinfo.value.__cause__, StorageError)
        assert (report.reviews_total, report.analyzed, report.cached, report.failed) == (6, 2, 0, 4)
        assert report.failures == {"save_failed": 2, "not_run": 2}
        assert report.reviews_total == report.analyzed + report.cached + report.failed
        saved = load_analyses(conn, provider="spy", model="spy-1", prompt_version="p1")
        assert set(saved) == {text_hash("review number 0"), text_hash("review number 1")}

    def test_runtime_error_from_classify_is_wrapped(self, conn: sqlite3.Connection) -> None:
        seed(conn, 6)
        spy = SpyClassifier(fail_calls={2: RuntimeError("boom ZEBRA")})

        with pytest.raises(AnalysisAborted) as excinfo:
            analyze_pending(conn, spy, batch_size=2)

        report = excinfo.value.report
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert (report.analyzed, report.failed, report.failures) == (2, 4, {"not_run": 4})

    def test_earlier_failures_are_kept_in_the_partial_report(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 6)
        spy = SpyClassifier(fail_calls={1: ClassifierError("timeout"), 2: RuntimeError("boom")})

        with pytest.raises(AnalysisAborted) as excinfo:
            analyze_pending(conn, spy, batch_size=2)

        report = excinfo.value.report
        assert report.failures == {"timeout": 2, "not_run": 4}
        assert report.reviews_total == report.analyzed + report.cached + report.failed

    def test_cached_reviews_stay_in_the_invariant(self, conn: sqlite3.Connection) -> None:
        seed(conn, 2)
        analyze_pending(conn, SpyClassifier())
        save_reviews(conn, [review(f"fresh {i}", day=11) for i in range(3)])
        spy = SpyClassifier(fail_calls={1: RuntimeError("boom")})

        with pytest.raises(AnalysisAborted) as excinfo:
            analyze_pending(conn, spy, batch_size=2)

        report = excinfo.value.report
        assert (report.cached, report.failed, report.reviews_total) == (2, 3, 5)

    def test_one_warning_with_counts_and_the_exception_type_only(
        self, conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
    ) -> None:
        seed(conn, 4)
        spy = SpyClassifier(fail_calls={2: RuntimeError("ZEBRA-QUOKKA-9917 leaked")})

        with caplog.at_level(logging.DEBUG), pytest.raises(AnalysisAborted):
            analyze_pending(conn, spy, batch_size=2)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "RuntimeError" in message
        assert "analyzed=2" in message
        assert "failed=2" in message
        assert "ZEBRA" not in caplog.text

    def test_base_exceptions_are_not_wrapped(self, conn: sqlite3.Connection) -> None:
        seed(conn, 2)
        spy = SpyClassifier(fail_calls={1: KeyboardInterrupt()})

        with pytest.raises(KeyboardInterrupt):
            analyze_pending(conn, spy)


class RawTopicClassifier:
    """Builds results from raw topic lists with `normalize_topics`, as a real classifier must."""

    provider = "raw"
    model = "raw-1"
    prompt_version = "p1"

    def __init__(self, raw_topics: list[list[object]]) -> None:
        self.raw_topics = raw_topics

    def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
        analyses: dict[str, Analysis] = {}
        unknown = truncated = 0
        for item, raw in zip(items, self.raw_topics, strict=False):
            normalized = normalize_topics(raw)
            unknown += normalized.unknown
            truncated += normalized.truncated
            analyses[item.item_id] = Analysis(
                sentiment=Sentiment.NEGATIVE, topics=normalized.topics, confidence=None
            )
        return ClassifyResult(analyses=analyses, unknown_topics=unknown, truncated_topics=truncated)


class TestTopicNormalizationEndToEnd:
    def test_counts_flow_into_the_report(self, conn: sqlite3.Connection) -> None:
        seed(conn, 2)
        classifier = RawTopicClassifier(
            [
                ["weather", "pricing"],
                ["delivery_delay", "pricing", "billing", "packaging", "nonsense"],
            ]
        )

        report = analyze_pending(conn, classifier)

        assert report.unknown_topics == 2
        assert report.truncated_topics == 2
        stored = load_analyses(conn, provider="raw", model="raw-1", prompt_version="p1")
        assert len(stored) == 2

    def test_clean_topics_report_zero(self, conn: sqlite3.Connection) -> None:
        seed(conn, 1)

        report = analyze_pending(conn, RawTopicClassifier([["pricing"]]))

        assert (report.unknown_topics, report.truncated_topics) == (0, 0)
