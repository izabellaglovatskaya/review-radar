"""Analysis service, review round: partial results on fatal/abort, circuit breaker, call budget,
save failures after a paid batch and Ctrl+C."""

import contextlib
import datetime as dt
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from anthropic_fakes import FAKE_KEY, FakeClient, make_message, result
from review_radar import analysis as analysis_module
from review_radar.analysis import AnalysisAborted, analyze_pending
from review_radar.anthropic_classifier import AnthropicClassifier
from review_radar.classifier import (
    ClassifierError,
    ClassifyItem,
    ClassifyResult,
    FatalClassifierError,
)
from review_radar.hashing import text_hash
from review_radar.models import Analysis, Review, Sentiment, Usage
from review_radar.storage import StorageError, connect, load_analyses, save_reviews

POSITIVE = Analysis(sentiment=Sentiment.POSITIVE, topics=(), confidence=None)
KEY = {"provider": "spy", "model": "spy-1", "prompt_version": "p1"}


class Scripted:
    """Per call number: raise an exception, or answer with every id analyzed."""

    provider = "spy"
    model = "spy-1"
    prompt_version = "p1"

    def __init__(
        self, raise_on: dict[int, BaseException] | None = None, usage: Usage | None = None
    ) -> None:
        self.calls: list[list[ClassifyItem]] = []
        self.raise_on = raise_on or {}
        self.usage = usage

    def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
        self.calls.append(list(items))
        if len(self.calls) in self.raise_on:
            raise self.raise_on[len(self.calls)]
        return ClassifyResult(analyses={i.item_id: POSITIVE for i in items}, usage=self.usage)


def review(text: str) -> Review:
    return Review.model_validate(
        {
            "external_id": None,
            "review_date": dt.date(2026, 11, 10),
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


def saved(conn: sqlite3.Connection) -> set[str]:
    return set(load_analyses(conn, **KEY))


def h(i: int) -> str:
    return text_hash(f"review number {i}")


def invariant(report: Any) -> bool:
    return report.reviews_total == (
        report.analyzed + report.cached + report.failed + report.not_analyzed_limit
    )


class TestPartialResultOnFatal:
    def test_valid_paid_analyses_are_saved_and_only_the_rest_fail_with_the_fatal_reason(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 6)
        partial = ClassifyResult(
            analyses={"r1": POSITIVE},
            failures={"r2": "refusal"},
            usage=Usage(input_tokens=5, output_tokens=1),
            unknown_topics=1,
            truncated_topics=2,
            unexpected_results=3,
        )
        fatal = FatalClassifierError("billing_error", "No credit.", partial=partial)

        report = analyze_pending(conn, Scripted({1: fatal}), batch_size=4)

        assert saved(conn) == {h(0)}
        assert report.analyzed == 1
        assert report.failures == {
            "refusal": 1,
            "billing_error": 2,
            "skipped_after_fatal": 2,
        }
        assert invariant(report)
        assert report.usage == Usage(input_tokens=5, output_tokens=1)
        assert (report.unknown_topics, report.truncated_topics, report.unexpected_results) == (
            1,
            2,
            3,
        )
        assert report.fatal_reason == "billing_error"
        assert report.stop_reason == "fatal"

    def test_the_partial_usage_is_not_counted_twice(self, conn: sqlite3.Connection) -> None:
        seed(conn, 2)
        spent = Usage(input_tokens=5, output_tokens=1)
        fatal = FatalClassifierError(
            "billing_error", "x", usage=spent, partial=ClassifyResult(analyses={}, usage=spent)
        )

        assert analyze_pending(conn, Scripted({1: fatal})).usage == spent

    def test_a_fatal_error_without_a_partial_fails_the_whole_batch(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 3)

        report = analyze_pending(
            conn, Scripted({1: FatalClassifierError("authentication_failed", "x")}), batch_size=2
        )

        assert report.failures == {"authentication_failed": 2, "skipped_after_fatal": 1}


class TestPartialResultOnAbort:
    def attach(self, exc: BaseException, result_: ClassifyResult) -> BaseException:
        exc.review_radar_partial = result_  # type: ignore[attr-defined]
        return exc

    def test_an_unexpected_exception_with_a_partial_saves_it_and_counts_usage(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 4)
        partial = ClassifyResult(
            analyses={"r1": POSITIVE}, usage=Usage(input_tokens=7, output_tokens=2)
        )
        boom = self.attach(RuntimeError("boom"), partial)

        with pytest.raises(AnalysisAborted) as excinfo:
            analyze_pending(conn, Scripted({1: boom}), batch_size=2)

        report = excinfo.value.report
        assert saved(conn) == {h(0)}
        assert report.analyzed == 1
        assert report.failures == {"not_run": 3}
        assert report.usage == Usage(input_tokens=7, output_tokens=2)
        assert report.stop_reason == "aborted"
        assert invariant(report)

    def test_failed_items_of_the_partial_keep_their_reasons(self, conn: sqlite3.Connection) -> None:
        seed(conn, 2)
        partial = ClassifyResult(analyses={}, failures={"r1": "refusal"})
        boom = self.attach(RuntimeError("boom"), partial)

        with pytest.raises(AnalysisAborted) as excinfo:
            analyze_pending(conn, Scripted({1: boom}))

        assert excinfo.value.report.failures == {"refusal": 1, "not_run": 1}

    def test_an_exception_without_a_partial_behaves_as_before(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 4)

        with pytest.raises(AnalysisAborted) as excinfo:
            analyze_pending(conn, Scripted({2: RuntimeError("boom")}), batch_size=2)

        assert excinfo.value.report.failures == {"not_run": 2}
        assert excinfo.value.report.analyzed == 2

    def test_a_storage_error_while_saving_the_partial_does_not_hide_the_abort(
        self, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed(conn, 2)
        boom = self.attach(
            RuntimeError("boom"), ClassifyResult(analyses={"r1": POSITIVE}, usage=None)
        )

        def broken(*_a: object, **_k: object) -> int:
            raise StorageError("Cannot save analyses: disk full.")

        monkeypatch.setattr(analysis_module, "save_analyses", broken)

        with pytest.raises(AnalysisAborted) as excinfo:
            analyze_pending(conn, Scripted({1: boom}))

        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert excinfo.value.report.failures == {"save_failed": 1, "not_run": 1}


class TestSaveFailureAfterAPaidBatch:
    def failing_second_save(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = analysis_module.save_analyses
        calls: list[int] = []

        def flaky(*args: object, **kwargs: object) -> int:
            calls.append(1)
            if len(calls) == 2:
                raise StorageError("Cannot save analyses: disk full.")
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(analysis_module, "save_analyses", flaky)

    def test_the_batch_is_save_failed_and_later_batches_not_run(
        self, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed(conn, 6)
        self.failing_second_save(monkeypatch)

        with pytest.raises(AnalysisAborted) as excinfo:
            analyze_pending(conn, Scripted(), batch_size=2)

        report = excinfo.value.report
        assert isinstance(excinfo.value.__cause__, StorageError)
        assert (report.analyzed, report.failed) == (2, 4)
        assert report.failures == {"save_failed": 2, "not_run": 2}
        assert invariant(report)

    def test_usage_of_the_unsaved_batch_is_still_counted(
        self, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed(conn, 6)
        self.failing_second_save(monkeypatch)
        spy = Scripted(usage=Usage(input_tokens=10, output_tokens=2))

        with pytest.raises(AnalysisAborted) as excinfo:
            analyze_pending(conn, spy, batch_size=2)

        assert excinfo.value.report.usage == Usage(input_tokens=20, output_tokens=4)

    def test_items_that_already_failed_keep_their_own_reason(
        self, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed(conn, 2)

        class Mixed(Scripted):
            def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
                return ClassifyResult(analyses={"r1": POSITIVE}, failures={"r2": "refusal"})

        def broken(*_a: object, **_k: object) -> int:
            raise StorageError("Cannot save analyses: disk full.")

        monkeypatch.setattr(analysis_module, "save_analyses", broken)

        with pytest.raises(AnalysisAborted) as excinfo:
            analyze_pending(conn, Mixed())

        assert excinfo.value.report.failures == {"save_failed": 1, "refusal": 1}


class TestCircuitBreaker:
    def rate_limited(self) -> ClassifierError:
        return ClassifierError("rate_limited", "Slow down.")

    def test_two_consecutive_transient_batches_stop_the_run(self, conn: sqlite3.Connection) -> None:
        seed(conn, 8)
        spy = Scripted({1: self.rate_limited(), 2: self.rate_limited()})

        report = analyze_pending(conn, spy, batch_size=2)

        assert len(spy.calls) == 2
        assert report.failures == {"rate_limited": 4, "skipped_after_transient": 4}
        assert report.stop_reason == "stopped_transient"
        assert report.outcome == "stopped_transient"
        assert invariant(report)

    def test_a_success_in_between_resets_the_count(self, conn: sqlite3.Connection) -> None:
        seed(conn, 8)
        spy = Scripted({1: self.rate_limited(), 3: self.rate_limited()})

        report = analyze_pending(conn, spy, batch_size=2)

        assert len(spy.calls) == 4
        assert report.stop_reason is None
        assert report.outcome == "completed"

    def test_non_transient_errors_do_not_trip_it(self, conn: sqlite3.Connection) -> None:
        seed(conn, 6)
        spy = Scripted({1: ClassifierError("bad_output"), 2: ClassifierError("bad_output")})

        report = analyze_pending(conn, spy, batch_size=2)

        assert len(spy.calls) == 3
        assert report.stop_reason is None

    @pytest.mark.parametrize("reason", ["rate_limited", "server_error", "network_error"])
    def test_every_transient_reason_counts(self, conn: sqlite3.Connection, reason: str) -> None:
        seed(conn, 6)
        spy = Scripted({1: ClassifierError(reason), 2: ClassifierError(reason)})

        report = analyze_pending(conn, spy, batch_size=2)

        assert report.stop_reason == "stopped_transient"
        assert report.failures == {reason: 4, "skipped_after_transient": 2}

    def test_a_batch_whose_items_all_failed_transiently_counts_too(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 6)

        class AllRateLimited(Scripted):
            def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
                self.calls.append(list(items))
                return ClassifyResult(
                    analyses={}, failures={i.item_id: "rate_limited" for i in items}
                )

        spy = AllRateLimited()

        report = analyze_pending(conn, spy, batch_size=2)

        assert len(spy.calls) == 2
        assert report.stop_reason == "stopped_transient"

    def test_a_batch_with_some_analyses_is_not_a_transient_failure(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 6)

        class Half(Scripted):
            def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
                self.calls.append(list(items))
                return ClassifyResult(
                    analyses={items[0].item_id: POSITIVE},
                    failures={items[1].item_id: "rate_limited"},
                )

        spy = Half()

        report = analyze_pending(conn, spy, batch_size=2)

        assert len(spy.calls) == 3
        assert report.stop_reason is None

    def test_two_failures_with_nothing_left_do_not_claim_a_stop(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 4)
        spy = Scripted({1: self.rate_limited(), 2: self.rate_limited()})

        report = analyze_pending(conn, spy, batch_size=2)

        assert report.stop_reason is None
        assert report.failures == {"rate_limited": 4}


def anthropic_classifier(*script: Any) -> tuple[AnthropicClassifier, FakeClient]:
    client = FakeClient(*script)
    classifier = AnthropicClassifier(
        api_key=SecretStr(FAKE_KEY), client=client, sleep=lambda _s: None
    )
    return classifier, client


class TestCallBudget:
    def test_the_default_budget_is_three_calls_per_batch_plus_ten(self) -> None:
        assert analysis_module.default_max_calls(1) == 13
        assert analysis_module.default_max_calls(5) == 25

    def test_a_20_item_batch_that_is_always_invalid_stops_at_the_budget(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 20)
        classifier, client = anthropic_classifier(*[make_message("not json") for _ in range(13)])

        report = analyze_pending(conn, classifier, max_calls="auto")

        assert len(client.calls) == 13
        assert (report.max_calls, report.api_calls) == (13, 13)
        assert (report.analyzed, report.failed) == (0, 20)
        assert report.stop_reason == "call_budget_exceeded"
        assert set(report.failures) <= {"invalid_output", "call_budget_exceeded"}
        assert "call_budget_exceeded" in report.failures
        assert invariant(report)

    def test_remaining_batches_are_not_run_once_the_budget_is_gone(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 6)
        classifier, client = anthropic_classifier(make_message([result("r1"), result("r2")]))

        report = analyze_pending(conn, classifier, batch_size=2, max_calls=1)

        assert len(client.calls) == 1
        assert report.analyzed == 2
        assert report.failures == {"call_budget_exceeded": 4}
        assert report.stop_reason == "call_budget_exceeded"
        assert report.api_calls == 1

    def test_an_explicit_budget_is_used_as_given(self, conn: sqlite3.Connection) -> None:
        seed(conn, 2)
        classifier, _client = anthropic_classifier(make_message([result("r1"), result("r2")]))

        report = analyze_pending(conn, classifier, max_calls=5)

        assert (report.max_calls, report.api_calls) == (5, 1)
        assert report.stop_reason is None

    def test_no_budget_means_none_in_the_report_but_calls_are_counted(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 2)
        classifier, _client = anthropic_classifier(make_message([result("r1"), result("r2")]))

        report = analyze_pending(conn, classifier)

        assert report.max_calls is None
        assert report.api_calls == 1

    def test_auto_is_computed_after_the_limit_is_applied(self, conn: sqlite3.Connection) -> None:
        seed(conn, 50)
        classifier, _client = anthropic_classifier(make_message([result("r1"), result("r2")]))

        report = analyze_pending(conn, classifier, limit=2, max_calls="auto")

        assert report.max_calls == 13

    def test_a_classifier_without_a_budget_ignores_max_calls(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 2)

        report = analyze_pending(conn, Scripted(), max_calls=3)

        assert (report.max_calls, report.api_calls) == (None, None)
        assert report.analyzed == 2

    @pytest.mark.parametrize("bad", [0, -2])
    def test_a_budget_below_one_is_rejected(self, conn: sqlite3.Connection, bad: int) -> None:
        seed(conn, 1)
        spy = Scripted()

        with pytest.raises(ValueError, match="max_calls"):
            analyze_pending(conn, spy, max_calls=bad)

        assert spy.calls == []


class TestInterrupt:
    def test_ctrl_c_raises_analysis_interrupted_with_the_partial_report(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 6)
        spy = Scripted({2: KeyboardInterrupt()})

        with pytest.raises(KeyboardInterrupt) as excinfo:
            analyze_pending(conn, spy, batch_size=2)

        assert isinstance(excinfo.value, analysis_module.AnalysisInterrupted)
        report = excinfo.value.report
        assert isinstance(excinfo.value.__cause__, KeyboardInterrupt)
        assert (report.analyzed, report.failed) == (2, 4)
        assert report.failures == {"interrupted": 4}
        assert report.stop_reason == "interrupted"
        assert report.outcome == "interrupted"
        assert invariant(report)
        assert saved(conn) == {h(0), h(1)}

    def test_a_partial_result_of_the_interrupted_batch_is_saved(
        self, conn: sqlite3.Connection
    ) -> None:
        seed(conn, 4)
        interrupt = KeyboardInterrupt()
        interrupt.review_radar_partial = ClassifyResult(  # type: ignore[attr-defined]
            analyses={"r1": POSITIVE}, usage=Usage(input_tokens=3, output_tokens=1)
        )

        with pytest.raises(analysis_module.AnalysisInterrupted) as excinfo:
            analyze_pending(conn, Scripted({1: interrupt}), batch_size=2)

        report = excinfo.value.report
        assert report.analyzed == 1
        assert report.failures == {"interrupted": 3}
        assert report.usage == Usage(input_tokens=3, output_tokens=1)
        assert saved(conn) == {h(0)}

    def test_interrupted_is_not_an_aborted_run(self, conn: sqlite3.Connection) -> None:
        seed(conn, 2)

        with pytest.raises(analysis_module.AnalysisInterrupted) as excinfo:
            analyze_pending(conn, Scripted({1: KeyboardInterrupt()}))

        assert not isinstance(excinfo.value, AnalysisAborted)


class TestFatalWithFailedPartialSave:
    def broken_save(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(*_a: object, **_k: object) -> int:
            raise StorageError("Cannot save analyses: disk full.")

        monkeypatch.setattr(analysis_module, "save_analyses", broken)

    def fatal(self) -> FatalClassifierError:
        partial = ClassifyResult(
            analyses={"r1": POSITIVE},
            failures={"r2": "refusal"},
            usage=Usage(input_tokens=5, output_tokens=1),
        )
        return FatalClassifierError("billing_error", "No credit.", partial=partial)

    def test_the_fatal_outcome_survives_a_failed_partial_save(
        self, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed(conn, 6)
        self.broken_save(monkeypatch)

        report = analyze_pending(conn, Scripted({1: self.fatal()}), batch_size=4)

        assert report.fatal_reason == "billing_error"
        assert report.fatal_message == "No credit."
        assert report.stop_reason == "fatal"
        assert report.failures == {
            "save_failed": 1,
            "refusal": 1,
            "billing_error": 2,
            "skipped_after_fatal": 2,
        }
        assert report.analyzed == 0
        assert report.usage == Usage(input_tokens=5, output_tokens=1)
        assert invariant(report)

    def test_the_failed_save_is_a_text_free_warning(
        self,
        conn: sqlite3.Connection,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        seed(conn, 2)
        self.broken_save(monkeypatch)

        with caplog.at_level("WARNING"):
            analyze_pending(conn, Scripted({1: self.fatal()}))

        assert any("Could not save" in r.getMessage() for r in caplog.records)
        assert "review number" not in caplog.text
