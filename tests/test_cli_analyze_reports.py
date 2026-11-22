"""CLI `analyze`: version invalidation, visible fallback/anomaly counters, aborts and logging."""

import contextlib
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from review_radar import analysis as analysis_module
from review_radar import cli
from review_radar.classifier import ClassifierError, ClassifyItem, ClassifyResult
from review_radar.cli import app
from review_radar.models import Analysis, Sentiment, normalize_topics
from review_radar.storage import StorageError, connect, count_reviews

runner = CliRunner()

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews.csv"
COMMITTED_FIXTURES = ROOT / "fixtures" / "demo_analyses.json"
FAKE_TEXT = "ZEBRA-QUOKKA-9917 distinctive fake review"
POSITIVE = Analysis(sentiment=Sentiment.POSITIVE, topics=(), confidence=None)


class ScriptedClassifier:
    """Positive answers; can add unexpected ids, report counters, or raise on a call."""

    provider = "fixture"
    model = "demo"
    prompt_version = "fixture-1+scripted"

    def __init__(
        self,
        *,
        fail_on: int | None = None,
        error: Exception | None = None,
        extra_ids: tuple[str, ...] = (),
        fallback: int = 0,
        unknown_topics: int = 0,
        truncated_topics: int = 0,
    ) -> None:
        self.calls = 0
        self.fail_on = fail_on
        self.error = error or ClassifierError("rate_limited", "Slow down.")
        self.extra_ids = extra_ids
        self.counters = {
            "fallback": fallback,
            "unknown_topics": unknown_topics,
            "truncated_topics": truncated_topics,
        }

    def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
        self.calls += 1
        if self.calls == self.fail_on:
            raise self.error
        analyses = {i.item_id: POSITIVE for i in items}
        analyses.update({extra: POSITIVE for extra in self.extra_ids})
        return ClassifyResult(analyses=analyses, **self.counters)


def use(monkeypatch: pytest.MonkeyPatch, classifier: ScriptedClassifier) -> None:
    monkeypatch.setattr(cli.FixtureClassifier, "from_file", lambda _path: classifier)


def ingest_many(tmp_path: Path, n: int, prefix: str = "item") -> None:
    rows = "".join(f"2026-11-10,{prefix} number {i} {FAKE_TEXT}\n" for i in range(n))
    path = tmp_path / "many.csv"
    path.write_text("review_date,text\n" + rows, encoding="utf-8")
    assert runner.invoke(app, ["ingest", str(path)]).exit_code == 0


def stored_total() -> int:
    with contextlib.closing(connect(Path(os.environ["DATABASE_PATH"]))) as conn:
        return count_reviews(conn)


class TestContentDerivedVersion:
    def test_changed_fixture_reanalyzes_everything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner.invoke(app, ["ingest", str(SAMPLE_CSV)])
        first = runner.invoke(app, ["analyze"])
        again = runner.invoke(app, ["analyze"])
        payload = json.loads(COMMITTED_FIXTURES.read_text(encoding="utf-8"))
        key = sorted(payload["analyses"])[0]
        payload["analyses"][key] = {"sentiment": "neutral", "topics": []}
        changed = tmp_path / "changed.json"
        changed.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(cli, "DEMO_FIXTURES", changed)

        result = runner.invoke(app, ["analyze"])

        assert "Analyzed now: 300" in first.output
        assert "Analyzed now: 0" in again.output
        assert result.exit_code == 0, result.output
        assert "Analyzed now: 300" in result.output
        assert "From cache: 0" in result.output
        assert result.output.splitlines()[0] != first.output.splitlines()[0]


class TestFallbackVisible:
    def test_fallback_line_is_printed_when_rules_answered(self, tmp_path: Path) -> None:
        ingest_many(tmp_path, 3)

        result = runner.invoke(app, ["analyze"])

        assert result.exit_code == 0, result.output
        assert "Fallback (keyword rules): 3" in result.output

    def test_no_fallback_line_for_the_sample(self) -> None:
        runner.invoke(app, ["ingest", str(SAMPLE_CSV)])

        result = runner.invoke(app, ["analyze"])

        assert "Fallback" not in result.output

    def test_fallback_counts_only_this_run(self, tmp_path: Path) -> None:
        ingest_many(tmp_path, 3)
        runner.invoke(app, ["analyze"])

        result = runner.invoke(app, ["analyze"])

        assert "Fallback" not in result.output
        assert "From cache: 3" in result.output


class TestAnomalyCounters:
    def test_unknown_and_truncated_topics_are_printed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        use(monkeypatch, ScriptedClassifier(unknown_topics=2, truncated_topics=1))

        result = runner.invoke(app, ["analyze"])

        assert result.exit_code == 0, result.output
        assert "Unknown topics: 2" in result.output
        assert "Truncated topics: 1" in result.output

    def test_zero_counters_are_not_printed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        use(monkeypatch, ScriptedClassifier())

        result = runner.invoke(app, ["analyze"])

        assert "Unknown topics" not in result.output
        assert "Truncated topics" not in result.output
        assert "Unexpected results" not in result.output

    def test_unexpected_results_are_printed_and_exit_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        use(monkeypatch, ScriptedClassifier(extra_ids=("r77",)))

        result = runner.invoke(app, ["analyze"])

        assert result.exit_code == 1
        assert "Unexpected results: 1" in result.output
        assert "Failed: 0" in result.output


class RawTopicClassifier(ScriptedClassifier):
    """Builds answers from raw topics with `normalize_topics`, like a real classifier."""

    def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
        normalized = normalize_topics(["weather", "pricing", "billing", "packaging", "other"])
        analysis = Analysis(sentiment=Sentiment.NEGATIVE, topics=normalized.topics, confidence=None)
        return ClassifyResult(
            analyses={i.item_id: analysis for i in items},
            unknown_topics=normalized.unknown * len(items),
            truncated_topics=normalized.truncated * len(items),
        )


class TestNormalizedTopicsEndToEnd:
    def test_counts_are_printed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ingest_many(tmp_path, 2)
        use(monkeypatch, RawTopicClassifier())

        result = runner.invoke(app, ["analyze"])

        assert result.exit_code == 0, result.output
        assert "Unknown topics: 2" in result.output
        assert "Truncated topics: 2" in result.output


class TestAbortedRun:
    def test_storage_error_prints_the_partial_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 45)
        real = analysis_module.save_analyses
        calls: list[int] = []

        def flaky(*args: object, **kwargs: object) -> int:
            calls.append(1)
            if len(calls) == 2:
                raise StorageError("Cannot save analyses: disk full.")
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(analysis_module, "save_analyses", flaky)

        result = runner.invoke(app, ["analyze"])

        assert result.exit_code == 1
        assert "Reviews: 45" in result.output
        assert "Analyzed now: 20" in result.output
        assert "Failed: 25" in result.output
        assert "  save_failed: 20" in result.output
        assert "  not_run: 5" in result.output
        assert "Error: analysis stopped: Cannot save analyses: disk full." in result.output
        assert "Traceback" not in result.output
        assert FAKE_TEXT not in result.output

    def test_other_exceptions_show_only_their_type(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 45)
        use(monkeypatch, ScriptedClassifier(fail_on=2, error=RuntimeError("ZEBRA leaked")))

        result = runner.invoke(app, ["analyze"])

        assert result.exit_code == 1
        assert "Analyzed now: 20" in result.output
        assert "  not_run: 25" in result.output
        assert "Error: analysis stopped: RuntimeError." in result.output
        assert "ZEBRA" not in result.output
        assert "Traceback" not in result.output
        assert stored_total() == 45


class TestLogging:
    def test_warnings_reach_stderr_by_default_and_info_does_not(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        use(monkeypatch, ScriptedClassifier(fail_on=1))

        result = runner.invoke(app, ["analyze"])

        assert "WARNING review_radar.analysis: Analysis failed for 2 review(s): rate_limited" in (
            result.output
        )
        assert "INFO" not in result.output
        assert FAKE_TEXT not in result.output

    def test_verbose_adds_info_lines(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ingest_many(tmp_path, 2)
        use(monkeypatch, ScriptedClassifier())

        result = runner.invoke(app, ["--verbose", "analyze"])

        assert "INFO review_radar.analysis: Analysis done: total=2 analyzed=2" in result.output
        assert FAKE_TEXT not in result.output

    def test_default_run_prints_no_info(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        use(monkeypatch, ScriptedClassifier())

        result = runner.invoke(app, ["analyze"])

        assert "INFO" not in result.output

    def test_verbose_flag_is_global_and_repeat_safe(self, tmp_path: Path) -> None:
        ingest_many(tmp_path, 1)

        first = runner.invoke(app, ["--verbose", "analyze"])
        second = runner.invoke(app, ["analyze"])

        assert "INFO" in first.output
        assert "INFO" not in second.output

    def test_a_broken_stderr_never_breaks_logging(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class BrokenStream:
            def write(self, _text: str) -> None:
                raise OSError("closed")

        monkeypatch.setattr(logging, "raiseExceptions", False)
        monkeypatch.setattr("sys.stderr", BrokenStream())
        record = logging.LogRecord("review_radar.x", logging.WARNING, "f", 1, "message", (), None)

        cli._StderrHandler().emit(record)  # must not raise
