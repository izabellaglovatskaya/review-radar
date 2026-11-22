"""CLI `analyze`, review round: call budget, circuit breaker, key safety, Ctrl+C, run records."""

import contextlib
import json
import os
import re
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import anthropic
import pytest
from typer.testing import CliRunner

from anthropic_fakes import FAKE_TEXT, make_message, status_error
from review_radar import cli
from review_radar.classifier import ClassifierError, ClassifyItem, ClassifyResult
from review_radar.cli import app
from review_radar.models import Analysis, Sentiment
from review_radar.storage import connect
from test_cli_analyze_live import Live, broken_save_run, ingest_many, ok, runs

runner = CliRunner()
POSITIVE = Analysis(sentiment=Sentiment.POSITIVE, topics=(), confidence=None)
LEAK_KEY = "sk-ant-FAKE-LEAK-0000"


def use_fixture_replacement(monkeypatch: pytest.MonkeyPatch, classifier: Any) -> None:
    monkeypatch.setattr(cli.FixtureClassifier, "from_file", lambda _p: classifier)


class Fake:
    provider = "fixture"
    model = "demo"
    prompt_version = "fixture-1+fake"

    def __init__(self, plan: dict[int, BaseException] | None = None) -> None:
        self.plan = plan or {}
        self.calls = 0

    def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
        self.calls += 1
        if self.calls in self.plan:
            raise self.plan[self.calls]
        return ClassifyResult(analyses={i.item_id: POSITIVE for i in items})


class TestMaxCalls:
    def test_live_default_budget_is_three_per_batch_plus_ten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        Live(monkeypatch, ok(2))

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 0, result_.output
        assert "API calls: 1 / 13" in result_.output.splitlines()

    def test_an_explicit_budget_is_shown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        Live(monkeypatch, ok(2))

        result_ = runner.invoke(app, ["analyze", "--max-calls", "5"])

        assert "API calls: 1 / 5" in result_.output.splitlines()

    def test_the_default_is_computed_after_the_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 50)
        Live(monkeypatch, ok(2))

        result_ = runner.invoke(app, ["analyze", "--limit", "2"])

        assert "API calls: 1 / 13" in result_.output.splitlines()

    @pytest.mark.parametrize("bad", ["0", "-1", "x"])
    def test_an_invalid_budget_is_a_usage_error(self, tmp_path: Path, bad: str) -> None:
        ingest_many(tmp_path, 1)

        result_ = runner.invoke(app, ["analyze", "--max-calls", bad])

        assert result_.exit_code == 2
        assert "Traceback" not in result_.output

    def test_demo_ignores_the_budget_and_prints_no_calls_line(self, tmp_path: Path) -> None:
        ingest_many(tmp_path, 3)

        result_ = runner.invoke(app, ["analyze", "--max-calls", "1"])

        assert result_.exit_code == 0, result_.output
        assert "API calls" not in result_.output
        assert "Analyzed now: 3" in result_.output

    def test_an_always_invalid_batch_hits_the_budget_and_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 20)
        live = Live(monkeypatch, *[make_message("not json") for _ in range(13)])

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 1
        assert len(live.client.calls) == 13
        lines = result_.output.splitlines()
        assert "API calls: 13 / 13" in lines
        assert "Stopped early: call_budget_exceeded" in lines
        assert any(line.startswith("  call_budget_exceeded: ") for line in lines)
        assert FAKE_TEXT not in result_.output
        (run,) = runs()
        assert run.outcome == "call_budget_exceeded"
        assert run.api_calls == 13


class TestCircuitBreakerCli:
    def test_two_transient_batches_stop_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 100)
        fake = Fake(
            {1: ClassifierError("rate_limited", "x"), 2: ClassifierError("rate_limited", "x")}
        )
        use_fixture_replacement(monkeypatch, fake)

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 1
        assert fake.calls == 2
        lines = result_.output.splitlines()
        assert "  rate_limited: 40" in lines
        assert "  skipped_after_transient: 60" in lines
        assert "Stopped early: stopped_transient" in lines
        (run,) = runs()
        assert run.outcome == "stopped_transient"
        assert run.failures == {"rate_limited": 40, "skipped_after_transient": 60}


class TestKeySafety:
    def test_the_app_never_shows_locals_in_tracebacks(self) -> None:
        assert app.pretty_exceptions_show_locals is False

    def test_a_client_construction_failure_is_one_clean_line_without_the_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 1)
        monkeypatch.setenv("REVIEW_RADAR_MODE", "live")
        monkeypatch.setenv("ANTHROPIC_API_KEY", LEAK_KEY)

        class Exploding:
            def __init__(self, **kwargs: Any) -> None:
                raise RuntimeError(f"cannot build with {kwargs['api_key']}")

        monkeypatch.setattr(anthropic, "Anthropic", Exploding)

        result_ = runner.invoke(app, ["--verbose", "analyze"])

        assert result_.exit_code == 1
        assert LEAK_KEY not in result_.output
        assert "sk-ant" not in result_.output
        assert "Traceback" not in result_.output
        assert result_.output.strip() == (
            "Error: Cannot create the Anthropic client (RuntimeError)."
        )
        assert runs() == []

    def test_an_unexpected_exception_never_shows_the_key_in_a_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 1)
        monkeypatch.setenv("REVIEW_RADAR_MODE", "live")
        monkeypatch.setenv("ANTHROPIC_API_KEY", LEAK_KEY)

        def explode(**_kwargs: Any) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(cli, "make_anthropic_classifier", explode)

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 1
        assert LEAK_KEY not in result_.output
        assert "Traceback" not in result_.output
        assert "Cannot create the Anthropic client (RuntimeError)." in result_.output


class TestCtrlC:
    def test_exit_130_report_printed_and_run_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 45)
        use_fixture_replacement(monkeypatch, Fake({2: KeyboardInterrupt()}))

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 130
        lines = result_.output.splitlines()
        assert "Analyzed now: 20" in lines
        assert "  interrupted: 25" in lines
        assert "Interrupted." in result_.output
        assert "Traceback" not in result_.output
        (run,) = runs()
        assert run.outcome == "interrupted"
        assert (run.analyzed, run.failed) == (20, 25)
        assert run.failures == {"interrupted": 25}


class TestRunRecords:
    def test_fatal_run_record(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ingest_many(tmp_path, 45)
        live = Live(monkeypatch, ok(20), status_error(402))
        runner.invoke(app, ["analyze"])
        assert live.client.remaining == 0
        (fatal,) = runs()
        assert fatal.outcome == "fatal"
        assert fatal.failures == {"billing_error": 20, "skipped_after_fatal": 5}

    def test_completed_run_record(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ingest_many(tmp_path, 5)
        Live(monkeypatch, ok(2), ok(2))
        runner.invoke(app, ["analyze", "--limit", "2"])

        (run,) = runs()

        assert run.outcome == "completed"
        assert run.not_analyzed_limit == 3
        assert run.failures == {}
        assert run.api_calls == 1
        assert run.cost_usd == Decimal("0.0002")

    def test_aborted_run_record(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ingest_many(tmp_path, 45)
        use_fixture_replacement(monkeypatch, Fake({2: RuntimeError("boom")}))

        runner.invoke(app, ["analyze"])

        (run,) = runs()
        assert run.outcome == "aborted"
        assert run.failures == {"not_run": 25}

    def test_the_failures_column_holds_reason_counts_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 1)
        Live(monkeypatch, make_message("not json"))
        runner.invoke(app, ["analyze"])

        with contextlib.closing(connect(Path(os.environ["DATABASE_PATH"]))) as conn:
            raw = conn.execute("SELECT failures FROM runs").fetchone()[0]

        assert json.loads(raw) == {"invalid_output": 1}
        assert FAKE_TEXT not in raw


class TestRunSaveFailureNeverHidesTheReport:
    def check(self, result_: Any, exit_code: int) -> None:
        assert result_.exit_code == exit_code
        out = result_.output
        assert "Reviews: " in out
        assert "Error: Cannot save the run: disk full." in out
        assert out.index("Reviews: ") < out.index("Error: Cannot save the run")
        assert "Traceback" not in out

    def test_normal_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ingest_many(tmp_path, 2)
        monkeypatch.setattr(cli, "save_run", broken_save_run)

        self.check(runner.invoke(app, ["analyze"]), 1)

    def test_fatal_run_prints_both_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        Live(monkeypatch, status_error(401))
        monkeypatch.setattr(cli, "save_run", broken_save_run)

        result_ = runner.invoke(app, ["analyze"])

        self.check(result_, 1)
        assert "Error: The Anthropic API key was rejected." in result_.output

    def test_aborted_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ingest_many(tmp_path, 2)
        use_fixture_replacement(monkeypatch, Fake({1: RuntimeError("boom")}))
        monkeypatch.setattr(cli, "save_run", broken_save_run)

        result_ = runner.invoke(app, ["analyze"])

        self.check(result_, 1)
        assert "Error: analysis stopped: RuntimeError." in result_.output

    def test_interrupted_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ingest_many(tmp_path, 2)
        use_fixture_replacement(monkeypatch, Fake({1: KeyboardInterrupt()}))
        monkeypatch.setattr(cli, "save_run", broken_save_run)

        self.check(runner.invoke(app, ["analyze"]), 130)

    def test_the_report_has_the_expected_shape_for_the_first_live_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        Live(monkeypatch, ok(2))

        output = runner.invoke(app, ["analyze"]).output

        assert re.search(r"^API calls: 1 / 13$", output, re.M)


class TestFatalWithFailedPartialSaveCli:
    def test_the_fatal_message_is_printed_then_a_line_about_the_failed_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from review_radar import analysis as analysis_module
        from review_radar.classifier import FatalClassifierError
        from review_radar.storage import StorageError

        ingest_many(tmp_path, 4)
        partial = ClassifyResult(analyses={"r1": POSITIVE})
        fatal = FatalClassifierError("billing_error", "No credit left.", partial=partial)
        use_fixture_replacement(monkeypatch, Fake({1: fatal}))

        def broken(*_a: object, **_k: object) -> int:
            raise StorageError("Cannot save analyses: disk full.")

        monkeypatch.setattr(analysis_module, "save_analyses", broken)

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 1
        lines = result_.output.splitlines()
        assert "analysis stopped" not in result_.output
        assert "  save_failed: 1" in lines
        assert "  billing_error: 3" in lines
        error_lines = [line for line in lines if line.startswith("Error: ")]
        assert error_lines[0] == "Error: No credit left."
        assert len(error_lines) == 2
        assert "could not be saved" in error_lines[1]
        assert FAKE_TEXT not in result_.output
        (run,) = runs()
        assert (run.outcome, run.fatal_reason) == ("fatal", "billing_error")
