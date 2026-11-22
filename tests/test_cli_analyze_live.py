"""CLI `analyze` in live mode (fake client only), `--limit`, fatal stops, cost lines and `runs`."""

import contextlib
import os
import re
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from anthropic_fakes import (
    FAKE_KEY,
    FAKE_TEXT,
    FakeClient,
    RecordingSleep,
    make_message,
    result,
    status_error,
)
from review_radar import cli
from review_radar.anthropic_classifier import AnthropicClassifier
from review_radar.classifier import (
    ClassifierError,
    ClassifyItem,
    ClassifyResult,
    FatalClassifierError,
)
from review_radar.cli import app
from review_radar.models import Analysis, Sentiment
from review_radar.storage import connect, list_runs

runner = CliRunner()
POSITIVE = Analysis(sentiment=Sentiment.POSITIVE, topics=(), confidence=None)


def ingest_many(tmp_path: Path, n: int) -> None:
    rows = "".join(f"2026-11-10,5,item number {i} {FAKE_TEXT}\n" for i in range(n))
    path = tmp_path / "many.csv"
    path.write_text("review_date,rating,text\n" + rows, encoding="utf-8")
    assert runner.invoke(app, ["ingest", str(path)]).exit_code == 0


def runs():
    with contextlib.closing(connect(Path(os.environ["DATABASE_PATH"]))) as conn:
        return list_runs(conn)


def broken_save_run(*_args: object, **_kwargs: object) -> int:
    raise cli.StorageError("Cannot save the run: disk full.")


class Live:
    """Sets up live mode with a fake key and a factory that records what it was given."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *script: object) -> None:
        monkeypatch.setenv("REVIEW_RADAR_MODE", "live")
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
        self.client = FakeClient(*script)  # type: ignore[arg-type]
        self.sleep = RecordingSleep()
        self.received: list[tuple[SecretStr, str]] = []
        monkeypatch.setattr(cli, "make_anthropic_classifier", self.factory)

    def factory(self, *, api_key: SecretStr, model: str) -> AnthropicClassifier:
        self.received.append((api_key, model))
        return AnthropicClassifier(
            api_key=api_key, model=model, client=self.client, sleep=self.sleep
        )


def ok(n: int, *, input_tokens: int = 1000, output_tokens: int = 200):
    return make_message(
        [result(f"r{i}") for i in range(1, n + 1)],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class TestLiveMode:
    def test_live_runs_the_anthropic_classifier_and_prints_its_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        live = Live(monkeypatch, ok(2))

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 0, result_.output
        lines = result_.output.splitlines()
        assert re.fullmatch(
            r"Classifier: anthropic/claude-haiku-5-5 \(prompt anthropic-1\+[0-9a-f]{8}\)", lines[0]
        )
        assert "Reviews: 2" in lines
        assert "Analyzed now: 2" in lines
        assert "From cache: 0" in lines
        assert "Failed: 0" in lines
        assert len(live.client.calls) == 1

    def test_the_factory_receives_a_secretstr_and_the_configured_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 1)
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        live = Live(monkeypatch, ok(1))

        runner.invoke(app, ["analyze"])

        assert len(live.received) == 1
        key, model = live.received[0]
        assert isinstance(key, SecretStr)
        assert key.get_secret_value() == FAKE_KEY
        assert model == "claude-sonnet-5-5"

    def test_the_key_never_appears_in_any_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        Live(monkeypatch, status_error(401))

        result_ = runner.invoke(app, ["--verbose", "analyze"])

        assert result_.exit_code == 1
        assert FAKE_KEY not in result_.output
        assert "sk-ant" not in result_.output
        assert FAKE_TEXT not in result_.output

    def test_auto_mode_with_a_key_is_live(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 1)
        live = Live(monkeypatch, ok(1))
        monkeypatch.setenv("REVIEW_RADAR_MODE", "auto")

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 0, result_.output
        assert "Classifier: anthropic/" in result_.output
        assert len(live.received) == 1

    def test_live_without_a_key_is_still_a_one_line_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("REVIEW_RADAR_MODE", "live")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 1
        assert result_.output.startswith("Error: ")
        assert "ANTHROPIC_API_KEY" in result_.output
        assert len(result_.output.strip().splitlines()) == 1

    def test_the_default_factory_builds_an_anthropic_classifier_without_a_network_call(
        self,
    ) -> None:
        classifier = cli.make_anthropic_classifier(api_key=SecretStr(FAKE_KEY), model="m-1")

        assert isinstance(classifier, AnthropicClassifier)
        assert classifier.model == "m-1"
        assert FAKE_KEY not in repr(classifier)

    def test_the_old_not_available_message_is_gone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 1)
        Live(monkeypatch, ok(1))

        result_ = runner.invoke(app, ["analyze"])

        assert "not available yet" not in result_.output
        assert result_.exit_code == 0

    def test_invalid_outputs_are_split_and_the_run_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        Live(
            monkeypatch,
            make_message("not json"),
            make_message([result("r1")], input_tokens=1000, output_tokens=200),
            make_message([result("r2")], input_tokens=1000, output_tokens=200),
        )

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 0, result_.output
        assert "Analyzed now: 2" in result_.output
        assert "Tokens: 2100 in / 420 out" in result_.output

    def test_item_failures_are_listed_by_reason_and_exit_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 1)
        Live(monkeypatch, make_message("not json"))

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 1
        assert "Failed: 1" in result_.output
        assert "  invalid_output: 1" in result_.output
        assert FAKE_TEXT not in result_.output


class TestUsageAndCostLines:
    def test_tokens_and_cost_are_printed_after_the_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        Live(monkeypatch, ok(2))

        lines = runner.invoke(app, ["analyze"]).output.splitlines()

        assert lines[-2:] == [
            "Tokens: 1000 in / 200 out",
            "Estimated cost: $0.000200 (claude-haiku-5-5 at $0.10/$0.50 per 1M tokens)",
        ]
        assert lines.index("Failed: 0") < lines.index("Tokens: 1000 in / 200 out")

    def test_sonnet_prices_are_shown(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ingest_many(tmp_path, 1)
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        Live(monkeypatch, ok(1, input_tokens=39_000, output_tokens=9_300))

        output = runner.invoke(app, ["analyze"]).output

        assert (
            "Estimated cost: $0.171000 (claude-sonnet-5-5 at $2.00/$10.00 per 1M tokens)" in output
        )

    def test_an_unknown_model_has_no_price(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 1)
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-future-9")
        Live(monkeypatch, ok(1))

        output = runner.invoke(app, ["analyze"]).output

        assert "Estimated cost: unknown (no price for claude-future-9)" in output
        assert "Tokens: 1000 in / 200 out" in output

    def test_a_fully_cached_live_run_costs_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        live = Live(monkeypatch, ok(2))
        runner.invoke(app, ["analyze"])

        result_ = runner.invoke(app, ["analyze"])

        assert len(live.client.calls) == 1
        assert "From cache: 2" in result_.output
        assert "Tokens: 0 in / 0 out" in result_.output
        assert "Estimated cost: $0.000000 (claude-haiku-5-5 at $0.10/$0.50 per 1M tokens)" in (
            result_.output
        )

    def test_demo_shows_zero_cost(self, tmp_path: Path) -> None:
        ingest_many(tmp_path, 2)

        lines = runner.invoke(app, ["analyze"]).output.splitlines()

        assert lines[-2:] == ["Tokens: 0 in / 0 out", "Estimated cost: $0 (demo)"]


class TestRunsAreRecorded:
    def test_a_live_run_writes_one_row(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        Live(monkeypatch, ok(2))

        runner.invoke(app, ["analyze"])

        (run,) = runs()
        assert (run.provider, run.model) == ("anthropic", "claude-haiku-5-5")
        assert re.fullmatch(r"anthropic-1\+[0-9a-f]{8}", run.prompt_version)
        assert (run.reviews_total, run.analyzed, run.cached, run.failed) == (2, 2, 0, 0)
        assert (run.input_tokens, run.output_tokens) == (1000, 200)
        assert run.cost_usd == Decimal("0.0002")
        assert run.fatal_reason is None
        assert run.started_at <= run.finished_at

    def test_each_run_adds_a_row_including_cached_ones(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        Live(monkeypatch, ok(2))
        runner.invoke(app, ["analyze"])
        runner.invoke(app, ["analyze"])

        first, second = runs()

        assert (second.analyzed, second.cached, second.input_tokens) == (0, 2, 0)
        assert second.cost_usd == Decimal(0)
        assert first.analyzed == 2

    def test_an_unknown_model_stores_no_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 1)
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-future-9")
        Live(monkeypatch, ok(1))

        runner.invoke(app, ["analyze"])

        assert runs()[0].cost_usd is None

    def test_a_demo_run_is_recorded_with_zero_cost(self, tmp_path: Path) -> None:
        ingest_many(tmp_path, 2)

        runner.invoke(app, ["analyze"])

        (run,) = runs()
        assert (run.provider, run.model) == ("fixture", "demo")
        assert run.cost_usd == Decimal(0)
        assert (run.input_tokens, run.output_tokens) == (0, 0)

    def test_an_empty_database_records_nothing(self) -> None:
        runner.invoke(app, ["analyze"])

        assert runs() == []

    def test_a_fatal_run_is_recorded_with_its_reason_and_partial_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 45)
        Live(monkeypatch, ok(20), status_error(402))

        runner.invoke(app, ["analyze"])

        (run,) = runs()
        assert run.fatal_reason == "billing_error"
        assert (run.reviews_total, run.analyzed, run.failed) == (45, 20, 25)
        assert (run.input_tokens, run.output_tokens) == (1000, 200)

    def test_an_aborted_run_is_recorded_with_partial_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 45)

        class Exploding:
            provider = "fixture"
            model = "demo"
            prompt_version = "fixture-1+x"
            calls = 0

            def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
                Exploding.calls += 1
                if Exploding.calls == 2:
                    raise RuntimeError("boom")
                return ClassifyResult(analyses={i.item_id: POSITIVE for i in items})

        monkeypatch.setattr(cli.FixtureClassifier, "from_file", lambda _p: Exploding())

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 1
        (run,) = runs()
        assert (run.analyzed, run.failed, run.fatal_reason) == (20, 25, None)

    def test_a_failure_to_save_the_run_is_a_clean_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        monkeypatch.setattr(cli, "save_run", broken_save_run)

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 1
        assert "Error: Cannot save the run: disk full." in result_.output
        assert "Traceback" not in result_.output

    def test_a_failure_to_save_the_run_does_not_hide_an_abort(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)

        class Exploding:
            provider = "fixture"
            model = "demo"
            prompt_version = "fixture-1+x"

            def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
                raise RuntimeError("boom")

        monkeypatch.setattr(cli.FixtureClassifier, "from_file", lambda _p: Exploding())
        monkeypatch.setattr(cli, "save_run", broken_save_run)

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 1
        assert "Error: analysis stopped: RuntimeError." in result_.output
        assert "Failed: 2" in result_.output
        assert "Error: Cannot save the run: disk full." in result_.output
        assert "Traceback" not in result_.output


class TestFatalStop:
    def test_live_401_prints_the_partial_report_then_the_error_and_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 45)
        live = Live(monkeypatch, status_error(401))

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 1
        lines = result_.output.splitlines()
        assert "Reviews: 45" in lines
        assert "Analyzed now: 0" in lines
        assert "Failed: 45" in lines
        assert "  authentication_failed: 20" in lines
        assert "  skipped_after_fatal: 25" in lines
        assert lines[-1].startswith(
            "Error: The Anthropic API key was rejected. Check ANTHROPIC_API_KEY."
        )
        assert len(live.client.calls) == 1
        assert live.sleep.delays == []

    def test_the_error_line_comes_after_the_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 2)
        Live(monkeypatch, status_error(401))

        output = runner.invoke(app, ["analyze"]).output

        assert output.index("Classifier:") < output.index("Failed:") < output.index("Error:")

    def test_a_fatal_stop_with_a_fake_classifier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 45)

        class Fatal:
            provider = "fixture"
            model = "demo"
            prompt_version = "fixture-1+fatal"
            calls = 0

            def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
                Fatal.calls += 1
                if Fatal.calls == 2:
                    raise FatalClassifierError("billing_error", "No credit left.")
                return ClassifyResult(analyses={i.item_id: POSITIVE for i in items})

        monkeypatch.setattr(cli.FixtureClassifier, "from_file", lambda _p: Fatal())

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 1
        assert Fatal.calls == 2
        assert "Analyzed now: 20" in result_.output
        assert "  billing_error: 20" in result_.output
        assert "  skipped_after_fatal: 5" in result_.output
        assert result_.output.strip().splitlines()[-1] == "Error: No credit left."

    def test_a_transient_error_is_not_a_fatal_stop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 25)

        class Flaky:
            provider = "fixture"
            model = "demo"
            prompt_version = "fixture-1+flaky"
            calls = 0

            def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
                Flaky.calls += 1
                if Flaky.calls == 1:
                    raise ClassifierError("rate_limited", "Slow down.")
                return ClassifyResult(analyses={i.item_id: POSITIVE for i in items})

        monkeypatch.setattr(cli.FixtureClassifier, "from_file", lambda _p: Flaky())

        result_ = runner.invoke(app, ["analyze"])

        assert result_.exit_code == 1
        assert Flaky.calls == 2
        assert "  rate_limited: 20" in result_.output
        assert "skipped_after_fatal" not in result_.output
        assert "Error:" not in result_.output


class TestLimit:
    def test_limit_classifies_only_the_first_n_and_reports_the_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ingest_many(tmp_path, 5)
        live = Live(monkeypatch, ok(2))

        result_ = runner.invoke(app, ["analyze", "--limit", "2"])

        assert result_.exit_code == 0, result_.output
        assert "Analyzed now: 2" in result_.output
        assert "Failed: 0" in result_.output
        assert "Not analyzed (limit): 3" in result_.output
        sent = live.client.calls[0]["messages"][0]["content"]
        assert "item number 0" in sent
        assert "item number 1" in sent
        assert "item number 2" not in sent

    def test_the_line_is_absent_without_a_limit(self, tmp_path: Path) -> None:
        ingest_many(tmp_path, 3)

        assert "Not analyzed" not in runner.invoke(app, ["analyze"]).output

    def test_a_limit_above_the_pending_count_prints_no_limit_line(self, tmp_path: Path) -> None:
        ingest_many(tmp_path, 3)

        result_ = runner.invoke(app, ["analyze", "--limit", "10"])

        assert result_.exit_code == 0
        assert "Not analyzed" not in result_.output

    def test_a_second_run_with_limit_continues_with_the_rest(self, tmp_path: Path) -> None:
        ingest_many(tmp_path, 5)
        runner.invoke(app, ["analyze", "--limit", "2"])

        result_ = runner.invoke(app, ["analyze", "--limit", "2"])

        assert "Analyzed now: 2" in result_.output
        assert "From cache: 2" in result_.output
        assert "Not analyzed (limit): 1" in result_.output

    @pytest.mark.parametrize("bad", ["0", "-3", "abc"])
    def test_an_invalid_limit_is_a_usage_error(self, tmp_path: Path, bad: str) -> None:
        ingest_many(tmp_path, 2)

        result_ = runner.invoke(app, ["analyze", "--limit", bad])

        assert result_.exit_code == 2
        assert "Traceback" not in result_.output

    def test_the_run_row_counts_only_what_was_processed(self, tmp_path: Path) -> None:
        ingest_many(tmp_path, 5)

        runner.invoke(app, ["analyze", "--limit", "2"])

        (run,) = runs()
        assert (run.reviews_total, run.analyzed, run.cached, run.failed) == (5, 2, 0, 0)
