"""`review-radar run`: the whole pipeline in one command (demo, and live with fakes only).

No network and no real key. Telegram goes through a mock transport (live) or is forbidden
(demo). All reviews, tokens and chat ids are fictional.
"""

import contextlib
import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import jinja2
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner, Result

from anthropic_fakes import FAKE_KEY, FakeClient, make_message, result, status_error
from review_radar import cli
from review_radar.anthropic_classifier import AnthropicClassifier
from review_radar.cli import app
from review_radar.storage import connect, count_reviews, list_summaries
from review_radar.telegram import TelegramSender

runner = CliRunner()
TOKEN = "123456:FAKE-token_value"  # noqa: S105 (fictional)
CHAT_ID = "123456789"
DAY = "2026-11-10"
LATEST_SAMPLE_DAY = "2026-11-18"
MARKER = "SECRET-REVIEW-TEXT"
STEPS = ("ingest", "analyze", "summary", "dashboard")


def main_db() -> Path:
    return Path(os.environ["DATABASE_PATH"])


def demo_db() -> Path:
    return main_db().with_name(f"{main_db().stem}-demo.db")


def count_rows(path: Path, table: str) -> int:
    with contextlib.closing(connect(path)) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # noqa: S608


def run_cmd(*args: str) -> Result:
    return runner.invoke(app, ["run", *args])


def step_positions(output: str) -> list[int]:
    lowered = output.lower()
    positions = [lowered.find(step) for step in STEPS]
    assert all(position >= 0 for position in positions), f"a step is not shown: {output}"
    return positions


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("no network in tests")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "site-out"


class TestDemo:
    @pytest.fixture
    def hostile_env(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """A live-looking setup: key, Telegram credentials. Nothing of it may be used."""
        monkeypatch.setenv("REVIEW_RADAR_MODE", "live")
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT_ID)
        attempts: list[str] = []

        def spy(name: str) -> Callable[..., None]:
            def inner(*_args: object, **_kwargs: object) -> None:
                attempts.append(name)
                raise AssertionError(f"demo mode must not use {name}")

            return inner

        monkeypatch.setattr(cli, "make_telegram_sender", spy("telegram"))
        monkeypatch.setattr(cli, "make_anthropic_classifier", spy("anthropic"))
        monkeypatch.setattr(cli, "make_eval_client", spy("eval client"), raising=False)
        return attempts

    def test_runs_every_step_and_writes_the_page(self, out: Path) -> None:
        result_ = run_cmd("--demo", "--out", str(out))

        assert result_.exit_code == 0, result_.output
        positions = step_positions(result_.output)
        assert positions == sorted(positions)
        assert "[DEMO]" in result_.output
        page = out / "index.html"
        assert page.is_file()
        assert str(page) in result_.output.replace("\n", "")
        assert "Demo data" in page.read_text(encoding="utf-8")

    def test_works_when_a_key_and_telegram_credentials_are_configured(
        self, out: Path, hostile_env: list[str]
    ) -> None:
        result_ = run_cmd("--demo", "--out", str(out))

        assert result_.exit_code == 0, result_.output
        assert hostile_env == []  # no Telegram sender, no Anthropic client
        assert "[DEMO]" in result_.output
        assert "Demo data" in (out / "index.html").read_text(encoding="utf-8")
        assert FAKE_KEY not in result_.output
        assert TOKEN not in result_.output

    def test_uses_its_own_database_and_never_creates_the_main_one(
        self, out: Path, hostile_env: list[str]
    ) -> None:
        assert run_cmd("--demo", "--out", str(out)).exit_code == 0

        assert not main_db().exists()
        assert demo_db().is_file()
        assert demo_db().name == f"{main_db().stem}-demo.db"
        with contextlib.closing(connect(demo_db())) as conn:
            assert count_reviews(conn) == 300  # the sample CSV
            assert list_summaries(conn) == []  # a dry run records nothing

    def test_the_summary_is_a_dry_run_for_the_latest_review_date(self, out: Path) -> None:
        result_ = run_cmd("--demo", "--out", str(out))

        assert result_.exit_code == 0, result_.output
        assert LATEST_SAMPLE_DAY in result_.output
        assert "dry run" in result_.output.lower()

    def test_date_selects_the_day_of_the_summary(self, out: Path) -> None:
        result_ = run_cmd("--demo", "--date", "2026-11-12", "--out", str(out))

        assert result_.exit_code == 0, result_.output
        assert "2026-11-12" in result_.output
        assert LATEST_SAMPLE_DAY not in result_.output

    def test_a_second_run_is_idempotent(self, out: Path) -> None:
        first = run_cmd("--demo", "--out", str(out))
        reviews, analyses = count_rows(demo_db(), "reviews"), count_rows(demo_db(), "analyses")

        second = run_cmd("--demo", "--out", str(out))

        assert first.exit_code == second.exit_code == 0, second.output
        assert count_rows(demo_db(), "reviews") == reviews == 300
        assert count_rows(demo_db(), "analyses") == analyses > 0
        assert (out / "index.html").is_file()

    def test_environment_variables_are_restored(self, out: Path, hostile_env: list[str]) -> None:
        before = dict(os.environ)

        assert run_cmd("--demo", "--out", str(out)).exit_code == 0

        assert dict(os.environ) == before

    def test_environment_variables_are_restored_after_a_failure(
        self, tmp_path: Path, hostile_env: list[str]
    ) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, not a folder", encoding="utf-8")
        before = dict(os.environ)

        result_ = run_cmd("--demo", "--out", str(blocker))

        assert result_.exit_code != 0
        assert demo_db().is_file()  # the run got as far as the dashboard step
        assert dict(os.environ) == before
        assert hostile_env == []

    def test_a_bad_date_is_a_usage_error(self, out: Path) -> None:
        result_ = run_cmd("--demo", "--date", "12/11/2026", "--out", str(out))

        assert result_.exit_code == 2
        assert "No such command" not in result_.output  # the command exists; the date is bad
        assert not demo_db().exists()


class TestWithoutCsvOrDemo:
    def test_is_a_clear_error_and_does_nothing(self, out: Path) -> None:
        result_ = run_cmd("--out", str(out))

        assert result_.exit_code in (1, 2)
        assert "csv" in result_.output.lower() or "--demo" in result_.output
        assert not main_db().exists()
        assert not demo_db().exists()
        assert not out.exists()


def write_csv(path: Path) -> Path:
    rows = "".join(
        f'x{i},{DAY},2,"{MARKER} number {i}: the invoice charged me twice",web\n' for i in (1, 2, 3)
    )
    path.write_text("review_id,review_date,rating,text,source\n" + rows, encoding="utf-8")
    return path


class LiveSetup:
    """Live mode with a fake classifier client and a fake Telegram transport."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *script: Any, out: Path) -> None:
        monkeypatch.setenv("REVIEW_RADAR_MODE", "live")
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT_ID)
        self.client = FakeClient(*script)
        self.requests: list[httpx2.Request] = []
        self.page_existed_when_sent: list[bool] = []
        self.reply: Callable[[httpx2.Request], httpx2.Response] = lambda _: httpx2.Response(
            200, json={"ok": True, "result": {"message_id": 42}}
        )

        def handle(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            self.page_existed_when_sent.append((out / "index.html").exists())
            return self.reply(request)

        def make_classifier(*, api_key: SecretStr, model: str) -> AnthropicClassifier:
            return AnthropicClassifier(
                api_key=api_key, model=model, client=self.client, sleep=lambda _: None
            )

        def make_sender(*, token: SecretStr, chat_id: SecretStr) -> TelegramSender:
            return TelegramSender(
                token=token, chat_id=chat_id, transport=httpx2.MockTransport(handle)
            )

        monkeypatch.setattr(cli, "make_anthropic_classifier", make_classifier)
        monkeypatch.setattr(cli, "make_telegram_sender", make_sender)


def three_ok() -> Any:
    return make_message([result("r1", "negative", ["billing"]), result("r2"), result("r3")])


def live_run(csv_path: Path, out: Path) -> Result:
    return run_cmd(str(csv_path), "--date", DAY, "--out", str(out))


class TestLiveWithCsv:
    def test_runs_ingest_analyze_summary_and_dashboard_in_order(
        self, tmp_path: Path, out: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setup = LiveSetup(monkeypatch, three_ok(), out=out)

        result_ = live_run(write_csv(tmp_path / "reviews.csv"), out)

        assert result_.exit_code == 0, result_.output
        positions = step_positions(result_.output)
        assert positions == sorted(positions)
        assert "[DEMO]" not in result_.output
        assert len(setup.client.calls) == 1
        assert len(setup.requests) == 1  # the summary was really sent (to the fake transport)
        assert setup.page_existed_when_sent == [False]  # the dashboard comes after the summary
        assert (out / "index.html").is_file()
        with contextlib.closing(connect(main_db())) as conn:
            assert count_reviews(conn) == 3
            assert [row.status for row in list_summaries(conn)] == ["sent"]
        assert not demo_db().exists()
        for secret in (FAKE_KEY, TOKEN, MARKER):
            assert secret not in result_.output

    def test_a_missing_csv_stops_at_ingest(
        self, tmp_path: Path, out: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setup = LiveSetup(monkeypatch, three_ok(), out=out)

        result_ = live_run(tmp_path / "missing.csv", out)

        assert result_.exit_code != 0
        assert "ingest" in result_.stderr.lower()
        assert setup.client.calls == []
        assert setup.requests == []
        assert not (out / "index.html").exists()

    def test_a_failed_analysis_stops_before_the_summary(
        self, tmp_path: Path, out: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setup = LiveSetup(monkeypatch, status_error(401), out=out)

        result_ = live_run(write_csv(tmp_path / "reviews.csv"), out)

        assert result_.exit_code != 0
        assert "analyze" in result_.stderr.lower()
        assert len(setup.client.calls) == 1
        assert setup.requests == []
        assert not (out / "index.html").exists()
        assert FAKE_KEY not in result_.output
        assert MARKER not in result_.output

    def test_a_failed_summary_stops_before_the_dashboard(
        self, tmp_path: Path, out: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setup = LiveSetup(monkeypatch, three_ok(), out=out)
        setup.reply = lambda _: httpx2.Response(
            400, json={"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}
        )

        result_ = live_run(write_csv(tmp_path / "reviews.csv"), out)

        assert result_.exit_code != 0
        assert "summary" in result_.stderr.lower()
        assert len(setup.requests) == 1
        assert not (out / "index.html").exists()
        assert TOKEN not in result_.output

    def test_a_failed_dashboard_is_named_after_the_earlier_steps_succeeded(
        self, tmp_path: Path, out: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        setup = LiveSetup(monkeypatch, three_ok(), out=out)

        def broken(_: object) -> str:
            raise jinja2.TemplateNotFound("dashboard.html.j2")

        monkeypatch.setattr(cli, "render_dashboard", broken)

        result_ = live_run(write_csv(tmp_path / "reviews.csv"), out)

        assert result_.exit_code != 0
        assert "dashboard" in result_.stderr.lower()
        assert len(setup.requests) == 1  # the summary step had already finished
        assert not (out / "index.html").exists()
