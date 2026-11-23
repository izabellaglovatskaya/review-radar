"""`review-radar summary`: dry run, credentials, idempotency and failures. No network."""

import contextlib
import datetime as dt
import logging
import os
from collections.abc import Callable
from pathlib import Path

import httpx2
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner, Result

from review_radar import cli
from review_radar.cli import app
from review_radar.config import Settings
from review_radar.hashing import text_hash
from review_radar.models import Analysis, Review
from review_radar.storage import (
    PendingText,
    StorageError,
    connect,
    list_summaries,
    save_analyses,
    save_reviews,
)
from review_radar.telegram import TelegramSender

runner = CliRunner()
TOKEN = "123456:FAKE-token_value"  # noqa: S105 (fictional)
CHAT_ID = "123456789"
DAY = "2026-11-10"
MARKER = "SECRET-REVIEW-TEXT"

Handler = Callable[[httpx2.Request], httpx2.Response]


def db() -> Path:
    return Path(os.environ["DATABASE_PATH"])


def seed(*, analyzed: int = 3, pending: int = 0, day: str = DAY) -> None:
    key = cli._analysis_key(Settings(_env_file=None))  # type: ignore[call-arg]
    date = dt.date.fromisoformat(day)
    with contextlib.closing(connect(db())) as conn:
        for index in range(analyzed + pending):
            text = f"{MARKER} {day} {index} https://bad.example"
            review = Review(
                external_id=f"{day}-{index}", review_date=date, rating=2, text=text, source="web"
            )
            save_reviews(conn, [review])
            if index < analyzed:
                sentiment = "negative" if index == 0 else "positive"
                topics = ["billing"] if index == 0 else []
                item = Analysis(sentiment=sentiment, topics=topics, confidence=0.9)
                save_analyses(conn, [(PendingText(text_hash(text), text, 2, 1), item)], **key)


def statuses() -> list[str]:
    with contextlib.closing(connect(db())) as conn:
        return [row.status for row in list_summaries(conn)]


def ok(_: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json={"ok": True, "result": {"message_id": 42}})


class Telegram:
    """Installs a fake HTTP transport behind the real sender and records the requests."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.requests: list[httpx2.Request] = []
        self.handler: Handler = ok
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT_ID)
        monkeypatch.setenv("REVIEW_RADAR_MODE", "live")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-FAKE-test")

        def handle(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(request)
            return self.handler(request)

        def make(*, token: SecretStr, chat_id: SecretStr) -> TelegramSender:
            return TelegramSender(
                token=token, chat_id=chat_id, transport=httpx2.MockTransport(handle)
            )

        monkeypatch.setattr(cli, "make_telegram_sender", make)


@pytest.fixture
def telegram(monkeypatch: pytest.MonkeyPatch) -> Telegram:
    return Telegram(monkeypatch)


def run(*args: str) -> Result:
    return runner.invoke(app, ["summary", *args])


def assert_no_leak(result: Result) -> None:
    assert TOKEN not in result.output
    assert CHAT_ID not in result.output
    assert "api.telegram.org" not in result.output
    assert MARKER not in result.output
    assert "bad.example" not in result.output


class TestDryRun:
    def test_prints_the_message_and_sends_nothing(self, telegram: Telegram) -> None:
        seed()

        result = run("--date", DAY, "--dry-run")

        assert result.exit_code == 0, result.output
        assert "Reviews: 3" in result.output
        assert "1. Billing: 1" in result.output
        assert "Dry run: nothing sent." in result.output
        assert telegram.requests == []
        assert statuses() == []
        assert_no_leak(result)

    def test_works_without_credentials(self) -> None:
        seed()

        result = run("--date", DAY, "--dry-run")

        assert result.exit_code == 0, result.output
        assert "Dry run: nothing sent." in result.output

    def test_shows_a_partial_summary_with_a_warning(self) -> None:
        seed(analyzed=2, pending=1)

        result = run("--date", DAY, "--dry-run")

        assert result.exit_code == 0, result.output
        assert "Not analyzed: 1" in result.output

    def test_does_not_send_after_a_real_send_either(self, telegram: Telegram) -> None:
        seed()
        assert run("--date", DAY).exit_code == 0
        sent = len(telegram.requests)

        result = run("--date", DAY, "--dry-run")

        assert result.exit_code == 0
        assert len(telegram.requests) == sent


class TestCredentials:
    def test_missing_both_is_a_clear_error(self) -> None:
        seed()

        result = run("--date", DAY)

        assert result.exit_code == 1
        assert "TELEGRAM_BOT_TOKEN" in result.output
        assert "TELEGRAM_CHAT_ID" in result.output
        assert "--dry-run" in result.output
        assert statuses() == []

    @pytest.mark.parametrize(
        ("present", "value", "missing"),
        [
            ("TELEGRAM_BOT_TOKEN", TOKEN, "TELEGRAM_CHAT_ID"),
            ("TELEGRAM_CHAT_ID", CHAT_ID, "TELEGRAM_BOT_TOKEN"),
        ],
    )
    def test_missing_one_is_a_clear_error(
        self, monkeypatch: pytest.MonkeyPatch, present: str, value: str, missing: str
    ) -> None:
        seed()
        monkeypatch.setenv(present, value)

        result = run("--date", DAY)

        assert result.exit_code == 1
        assert missing in result.output
        assert value not in result.output
        assert statuses() == []


class TestSend:
    def test_sends_once_and_records_it(self, telegram: Telegram) -> None:
        seed()

        result = run("--date", DAY)

        assert result.exit_code == 0, result.output
        assert "Sent" in result.output
        assert len(telegram.requests) == 1
        assert statuses() == ["sent"]
        body = telegram.requests[0].content.decode()
        assert MARKER not in body
        assert "bad.example" not in body
        assert_no_leak(result)

    def test_a_second_run_does_not_send(self, telegram: Telegram) -> None:
        seed()
        run("--date", DAY)

        result = run("--date", DAY)

        assert result.exit_code == 0, result.output
        assert "Already sent" in result.output
        assert len(telegram.requests) == 1
        assert statuses() == ["sent"]

    def test_force_sends_again(self, telegram: Telegram) -> None:
        seed()
        run("--date", DAY)

        result = run("--date", DAY, "--force")

        assert result.exit_code == 0, result.output
        assert len(telegram.requests) == 2
        assert statuses() == ["sent", "sent"]

    def test_default_date_is_yesterday(
        self, telegram: Telegram, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed()
        monkeypatch.setattr(cli, "_today", lambda: dt.date(2026, 11, 11))

        result = run("--dry-run")

        assert result.exit_code == 0, result.output
        assert f"Review Radar: {DAY}" in result.output

    def test_a_range_is_summarized(self, telegram: Telegram) -> None:
        seed()
        seed(day="2026-11-11")

        result = run("--from", DAY, "--to", "2026-11-11", "--dry-run")

        assert result.exit_code == 0, result.output
        assert "Reviews: 6" in result.output


class TestDemoMode:
    def test_a_real_send_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        telegram = Telegram(monkeypatch)
        monkeypatch.setenv("REVIEW_RADAR_MODE", "demo")
        seed()

        result = run("--date", DAY, "--force", "--allow-partial")

        assert result.exit_code == 1
        assert "demo" in result.output.lower()
        assert "--dry-run" in result.output
        assert telegram.requests == []
        assert statuses() == []

    def test_auto_mode_without_a_key_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        telegram = Telegram(monkeypatch)
        monkeypatch.setenv("REVIEW_RADAR_MODE", "auto")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        seed()

        result = run("--date", DAY)

        assert result.exit_code == 1
        assert telegram.requests == []

    def test_a_dry_run_is_labelled_demo(self) -> None:
        seed()

        result = run("--date", DAY, "--dry-run")

        assert result.exit_code == 0, result.output
        assert "[DEMO]" in result.output

    def test_a_live_message_has_no_demo_label(self, telegram: Telegram) -> None:
        seed()

        result = run("--date", DAY)

        assert result.exit_code == 0, result.output
        assert "DEMO" not in telegram.requests[0].content.decode()

    def test_the_sent_row_names_the_classifier(self, telegram: Telegram) -> None:
        seed()
        run("--date", DAY)

        with contextlib.closing(connect(db())) as conn:
            row = list_summaries(conn)[0]

        assert (row.provider, row.model) == ("anthropic", "claude-haiku-5-5")
        assert row.prompt_version.startswith("anthropic-1+")


class TestNothingToSend:
    def test_no_reviews_is_a_distinct_exit_code(self, telegram: Telegram) -> None:
        seed()

        result = run("--date", "2026-01-01")

        assert result.exit_code == 3
        assert "No reviews" in result.stderr
        assert "nothing sent" in result.stderr.lower()
        assert "--allow-empty" in result.stderr
        assert telegram.requests == []
        assert statuses() == []

    @pytest.mark.parametrize("extra", [(), ("--dry-run",)])
    def test_allow_empty_exits_zero_and_sends_nothing(
        self, telegram: Telegram, extra: tuple[str, ...]
    ) -> None:
        seed()

        result = run("--date", "2026-01-01", "--allow-empty", *extra)

        assert result.exit_code == 0, result.output
        assert "nothing sent" in result.output.lower()
        assert telegram.requests == []
        assert statuses() == []

    def test_a_dry_run_of_an_empty_day_is_also_code_3(self) -> None:
        seed()

        assert run("--date", "2026-01-01", "--dry-run").exit_code == 3

    def test_a_sent_day_with_a_late_unanalyzed_review_stays_already_sent(
        self, telegram: Telegram
    ) -> None:
        seed(analyzed=2)
        assert run("--date", DAY).exit_code == 0
        with contextlib.closing(connect(db())) as conn:
            late = Review(
                external_id="late",
                review_date=dt.date(2026, 11, 10),
                rating=2,
                text="Late review",
                source="web",
            )
            save_reviews(conn, [late])

        result = run("--date", DAY)

        assert result.exit_code == 0, result.output
        assert "Already sent" in result.output
        assert len(telegram.requests) == 1

    def test_reviews_without_any_analysis_are_an_error(self, telegram: Telegram) -> None:
        seed(analyzed=0, pending=2)

        result = run("--date", DAY, "--allow-partial")

        assert result.exit_code == 1
        assert "analyze" in result.output
        assert telegram.requests == []

    def test_partial_data_is_refused_by_default(self, telegram: Telegram) -> None:
        seed(analyzed=2, pending=1)

        result = run("--date", DAY)

        assert result.exit_code == 1
        assert "--allow-partial" in result.output
        assert telegram.requests == []
        assert statuses() == []

    def test_partial_data_can_be_allowed(self, telegram: Telegram) -> None:
        seed(analyzed=2, pending=1)

        result = run("--date", DAY, "--allow-partial")

        assert result.exit_code == 0, result.output
        assert len(telegram.requests) == 1
        assert "Not analyzed: 1" in telegram.requests[0].content.decode()


class TestFailures:
    def test_a_refusal_is_recorded_and_can_be_retried(self, telegram: Telegram) -> None:
        seed()
        telegram.handler = lambda _: httpx2.Response(401, json={"ok": False, "error_code": 401})

        result = run("--date", DAY)

        assert result.exit_code == 1
        assert "unauthorized" in result.output
        assert "401" in result.output
        assert statuses() == ["failed"]
        assert_no_leak(result)

        telegram.handler = ok
        retried = run("--date", DAY)

        assert retried.exit_code == 0, retried.output
        assert statuses() == ["failed", "sent"]

    def test_an_unknown_delivery_blocks_the_next_run(self, telegram: Telegram) -> None:
        seed()

        def timeout(_: httpx2.Request) -> httpx2.Response:
            raise httpx2.ReadTimeout(f"timed out bot{TOKEN}")

        telegram.handler = timeout

        result = run("--date", DAY)

        assert result.exit_code == 1
        assert "may have been delivered" in result.output
        assert statuses() == ["unknown"]
        assert_no_leak(result)

        telegram.handler = ok
        again = run("--date", DAY)

        assert again.exit_code == 1
        assert "--force" in again.output
        assert len(telegram.requests) == 1

        forced = run("--date", DAY, "--force")

        assert forced.exit_code == 0, forced.output
        assert statuses() == ["unknown", "sent"]

    def test_sent_but_not_recorded_is_loud_and_blocks_a_resend(
        self, telegram: Telegram, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed()

        def broken(*args: object, **kwargs: object) -> None:
            raise StorageError("Cannot record the summary: disk full.")

        with monkeypatch.context() as patch:
            patch.setattr(cli, "finish_summary", broken)
            result = run("--date", DAY)

        assert result.exit_code == 1
        assert "was sent" in result.output
        assert statuses() == ["sending"]

        again = run("--date", DAY)

        assert again.exit_code == 1
        assert "--force" in again.output
        assert len(telegram.requests) == 1

    def test_a_failure_that_cannot_be_recorded_gives_the_right_advice(
        self, telegram: Telegram, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed()
        telegram.handler = lambda _: httpx2.Response(403, json={"ok": False})

        def broken(*args: object, **kwargs: object) -> None:
            raise StorageError("Cannot record the summary: database is locked.")

        with monkeypatch.context() as patch:
            patch.setattr(cli, "finish_summary", broken)
            result = run("--date", DAY)

        assert result.exit_code == 1
        assert "database is locked" in result.output
        assert "--force" in result.output
        assert "run the command again" not in result.output
        assert statuses() == ["sending"]

    def test_ctrl_c_during_send_is_recorded_as_unknown(self, telegram: Telegram) -> None:
        seed()

        def interrupt(_: httpx2.Request) -> httpx2.Response:
            raise KeyboardInterrupt

        telegram.handler = interrupt

        result = run("--date", DAY)

        assert result.exit_code == 130
        assert "may have been delivered" in result.output
        assert "--force" in result.output
        assert statuses() == ["unknown"]
        with contextlib.closing(connect(db())) as conn:
            assert list_summaries(conn)[0].reason == "interrupted"

    def test_an_unexpected_send_error_is_recorded_without_its_text(
        self, telegram: Telegram, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed()

        class Exploding:
            def send(self, _: str) -> int:
                raise RuntimeError(f"boom https://api.telegram.org/bot{TOKEN}/x")

        monkeypatch.setattr(cli, "make_telegram_sender", lambda **_: Exploding())

        result = run("--date", DAY)

        assert result.exit_code == 1
        assert "may have been delivered" in result.output
        assert statuses() == ["unknown"]
        assert_no_leak(result)

    def test_a_token_is_not_logged_with_verbose(
        self, telegram: Telegram, caplog: pytest.LogCaptureFixture
    ) -> None:
        seed()
        caplog.set_level(logging.DEBUG)

        result = runner.invoke(app, ["--verbose", "summary", "--date", DAY])

        assert result.exit_code == 0, result.output
        assert TOKEN not in caplog.text
        assert "api.telegram.org" not in caplog.text
        assert_no_leak(result)


class TestPeriodOptions:
    @pytest.mark.parametrize(
        "args",
        [
            ("--date", DAY, "--from", DAY, "--to", DAY),
            ("--from", DAY),
            ("--to", DAY),
            ("--from", "2026-11-11", "--to", DAY),
        ],
    )
    def test_invalid_combinations_are_rejected(self, args: tuple[str, ...]) -> None:
        seed()

        result = run(*args, "--dry-run")

        assert result.exit_code != 0

    def test_a_bad_date_is_rejected(self) -> None:
        assert run("--date", "10.11.2026", "--dry-run").exit_code != 0
