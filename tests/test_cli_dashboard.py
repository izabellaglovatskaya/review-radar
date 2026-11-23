"""`review-radar dashboard`: output, exit codes, demo labelling and failures. No network."""

import contextlib
import datetime as dt
import os
import socket
from pathlib import Path

import jinja2
import pytest
from typer.testing import CliRunner, Result

from review_radar import cli, dashboard
from review_radar.cli import app
from review_radar.config import Settings
from review_radar.hashing import text_hash
from review_radar.models import Analysis, Review
from review_radar.storage import PendingText, connect, save_analyses, save_reviews

runner = CliRunner()
MARKER = "SECRET-REVIEW-TEXT"
BAD_URL = "https://bad.example/x"
DAY = "2026-11-10"
OLD_PAGE = "<html>previous dashboard</html>"


def db() -> Path:
    return Path(os.environ["DATABASE_PATH"])


def seed(*, analyzed: int = 3, pending: int = 0, day: str = DAY) -> None:
    key = cli._analysis_key(Settings(_env_file=None))  # type: ignore[call-arg]
    date = dt.date.fromisoformat(day)
    with contextlib.closing(connect(db())) as conn:
        for index in range(analyzed + pending):
            text = f"{MARKER} {day} {index} {BAD_URL}"
            review = Review(
                external_id=f"{day}-{index}", review_date=date, rating=2, text=text, source="web"
            )
            save_reviews(conn, [review])
            if index < analyzed:
                sentiment = "negative" if index == 0 else "positive"
                topics = ["billing"] if index == 0 else []
                item = Analysis(sentiment=sentiment, topics=topics, confidence=0.9)
                save_analyses(conn, [(PendingText(text_hash(text), text, 2, 1), item)], **key)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command must neither build an API client nor open a socket."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the dashboard command must not use the network")

    monkeypatch.setattr(cli, "_build_classifier", forbidden)
    monkeypatch.setattr(cli, "make_telegram_sender", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "site-out"


@pytest.fixture
def old_page(out: Path) -> Path:
    out.mkdir()
    page = out / "index.html"
    page.write_text(OLD_PAGE, encoding="utf-8")
    return page


def run(out: Path) -> Result:
    return runner.invoke(app, ["dashboard", "--out", str(out)])


def go_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_RADAR_MODE", "live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-FAKE-test")


class TestSuccess:
    def test_writes_the_page_and_prints_its_path(self, out: Path) -> None:
        seed()

        result = run(out)

        assert result.exit_code == 0, result.output
        page = out / "index.html"
        assert page.is_file()
        assert "<html" in page.read_text(encoding="utf-8")
        assert str(page) in result.stdout.replace("\n", "")

    def test_the_assets_are_copied_next_to_the_page(self, out: Path) -> None:
        seed()

        assert run(out).exit_code == 0

        assert (out / "assets" / "vendor" / "chart-4.5.1.umd.min.js").is_file()

    def test_works_without_any_credentials(self, out: Path) -> None:
        seed()

        result = run(out)

        assert result.exit_code == 0, result.output

    def test_output_and_page_hold_no_review_text(self, out: Path) -> None:
        seed()

        result = run(out)

        html = (out / "index.html").read_text(encoding="utf-8")
        for text in (result.output, html):
            assert MARKER not in text
            assert "bad.example" not in text

    def test_a_rerun_replaces_the_previous_page(self, out: Path, old_page: Path) -> None:
        seed()

        assert run(out).exit_code == 0

        assert old_page.read_text(encoding="utf-8") != OLD_PAGE


class TestModeLabelling:
    def test_a_demo_page_is_labelled(self, out: Path) -> None:
        seed()

        assert run(out).exit_code == 0

        assert "Demo data" in (out / "index.html").read_text(encoding="utf-8")

    def test_a_live_page_is_not_labelled(self, out: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        go_live(monkeypatch)
        seed()

        result = run(out)

        assert result.exit_code == 0, result.output
        assert "demo data" not in (out / "index.html").read_text(encoding="utf-8").casefold()

    def test_live_mode_reads_only_analyses_of_the_live_key(
        self, out: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed()  # stored under the demo (fixture) key
        go_live(monkeypatch)

        result = run(out)

        assert result.exit_code == 1
        assert "analyze" in result.output

    def test_a_fake_key_never_builds_a_client(
        self, out: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        go_live(monkeypatch)
        seed()

        # The autouse guard raises if `_build_classifier` is called or a socket is opened.
        assert run(out).exit_code == 0


class TestNothingToShow:
    def test_no_reviews_exits_3_and_names_ingest(self, out: Path) -> None:
        result = run(out)

        assert result.exit_code == 3
        assert "ingest" in result.stderr
        assert not (out / "index.html").exists()

    def test_no_reviews_keeps_an_existing_page(self, out: Path, old_page: Path) -> None:
        result = run(out)

        assert result.exit_code == 3
        assert old_page.read_text(encoding="utf-8") == OLD_PAGE

    def test_reviews_but_none_analyzed_exits_1_and_names_analyze(self, out: Path) -> None:
        seed(analyzed=0, pending=2)

        result = run(out)

        assert result.exit_code == 1
        assert "analyze" in result.stderr
        assert not (out / "index.html").exists()

    def test_none_analyzed_keeps_an_existing_page(self, out: Path, old_page: Path) -> None:
        seed(analyzed=0, pending=2)

        result = run(out)

        assert result.exit_code == 1
        assert old_page.read_text(encoding="utf-8") == OLD_PAGE


class TestPartialAnalysis:
    def test_writes_the_page_and_warns_with_the_count(self, out: Path) -> None:
        seed(analyzed=2, pending=3)

        result = run(out)

        assert result.exit_code == 0, result.output
        assert (out / "index.html").is_file()
        assert "3" in result.stderr
        assert "not analyzed" in result.stderr.lower()

    def test_a_fully_analyzed_run_has_no_warning(self, out: Path) -> None:
        seed()

        result = run(out)

        assert result.exit_code == 0, result.output
        assert "not analyzed" not in result.stderr.lower()


class TestFailures:
    def test_an_unwritable_output_is_a_clean_error(self, tmp_path: Path) -> None:
        seed()
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, not a directory", encoding="utf-8")

        result = run(blocker)

        assert result.exit_code == 1
        assert "Error" in result.stderr
        assert "Traceback" not in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert MARKER not in result.output

    @pytest.mark.parametrize(
        "error",
        [
            jinja2.TemplateNotFound("dashboard.html.j2"),
            jinja2.UndefinedError(f"missing {MARKER}"),
            jinja2.TemplateSyntaxError(f"bad {MARKER}", 1),
        ],
    )
    def test_a_template_failure_is_a_clean_error(
        self,
        out: Path,
        old_page: Path,
        monkeypatch: pytest.MonkeyPatch,
        error: jinja2.TemplateError,
    ) -> None:
        seed()

        def broken(_: object) -> str:
            raise error

        monkeypatch.setattr(cli, "render_dashboard", broken)

        result = run(out)

        assert result.exit_code == 1
        assert "template" in result.stderr
        assert isinstance(result.exception, SystemExit)
        assert MARKER not in result.output
        assert old_page.read_text(encoding="utf-8") == OLD_PAGE

    def test_missing_assets_are_a_clean_error(
        self, out: Path, old_page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed()
        monkeypatch.setattr(dashboard, "SITE_DIR", tmp_path / "no-assets")

        result = run(out)

        assert result.exit_code == 1
        assert "assets/dashboard.js" in result.stderr
        assert "checkout" in result.stderr
        assert old_page.read_text(encoding="utf-8") == OLD_PAGE

    def test_a_corrupt_database_row_is_a_clean_error(self, out: Path) -> None:
        seed()
        with contextlib.closing(connect(db())) as conn:
            conn.execute("UPDATE analyses SET topics = 'not json'")
            conn.commit()

        result = run(out)

        assert result.exit_code == 1
        assert "Traceback" not in result.output
        assert MARKER not in result.output
        assert not (out / "index.html").exists()
