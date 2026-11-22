import contextlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from review_radar import cli
from review_radar.classifier import ClassifierError, ClassifyItem, ClassifyResult
from review_radar.cli import app
from review_radar.fixture_classifier import FixtureClassifier
from review_radar.hashing import text_hash
from review_radar.ingest import read_reviews
from review_radar.storage import connect, load_analyses

runner = CliRunner()

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews.csv"
SAMPLE_TRUTH = ROOT / "data" / "sample" / "ground_truth.json"
FAKE_TEXT = "ZEBRA-QUOKKA-9917 distinctive fake review"


def demo_version() -> str:
    return FixtureClassifier.from_file(cli.DEMO_FIXTURES).prompt_version


def db_path() -> Path:
    return Path(os.environ["DATABASE_PATH"])


def ingest_text(tmp_path: Path, content: str) -> None:
    path = tmp_path / "in.csv"
    path.write_text(content, encoding="utf-8")
    assert runner.invoke(app, ["ingest", str(path)]).exit_code == 0


def test_sample_ingest_then_analyze(tmp_path: Path) -> None:
    runner.invoke(app, ["ingest", str(SAMPLE_CSV)])

    result = runner.invoke(app, ["analyze"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert re.fullmatch(r"Classifier: fixture/demo \(prompt fixture-1\+[0-9a-f]{8}\)", lines[0])
    assert lines[1:] == [
        "Reviews: 300",
        "Analyzed now: 300",
        "From cache: 0",
        "Failed: 0",
        "Tokens: 0 in / 0 out",
        "Estimated cost: $0 (demo)",
    ]


def test_second_analyze_is_all_cache() -> None:
    runner.invoke(app, ["ingest", str(SAMPLE_CSV)])
    runner.invoke(app, ["analyze"])

    result = runner.invoke(app, ["analyze"])

    assert result.exit_code == 0
    assert "Analyzed now: 0" in result.output
    assert "From cache: 300" in result.output
    assert "Failed: 0" in result.output


def test_demo_on_the_sample_reproduces_the_ground_truth() -> None:
    runner.invoke(app, ["ingest", str(SAMPLE_CSV)])
    runner.invoke(app, ["analyze"])
    truth = json.loads(SAMPLE_TRUTH.read_text(encoding="utf-8"))["reviews"]

    with contextlib.closing(connect(db_path())) as conn:
        stored = load_analyses(
            conn, provider="fixture", model="demo", prompt_version=demo_version()
        )
    for review in read_reviews(SAMPLE_CSV).reviews:
        expected = truth[review.external_id or ""]
        analysis = stored[text_hash(review.text)]
        assert analysis.sentiment.value == expected["sentiment"]
        assert [t.value for t in analysis.topics] == expected["topics"]


def test_unknown_texts_use_the_fallback(tmp_path: Path) -> None:
    ingest_text(
        tmp_path,
        "review_date,rating,text\n"
        f"2026-11-10,1,{FAKE_TEXT} it was late\n"
        "2026-11-10,5,Lovely thing\n"
        "2026-11-10,,Arrived on Tuesday\n",
    )

    result = runner.invoke(app, ["analyze"])

    assert result.exit_code == 0, result.output
    assert "Reviews: 3" in result.output
    assert "Analyzed now: 3" in result.output
    assert "Failed: 0" in result.output
    with contextlib.closing(connect(db_path())) as conn:
        stored = load_analyses(
            conn, provider="fixture", model="demo", prompt_version=demo_version()
        )
    assert len(stored) == 3
    assert all(a.confidence == 0.5 for a in stored.values())


def test_output_never_contains_review_text(tmp_path: Path) -> None:
    ingest_text(tmp_path, f"review_date,rating,text\n2026-11-10,1,{FAKE_TEXT}\n")

    result = runner.invoke(app, ["analyze"])

    assert FAKE_TEXT not in result.output
    assert "ZEBRA" not in result.output


def test_live_without_a_key_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_RADAR_MODE", "live")

    result = runner.invoke(app, ["analyze"])

    assert result.exit_code == 1
    assert result.output.startswith("Error: ")
    assert "ANTHROPIC_API_KEY" in result.output
    assert not db_path().exists()


def test_invalid_setting_is_named_without_its_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_RADAR_MODE", "bogus-value")

    result = runner.invoke(app, ["analyze"])

    assert result.exit_code == 1
    assert "Error: Invalid setting(s): REVIEW_RADAR_MODE." in result.output
    assert "bogus-value" not in result.output


def test_empty_database_asks_for_ingest() -> None:
    result = runner.invoke(app, ["analyze"])

    assert result.exit_code == 1
    assert "No reviews in the database; run ingest first." in result.output
    assert "Traceback" not in result.output


def test_corrupt_database_is_a_clean_error() -> None:
    db_path().write_bytes(os.urandom(2048))

    result = runner.invoke(app, ["analyze"])

    assert result.exit_code == 1
    assert result.output.startswith("Error: ")
    assert "Traceback" not in result.output


def test_missing_fixture_file_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingest_text(tmp_path, "review_date,text\n2026-11-10,Fine\n")
    monkeypatch.setattr(cli, "DEMO_FIXTURES", tmp_path / "missing.json")

    result = runner.invoke(app, ["analyze"])

    assert result.exit_code == 1
    assert result.output.startswith("Error: ")
    assert "missing.json" in result.output
    assert "Traceback" not in result.output


class FailingClassifier:
    provider = "fixture"
    model = "demo"
    prompt_version = "fixture-1"

    def classify(self, items: Sequence[ClassifyItem]) -> ClassifyResult:
        raise ClassifierError("rate_limited", "Slow down.")


def test_failures_are_listed_and_exit_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ingest_text(tmp_path, f"review_date,text\n2026-11-10,{FAKE_TEXT}\n2026-11-10,Other\n")
    monkeypatch.setattr(cli.FixtureClassifier, "from_file", lambda _path: FailingClassifier())

    result = runner.invoke(app, ["analyze"])

    assert result.exit_code == 1
    assert "Analyzed now: 0" in result.output
    assert "Failed: 2" in result.output
    assert "  rate_limited: 2" in result.output
    assert FAKE_TEXT not in result.output
