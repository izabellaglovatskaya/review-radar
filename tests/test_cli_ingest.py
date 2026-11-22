import contextlib
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from review_radar.cli import app
from review_radar.storage import connect, count_reviews

runner = CliRunner()

HEADER = b"review_id,review_date,rating,text,source\n"
FAKE_TEXT = "ZEBRA-QUOKKA-9917 distinctive fake review"
SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "sample" / "reviews.csv"


def stored_count() -> int:
    with contextlib.closing(connect(Path(os.environ["DATABASE_PATH"]))) as conn:
        return count_reviews(conn)


def write(tmp_path: Path, content: bytes) -> Path:
    path = tmp_path / "reviews.csv"
    path.write_bytes(content)
    return path


def test_dry_run_prints_summary(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Great,web\nr2,2026-11-10,4,Good,app\n")

    result = runner.invoke(app, ["ingest", str(path), "--dry-run"])

    assert result.exit_code == 0
    assert "Rows read: 2" in result.output
    assert "Valid: 2" in result.output
    assert "Skipped: 0" in result.output


def test_skipped_rows_listed_per_reason_with_lines(tmp_path: Path) -> None:
    content = (
        HEADER
        + b"r1,2026-11-10,5,Great,web\n"
        + b"r2,bad,5,Fine,web\n"
        + b"r3,2026-11-10,5,,web\n"
        + b"r4,worse,5,Fine,web\n"
    )

    result = runner.invoke(app, ["ingest", str(write(tmp_path, content)), "--dry-run"])

    assert result.exit_code == 0
    assert "Rows read: 4" in result.output
    assert "Valid: 1" in result.output
    assert "Skipped: 3" in result.output
    assert "invalid_date: 2 (lines 3, 5)" in result.output
    assert "missing_text: 1 (line 4)" in result.output


def test_line_numbers_are_capped_at_ten(tmp_path: Path) -> None:
    content = HEADER + b"".join(b"r%d,bad,5,Fine,web\n" % i for i in range(15))

    result = runner.invoke(app, ["ingest", str(write(tmp_path, content)), "--dry-run"])

    assert "invalid_date: 15 (lines 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, and 5 more)" in result.output


def test_output_never_contains_review_text(tmp_path: Path) -> None:
    content = HEADER + f"r1,bad,5,{FAKE_TEXT},web\nr2,2026-11-10,5,{FAKE_TEXT},web\n".encode()

    result = runner.invoke(app, ["ingest", str(write(tmp_path, content)), "--dry-run"])

    assert FAKE_TEXT not in result.output


def test_delimiter_and_encoding_options(tmp_path: Path) -> None:
    content = b"review_date;text\n2026-11-10;" + "Отличный товар".encode("cp1251") + b"\n"

    result = runner.invoke(
        app,
        [
            "ingest",
            str(write(tmp_path, content)),
            "--delimiter",
            ";",
            "--encoding",
            "cp1251",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Valid: 1" in result.output


def test_wrong_encoding_is_one_line_error(tmp_path: Path) -> None:
    content = HEADER + "r1,2026-11-10,5,Отличный товар,web\n".encode("cp1251")

    result = runner.invoke(app, ["ingest", str(write(tmp_path, content)), "--dry-run"])

    assert result.exit_code == 1
    assert "cp1251" in result.output
    assert "Traceback" not in result.output
    assert len(result.output.strip().splitlines()) == 1


@pytest.mark.parametrize("content", [b"", b"just,some,columns\n1,2,3\n"])
def test_ingest_error_exits_1(tmp_path: Path, content: bytes) -> None:
    result = runner.invoke(app, ["ingest", str(write(tmp_path, content)), "--dry-run"])

    assert result.exit_code == 1
    assert result.output.startswith("Error: ")
    assert "Traceback" not in result.output


def test_invalid_delimiter_is_a_clean_error(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Great,web\n")

    result = runner.invoke(app, ["ingest", str(path), "--delimiter", '"', "--dry-run"])

    assert result.exit_code == 1
    assert result.output.startswith("Error: ")
    assert "delimiter" in result.output
    assert "Traceback" not in result.output


def test_prints_ignored_columns(tmp_path: Path) -> None:
    path = write(tmp_path, b"review_date,text,ratting\n2026-11-10,Fine,5\n")

    result = runner.invoke(app, ["ingest", str(path), "--dry-run"])

    assert result.exit_code == 0
    assert "Ignored columns: ratting" in result.output


def test_no_ignored_columns_line_when_none(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Great,web\n")

    result = runner.invoke(app, ["ingest", str(path), "--dry-run"])

    assert "Ignored columns" not in result.output


def test_all_rows_skipped_prints_summary_then_exits_1(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + b"r1,bad,5,Fine,web\nr2,bad,5,Fine,web\n")

    result = runner.invoke(app, ["ingest", str(path), "--dry-run"])

    assert result.exit_code == 1
    assert "Rows read: 2" in result.output
    assert "Valid: 0" in result.output
    assert "invalid_date: 2 (lines 2, 3)" in result.output


def test_header_only_file_exits_1(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ingest", str(write(tmp_path, HEADER)), "--dry-run"])

    assert result.exit_code == 1
    assert "The file has no data rows." in result.output
    assert "Traceback" not in result.output


def test_malformed_and_duplicate_rows_are_listed_without_text(tmp_path: Path) -> None:
    content = (
        HEADER
        + f"r1,2026-11-10,5,{FAKE_TEXT},web\n".encode()
        + f'r2,2026-11-10,5,"{FAKE_TEXT}"x,web\n'.encode()
        + f"r1,2026-11-10,5,{FAKE_TEXT},web\n".encode()
    )

    result = runner.invoke(app, ["ingest", str(write(tmp_path, content)), "--dry-run"])

    assert result.exit_code == 0
    assert "malformed_row: 1 (line 3)" in result.output
    assert "duplicate_review_id: 1 (line 4)" in result.output
    assert FAKE_TEXT not in result.output


def test_missing_file_exits_1(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ingest", str(tmp_path / "nope.csv"), "--dry-run"])

    assert result.exit_code == 1
    assert result.output.startswith("Error: ")
    assert "Traceback" not in result.output


def test_committed_sample_dry_run() -> None:
    result = runner.invoke(app, ["ingest", str(SAMPLE_CSV), "--dry-run"])

    assert result.exit_code == 0
    assert "Rows read: 300" in result.output
    assert "Valid: 300" in result.output
    assert "Skipped: 0" in result.output


def test_dry_run_prints_notice_and_no_storage_lines(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Great,web\n")

    result = runner.invoke(app, ["ingest", str(path), "--dry-run"])

    assert "Dry run: nothing saved." in result.output
    assert "New:" not in result.output
    assert "Already stored:" not in result.output
    assert "Imported:" not in result.output


def test_dry_run_does_not_create_the_database(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Great,web\n")

    result = runner.invoke(app, ["ingest", str(path), "--dry-run"])

    assert result.exit_code == 0
    assert not Path(os.environ["DATABASE_PATH"]).exists()


def test_default_run_saves_to_database(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Great,web\nr2,2026-11-10,4,Good,app\n")

    result = runner.invoke(app, ["ingest", str(path)])

    assert result.exit_code == 0
    assert "Rows read: 2" in result.output
    assert "Valid: 2" in result.output
    assert "New: 2" in result.output
    assert "Already stored: 0" in result.output
    assert "Dry run" not in result.output
    assert stored_count() == 2


def test_reimporting_the_same_file_adds_nothing(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Great,web\nr2,2026-11-10,4,Good,app\n")
    runner.invoke(app, ["ingest", str(path)])

    result = runner.invoke(app, ["ingest", str(path)])

    assert result.exit_code == 0
    assert "New: 0" in result.output
    assert "Already stored: 2" in result.output
    assert stored_count() == 2


def test_reimporting_idless_file_adds_no_duplicates(tmp_path: Path) -> None:
    path = write(tmp_path, b"review_date,text\n2026-11-10,Great\n2026-11-10,Good\n")
    runner.invoke(app, ["ingest", str(path)])

    result = runner.invoke(app, ["ingest", str(path)])

    assert result.exit_code == 0
    assert "New: 0" in result.output
    assert "Already stored: 2" in result.output
    assert stored_count() == 2


def test_same_id_from_two_sources_is_two_reviews(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + b"1,2026-11-10,5,Great,web\n1,2026-11-10,2,Slow,app\n")

    first = runner.invoke(app, ["ingest", str(path)])
    second = runner.invoke(app, ["ingest", str(path)])

    assert "New: 2" in first.output
    assert "New: 0" in second.output
    assert "Already stored: 2" in second.output
    assert stored_count() == 2


def test_committed_sample_is_idempotent() -> None:
    first = runner.invoke(app, ["ingest", str(SAMPLE_CSV)])
    second = runner.invoke(app, ["ingest", str(SAMPLE_CSV)])

    assert first.exit_code == 0
    assert "New: 300" in first.output
    assert second.exit_code == 0
    assert "New: 0" in second.output
    assert "Already stored: 300" in second.output
    assert stored_count() == 300


def test_corrupt_database_is_a_clean_error(tmp_path: Path) -> None:
    Path(os.environ["DATABASE_PATH"]).write_bytes(os.urandom(2048))
    path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Great,web\n")

    result = runner.invoke(app, ["ingest", str(path)])

    assert result.exit_code == 1
    assert "Error: " in result.output
    assert "Traceback" not in result.output


def test_zero_valid_rows_does_not_open_the_database(tmp_path: Path) -> None:
    path = write(tmp_path, HEADER + b"r1,bad,5,Fine,web\n")

    result = runner.invoke(app, ["ingest", str(path)])

    assert result.exit_code == 1
    assert "Valid: 0" in result.output
    assert "New:" not in result.output
    assert not Path(os.environ["DATABASE_PATH"]).exists()


def test_file_error_does_not_create_the_database(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ingest", str(tmp_path / "nope.csv")])

    assert result.exit_code == 1
    assert not Path(os.environ["DATABASE_PATH"]).exists()


def test_saving_output_never_contains_review_text(tmp_path: Path) -> None:
    content = HEADER + f"r1,2026-11-10,5,{FAKE_TEXT},web\n".encode()

    result = runner.invoke(app, ["ingest", str(write(tmp_path, content))])

    assert result.exit_code == 0
    assert FAKE_TEXT not in result.output


def test_invalid_database_path_setting_is_a_clean_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    monkeypatch.setenv("DATABASE_PATH", str(blocker / "sub" / "radar.db"))
    path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Great,web\n")

    result = runner.invoke(app, ["ingest", str(path)])

    assert result.exit_code == 1
    assert "Error: " in result.output
    assert "Traceback" not in result.output


def test_invalid_setting_is_a_clean_error_naming_only_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REVIEW_RADAR_MODE", "bogus-SECRETVALUE")
    path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Great,web\n")

    result = runner.invoke(app, ["ingest", str(path)])

    assert result.exit_code == 1
    assert "Error: Invalid setting(s): REVIEW_RADAR_MODE." in result.output
    assert "SECRETVALUE" not in result.output
    assert "Traceback" not in result.output
    assert "Rows read" not in result.output
    assert not Path(os.environ["DATABASE_PATH"]).exists()


def test_dry_run_does_not_load_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_RADAR_MODE", "bogus")
    path = write(tmp_path, HEADER + b"r1,2026-11-10,5,Great,web\n")

    result = runner.invoke(app, ["ingest", str(path), "--dry-run"])

    assert result.exit_code == 0


def test_identical_idless_rows_in_one_file_count_once(tmp_path: Path) -> None:
    content = b"review_date,text\n2026-11-10,Great\n2026-11-10,  GREAT \n"

    result = runner.invoke(app, ["ingest", str(write(tmp_path, content))])

    assert result.exit_code == 0
    assert "Valid: 1" in result.output
    assert "Skipped: 1" in result.output
    assert "duplicate_review: 1 (line 3)" in result.output
    assert "New: 1" in result.output
    assert "Already stored: 0" in result.output
