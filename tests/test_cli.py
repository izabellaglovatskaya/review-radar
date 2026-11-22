import runpy
import subprocess
import sys
from importlib.metadata import version

import pytest
from typer.testing import CliRunner

from review_radar.cli import app

runner = CliRunner()
EXPECTED_VERSION_LINE = f"review-radar {version('review-radar')}"


def test_version_option_prints_name_and_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == EXPECTED_VERSION_LINE


def test_package_version_comes_from_metadata() -> None:
    import review_radar

    assert review_radar.__version__ == version("review-radar")


def test_python_dash_m_version() -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "review_radar", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == EXPECTED_VERSION_LINE


def test_no_args_shows_help_without_traceback() -> None:
    result = runner.invoke(app, [])

    # Click exits with code 2 when no_args_is_help shows the help text.
    assert result.exit_code == 2
    assert "Usage" in result.output
    assert "Traceback" not in result.output


def test_main_module_runs_app_in_process(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["review-radar", "--version"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("review_radar", run_name="__main__")

    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == EXPECTED_VERSION_LINE
