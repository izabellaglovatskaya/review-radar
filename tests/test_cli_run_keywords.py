"""`run` hands every option to the step commands by keyword, so a reordered option cannot shift.

The fakes accept keyword arguments only: a positional call raises TypeError.
"""

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from review_radar import cli

DAY = dt.datetime(2026, 11, 10)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    recorded: dict[str, dict[str, Any]] = {}

    def ingest(*, path: Path, delimiter: str | None, encoding: str, dry_run: bool) -> None:
        recorded["ingest"] = {
            "path": path,
            "delimiter": delimiter,
            "encoding": encoding,
            "dry_run": dry_run,
        }

    def analyze(*, limit: int | None, max_calls: int | None) -> None:
        recorded["analyze"] = {"limit": limit, "max_calls": max_calls}

    def summary(
        *,
        date: dt.datetime | None,
        start: dt.datetime | None,
        end: dt.datetime | None,
        dry_run: bool,
        force: bool,
        allow_partial: bool,
        allow_empty: bool,
    ) -> None:
        recorded["summary"] = {
            "date": date,
            "start": start,
            "end": end,
            "dry_run": dry_run,
            "force": force,
            "allow_partial": allow_partial,
            "allow_empty": allow_empty,
        }

    def dashboard(*, out: Path | None) -> None:
        recorded["dashboard"] = {"out": out}

    monkeypatch.setattr(cli, "ingest", ingest)
    monkeypatch.setattr(cli, "analyze", analyze)
    monkeypatch.setattr(cli, "summary_command", summary)
    monkeypatch.setattr(cli, "dashboard_command", dashboard)
    return recorded


def test_demo_run_passes_dry_run_and_the_date_by_keyword(
    calls: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
    csv_path = tmp_path / "reviews.csv"
    out = tmp_path / "site"

    cli._run_steps(csv_path, DAY, out, dry_run=True)

    assert calls["ingest"] == {
        "path": csv_path,
        "delimiter": None,
        "encoding": "utf-8-sig",
        "dry_run": False,
    }
    assert calls["analyze"] == {"limit": None, "max_calls": None}
    assert calls["summary"] == {
        "date": DAY,
        "start": None,
        "end": None,
        "dry_run": True,
        "force": False,
        "allow_partial": False,
        "allow_empty": False,
    }
    assert calls["dashboard"] == {"out": out}


def test_live_run_does_not_make_the_summary_a_dry_run(
    calls: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
    cli._run_steps(tmp_path / "reviews.csv", DAY, None, dry_run=False)

    assert calls["summary"]["dry_run"] is False
    assert calls["summary"]["date"] == DAY
    assert calls["dashboard"] == {"out": None}
