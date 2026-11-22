from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_database_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point DATABASE_PATH at a temp file so no test can touch the real database."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
