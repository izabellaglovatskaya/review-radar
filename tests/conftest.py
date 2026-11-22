import logging
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_database_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point DATABASE_PATH at a temp file so no test can touch the real database."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))


@pytest.fixture(autouse=True)
def _no_real_secrets(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Blank every secret and force demo mode, so a developer's real `.env` is never used.

    Environment variables take priority over `.env`, and blank secrets count as unset.
    The model names are always pinned (the cache key depends on them). Tests marked `live`
    are the one exception to the blanking: they read the developer's key through `Settings()`.
    """
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("ANTHROPIC_COMPARE_MODEL", "claude-sonnet-5-5")
    if request.node.get_closest_marker("live"):
        return
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    monkeypatch.setenv("REVIEW_RADAR_MODE", "demo")


@pytest.fixture(autouse=True)
def _restore_package_logger() -> Iterator[None]:
    """The CLI configures the `review_radar` logger; undo it so tests stay independent."""
    package_logger = logging.getLogger("review_radar")
    level, handlers = package_logger.level, list(package_logger.handlers)
    yield
    package_logger.setLevel(level)
    package_logger.handlers[:] = handlers
