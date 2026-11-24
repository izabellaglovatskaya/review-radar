"""In CI a missing browser must fail the job, not skip every test and pass."""

import pytest

from browser_support import REQUIRE_BROWSER_ENV, unavailable


def test_a_missing_browser_skips_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REQUIRE_BROWSER_ENV, raising=False)

    with pytest.raises(pytest.skip.Exception, match="Chrome is not available"):
        unavailable("Chrome is not available")


def test_a_missing_browser_fails_when_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REQUIRE_BROWSER_ENV, "1")

    with pytest.raises(pytest.fail.Exception, match="Chrome is not available"):
        unavailable("Chrome is not available")


@pytest.mark.parametrize("value", ["", "0", "yes"])
def test_only_the_value_1_requires_the_browser(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(REQUIRE_BROWSER_ENV, value)

    with pytest.raises(pytest.skip.Exception):
        unavailable("Playwright is not installed")
