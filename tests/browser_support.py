"""Skip browser tests when Playwright or Chrome is missing, unless the run requires them.

CI sets `REVIEW_RADAR_REQUIRE_BROWSER=1`, so a broken browser setup fails the job instead of
skipping every test and passing.
"""

import os
from typing import NoReturn

import pytest

REQUIRE_BROWSER_ENV = "REVIEW_RADAR_REQUIRE_BROWSER"


def unavailable(reason: str) -> NoReturn:
    """Fail when the browser is required, otherwise skip."""
    if os.environ.get(REQUIRE_BROWSER_ENV) == "1":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason, allow_module_level=True)
