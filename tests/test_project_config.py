"""Guards for project configuration that the workflow relies on."""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _config() -> dict:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)


def test_pytest_always_measures_coverage() -> None:
    addopts = _config()["tool"]["pytest"]["ini_options"]["addopts"]

    assert "--cov" in addopts


def test_coverage_threshold_is_at_least_80() -> None:
    assert _config()["tool"]["coverage"]["report"]["fail_under"] >= 80


def test_live_marker_is_registered_and_markers_are_strict() -> None:
    ini = _config()["tool"]["pytest"]["ini_options"]

    assert any(marker.startswith("live:") for marker in ini.get("markers", []))
    assert "--strict-markers" in ini["addopts"]
