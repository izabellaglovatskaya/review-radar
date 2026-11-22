"""Guards for project configuration that the workflow relies on."""

import os
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


def test_model_names_are_pinned_for_every_test() -> None:
    assert os.environ["ANTHROPIC_MODEL"] == "claude-haiku-5-5"
    assert os.environ["ANTHROPIC_COMPARE_MODEL"] == "claude-sonnet-5-5"


def test_live_tests_are_deselected_by_default() -> None:
    assert "not live" in _config()["tool"]["pytest"]["ini_options"]["addopts"]


def test_the_anthropic_sdk_is_a_runtime_dependency() -> None:
    dependencies = _config()["project"]["dependencies"]

    assert any(dep.startswith("anthropic>=") for dep in dependencies)
