"""Guards for the GitHub Actions workflow, checked locally without running it on GitHub.

These tests pin the properties the plan relies on: Ubuntu only, a locked install, the three
checks, gitleaks over the full history, no secrets, pinned actions and read-only permissions.
Syntax and expressions are checked separately with actionlint:
`uvx --from actionlint-py==1.7.12.24 actionlint .github/workflows/ci.yml`.
"""

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PRE_COMMIT = ROOT / ".pre-commit-config.yaml"
PINNED_ACTION = re.compile(r"^[\w.-]+/[\w.-]+(/[\w./-]+)?@[0-9a-f]{40}$")


def _workflow() -> dict[str, Any]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _jobs() -> dict[str, dict[str, Any]]:
    return _workflow()["jobs"]


def _steps(job: str) -> list[dict[str, Any]]:
    return _jobs()[job]["steps"]


def _runs(job: str) -> str:
    return "\n".join(step.get("run", "") for step in _steps(job))


def _all_steps() -> list[dict[str, Any]]:
    return [step for job in _jobs().values() for step in job["steps"]]


def _env(job: str) -> dict[str, Any]:
    env = _jobs()[job].get("env")
    assert isinstance(env, dict), f"job {job} has no env block"
    return env


def _strings(node: object) -> list[str]:
    """Every key and value of the parsed workflow (comments are not part of it)."""
    if isinstance(node, dict):
        return [s for k, v in node.items() for s in (*_strings(k), *_strings(v))]
    if isinstance(node, list):
        return [s for item in node for s in _strings(item)]
    return [node] if isinstance(node, str) else []


def test_runs_on_push_and_pull_request() -> None:
    workflow = _workflow()
    # PyYAML reads a bare `on` key as True (YAML 1.1); a quoted "on" stays a string.
    triggers = workflow.get("on", workflow.get(True))

    assert "push" in triggers
    assert "pull_request" in triggers


def test_every_job_runs_on_ubuntu_with_a_timeout() -> None:
    for name, job in _jobs().items():
        assert job["runs-on"].startswith("ubuntu-"), name
        assert 0 < job["timeout-minutes"] <= 30, name


def test_token_permissions_are_read_only() -> None:
    assert _workflow()["permissions"] == {"contents": "read"}
    for name, job in _jobs().items():
        assert "permissions" not in job, name


def test_no_secrets_are_used() -> None:
    for value in _strings(_workflow()):
        assert "secrets." not in value, value
        assert "GITHUB_TOKEN" not in value, value


@pytest.mark.parametrize("job", ["test", "browser"])
def test_test_jobs_never_call_a_paid_api_or_telegram(job: str) -> None:
    env = _env(job)

    assert env["REVIEW_RADAR_MODE"] == "demo"
    for name in ("ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        assert env[name] == "", name


def test_only_pull_request_runs_are_cancelled_by_a_newer_push() -> None:
    # A cancelled run on main would leave that commit unchecked.
    cancel = _workflow()["concurrency"]["cancel-in-progress"]

    assert cancel == "${{ github.event_name == 'pull_request' }}"


def test_every_action_is_pinned_to_a_commit_sha() -> None:
    uses = [step["uses"] for step in _all_steps() if "uses" in step]

    assert uses
    for action in uses:
        assert PINNED_ACTION.fullmatch(action), action


def test_checkout_does_not_keep_credentials() -> None:
    checkouts = [s for s in _all_steps() if s.get("uses", "").startswith("actions/checkout@")]

    assert checkouts
    for step in checkouts:
        assert step["with"]["persist-credentials"] is False


def test_python_jobs_install_with_the_lock_file() -> None:
    for job in ("lint", "test", "browser"):
        assert "uv sync --locked" in _runs(job), job


def test_lint_job_runs_ruff_check_and_format_check() -> None:
    runs = _runs("lint")

    assert "uv run ruff check ." in runs
    assert "uv run ruff format --check ." in runs


def test_test_job_runs_the_default_pytest_suite() -> None:
    lines = [line.strip() for line in _runs("test").splitlines()]

    assert "uv run pytest" in lines  # default markers: no live, no browser tests


def test_browser_job_is_separate_locked_and_cannot_pass_by_skipping() -> None:
    runs = _runs("browser")

    assert "uv sync --locked --group browser" in runs
    assert "uv run --locked --group browser pytest -m browser" in runs
    assert "--with" not in runs  # everything installed comes from uv.lock
    assert "-m live" not in runs
    assert _env("browser")["REVIEW_RADAR_REQUIRE_BROWSER"] == "1"


def test_playwright_is_pinned_in_its_own_dependency_group() -> None:
    with (ROOT / "pyproject.toml").open("rb") as f:
        groups = tomllib.load(f)["dependency-groups"]

    assert any(re.fullmatch(r"playwright==\d+\.\d+\.\d+", dep) for dep in groups["browser"])
    assert not any(dep.startswith("playwright") for dep in groups["dev"])


def test_gitleaks_scans_the_full_history() -> None:
    checkout = next(s for s in _steps("gitleaks") if s.get("uses", "").startswith("actions/"))
    runs = _runs("gitleaks")

    assert checkout["with"]["fetch-depth"] == 0
    assert "gitleaks git" in runs
    assert "--redact" in runs
    assert "--pre-commit" not in runs
    assert "--staged" not in runs
    assert "--log-opts" not in runs  # no range limit: every commit


def test_gitleaks_download_is_verified_by_checksum() -> None:
    env = _env("gitleaks")
    runs = _runs("gitleaks")

    assert re.fullmatch(r"[0-9a-f]{64}", env["GITLEAKS_SHA256"])
    assert "${GITLEAKS_SHA256}" in runs
    assert "sha256sum --check" in runs


def test_gitleaks_version_matches_the_pre_commit_hook() -> None:
    hook = PRE_COMMIT.read_text(encoding="utf-8")
    frozen = re.search(r"frozen: v(\d+\.\d+\.\d+)", hook)
    assert frozen is not None

    env = _env("gitleaks")

    assert env["GITLEAKS_VERSION"] == frozen.group(1)


def test_uv_version_is_pinned() -> None:
    setups = [s for s in _all_steps() if s.get("uses", "").startswith("astral-sh/setup-uv@")]

    assert setups
    for step in setups:
        assert re.fullmatch(r"\d+\.\d+\.\d+", str(step["with"]["version"]))


@pytest.mark.parametrize("job", ["lint", "test", "browser"])
def test_python_version_comes_from_the_project(job: str) -> None:
    # uv reads .python-version; a hard-coded version here could drift from it.
    for step in _steps(job):
        assert "python-version" not in step.get("with", {}), step
        assert not step.get("uses", "").startswith("actions/setup-python@"), step


def test_pyyaml_is_a_direct_dev_dependency() -> None:
    with (ROOT / "pyproject.toml").open("rb") as f:
        dev = tomllib.load(f)["dependency-groups"]["dev"]

    assert any(dep.lower().startswith("pyyaml") for dep in dev)
