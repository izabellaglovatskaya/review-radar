"""`build_client` warns, by variable name only, about SDK environment overrides (fictional)."""

import logging

import pytest
from pydantic import SecretStr

from review_radar.anthropic_classifier import build_client

KEY = SecretStr("sk-ant-fictional-test-key")
BASE_URL_VALUE = "https://proxy.example.invalid/fictional-path"
TOKEN_VALUE = "fictional-bearer-value-9876"  # noqa: S105

BASE_URL_WARNING = (
    "ANTHROPIC_BASE_URL is set: requests and the API key go to that address "
    "instead of the default API."
)
TOKEN_WARNING = "ANTHROPIC_AUTH_TOKEN is set: the SDK also sends it as a bearer token."  # noqa: S105

# Variable -> (a value that is risky, the warning). Each value is fictional.
EXTRA_WARNINGS = {
    "ANTHROPIC_LOG": (
        "debug",
        "ANTHROPIC_LOG=debug is set: the SDK logs request bodies, which hold review text.",
    ),
    "ANTHROPIC_CUSTOM_HEADERS": (
        "X-Fictional: fictional-header-value-1234",
        "ANTHROPIC_CUSTOM_HEADERS is set: the SDK adds those headers to every request.",
    ),
    "ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE": (
        "content",
        "ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE=content is set: traces can hold review text.",
    ),
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    for name in EXTRA_WARNINGS:
        monkeypatch.delenv(name, raising=False)


def warning_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_no_warning_when_both_are_unset(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        build_client(KEY)

    assert warning_messages(caplog) == []


@pytest.mark.parametrize("blank", ["", "   "])
def test_no_warning_when_both_are_blank(
    blank: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", blank)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", blank)

    with caplog.at_level(logging.DEBUG):
        build_client(KEY)

    assert warning_messages(caplog) == []


def test_warns_once_about_a_base_url(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", BASE_URL_VALUE)

    with caplog.at_level(logging.DEBUG):
        build_client(KEY)

    assert warning_messages(caplog) == [BASE_URL_WARNING]


def test_warns_once_about_an_auth_token(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", TOKEN_VALUE)

    with caplog.at_level(logging.DEBUG):
        build_client(KEY)

    assert warning_messages(caplog) == [TOKEN_WARNING]


def test_warns_about_each_variable_and_never_logs_a_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ANTHROPIC_BASE_URL", BASE_URL_VALUE)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", TOKEN_VALUE)

    with caplog.at_level(logging.DEBUG):
        build_client(KEY)

    assert warning_messages(caplog) == [BASE_URL_WARNING, TOKEN_WARNING]
    assert BASE_URL_VALUE not in caplog.text
    assert "proxy.example.invalid" not in caplog.text
    assert TOKEN_VALUE not in caplog.text
    assert KEY.get_secret_value() not in caplog.text


@pytest.mark.parametrize("name", sorted(EXTRA_WARNINGS))
def test_warns_about_sdk_settings_that_can_leak_review_text(
    name: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    value, warning = EXTRA_WARNINGS[name]
    monkeypatch.setenv(name, value)

    with caplog.at_level(logging.DEBUG):
        build_client(KEY)

    assert warning_messages(caplog) == [warning]
    if name == "ANTHROPIC_CUSTOM_HEADERS":
        assert value not in caplog.text


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ANTHROPIC_LOG", "info"),
        ("ANTHROPIC_OPEN_TELEMETRY_TRACES_CONTENT_MODE", "metadata"),
        ("ANTHROPIC_CUSTOM_HEADERS", "  "),
    ],
)
def test_harmless_sdk_settings_do_not_warn(
    name: str, value: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(name, value)

    with caplog.at_level(logging.DEBUG):
        build_client(KEY)

    assert warning_messages(caplog) == []
