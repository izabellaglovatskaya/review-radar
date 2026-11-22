from pathlib import Path

import pytest
from pydantic import SecretStr

from review_radar.config import PROJECT_ROOT, ConfigError, Mode, Settings

FAKE_KEY = "sk-ant-test-FAKE"
FAKE_TOKEN = "123456:FAKE-TOKEN"  # noqa: S105 (fictional)
FAKE_CHAT = "987654321"
ENV_NAMES = (
    "REVIEW_RADAR_MODE",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_COMPARE_MODEL",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "DATABASE_PATH",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def make() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_defaults() -> None:
    s = make()
    assert s.mode is Mode.AUTO
    assert s.anthropic_api_key is None
    assert s.anthropic_model == "claude-haiku-5-5"
    assert s.anthropic_compare_model == "claude-sonnet-5-5"
    assert s.telegram_bot_token is None
    assert s.telegram_chat_id is None
    assert s.database_path == PROJECT_ROOT / "data" / "review_radar.db"


def test_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_RADAR_MODE", "demo")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("ANTHROPIC_MODEL", "m1")
    monkeypatch.setenv("ANTHROPIC_COMPARE_MODEL", "m2")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", FAKE_CHAT)
    monkeypatch.setenv("DATABASE_PATH", "./tmp/x.db")
    s = make()
    assert s.mode is Mode.DEMO
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == FAKE_KEY
    assert s.anthropic_model == "m1"
    assert s.anthropic_compare_model == "m2"
    assert s.telegram_bot_token is not None
    assert s.telegram_bot_token.get_secret_value() == FAKE_TOKEN
    assert s.telegram_chat_id is not None
    assert s.telegram_chat_id.get_secret_value() == FAKE_CHAT
    assert s.database_path == PROJECT_ROOT / "tmp" / "x.db"


def test_empty_strings_become_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "")
    s = make()
    assert s.anthropic_api_key is None
    assert s.telegram_bot_token is None
    assert s.telegram_chat_id is None


def test_whitespace_only_key_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert make().anthropic_api_key is None


def test_env_file_is_read_and_extra_ignored(tmp_path: Path) -> None:
    env = tmp_path / "test.env"
    env.write_text(f"ANTHROPIC_API_KEY={FAKE_KEY}\nUNRELATED=1\n", encoding="utf-8")
    s = Settings(_env_file=env)  # type: ignore[call-arg]
    assert s.anthropic_api_key is not None


def test_invalid_mode_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_RADAR_MODE", "bogus")
    with pytest.raises(ValueError, match="REVIEW_RADAR_MODE"):
        make()


def test_auto_without_key_is_demo() -> None:
    assert make().resolved_mode() is Mode.DEMO


def test_auto_with_key_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    assert make().resolved_mode() is Mode.LIVE


def test_demo_with_key_is_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_RADAR_MODE", "demo")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    assert make().resolved_mode() is Mode.DEMO


def test_live_with_key_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_RADAR_MODE", "live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    assert make().resolved_mode() is Mode.LIVE


def test_live_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_RADAR_MODE", "live")
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        make().resolved_mode()


def test_live_with_empty_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_RADAR_MODE", "live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    with pytest.raises(ConfigError):
        make().resolved_mode()


def test_secrets_not_leaked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", FAKE_CHAT)
    s = make()
    for rendered in (repr(s), str(s), s.model_dump_json(), str(s.model_dump())):
        for secret in (FAKE_KEY, FAKE_TOKEN, FAKE_CHAT):
            assert secret not in rendered


def test_config_error_message_has_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_RADAR_MODE", "live")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    with pytest.raises(ConfigError) as exc:
        make().resolved_mode()
    assert FAKE_TOKEN not in str(exc.value)


def test_project_root_contains_pyproject() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_env_file_is_anchored_to_project_root() -> None:
    assert Settings.model_config["env_file"] == PROJECT_ROOT / ".env"


def test_default_database_path_ignores_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert make().database_path == PROJECT_ROOT / "data" / "review_radar.db"


def test_relative_database_path_resolved_against_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_PATH", "sub/x.db")
    path = make().database_path
    assert path == PROJECT_ROOT / "sub" / "x.db"
    assert path.is_absolute()


def test_absolute_database_path_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "abs.db"
    monkeypatch.setenv("DATABASE_PATH", str(target))
    assert make().database_path == target


def test_telegram_enabled_when_both_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", FAKE_CHAT)
    assert make().telegram_enabled() is True


def test_telegram_disabled_when_neither_set() -> None:
    assert make().telegram_enabled() is False


def test_telegram_token_without_chat_id_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    with pytest.raises(ConfigError, match="TELEGRAM_CHAT_ID") as exc:
        make().telegram_enabled()
    assert FAKE_TOKEN not in str(exc.value)


def test_telegram_chat_id_without_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_CHAT_ID", FAKE_CHAT)
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN") as exc:
        make().telegram_enabled()
    assert FAKE_CHAT not in str(exc.value)


@pytest.mark.parametrize("name", ["ANTHROPIC_MODEL", "ANTHROPIC_COMPARE_MODEL"])
@pytest.mark.parametrize("value", ["", "   "])
def test_blank_model_name_is_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        make()


def test_model_names_are_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "  claude-haiku-5-5\n")
    monkeypatch.setenv("ANTHROPIC_COMPARE_MODEL", "\tclaude-sonnet-5-5 ")

    s = make()

    assert s.anthropic_model == "claude-haiku-5-5"
    assert s.anthropic_compare_model == "claude-sonnet-5-5"


@pytest.mark.parametrize(
    ("name", "attr"),
    [
        ("ANTHROPIC_API_KEY", "anthropic_api_key"),
        ("TELEGRAM_BOT_TOKEN", "telegram_bot_token"),
        ("TELEGRAM_CHAT_ID", "telegram_chat_id"),
    ],
)
def test_secrets_are_stripped(monkeypatch: pytest.MonkeyPatch, name: str, attr: str) -> None:
    monkeypatch.setenv(name, "  value-123\r\n")

    secret = getattr(make(), attr)

    assert secret.get_secret_value() == "value-123"


def test_a_secret_passed_as_secretstr_is_left_alone() -> None:
    s = Settings(_env_file=None, ANTHROPIC_API_KEY=SecretStr("abc"))  # type: ignore[call-arg]

    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "abc"
