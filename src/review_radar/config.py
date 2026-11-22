"""Application settings loaded from the environment and `.env`."""

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(Exception):
    """Raised when the configuration is inconsistent. Messages never contain secrets."""


class Mode(StrEnum):
    AUTO = "auto"
    DEMO = "demo"
    LIVE = "live"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore", frozen=True)

    mode: Mode = Field(default=Mode.AUTO, validation_alias="REVIEW_RADAR_MODE")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-haiku-5-5", validation_alias="ANTHROPIC_MODEL")
    anthropic_compare_model: str = Field(
        default="claude-sonnet-5-5", validation_alias="ANTHROPIC_COMPARE_MODEL"
    )
    telegram_bot_token: SecretStr | None = Field(
        default=None, validation_alias="TELEGRAM_BOT_TOKEN"
    )
    telegram_chat_id: SecretStr | None = Field(default=None, validation_alias="TELEGRAM_CHAT_ID")
    database_path: Path = Field(
        default=PROJECT_ROOT / "data" / "review_radar.db", validation_alias="DATABASE_PATH"
    )

    @field_validator("anthropic_api_key", "telegram_bot_token", "telegram_chat_id", mode="before")
    @classmethod
    def _blank_secret_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("database_path", mode="after")
    @classmethod
    def _anchor_database_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    def telegram_enabled(self) -> bool:
        """True when token and chat id are both set, False when neither; error if only one."""
        has_token = self.telegram_bot_token is not None
        has_chat = self.telegram_chat_id is not None
        if has_token and not has_chat:
            raise ConfigError("TELEGRAM_BOT_TOKEN is set but TELEGRAM_CHAT_ID is missing.")
        if has_chat and not has_token:
            raise ConfigError("TELEGRAM_CHAT_ID is set but TELEGRAM_BOT_TOKEN is missing.")
        return has_token and has_chat

    def resolved_mode(self) -> Mode:
        """Return DEMO or LIVE. AUTO picks LIVE only when an API key is set."""
        has_key = self.anthropic_api_key is not None
        if self.mode is Mode.AUTO:
            return Mode.LIVE if has_key else Mode.DEMO
        if self.mode is Mode.LIVE and not has_key:
            raise ConfigError(
                "REVIEW_RADAR_MODE=live requires ANTHROPIC_API_KEY to be set "
                "(or use REVIEW_RADAR_MODE=demo)."
            )
        return self.mode
