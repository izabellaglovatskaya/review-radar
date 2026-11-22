from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "CHAT_ID")
REQUIRED_KEYS = {
    "ANTHROPIC_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "DATABASE_PATH",
    "ANTHROPIC_MODEL",
}


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def test_required_keys_present() -> None:
    env = _parse_env(REPO_ROOT / ".env.example")

    assert env.keys() >= REQUIRED_KEYS


def test_secret_values_are_empty() -> None:
    env = _parse_env(REPO_ROOT / ".env.example")
    secret_keys = [k for k in env if any(m in k for m in SECRET_MARKERS)]

    assert secret_keys, "expected at least one secret-like key"
    for key in secret_keys:
        assert env[key] == "", f"{key} must be empty in .env.example"


def test_gitignore_ignores_dotenv() -> None:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in [line.strip() for line in lines]
