"""Send one message through the Telegram Bot API.

Security rule: the bot token is part of the request URL, so a URL must never reach a log or an
error. HTTP client loggers are held at WARNING, and `TelegramError` carries only a reason code
and a status code. It is raised outside the `except` block, so no original exception (whose text
can quote the URL) stays attached to it.

A send is never retried here: repeating a request whose outcome is unclear could deliver the
summary twice. `DeliveryState` tells the caller whether a later run may safely send again.

Proxy and CA environment variables are honoured (`trust_env=True`), as the Anthropic SDK does.
"""

import logging
import re
from enum import StrEnum

import httpx2
from pydantic import SecretStr

API_BASE = "https://api.telegram.org"
HTTP_LOGGERS = ("httpx", "httpx2", "httpcore", "httpcore2")
REQUEST_TIMEOUT_SECONDS = 10.0

_TOKEN_FORMAT = re.compile(r"[0-9]+:[A-Za-z0-9_-]+")
_CLIENT_ERRORS = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    429: "rate_limited",
}


class DeliveryState(StrEnum):
    """What a failed send means for the chat."""

    FAILED = "failed"  # certainly not delivered: safe to send again
    UNKNOWN = "unknown"  # may have been delivered: a person must check


class TelegramError(Exception):
    """A send failed. Holds a reason code and status only: no URL, token, chat id or body."""

    def __init__(self, reason: str, state: DeliveryState, status_code: int | None = None) -> None:
        detail = f"{reason}, HTTP {status_code}" if status_code is not None else reason
        super().__init__(f"Telegram send failed ({detail}).")
        self.reason = reason
        self.state = state
        self.status_code = status_code


def quiet_http_loggers() -> None:
    """Hold HTTP client loggers at WARNING: at INFO they log request URLs with the token."""
    for name in HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


class TelegramSender:
    """Sends HTML messages to one chat. `transport` lets tests replace the network."""

    def __init__(
        self,
        *,
        token: SecretStr,
        chat_id: SecretStr,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        quiet_http_loggers()
        self._token = token
        self._chat_id = chat_id
        self._transport = transport

    def send(self, text: str) -> int:
        """Send `text` once and return the Telegram message id; raise TelegramError on failure."""
        token = self._token.get_secret_value()
        if _TOKEN_FORMAT.fullmatch(token) is None:
            raise TelegramError("invalid_token_format", DeliveryState.FAILED)
        outcome = self._post(token, text)
        if isinstance(outcome, TelegramError):
            raise outcome  # outside `except`, so no exception holding the URL is chained
        return outcome

    def _post(self, token: str, text: str) -> int | TelegramError:
        payload = {
            "chat_id": self._chat_id.get_secret_value(),
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        try:
            with httpx2.Client(
                transport=self._transport,
                timeout=REQUEST_TIMEOUT_SECONDS,
                follow_redirects=False,
                # Honour proxy/CA variables: whoever controls the process environment can already
                # run code (PYTHONPATH, the venv), so disabling this adds no protection and would
                # break a legitimate proxy.
                trust_env=True,
            ) as client:
                response = client.post(f"{API_BASE}/bot{token}/sendMessage", json=payload)
        except (httpx2.ConnectError, httpx2.ConnectTimeout, httpx2.PoolTimeout):
            return TelegramError("connect_error", DeliveryState.FAILED)
        except httpx2.TransportError:
            return TelegramError("network_error", DeliveryState.UNKNOWN)
        except Exception:  # noqa: BLE001 (any other text could quote the URL)
            return TelegramError("unexpected_error", DeliveryState.UNKNOWN)
        return _parse(response)


def _parse(response: httpx2.Response) -> int | TelegramError:
    """Return the message id, or the error this response stands for."""
    code = response.status_code
    if code >= 500:
        return TelegramError("server_error", DeliveryState.UNKNOWN, code)
    if code != 200:
        reason = _CLIENT_ERRORS.get(code, "http_error")
        return TelegramError(reason, DeliveryState.FAILED, code)
    invalid = TelegramError("invalid_response", DeliveryState.UNKNOWN, code)
    try:
        body = response.json()
    except ValueError:
        return invalid
    if not isinstance(body, dict):
        return invalid
    if body.get("ok") is False:
        return TelegramError("api_error", DeliveryState.FAILED, code)
    result = body.get("result")
    if body.get("ok") is not True or not isinstance(result, dict):
        return invalid
    message_id = result.get("message_id")
    if not isinstance(message_id, int) or isinstance(message_id, bool):
        return invalid
    return message_id
