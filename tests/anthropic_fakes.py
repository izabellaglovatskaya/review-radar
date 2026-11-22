"""Test doubles for the Anthropic SDK: a scripted fake client and error/message builders.

Nothing here touches the network. All keys and review texts are fictional.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anthropic
import httpx2
from anthropic.types import Message

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "anthropic"
FAKE_KEY = "sk-ant-FAKE-0000"
FAKE_TEXT = "ZEBRA-QUOKKA-9917 distinctive fake review"
URL = "https://api.anthropic.com/v1/messages"
_UNSET: Any = object()


def fixture_message(name: str) -> Message:
    """Build a `Message` from a recorded payload in `tests/fixtures/anthropic`."""
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return Message.model_validate(payload)


def make_message(
    results: Sequence[dict[str, Any]] | str | None,
    *,
    stop_reason: str | None = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 20,
    blocks: list[dict[str, Any]] | None = None,
) -> Message:
    """A `Message` whose single text block holds `{"results": results}` (or a raw string)."""
    if blocks is None:
        text = results if isinstance(results, str) else json.dumps({"results": list(results or [])})
        blocks = [{"type": "text", "text": text}]
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    )


def result(item_id: str, sentiment: str = "positive", topics: Sequence[str] = (), conf: Any = 0.9):
    return {"id": item_id, "sentiment": sentiment, "topics": list(topics), "confidence": conf}


def status_error(
    status: int,
    *,
    error_type: str | None = None,
    headers: dict[str, str] | None = None,
    message: str = f"SDK says: {FAKE_TEXT} {FAKE_KEY}",
    request_id: str | None = "req_test_1",
    error_code: str | None = None,
    body: Any = _UNSET,
) -> anthropic.APIStatusError:
    """An `APIStatusError` of the right subclass; its message deliberately echoes secrets.

    `error_code` goes to `error.details.error_code`; `body` replaces the whole body.
    """
    all_headers = dict(headers or {})
    if request_id is not None:
        all_headers["request-id"] = request_id
    error: dict[str, Any] = {"type": error_type or "api_error", "message": message}
    if error_code is not None:
        error["details"] = {"error_code": error_code}
    if body is _UNSET:
        body = {"type": "error", "error": error}
    response = httpx2.Response(
        status,
        request=httpx2.Request("POST", URL),
        headers=all_headers,
        content=json.dumps(body).encode("utf-8"),
    )
    classes: dict[int, type[anthropic.APIStatusError]] = {
        400: anthropic.BadRequestError,
        401: anthropic.AuthenticationError,
        402: anthropic.PaymentRequiredError,
        403: anthropic.PermissionDeniedError,
        404: anthropic.NotFoundError,
        408: anthropic.RequestTimeoutError,
        409: anthropic.ConflictError,
        413: anthropic.ContentTooLargeError,
        429: anthropic.RateLimitError,
        500: anthropic.InternalServerError,
    }
    cls = classes.get(status, anthropic.InternalServerError if status >= 500 else None)
    if cls is None:
        cls = anthropic.APIStatusError
    return cls(message, response=response, body=body)


def response_validation_error() -> anthropic.APIResponseValidationError:
    response = httpx2.Response(200, request=httpx2.Request("POST", URL), json={"x": 1})
    return anthropic.APIResponseValidationError(response, {"x": 1}, message=f"bad {FAKE_TEXT}")


def generic_api_error() -> anthropic.APIError:
    return anthropic.APIError(f"odd {FAKE_TEXT} {FAKE_KEY}", httpx2.Request("POST", URL), body=None)


def connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(
        message=f"boom {FAKE_TEXT}", request=httpx2.Request("POST", URL)
    )


def timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(httpx2.Request("POST", URL))


class _Messages:
    def __init__(self, script: list[Message | Exception]) -> None:
        self._script = script
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("the fake client has no scripted response left")
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


class FakeClient:
    """Stands in for `anthropic.Anthropic`: replays a script of messages / exceptions in order."""

    def __init__(self, *script: Message | Exception) -> None:
        self.messages = _Messages(list(script))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls

    @property
    def remaining(self) -> int:
        return len(self.messages._script)


class RecordingSleep:
    """Replaces `time.sleep`: records the delays instead of waiting."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
