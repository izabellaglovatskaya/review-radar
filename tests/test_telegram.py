"""Telegram sender: all HTTP is faked with a mock transport, nothing reaches the network."""

import json
import logging
import traceback
from collections.abc import Callable

import httpx2
import pytest
from pydantic import SecretStr

from review_radar.telegram import (
    HTTP_LOGGERS,
    DeliveryState,
    TelegramError,
    TelegramSender,
    quiet_http_loggers,
)

TOKEN = "123456:FAKE-token_value"  # noqa: S105 (fictional)
CHAT_ID = "123456789"

Handler = Callable[[httpx2.Request], httpx2.Response]


def sender(handler: Handler, token: str = TOKEN) -> TelegramSender:
    return TelegramSender(
        token=SecretStr(token),
        chat_id=SecretStr(CHAT_ID),
        transport=httpx2.MockTransport(handler),
    )


def ok(_: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json={"ok": True, "result": {"message_id": 42}})


def status(code: int) -> Handler:
    def handle(_: httpx2.Request) -> httpx2.Response:
        body = {"ok": False, "error_code": code, "description": f"secret detail {TOKEN}"}
        return httpx2.Response(code, json=body)

    return handle


def raising(exc: Exception) -> Handler:
    def handle(_: httpx2.Request) -> httpx2.Response:
        raise exc

    return handle


def failure(handler: Handler) -> TelegramError:
    with pytest.raises(TelegramError) as caught:
        sender(handler).send("hello")
    return caught.value


def assert_clean(error: BaseException) -> None:
    rendered = "".join(traceback.format_exception(error))
    assert TOKEN not in rendered
    assert "api.telegram.org" not in rendered
    assert CHAT_ID not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None


class TestSend:
    def test_posts_html_without_link_previews(self) -> None:
        seen: list[httpx2.Request] = []

        def handle(request: httpx2.Request) -> httpx2.Response:
            seen.append(request)
            return ok(request)

        message_id = sender(handle).send("<b>hello</b>")

        assert message_id == 42
        assert len(seen) == 1
        assert seen[0].method == "POST"
        assert seen[0].url.path == f"/bot{TOKEN}/sendMessage"
        assert seen[0].url.host == "api.telegram.org"
        assert json.loads(seen[0].content) == {
            "chat_id": CHAT_ID,
            "text": "<b>hello</b>",
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }

    def test_a_failed_send_is_not_retried(self) -> None:
        calls = 0

        def handle(request: httpx2.Request) -> httpx2.Response:
            nonlocal calls
            calls += 1
            return status(500)(request)

        failure(handle)

        assert calls == 1

    def test_a_redirect_is_not_followed(self) -> None:
        calls = 0

        def handle(_: httpx2.Request) -> httpx2.Response:
            nonlocal calls
            calls += 1
            return httpx2.Response(302, headers={"location": "https://evil.example/steal"})

        error = failure(handle)

        assert calls == 1
        assert error.state is DeliveryState.FAILED
        assert_clean(error)


class TestRefusals:
    @pytest.mark.parametrize(
        ("code", "reason"),
        [
            (400, "bad_request"),
            (401, "unauthorized"),
            (403, "forbidden"),
            (404, "not_found"),
            (429, "rate_limited"),
            (418, "http_error"),
        ],
    )
    def test_a_client_error_was_not_delivered(self, code: int, reason: str) -> None:
        error = failure(status(code))

        assert (error.reason, error.status_code) == (reason, code)
        assert error.state is DeliveryState.FAILED
        assert_clean(error)

    @pytest.mark.parametrize("code", [500, 502, 503, 504])
    def test_a_server_error_may_have_been_delivered(self, code: int) -> None:
        error = failure(status(code))

        assert (error.reason, error.status_code) == ("server_error", code)
        assert error.state is DeliveryState.UNKNOWN
        assert_clean(error)

    def test_ok_false_on_200_was_not_delivered(self) -> None:
        def handle(_: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, json={"ok": False, "description": TOKEN})

        error = failure(handle)

        assert error.reason == "api_error"
        assert error.state is DeliveryState.FAILED
        assert_clean(error)

    @pytest.mark.parametrize(
        "response",
        [
            httpx2.Response(200, text="not json"),
            httpx2.Response(200, json=[1]),
            httpx2.Response(200, json={"ok": True}),
            httpx2.Response(200, json={"ok": True, "result": {"message_id": "x"}}),
            httpx2.Response(200, json={"ok": True, "result": {"message_id": True}}),
        ],
    )
    def test_an_unreadable_success_has_an_unknown_state(self, response: httpx2.Response) -> None:
        error = failure(lambda _: response)

        assert error.reason == "invalid_response"
        assert error.state is DeliveryState.UNKNOWN
        assert_clean(error)


class TestTransportErrors:
    @pytest.mark.parametrize(
        ("exc", "reason"),
        [
            (httpx2.ConnectError(f"failed https://api.telegram.org/bot{TOKEN}/x"), "connect_error"),
            (httpx2.ConnectTimeout(f"failed bot{TOKEN}"), "connect_error"),
            (httpx2.PoolTimeout(f"failed bot{TOKEN}"), "connect_error"),
        ],
    )
    def test_no_connection_means_not_delivered(self, exc: Exception, reason: str) -> None:
        error = failure(raising(exc))

        assert error.reason == reason
        assert error.state is DeliveryState.FAILED
        assert_clean(error)

    @pytest.mark.parametrize(
        "exc",
        [
            httpx2.ReadTimeout(f"failed bot{TOKEN}"),
            httpx2.WriteTimeout(f"failed bot{TOKEN}"),
            httpx2.RemoteProtocolError(f"failed bot{TOKEN}"),
            httpx2.ReadError(f"failed bot{TOKEN}"),
        ],
    )
    def test_a_broken_exchange_has_an_unknown_state(self, exc: Exception) -> None:
        error = failure(raising(exc))

        assert error.reason == "network_error"
        assert error.state is DeliveryState.UNKNOWN
        assert_clean(error)

    def test_an_unexpected_exception_is_clean_and_unknown(self) -> None:
        error = failure(raising(RuntimeError(f"boom https://api.telegram.org/bot{TOKEN}/x")))

        assert error.reason == "unexpected_error"
        assert error.state is DeliveryState.UNKNOWN
        assert_clean(error)


class TestTokenFormat:
    @pytest.mark.parametrize(
        "token", ["no-colon", "abc:def", "123:", "123:a/b", "123:a b", "123:a?x=1", "123:a#b"]
    )
    def test_a_malformed_token_never_reaches_the_network(self, token: str) -> None:
        calls = 0

        def handle(request: httpx2.Request) -> httpx2.Response:
            nonlocal calls
            calls += 1
            return ok(request)

        with pytest.raises(TelegramError) as caught:
            sender(handle, token=token).send("hello")

        assert caught.value.reason == "invalid_token_format"
        assert caught.value.state is DeliveryState.FAILED
        assert token not in str(caught.value)
        assert calls == 0


class TestSecretsInLogs:
    def test_http_loggers_are_at_warning(self) -> None:
        for name in HTTP_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)

        quiet_http_loggers()

        assert {"httpx", "httpx2", "httpcore", "httpcore2"} <= set(HTTP_LOGGERS)
        for name in HTTP_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING

    def test_token_is_not_logged_on_success(self, caplog: pytest.LogCaptureFixture) -> None:
        logging.getLogger("httpx2").setLevel(logging.DEBUG)
        caplog.set_level(logging.DEBUG)

        sender(ok).send("hello")

        assert TOKEN not in caplog.text
        assert "api.telegram.org" not in caplog.text

    def test_token_is_not_logged_on_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        logging.getLogger("httpx2").setLevel(logging.DEBUG)
        caplog.set_level(logging.DEBUG)

        failure(status(401))
        failure(raising(httpx2.ConnectError(f"failed bot{TOKEN}")))

        assert TOKEN not in caplog.text
        assert "api.telegram.org" not in caplog.text

    def test_repr_hides_the_secrets(self) -> None:
        rendered = repr(sender(ok))

        assert TOKEN not in rendered
        assert CHAT_ID not in rendered


class TestClientSettings:
    def test_honours_proxy_environment_and_never_follows_redirects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}
        real_client = httpx2.Client

        def recording_client(**kwargs: object) -> httpx2.Client:
            captured.update(kwargs)
            return real_client(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr("review_radar.telegram.httpx2.Client", recording_client)

        assert sender(ok).send("hello") == 42

        assert captured["trust_env"] is True
        assert captured["follow_redirects"] is False
