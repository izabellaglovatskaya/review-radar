"""The durable spend ledger: a call is reserved before it is made and settled after it."""

import contextlib
import sqlite3
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import anthropic
import pytest

from anthropic_fakes import (
    FakeClient,
    connection_error,
    make_message,
    response_validation_error,
    result,
    status_error,
    timeout_error,
)
from review_radar import storage
from review_radar.classifier import FatalClassifierError
from review_radar.evaluation import GuardedClient, SpendGuard, SpendLedger
from review_radar.models import Usage
from review_radar.pricing import estimate_cost
from review_radar.storage import (
    RunRecord,
    StorageError,
    connect,
    ledger_total,
    open_spend_ledger,
    reserve_spend,
    save_run,
    settle_spend,
)

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "ledger.db"


@pytest.fixture
def conn(path: Path) -> Iterator[sqlite3.Connection]:
    with contextlib.closing(connect(path)) as connection:
        yield connection


def request(*, text: str = "u" * 100, max_tokens: int = 1000, **extra: Any) -> dict[str, Any]:
    return {
        "model": HAIKU,
        "max_tokens": max_tokens,
        "system": "s" * 100,
        "messages": [{"role": "user", "content": text}],
        **extra,
    }


def worst(input_tokens: int = 200, output_tokens: int = 1000) -> Decimal:
    cost = estimate_cost(HAIKU, Usage(input_tokens=input_tokens, output_tokens=output_tokens))
    assert cost is not None
    return cost


def ok():
    return make_message([result("r1")], input_tokens=100, output_tokens=20)


ACTUAL = Decimal("0.00002")  # 100 input and 20 output tokens on Haiku


def guarded(
    conn: sqlite3.Connection, client: Any, limit: str = "1.00"
) -> tuple[GuardedClient, SpendGuard]:
    guard = SpendGuard(Decimal(limit), ledger_total(conn), SpendLedger(conn))
    return GuardedClient(client, guard, HAIKU), guard


class TestStorage:
    def test_an_empty_ledger_is_zero(self, conn: sqlite3.Connection) -> None:
        assert ledger_total(conn) == 0

    def test_a_reservation_counts_until_settled(self, conn: sqlite3.Connection) -> None:
        entry = reserve_spend(conn, HAIKU, Decimal("0.05"))
        assert ledger_total(conn) == Decimal("0.05")

        settle_spend(conn, entry, Decimal("0.01"), input_tokens=10, output_tokens=5)

        assert ledger_total(conn) == Decimal("0.01")

    def test_a_reservation_survives_a_crash(self, path: Path) -> None:
        with contextlib.closing(connect(path)) as first:
            reserve_spend(first, HAIKU, Decimal("0.05"))

        with contextlib.closing(connect(path)) as second:
            assert ledger_total(second) == Decimal("0.05")

    def test_an_entry_cannot_be_settled_twice(self, conn: sqlite3.Connection) -> None:
        entry = reserve_spend(conn, HAIKU, Decimal("0.05"))
        settle_spend(conn, entry, Decimal("0.01"))

        with pytest.raises(StorageError):
            settle_spend(conn, entry, Decimal("0.02"))

        assert ledger_total(conn) == Decimal("0.01")

    def test_an_unknown_entry_is_a_clean_error(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(StorageError):
            settle_spend(conn, 999, Decimal("0.01"))

    def test_amounts_keep_their_exact_decimal_value(self, conn: sqlite3.Connection) -> None:
        for _ in range(3):
            settle_spend(conn, reserve_spend(conn, HAIKU, Decimal("1")), Decimal("0.1"))

        assert ledger_total(conn) == Decimal("0.3")

    def test_the_schema_version_is_six(self) -> None:
        assert storage.SCHEMA_VERSION == 6
        assert storage.MIN_UPGRADABLE_VERSION == 2


def run_row(cost: str) -> RunRecord:
    import datetime as dt

    now = dt.datetime(2026, 11, 23, tzinfo=dt.UTC)
    return RunRecord(
        started_at=now,
        finished_at=now,
        provider="anthropic",
        model=HAIKU,
        prompt_version="p1",
        reviews_total=1,
        analyzed=1,
        cached=0,
        failed=0,
        input_tokens=1,
        output_tokens=1,
        cost_usd=Decimal(cost),
        fatal_reason=None,
    )


class TestOpeningBalance:
    def test_spend_recorded_before_the_ledger_is_carried_over(
        self, conn: sqlite3.Connection
    ) -> None:
        save_run(conn, run_row("0.215"))

        open_spend_ledger(conn)

        assert ledger_total(conn) == Decimal("0.215")

    def test_it_is_idempotent(self, conn: sqlite3.Connection) -> None:
        save_run(conn, run_row("0.215"))

        open_spend_ledger(conn)
        open_spend_ledger(conn)

        assert ledger_total(conn) == Decimal("0.215")

    def test_later_runs_are_not_counted_twice(self, conn: sqlite3.Connection) -> None:
        open_spend_ledger(conn)
        entry = reserve_spend(conn, HAIKU, Decimal("0.5"))
        settle_spend(conn, entry, Decimal("0.1"))
        save_run(conn, run_row("0.1"))

        open_spend_ledger(conn)

        assert ledger_total(conn) == Decimal("0.1")

    def test_no_spend_adds_no_row(self, conn: sqlite3.Connection) -> None:
        open_spend_ledger(conn)

        assert conn.execute("SELECT COUNT(*) FROM spend_ledger").fetchone()[0] == 0


class TestGuardWithLedger:
    def test_the_reservation_is_durable_before_the_call_is_made(
        self, conn: sqlite3.Connection
    ) -> None:
        seen: list[Decimal] = []

        class Spy:
            messages = None

            def create(self, **_: Any) -> Any:
                seen.append(ledger_total(conn))
                return ok()

        spy = Spy()
        spy.messages = spy
        client, _ = guarded(conn, spy)

        client.messages.create(**request())

        assert seen == [worst()]

    def test_a_success_settles_at_the_real_cost(self, conn: sqlite3.Connection) -> None:
        client, guard = guarded(conn, FakeClient(ok()))

        client.messages.create(**request())

        assert ledger_total(conn) == ACTUAL
        assert guard.spent == ACTUAL

    @pytest.mark.parametrize("code", [400, 401, 429, 500, 529])
    def test_an_error_response_releases_the_reservation(
        self, conn: sqlite3.Connection, code: int
    ) -> None:
        client, guard = guarded(conn, FakeClient(status_error(code)))

        with pytest.raises(anthropic.APIStatusError):
            client.messages.create(**request())

        assert ledger_total(conn) == 0
        assert guard.spent == 0

    @pytest.mark.parametrize("error", [timeout_error, connection_error, response_validation_error])
    def test_a_call_that_may_have_been_billed_keeps_the_worst_case(
        self, conn: sqlite3.Connection, error: Any
    ) -> None:
        client, guard = guarded(conn, FakeClient(error()))

        with pytest.raises(anthropic.APIError):
            client.messages.create(**request())

        assert ledger_total(conn) == worst()
        assert guard.spent == worst()

    def test_a_response_without_usage_keeps_the_worst_case(self, conn: sqlite3.Connection) -> None:
        class NoUsage:
            usage = None

        fake = FakeClient(NoUsage())  # type: ignore[arg-type]
        client, guard = guarded(conn, fake)

        client.messages.create(**request())

        assert ledger_total(conn) == worst()
        assert guard.spent == worst()

    def test_ctrl_c_during_the_call_keeps_the_reservation(self, path: Path) -> None:
        with contextlib.closing(connect(path)) as conn:

            class Interrupting:
                @property
                def messages(self) -> "Interrupting":
                    return self

                def create(self, **_: Any) -> Any:
                    raise KeyboardInterrupt

            client, _ = guarded(conn, Interrupting())
            with pytest.raises(KeyboardInterrupt):
                client.messages.create(**request())

        with contextlib.closing(connect(path)) as again:
            assert ledger_total(again) == worst()

    def test_a_ledger_that_cannot_reserve_stops_before_the_call(
        self, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeClient(ok())
        client, guard = guarded(conn, fake)

        def broken(*_: object, **__: object) -> int:
            raise StorageError("Cannot record the spend: disk full.")

        monkeypatch.setattr(guard.ledger, "reserve", broken)

        with pytest.raises(FatalClassifierError) as caught:
            client.messages.create(**request())

        assert caught.value.reason == "spend_ledger_failed"
        assert fake.calls == []

    def test_a_failed_settle_keeps_the_paid_answer_and_the_safe_total(
        self, conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        message = ok()
        client, guard = guarded(conn, FakeClient(message))

        def broken(*_: object, **__: object) -> None:
            raise StorageError("Cannot record the spend: disk full.")

        monkeypatch.setattr(guard.ledger, "settle", broken)

        assert client.messages.create(**request()) is message
        assert ledger_total(conn) == worst()  # stays reserved: over-counted, never under-counted
        assert guard.spent == worst()


class TestWorstCase:
    def test_the_output_schema_counts_as_input(self) -> None:
        schema = {"format": {"type": "json_schema", "schema": {"x": "y" * 2000}}}
        fake = FakeClient(ok())
        guard = SpendGuard(worst() + Decimal("0.00001"))
        client = GuardedClient(fake, guard, HAIKU)

        with pytest.raises(FatalClassifierError):
            client.messages.create(**request(output_config=schema))

        assert fake.calls == []

    def test_non_ascii_text_is_counted_in_bytes(self) -> None:
        fake = FakeClient(ok())
        guard = SpendGuard(worst() + Decimal("0.000005"))  # room for 50 more input tokens
        client = GuardedClient(fake, guard, HAIKU)

        with pytest.raises(FatalClassifierError):
            client.messages.create(**request(text="\U0001f600" * 100))

        assert fake.calls == []
