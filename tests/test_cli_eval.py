"""`review-radar eval`: estimate, spend limits, caching, reports and demo mode.

No test calls the network: paid runs go through the `cli.make_eval_client` seam with a fake
client behind a real `GuardedClient`. Results live in a separate `<stem>-eval.db` that sits next
to the (temporary) DATABASE_PATH. All reviews and keys are fictional.
"""

import contextlib
import csv
import json
import os
import re
import socket
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner, Result

from anthropic_fakes import FAKE_KEY, FAKE_TEXT, make_message, result, status_error
from review_radar import cli
from review_radar.anthropic_classifier import PROMPT_VERSION
from review_radar.cli import app
from review_radar.evaluation import DATASETS, GuardedClient, SpendGuard, estimate_cost_for
from review_radar.hashing import text_hash
from review_radar.models import Review, Usage
from review_radar.pricing import estimate_cost
from review_radar.storage import (
    RunRecord,
    StorageError,
    connect,
    list_runs,
    load_analyses,
    save_reviews,
)

runner = CliRunner()
HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"


def main_db() -> Path:
    return Path(os.environ["DATABASE_PATH"])


def eval_db() -> Path:
    return main_db().with_name(f"{main_db().stem}-eval.db")


def read_texts(*names: str) -> list[str]:
    texts: list[str] = []
    for name in names:
        with DATASETS[name].csv.open(encoding="utf-8-sig", newline="") as handle:
            texts += [row["text"] for row in csv.DictReader(handle)]
    return texts


def distinct_texts(*names: str) -> list[str]:
    """One text per cache key: the texts the model has to see."""
    return list({text_hash(text): text for text in read_texts(*names)}.values())


def snippets() -> list[str]:
    """Distinctive pieces of real fixture reviews that must never be printed."""
    return [text[:30] for text in read_texts("regular")[:20] + read_texts("hard")[:20]]


def runs() -> list[RunRecord]:
    with contextlib.closing(connect(eval_db())) as conn:
        return list_runs(conn)


def stored(model: str) -> int:
    with contextlib.closing(connect(eval_db())) as conn:
        return len(
            load_analyses(conn, provider="anthropic", model=model, prompt_version=PROMPT_VERSION)
        )


def spent() -> Decimal:
    return sum((record.cost_usd or Decimal(0) for record in runs()), Decimal(0))


def shows(output: str, amount: Decimal) -> bool:
    """True if `amount` appears in the output rounded or cut to 3 to 6 decimals."""
    candidates: set[str] = set()
    for places in range(3, 7):
        for rounding in (ROUND_HALF_UP, ROUND_DOWN):
            text = format(amount.quantize(Decimal(1).scaleb(-places), rounding), "f")
            candidates |= {text, text.rstrip("0")}
    return any(candidate in output for candidate in candidates)


def flat(result_: Result) -> str:
    """Output with thousands separators removed, so token counts can be searched as digits."""
    return result_.output.replace(",", "").replace("_", "")


def run_eval(*args: str) -> Result:
    return runner.invoke(app, ["eval", *args])


def assert_clean(result_: Result, *extra: str) -> None:
    """No review text, key or SDK message anywhere in what the command printed or wrote."""
    for text in (result_.output, *extra):
        assert FAKE_KEY not in text
        assert FAKE_TEXT not in text
        assert "Traceback" not in text
        for snippet in snippets():
            assert snippet not in text


def forbid(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("this must not be called")


def go_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REVIEW_RADAR_MODE", "live")
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", forbid)
    monkeypatch.setattr(socket, "getaddrinfo", forbid)
    monkeypatch.setattr(cli, "make_telegram_sender", forbid)


class SmartClient:
    """A fake Anthropic client that answers every id it is asked about (always `positive`).

    It reports `factor` times the measured usage per item, so a test can make the real spend
    bigger than the estimate.
    """

    def __init__(self, *, factor: int = 1, error: BaseException | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.items_sent = 0
        self._factor = factor
        self._error = error
        self.messages = self

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        payload = json.loads(kwargs["messages"][0]["content"])
        self.items_sent += len(payload)
        return make_message(
            [result(item["id"], "positive") for item in payload],
            input_tokens=135 * len(payload) * self._factor,
            output_tokens=36 * len(payload) * self._factor,
        )


class Paid:
    """Live mode with a fake key; `cli.make_eval_client` returns a guarded fake client."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        factor: int = 1,
        error: BaseException | None = None,
    ) -> None:
        go_live(monkeypatch)
        self.client = SmartClient(factor=factor, error=error)
        self.built: list[tuple[Any, str, SpendGuard]] = []
        monkeypatch.setattr(cli, "make_eval_client", self.make)
        monkeypatch.setattr(cli, "make_anthropic_classifier", forbid)

    def make(self, api_key: Any = None, model: str = "", guard: Any = None, *_a: Any, **_k: Any):
        self.built.append((api_key, model, guard))
        return GuardedClient(self.client, guard, model)

    @property
    def models_called(self) -> set[str]:
        return {call["model"] for call in self.client.calls}


@pytest.fixture
def paid(monkeypatch: pytest.MonkeyPatch) -> Paid:
    return Paid(monkeypatch)


@pytest.fixture
def no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live mode in which building any client fails the test."""
    go_live(monkeypatch)
    monkeypatch.setattr(cli, "make_eval_client", forbid)
    monkeypatch.setattr(cli, "make_anthropic_classifier", forbid)
    monkeypatch.setattr(cli, "_build_classifier", forbid)


class TestEstimateWithoutYes:
    def test_prints_the_estimate_and_sends_nothing(self, no_client: None) -> None:
        texts = distinct_texts("regular", "hard")
        estimate = estimate_cost_for(HAIKU, texts)

        result_ = run_eval()

        assert result_.exit_code == 0, result_.output
        text = flat(result_)
        assert HAIKU in text
        assert str(len(texts)) in text
        assert str(estimate.input_tokens) in text
        assert str(estimate.output_tokens) in text
        assert shows(text, estimate.expected_usd)
        assert shows(text, estimate.upper_usd)
        assert re.search(r"nothing[^.\n]*sent", text.lower())
        assert_clean(result_)

    def test_the_main_database_is_never_created(self, no_client: None) -> None:
        assert run_eval().exit_code == 0

        assert not main_db().exists()

    def test_one_dataset_estimates_only_its_texts(self, no_client: None) -> None:
        texts = distinct_texts("hard")
        estimate = estimate_cost_for(HAIKU, texts)

        result_ = run_eval("--dataset", "hard")

        assert result_.exit_code == 0, result_.output
        text = flat(result_)
        assert str(estimate.input_tokens) in text
        assert shows(text, estimate.upper_usd)

    def test_compare_estimates_both_models_and_a_total(self, no_client: None) -> None:
        texts = distinct_texts("regular", "hard")
        haiku, sonnet = estimate_cost_for(HAIKU, texts), estimate_cost_for(SONNET, texts)

        result_ = run_eval("--compare")

        assert result_.exit_code == 0, result_.output
        text = flat(result_)
        assert HAIKU in text
        assert SONNET in text
        assert shows(text, sonnet.expected_usd)
        assert shows(text, sonnet.upper_usd)
        assert shows(text, haiku.expected_usd + sonnet.expected_usd)  # the total
        assert shows(text, haiku.upper_usd + sonnet.upper_usd)

    def test_two_estimates_in_a_row_are_identical(self, no_client: None) -> None:
        # An estimate records nothing, so repeating it gives the same numbers and still no call.
        first, second = run_eval(), run_eval()

        assert first.exit_code == second.exit_code == 0
        assert first.output == second.output


class TestRefusals:
    def test_an_upper_estimate_above_max_cost_refuses_before_any_call(self, paid: Paid) -> None:
        upper = estimate_cost_for(HAIKU, distinct_texts("regular", "hard")).upper_usd

        result_ = run_eval("--yes", "--max-cost", "0.001")

        assert result_.exit_code == 1
        assert paid.client.calls == []
        text = flat(result_)
        assert shows(text, Decimal("0.001"))
        assert shows(text, upper)
        assert_clean(result_)

    def test_the_cumulative_spend_of_earlier_runs_counts_against_max_cost(self, paid: Paid) -> None:
        first = run_eval("--dataset", "regular", "--yes", "--max-cost", "1.00")
        assert first.exit_code == 0, first.output
        already = spent()
        assert already > 0
        calls_before = len(paid.client.calls)
        cap = already + Decimal("0.001")  # the hard set alone would fit under this cap
        assert estimate_cost_for(HAIKU, distinct_texts("hard")).upper_usd < cap

        second = run_eval("--dataset", "hard", "--yes", "--max-cost", str(cap))

        assert second.exit_code == 1
        assert spent() <= cap
        assert len(paid.client.calls) == calls_before  # not even one more call fits
        assert_clean(second)

    def test_an_unknown_dataset_is_a_usage_error(self) -> None:
        assert run_eval("--dataset", "bogus").exit_code == 2

    def test_a_bad_max_cost_is_a_usage_error(self) -> None:
        assert run_eval("--max-cost", "plenty").exit_code == 2


class TestPaidRun:
    def test_one_dataset_saves_its_analyses_and_one_run_row(self, paid: Paid) -> None:
        result_ = run_eval("--dataset", "regular", "--yes")

        assert result_.exit_code == 0, result_.output
        rows = runs()
        assert len(rows) == 1
        assert rows[0].provider == "anthropic"
        assert rows[0].model == HAIKU
        assert rows[0].prompt_version == PROMPT_VERSION
        assert rows[0].cost_usd is not None
        assert rows[0].cost_usd > 0
        assert rows[0].cost_usd == estimate_cost(
            HAIKU, Usage(input_tokens=rows[0].input_tokens, output_tokens=rows[0].output_tokens)
        )
        assert stored(HAIKU) == len(distinct_texts("regular"))
        assert paid.client.items_sent == len(distinct_texts("regular"))
        assert [model for _, model, _ in paid.built] == [HAIKU]
        assert_clean(result_)

    def test_all_datasets_record_a_run_row_per_dataset(self, paid: Paid) -> None:
        result_ = run_eval("--yes")

        assert result_.exit_code == 0, result_.output
        assert len(runs()) == 2
        assert {row.model for row in runs()} == {HAIKU}
        assert stored(HAIKU) == len(distinct_texts("regular", "hard"))
        assert sum(row.input_tokens for row in runs()) == 135 * len(
            distinct_texts("regular", "hard")
        )

    def test_the_client_gets_the_key_the_model_and_a_guard(self, paid: Paid) -> None:
        assert run_eval("--dataset", "hard", "--yes").exit_code == 0

        key, model, guard = paid.built[0]
        assert key.get_secret_value() == FAKE_KEY
        assert model == HAIKU
        assert isinstance(guard, SpendGuard)
        assert 0 < guard.limit <= Decimal("1.00")

    def test_compare_runs_both_models(self, paid: Paid) -> None:
        result_ = run_eval("--compare", "--yes")

        assert result_.exit_code == 0, result_.output
        assert {model for _, model, _ in paid.built} == {HAIKU, SONNET}
        assert paid.models_called == {HAIKU, SONNET}
        assert len(runs()) == 4
        assert {row.model for row in runs()} == {HAIKU, SONNET}
        assert stored(HAIKU) == stored(SONNET) == len(distinct_texts("regular", "hard"))

    def test_without_compare_only_the_configured_model_is_used(self, paid: Paid) -> None:
        assert run_eval("--yes").exit_code == 0

        assert paid.models_called == {HAIKU}

    def test_a_second_identical_run_makes_no_call_and_costs_nothing(self, paid: Paid) -> None:
        assert run_eval("--yes").exit_code == 0
        calls, total = len(paid.client.calls), spent()
        assert calls > 0

        second = run_eval("--yes")

        assert second.exit_code == 0, second.output
        assert len(paid.client.calls) == calls
        assert spent() == total

    def test_the_main_database_is_never_written(self, paid: Paid) -> None:
        with contextlib.closing(connect(main_db())) as conn:
            save_reviews(
                conn,
                [
                    Review(
                        external_id="m1",
                        review_date="2026-11-10",
                        rating=3,
                        text="mine",
                        source="web",
                    )
                ],
            )
        before = main_db().read_bytes()

        assert run_eval("--yes").exit_code == 0

        assert main_db().read_bytes() == before
        assert eval_db().is_file()
        assert eval_db() != main_db()

    def test_the_main_database_is_not_created(self, paid: Paid) -> None:
        assert run_eval("--yes").exit_code == 0

        assert not main_db().exists()
        assert eval_db().name == f"{main_db().stem}-eval.db"
        assert eval_db().is_file()


class TestBudgetDuringTheRun:
    def test_the_guard_stops_the_run_and_the_partial_result_is_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The real usage is four times the estimate, so the estimate passes but the guard stops.
        paid = Paid(monkeypatch, factor=4)
        texts = distinct_texts("regular")
        cap = Decimal("0.0145")
        assert estimate_cost_for(HAIKU, texts).upper_usd < cap

        result_ = run_eval("--dataset", "regular", "--yes", "--max-cost", str(cap))

        assert result_.exit_code == 1
        assert re.search(r"(?i)limit|budget", result_.output)
        failed = re.search(r"(?i)failed\D{0,30}([1-9]\d*)", result_.output)
        assert failed is not None, result_.output  # the partial report counts the failed reviews
        assert paid.client.calls  # something ran before the stop
        assert 0 < stored(HAIKU) < len(texts)
        recorded = runs()
        assert recorded
        assert any(row.fatal_reason == "budget_exceeded" for row in recorded)
        assert 0 < spent() <= cap  # the spend is recorded and the cap held
        assert_clean(result_)

    def test_a_fatal_api_error_is_reported_without_leaking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paid = Paid(monkeypatch, error=status_error(401))

        result_ = run_eval("--dataset", "hard", "--yes")

        assert result_.exit_code == 1
        assert len(paid.client.calls) >= 1
        assert_clean(result_)


class TestReport:
    def test_a_paid_run_writes_a_live_markdown_report(self, paid: Paid, tmp_path: Path) -> None:
        path = tmp_path / "reports" / "eval.md"
        path.parent.mkdir()

        result_ = run_eval("--yes", "--report", str(path))

        assert result_.exit_code == 0, result_.output
        text = path.read_text(encoding="utf-8")
        assert HAIKU in text
        assert "live" in text.lower()
        assert "regular" in text
        assert "hard" in text
        assert "|" in text
        assert [item.name for item in path.parent.iterdir()] == ["eval.md"]  # no temp file left
        assert_clean(result_, text)

    def test_a_demo_run_writes_a_report_that_says_demo(self, tmp_path: Path) -> None:
        path = tmp_path / "demo.md"

        result_ = run_eval("--report", str(path))

        assert result_.exit_code == 0, result_.output
        text = path.read_text(encoding="utf-8")
        assert "demo" in text.lower()
        assert_clean(result_, text)

    def test_a_failed_replace_keeps_the_old_report_and_leaves_no_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "keep.md"
        path.write_text("OLD REPORT", encoding="utf-8")

        def broken_replace(*_args: object, **_kwargs: object) -> None:
            raise OSError("disk says no")

        monkeypatch.setattr(os, "replace", broken_replace)

        result_ = run_eval("--report", str(path))

        assert result_.exit_code != 0
        assert result_.exception is None or isinstance(result_.exception, SystemExit)
        assert path.read_text(encoding="utf-8") == "OLD REPORT"
        assert [item.name for item in tmp_path.iterdir() if item.suffix != ".db"] == ["keep.md"]


class TestDurableSpend:
    def test_spend_of_a_run_killed_before_its_row_is_saved_still_counts(
        self, paid: Paid, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with monkeypatch.context() as patch:
            patch.setattr(cli, "_record_run", lambda *_a, **_k: None)  # the row is never written
            assert run_eval("--dataset", "regular", "--yes").exit_code == 0
        assert runs() == []
        calls = len(paid.client.calls)

        result_ = run_eval("--dataset", "hard", "--yes", "--max-cost", "0.002")

        assert result_.exit_code == 1
        assert len(paid.client.calls) == calls
        assert_clean(result_)

    def test_spend_recorded_before_the_ledger_counts(self, paid: Paid) -> None:
        with contextlib.closing(connect(eval_db())) as conn:
            conn.execute("DELETE FROM spend_ledger")
            conn.commit()
        assert run_eval("--dataset", "regular", "--yes").exit_code == 0
        with contextlib.closing(connect(eval_db())) as conn:
            conn.execute("DELETE FROM spend_ledger")  # as in a database from before the ledger
            conn.commit()
        calls = len(paid.client.calls)

        result_ = run_eval("--dataset", "hard", "--yes", "--max-cost", str(spent()))

        assert result_.exit_code == 1
        assert len(paid.client.calls) == calls

    def test_a_run_row_that_cannot_be_saved_on_interrupt_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        Paid(monkeypatch, error=KeyboardInterrupt())  # type: ignore[arg-type]

        def broken(_conn: object, _record: object) -> int:
            raise StorageError("Cannot save the run: disk full.")

        monkeypatch.setattr(cli, "save_run", broken)

        result_ = run_eval("--dataset", "hard", "--yes")

        assert result_.exit_code == 130
        assert "Cannot save the run" in result_.stderr

    def test_a_report_of_a_stopped_run_is_marked_incomplete(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        Paid(monkeypatch, factor=4)
        path = tmp_path / "partial.md"

        result_ = run_eval(
            "--dataset", "regular", "--yes", "--max-cost", "0.0145", "--report", str(path)
        )

        assert result_.exit_code == 1
        text = path.read_text(encoding="utf-8")
        assert "INCOMPLETE" in text.split("|")[0]
        assert "INCOMPLETE" in result_.stdout
        assert_clean(result_, text)

    def test_an_interrupted_run_says_the_report_was_not_updated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        Paid(monkeypatch, error=KeyboardInterrupt())  # type: ignore[arg-type]
        path = tmp_path / "old.md"
        path.write_text("OLD REPORT", encoding="utf-8")

        result_ = run_eval("--dataset", "hard", "--yes", "--report", str(path))

        assert result_.exit_code == 130
        assert "report was not updated" in result_.stderr.lower()
        assert path.read_text(encoding="utf-8") == "OLD REPORT"


class TestDemoMode:
    @pytest.fixture(autouse=True)
    def no_client_in_demo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "make_eval_client", forbid)
        monkeypatch.setattr(cli, "make_anthropic_classifier", forbid)

    def test_works_without_a_key_and_says_it_is_demo(self) -> None:
        result_ = run_eval()

        assert result_.exit_code == 0, result_.output
        assert "demo" in result_.output.lower()
        assert "$0" in result_.output
        assert_clean(result_)

    def test_uses_the_fixture_classifier_at_zero_cost(self) -> None:
        assert run_eval().exit_code == 0

        rows = runs()
        assert len(rows) == 2  # regular and hard
        assert {row.provider for row in rows} == {"fixture"}
        assert all(row.cost_usd == 0 for row in rows)

    def test_one_dataset(self) -> None:
        assert run_eval("--dataset", "regular").exit_code == 0

        assert len(runs()) == 1

    def test_never_creates_the_main_database(self) -> None:
        assert run_eval().exit_code == 0

        assert not main_db().exists()
        assert eval_db().is_file()

    def test_a_second_run_works_too(self) -> None:
        assert run_eval().exit_code == 0

        assert run_eval().exit_code == 0

    def test_compare_in_demo_mode_is_a_clear_error(self) -> None:
        result_ = run_eval("--compare")

        assert result_.exit_code == 1
        assert "compare" in result_.output.lower()
        assert "demo" in result_.output.lower()
        assert not main_db().exists()
        assert_clean(result_)
