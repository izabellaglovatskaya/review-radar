"""Command line interface for Review Radar."""

import contextlib
import datetime as dt
import logging
import os
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import typer
from pydantic import SecretStr, ValidationError

from review_radar import __version__
from review_radar.analysis import (
    AUTO,
    AnalysisAborted,
    AnalysisInterrupted,
    AnalysisReport,
    analyze_pending,
)
from review_radar.anthropic_classifier import AnthropicClassifier
from review_radar.classifier import Classifier, ClassifierError
from review_radar.config import PROJECT_ROOT, ConfigError, Mode, Settings
from review_radar.fixture_classifier import FixtureClassifier
from review_radar.ingest import IngestError, IngestResult, read_reviews
from review_radar.models import Usage
from review_radar.pricing import PRICES_PER_MTOK, estimate_cost
from review_radar.storage import (
    RunRecord,
    SaveResult,
    StorageError,
    connect,
    count_reviews,
    save_reviews,
    save_run,
)

logger = logging.getLogger(__name__)

MAX_LISTED_LINES = 10
DEMO_PROVIDER = "fixture"
_EARLY_STOPS = frozenset({"stopped_transient", "call_budget_exceeded"})
DEMO_FIXTURES = PROJECT_ROOT / "fixtures" / "demo_analyses.json"

app = typer.Typer(
    name="review-radar",
    help="Turn a CSV of customer reviews into a daily summary and a sentiment dashboard.",
    no_args_is_help=True,
    add_completion=False,
    # A traceback with locals could print the API key; never show locals.
    pretty_exceptions_show_locals=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"review-radar {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Also log progress (INFO) to stderr.")
    ] = False,
) -> None:
    """Review Radar command line interface."""
    _configure_logging(verbose)


class _StderrHandler(logging.Handler):
    """Write to whatever `sys.stderr` is at emit time (so it survives stream swapping)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + os.linesep)
        except Exception:  # noqa: BLE001 (logging must never break the command)
            self.handleError(record)


def _configure_logging(verbose: bool) -> None:
    """Log the `review_radar` package to stderr: WARNING by default, INFO with --verbose.

    Only the package logger is configured, and any handler of ours from an earlier call is
    replaced, so calling this repeatedly (e.g. in tests) never stacks handlers.
    """
    package_logger = logging.getLogger("review_radar")
    for handler in [h for h in package_logger.handlers if isinstance(h, _StderrHandler)]:
        package_logger.removeHandler(handler)
    handler = _StderrHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO if verbose else logging.WARNING)


def _format_lines(lines: list[int]) -> str:
    shown = ", ".join(str(n) for n in lines[:MAX_LISTED_LINES])
    extra = len(lines) - MAX_LISTED_LINES
    label = "line" if len(lines) == 1 else "lines"
    return f"{label} {shown}" + (f", and {extra} more" if extra > 0 else "")


def _echo_summary(result: IngestResult) -> None:
    typer.echo(f"Rows read: {result.rows_read}")
    typer.echo(f"Valid: {len(result.reviews)}")
    typer.echo(f"Skipped: {len(result.errors)}")
    if result.ignored_columns:
        typer.echo(f"Ignored columns: {', '.join(result.ignored_columns)}")
    for reason, count in result.skipped_by_reason().items():
        lines = [e.line for e in result.errors if e.reason is reason]
        typer.echo(f"  {reason.value}: {count} ({_format_lines(lines)})")


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="CSV file with reviews.")],
    delimiter: Annotated[
        str | None,
        typer.Option(help="Field delimiter (, ; tab or |); detected (, or ;) when omitted."),
    ] = None,
    encoding: Annotated[str, typer.Option(help="File encoding, e.g. cp1251.")] = "utf-8-sig",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Read and validate only; do not save.")
    ] = False,
) -> None:
    """Read a reviews CSV, save it to the database and print a summary (no review text)."""
    settings = None if dry_run else _load_settings()
    try:
        result = read_reviews(path, delimiter=delimiter, encoding=encoding)
    except IngestError as exc:
        _fail(exc)
    _echo_summary(result)
    if not result.reviews:
        raise typer.Exit(1)
    if settings is None:
        typer.echo("Dry run: nothing saved.")
        return
    try:
        saved = _save(result, settings)
    except StorageError as exc:
        _fail(exc)
    typer.echo(f"New: {saved.new}")
    typer.echo(f"Already stored: {saved.existing}")


def make_anthropic_classifier(*, api_key: SecretStr, model: str) -> Classifier:
    """Build the live classifier. Tests replace this function; nothing else builds one."""
    return AnthropicClassifier(api_key=api_key, model=model)


@app.command()
def analyze(
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Classify at most N pending texts this run (the rest wait for a later run).",
        ),
    ] = None,
    max_calls: Annotated[
        int | None,
        typer.Option(
            "--max-calls",
            min=1,
            help="Live mode: stop after N API calls (default: 3 per batch plus 10).",
        ),
    ] = None,
) -> None:
    """Classify stored reviews that have no cached analysis yet (no review text is printed)."""
    settings = _load_settings()
    try:
        mode = settings.resolved_mode()
    except ConfigError as exc:
        _fail(exc)
    run_errors: list[StorageError] = []
    try:
        classifier = _build_classifier(settings, mode)
        report = _analyze(settings, classifier, limit, _call_budget(max_calls, mode), run_errors)
    except AnalysisInterrupted as exc:
        _echo_analysis(classifier, exc.report)
        _echo_run_errors(run_errors)
        typer.echo("Interrupted.", err=True)
        raise typer.Exit(130) from None
    except AnalysisAborted as exc:
        _echo_analysis(classifier, exc.report)
        _echo_run_errors(run_errors)
        _fail(f"analysis stopped: {_abort_detail(exc)}.")
    except (StorageError, ClassifierError) as exc:
        _fail(exc)
    if report is None:
        typer.echo("No reviews in the database; run ingest first.", err=True)
        raise typer.Exit(1)
    _echo_analysis(classifier, report)
    if report.fatal_message is not None:
        typer.echo(f"Error: {report.fatal_message}", err=True)
        lost = report.failures.get("save_failed", 0)
        if lost:
            typer.echo(
                f"Error: {lost} review(s) could not be saved (save_failed); "
                "their paid analyses are lost.",
                err=True,
            )
    _echo_run_errors(run_errors)
    if report.fatal_message is not None or run_errors or report.failed or report.unexpected_results:
        raise typer.Exit(1)


def _call_budget(max_calls: int | None, mode: Mode) -> int | Literal["auto"] | None:
    """The user's budget, else the automatic one in live mode; demo makes no API calls."""
    if mode is not Mode.LIVE:
        return None
    return max_calls if max_calls is not None else AUTO


def _echo_run_errors(run_errors: list[StorageError]) -> None:
    for error in run_errors:
        typer.echo(f"Error: {error}", err=True)


def _build_classifier(settings: Settings, mode: Mode) -> Classifier:
    if mode is Mode.LIVE:
        api_key = settings.anthropic_api_key
        if api_key is None:  # pragma: no cover (resolved_mode() guarantees a key in live mode)
            raise ConfigError("ANTHROPIC_API_KEY is not set.")
        try:
            return make_anthropic_classifier(api_key=api_key, model=settings.anthropic_model)
        except Exception as exc:  # noqa: BLE001 (any failure: name its type, never its text)
            _fail(f"Cannot create the Anthropic client ({type(exc).__name__}).")
    return FixtureClassifier.from_file(DEMO_FIXTURES)


def _abort_detail(exc: AnalysisAborted) -> str:
    """Name the cause: a StorageError message is text-free; anything else shows only its type."""
    cause = exc.__cause__
    if isinstance(cause, StorageError):
        return str(cause).rstrip(".")
    return type(cause).__name__


def _analyze(
    settings: Settings,
    classifier: Classifier,
    limit: int | None,
    max_calls: int | Literal["auto"] | None,
    run_errors: list[StorageError],
) -> AnalysisReport | None:
    """Return the report, or None when the database holds no reviews. Records the run.

    A failure to save the run row is appended to `run_errors` (never raised), so the report
    can still be printed first.
    """
    with contextlib.closing(connect(settings.database_path)) as conn:
        if count_reviews(conn) == 0:
            return None
        started = dt.datetime.now(dt.UTC)
        try:
            report = analyze_pending(
                conn,
                classifier,
                limit=limit,
                max_calls=max_calls,
            )
        except (AnalysisAborted, AnalysisInterrupted) as exc:
            _record_run(conn, classifier, exc.report, started, run_errors)
            raise
        _record_run(conn, classifier, report, started, run_errors)
        return report


def _record_run(
    conn: sqlite3.Connection,
    classifier: Classifier,
    report: AnalysisReport,
    started: dt.datetime,
    run_errors: list[StorageError],
) -> None:
    try:
        save_run(conn, _run_record(classifier, report, started))
    except StorageError as exc:
        logger.warning("Could not record the run: %s", exc)
        run_errors.append(exc)


def _run_cost(classifier: Classifier, usage: Usage) -> Decimal | None:
    """Demo runs cost nothing; live runs use the price table (None for an unknown model)."""
    if classifier.provider == DEMO_PROVIDER:
        return Decimal(0)
    return estimate_cost(classifier.model, usage)


def _run_record(classifier: Classifier, report: AnalysisReport, started: dt.datetime) -> RunRecord:
    usage = report.usage or Usage(input_tokens=0, output_tokens=0)
    return RunRecord(
        started_at=started,
        finished_at=dt.datetime.now(dt.UTC),
        provider=classifier.provider,
        model=classifier.model,
        prompt_version=classifier.prompt_version,
        reviews_total=report.reviews_total,
        analyzed=report.analyzed,
        cached=report.cached,
        failed=report.failed,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=_run_cost(classifier, usage),
        fatal_reason=report.fatal_reason,
        not_analyzed_limit=report.not_analyzed_limit,
        outcome=report.outcome,
        failures=report.failures,
        api_calls=report.api_calls or 0,
    )


def _cost_line(classifier: Classifier, usage: Usage) -> str:
    cost = _run_cost(classifier, usage)
    if classifier.provider == DEMO_PROVIDER:
        return "Estimated cost: $0 (demo)"
    if cost is None:
        return f"Estimated cost: unknown (no price for {classifier.model})"
    price_in, price_out = PRICES_PER_MTOK[classifier.model]
    return (
        f"Estimated cost: ${cost:.6f} "
        f"({classifier.model} at ${price_in}/${price_out} per 1M tokens)"
    )


def _echo_analysis(classifier: Classifier, report: AnalysisReport) -> None:
    typer.echo(
        f"Classifier: {classifier.provider}/{classifier.model} (prompt {classifier.prompt_version})"
    )
    typer.echo(f"Reviews: {report.reviews_total}")
    typer.echo(f"Analyzed now: {report.analyzed}")
    typer.echo(f"From cache: {report.cached}")
    typer.echo(f"Failed: {report.failed}")
    for reason, count in report.failures.items():
        typer.echo(f"  {reason}: {count}")
    for label, value in (
        ("Fallback (keyword rules)", report.fallback),
        ("Unknown topics", report.unknown_topics),
        ("Truncated topics", report.truncated_topics),
        ("Unexpected results", report.unexpected_results),
        ("Not analyzed (limit)", report.not_analyzed_limit),
    ):
        if value:
            typer.echo(f"{label}: {value}")
    if report.stop_reason in _EARLY_STOPS:
        typer.echo(f"Stopped early: {report.stop_reason}")
    if report.max_calls is not None and report.api_calls is not None:
        typer.echo(f"API calls: {report.api_calls} / {report.max_calls}")
    usage = report.usage or Usage(input_tokens=0, output_tokens=0)
    typer.echo(f"Tokens: {usage.input_tokens} in / {usage.output_tokens} out")
    typer.echo(_cost_line(classifier, usage))


def _load_settings() -> Settings:
    """Load settings; on invalid values name only the variables, never the values."""
    try:
        return Settings()
    except ValidationError as exc:
        names = sorted({_setting_name(str(err["loc"][0])) for err in exc.errors() if err["loc"]})
        _fail(f"Invalid setting(s): {', '.join(names)}.")


def _setting_name(field: str) -> str:
    alias = (
        Settings.model_fields[field].validation_alias if field in Settings.model_fields else None
    )
    return alias if isinstance(alias, str) else field


def _save(result: IngestResult, settings: Settings) -> SaveResult:
    with contextlib.closing(connect(settings.database_path)) as conn:
        return save_reviews(conn, result.reviews)


def _fail(exc: object) -> NoReturn:
    typer.echo(f"Error: {exc}", err=True)
    raise typer.Exit(1) from None
