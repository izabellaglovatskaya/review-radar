"""Command line interface for Review Radar."""

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError

from review_radar import __version__
from review_radar.analysis import AnalysisAborted, AnalysisReport, analyze_pending
from review_radar.classifier import Classifier, ClassifierError
from review_radar.config import PROJECT_ROOT, ConfigError, Mode, Settings
from review_radar.fixture_classifier import FixtureClassifier
from review_radar.ingest import IngestError, IngestResult, read_reviews
from review_radar.storage import SaveResult, StorageError, connect, count_reviews, save_reviews

MAX_LISTED_LINES = 10
DEMO_FIXTURES = PROJECT_ROOT / "fixtures" / "demo_analyses.json"

app = typer.Typer(
    name="review-radar",
    help="Turn a CSV of customer reviews into a daily summary and a sentiment dashboard.",
    no_args_is_help=True,
    add_completion=False,
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


@app.command()
def analyze() -> None:
    """Classify stored reviews that have no cached analysis yet (no review text is printed)."""
    settings = _load_settings()
    try:
        mode = settings.resolved_mode()
    except ConfigError as exc:
        _fail(exc)
    if mode is Mode.LIVE:
        typer.echo(
            "Error: Live mode is not available yet (stage 5); set REVIEW_RADAR_MODE=demo.",
            err=True,
        )
        raise typer.Exit(2)
    try:
        classifier = FixtureClassifier.from_file(DEMO_FIXTURES)
        report = _analyze(settings, classifier)
    except AnalysisAborted as exc:
        _echo_analysis(classifier, exc.report)
        _fail(f"analysis stopped: {_abort_detail(exc)}.")
    except (StorageError, ClassifierError) as exc:
        _fail(exc)
    if report is None:
        typer.echo("No reviews in the database; run ingest first.", err=True)
        raise typer.Exit(1)
    _echo_analysis(classifier, report)
    if report.failed or report.unexpected_results:
        raise typer.Exit(1)


def _abort_detail(exc: AnalysisAborted) -> str:
    """Name the cause: a StorageError message is text-free; anything else shows only its type."""
    cause = exc.__cause__
    if isinstance(cause, StorageError):
        return str(cause).rstrip(".")
    return type(cause).__name__


def _analyze(settings: Settings, classifier: Classifier) -> AnalysisReport | None:
    """Return the report, or None when the database holds no reviews."""
    with contextlib.closing(connect(settings.database_path)) as conn:
        if count_reviews(conn) == 0:
            return None
        return analyze_pending(conn, classifier)


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
    ):
        if value:
            typer.echo(f"{label}: {value}")


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
