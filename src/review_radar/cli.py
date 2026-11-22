"""Command line interface for Review Radar."""

import contextlib
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError

from review_radar import __version__
from review_radar.config import Settings
from review_radar.ingest import IngestError, IngestResult, read_reviews
from review_radar.storage import SaveResult, StorageError, connect, save_reviews

MAX_LISTED_LINES = 10

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
) -> None:
    """Review Radar command line interface."""


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
