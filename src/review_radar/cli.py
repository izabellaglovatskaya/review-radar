"""Command line interface for Review Radar."""

from pathlib import Path
from typing import Annotated

import typer

from review_radar import __version__
from review_radar.ingest import IngestError, IngestResult, read_reviews

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
    typer.echo(f"Imported: {len(result.reviews)}")
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
    """Read a reviews CSV and print a summary (no review text is printed)."""
    if not dry_run:
        typer.echo(
            "Saving to the database is not implemented yet (stage 3); use --dry-run.", err=True
        )
        raise typer.Exit(2)
    try:
        result = read_reviews(path, delimiter=delimiter, encoding=encoding)
    except IngestError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None
    _echo_summary(result)
    if not result.reviews:
        raise typer.Exit(1)
