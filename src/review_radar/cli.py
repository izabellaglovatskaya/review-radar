"""Command line interface for Review Radar."""

from typing import Annotated

import typer

from review_radar import __version__

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
