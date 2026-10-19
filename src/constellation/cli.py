"""Command-line entry point for Constellation."""

import typer

from constellation import __version__

app = typer.Typer(help="Constellation: spacecraft telemetry health monitoring.")


@app.callback()
def main() -> None:
    """Constellation command-line interface."""


@app.command()
def version() -> None:
    """Print the installed Constellation version."""
    typer.echo(__version__)
