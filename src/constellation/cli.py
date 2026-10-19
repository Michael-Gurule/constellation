"""Command-line entry point for Constellation."""

import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from constellation import __version__
from constellation.core.logging import configure_logging
from constellation.core.settings import get_settings
from constellation.core.store import ParquetStore
from constellation.core.timeutils import parse_duration

app = typer.Typer(help="Constellation: spacecraft telemetry health monitoring.")
collect_app = typer.Typer(help="Collect telemetry into the event-time store.")
app.add_typer(collect_app, name="collect")

DEFAULT_CHANNELS = Path("configs/ingest/iss.yaml")
STALE_AFTER = pd.Timedelta(5, unit="min")


def readings_store() -> ParquetStore:
    """The store that collectors write to, under the configured data directory."""
    return ParquetStore(get_settings().data_dir / "readings")


@app.callback()
def main() -> None:
    """Constellation command-line interface."""
    configure_logging(get_settings().log_level)


@app.command()
def version() -> None:
    """Print the installed Constellation version."""
    typer.echo(__version__)


@collect_app.command("iss")
def collect_iss(
    duration: Annotated[
        str | None,
        typer.Option(help="Stop after this long, for example 90s, 2h or 7d."),
    ] = None,
    probe: Annotated[
        int | None,
        typer.Option(help="Record raw updates for this many seconds and exit."),
    ] = None,
    probe_out: Annotated[
        Path, typer.Option(help="Where --probe saves its trimmed JSON lines.")
    ] = Path("tests/fixtures/isslive_probe.jsonl"),
    config: Annotated[
        Path, typer.Option(help="Channel configuration.")
    ] = DEFAULT_CHANNELS,
) -> None:
    """Stream the configured ISS channels into the store until stopped."""
    from constellation.ingest.channels import load_channels
    from constellation.ingest.isslive import TIME_ITEM, YEAR_ITEM, IssLiveCollector
    from constellation.ingest.isslive import probe as run_probe

    channels = load_channels(config).channels
    settings = get_settings()

    if probe is not None:
        items = [c.id for c in channels] + [TIME_ITEM, YEAR_ITEM]
        typer.echo(f"probing {len(items)} items for {probe} s ...")
        counts = run_probe(items, probe, probe_out)
        silent = sorted(i for i in items if i not in counts)
        static = sorted(i for i, n in counts.items() if n == 1)
        typer.echo(
            f"{sum(counts.values())} updates on {len(counts)} items; saved {probe_out}"
        )
        typer.echo(f"no update at all: {silent}")
        typer.echo(f"one update only (snapshot): {static}")
        return

    collector = IssLiveCollector(
        channels,
        readings_store(),
        flush_seconds=settings.iss_flush_seconds,
    )
    seconds = parse_duration(duration).total_seconds() if duration else None
    logging.getLogger(__name__).info(
        "collecting %d channels into %s", len(channels), collector.store.root
    )
    collector.run(seconds)
    typer.echo(f"stopped; counts: {collector.counts}")


@collect_app.command("status")
def collect_status(
    by_channel: Annotated[bool, typer.Option(help="List every channel.")] = False,
) -> None:
    """Summarize what the store holds and how fresh it is."""
    store = readings_store()
    stats = store.stats()
    if stats.empty:
        typer.echo(f"no readings under {store.root}")
        raise typer.Exit(code=1)

    now = pd.Timestamp.now(tz="UTC")
    typer.echo(f"store: {store.root}")
    for (source, asset), group in stats.groupby(["source", "asset_id"]):
        last = group["last_t"].max()
        age = now - last
        flag = "STALE" if age > STALE_AFTER else "ok"
        typer.echo(
            f"{source}/{asset}: {int(group['rows'].sum())} rows, "
            f"{len(group)} channels, {group['first_t'].min():%Y-%m-%d %H:%M} to "
            f"{last:%Y-%m-%d %H:%M} UTC, last update {age.total_seconds():.0f} s ago [{flag}]"
        )
        if by_channel:
            for row in group.itertuples():
                typer.echo(
                    f"  {row.channel:28s} {row.rows:>9d}  last {row.last_t:%H:%M:%S}"
                )
