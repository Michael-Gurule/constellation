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
tle_app = typer.Typer(help="Orbital element sets.")
app.add_typer(tle_app, name="tle")
data_app = typer.Typer(help="Datasets.")
app.add_typer(data_app, name="data")
store_app = typer.Typer(help="Maintain the readings store.")
app.add_typer(store_app, name="store")

DEFAULT_CHANNELS = Path("configs/ingest/iss.yaml")
STALE_AFTER = pd.Timedelta(5, unit="min")
PROGRESS_STEP = 100 << 20  # report every 100 MB


def readings_store() -> ParquetStore:
    """The store that collectors write to, under the configured data directory."""
    return ParquetStore(get_settings().data_dir / "readings")


@app.callback()
def main() -> None:
    """Constellation command-line interface."""
    settings = get_settings()
    configure_logging(settings.log_level, file=settings.log_file)


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


@tle_app.command("fetch")
def tle_fetch(
    norad: Annotated[int, typer.Option(help="NORAD catalogue number.")] = 25544,
    force: Annotated[
        bool, typer.Option(help="Download even if the saved copy is recent.")
    ] = False,
) -> None:
    """Download the current TLE from CelesTrak (at most once per 12 hours)."""
    from constellation.ingest.tle import fetch_tle

    directory = get_settings().data_dir / "external" / "tle"
    path, downloaded = fetch_tle(norad, directory, force=force)
    typer.echo(f"{'downloaded' if downloaded else 'kept recent copy:'} {path}")


@data_app.command("fetch")
def data_fetch(
    name: Annotated[
        str | None, typer.Argument(help="smap_msl, esa_adb or opssat; omit with --all.")
    ] = None,
    all_: Annotated[bool, typer.Option("--all", help="Fetch every benchmark.")] = False,
    config: Annotated[Path, typer.Option(help="Benchmark sources.")] = Path(
        "configs/data/benchmarks.yaml"
    ),
) -> None:
    """Download, verify and extract a benchmark into data/external/<name>/."""
    from constellation.data.benchmarks import fetch, load_benchmark_config
    from constellation.data.benchmarks.base import ManualDownloadError, update_manifest

    sources = load_benchmark_config(config).benchmarks
    if all_ and name is not None:
        raise typer.BadParameter("give a benchmark name or --all, not both")
    if not all_ and name is None:
        raise typer.BadParameter("give a benchmark name or --all")
    names = sorted(sources) if all_ else [str(name)]
    unknown = [n for n in names if n not in sources]
    if unknown:
        raise typer.BadParameter(
            f"unknown benchmark {unknown[0]!r}; known: {sorted(sources)}"
        )

    settings = get_settings()
    manifest = settings.data_dir / "manifests" / "benchmarks.json"
    failed = False
    for benchmark in names:
        source = sources[benchmark]
        typer.echo(f"{benchmark}: {source.title} (doi:{source.doi})")
        counter = {"bytes": 0}

        def progress(chunk: int, counter: dict[str, int] = counter) -> None:
            before = counter["bytes"]
            counter["bytes"] += chunk
            if counter["bytes"] // PROGRESS_STEP != before // PROGRESS_STEP:
                typer.echo(f"  {counter['bytes'] / 1e6:.0f} MB downloaded", err=True)

        try:
            files = fetch(
                benchmark,
                source,
                settings.data_dir / "external" / benchmark,
                progress=progress,
            )
        except ManualDownloadError as error:
            typer.echo(f"  needs a manual step: {error}", err=True)
            failed = True
            continue
        update_manifest(manifest, benchmark, files)
        for item in files:
            if item.tree:
                typer.echo(
                    f"  {item.name}: found extracted ({item.digests['files']} files, "
                    f"{int(item.digests['size']):,} bytes)"
                )
                continue
            state = "downloaded" if item.downloaded else "verified"
            typer.echo(f"  {item.name}: {state} ({int(item.digests['size']):,} bytes)")
    if failed:
        raise typer.Exit(code=2)


@store_app.command("compact")
def store_compact(
    before: Annotated[
        str | None,
        typer.Option(
            help="Compact days before this date (YYYY-MM-DD); default today (UTC)."
        ),
    ] = None,
    source: Annotated[str | None, typer.Option(help="Only this source.")] = None,
    asset: Annotated[str | None, typer.Option(help="Only this asset.")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report only; change nothing.")
    ] = False,
) -> None:
    """Merge the part files of each finished day into one deduplicated file."""
    from constellation.core.compact import compact

    store = readings_store()
    try:
        report = compact(
            store, before=before, source=source, asset=asset, dry_run=dry_run
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    verb = "would compact" if dry_run else "compacted"
    for plan in report.partitions:
        typer.echo(
            f"{verb} {plan.source}/{plan.asset}/{plan.date}: {len(plan.files)} files, "
            f"{plan.rows_before:,} rows -> 1 file, {plan.rows_after:,} rows"
        )
    for skipped in report.skipped_open:
        typer.echo(f"skipped {skipped}: a writer has left a temporary file in it")
    typer.echo(
        f"{verb}: {report.files_before} files -> {report.files_after}, "
        f"{report.rows_before:,} rows -> {report.rows_after:,} "
        f"({report.rows_before - report.rows_after:,} duplicates)"
    )


@data_app.command("snapshot")
def data_snapshot(
    dataset: Annotated[str, typer.Argument(help="Only `iss` is supported.")],
    start: Annotated[str, typer.Option(help="First date, YYYY-MM-DD.")],
    end: Annotated[str, typer.Option(help="Last date, YYYY-MM-DD (inclusive).")],
    name: Annotated[
        str | None, typer.Option(help="Snapshot name; default <start>_<end>.")
    ] = None,
    verify: Annotated[
        bool,
        typer.Option("--verify", help="Check the files against the saved snapshot."),
    ] = False,
) -> None:
    """Record, or verify, a checksummed snapshot of the ISS readings in a date range."""
    import json

    from constellation.data.snapshot import (
        build_snapshot,
        verify_snapshot,
        write_snapshot,
    )

    if dataset != "iss":
        raise typer.BadParameter("only `iss` is supported")
    label = name or f"{start}_{end}"
    store = readings_store()
    directory = get_settings().data_dir / "manifests"
    path = directory / f"iss_{label}.json"

    if verify:
        if not path.exists():
            raise typer.BadParameter(f"no snapshot at {path}")
        problems = verify_snapshot(store, json.loads(path.read_text()))
        if problems:
            for problem in problems:
                typer.echo(problem)
            typer.echo(f"snapshot {label}: {len(problems)} problems")
            raise typer.Exit(code=1)
        typer.echo(f"snapshot {label}: all files match")
        return

    snapshot = build_snapshot(store, start=start, end=end, name=label)
    written = write_snapshot(directory, label, snapshot)
    totals = snapshot["totals"]
    typer.echo(
        f"wrote {written}: {totals['partitions']} partitions, {totals['files']} files, "
        f"{totals['rows_unique']:,} unique readings on {totals['channels']} channels, "
        f"{len(snapshot['gaps'])} gaps over {int(snapshot['gap_threshold_s'])} s"
    )
    if not snapshot["complete"]:
        typer.echo(
            "warning: the range includes today's open partition; snapshot again once it is finished"
        )
    if not snapshot["compacted"]:
        typer.echo(
            "warning: some days have several files; run `constellation store compact` first, or the hashes change"
        )
