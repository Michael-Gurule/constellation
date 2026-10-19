"""Collect ISS telemetry from NASA's public ISSLive Lightstreamer feed.

Updates are subscribed in MERGE mode through the official Lightstreamer client
library. Every update names its item, so a reading is always attributed to the
channel it belongs to. The event time `t` comes from the update's own
`TimeStamp` field, never from the wall clock.
"""

import json
import logging
import math
import signal
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

import pandas as pd
from lightstreamer.client import (
    ClientListener,
    LightstreamerClient,
    Subscription,
    SubscriptionListener,
)

from constellation.core.schemas import READINGS
from constellation.core.store import ParquetStore
from constellation.ingest.channels import Channel

logger = logging.getLogger(__name__)

SERVER = "https://push.lightstreamer.com"
ADAPTER_SET = "ISSLIVE"
DATA_ADAPTER = "DEFAULT"
FIELDS = ["TimeStamp", "Value", "Status.Class", "Status.Indicator"]
TIME_ITEM = "TIME_000001"
YEAR_ITEM = "TIME_000002"
SOURCE = "iss"
ASSET_ID = "ISS"
GAP_SECONDS = 300.0
_UNMAPPED_LOG_LIMIT = 5


class UpdateLike(Protocol):
    """The methods of the SDK's `ItemUpdate` that the collector uses."""

    def getItemName(self) -> str: ...
    def getValue(self, field: str) -> str | None: ...
    def isSnapshot(self) -> bool: ...


def parse_source_time(hours: float, year: int) -> pd.Timestamp:
    """Convert an ISSLive `TimeStamp` to UTC.

    The feed reports decimal hours since 00:00 UTC on 31 December of the
    previous year, so 1 January 00:00 is hour 24. The result is rounded to the
    microsecond, which is finer than the feed's own resolution.
    """
    origin = pd.Timestamp(year=year - 1, month=12, day=31, tz="UTC")
    return origin + pd.Timedelta(round(hours * 3600.0 * 1e6), unit="us")


def resolve_source_time(
    hours: float, year: int | None, now: pd.Timestamp
) -> pd.Timestamp:
    """Like `parse_source_time`, choosing the year nearest `now` if `year` is unknown
    or leaves the result more than half a year from `now` (year rollover).
    """
    candidates = [year] if year is not None else []
    candidates += [now.year - 1, now.year, now.year + 1]
    parsed = [parse_source_time(hours, y) for y in dict.fromkeys(candidates)]
    if year is not None:
        first = parsed[0]
        if abs(first - now) <= pd.Timedelta(183, unit="D"):
            return first
    return min(parsed, key=lambda t: abs(t - now))


@dataclass(frozen=True)
class Observation:
    """What the collector did with one update."""

    item: str
    outcome: str  # "stored", "duplicate", "time", "nodata", "unmapped" or "invalid"
    t: pd.Timestamp | None = None
    snapshot: bool = False


class IssLiveCollector:
    """Buffers ISSLive updates and writes them to a `ParquetStore`."""

    def __init__(
        self,
        channels: list[Channel],
        store: ParquetStore,
        flush_seconds: float = 60,
        max_buffer: int = 20_000,
        *,
        clock: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
        observer: Callable[[Observation], None] | None = None,
    ) -> None:
        self.channels = {c.id: c for c in channels}
        self.store = store
        self.flush_seconds = flush_seconds
        self.max_buffer = max_buffer
        self.observer = observer
        self._clock = clock
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._year: int | None = None
        self._last_seen: dict[str, tuple[pd.Timestamp, str]] = {}
        self._last_update = time.monotonic()
        self._gap_logged = False
        self._unmapped_logged = 0
        self._stop = threading.Event()
        self._flusher: threading.Thread | None = None
        self.counts = {
            "stored": 0,
            "duplicate": 0,
            "time": 0,
            "nodata": 0,
            "unmapped": 0,
            "invalid": 0,
            "flushed": 0,
        }

    # -- update handling -----------------------------------------------------

    @property
    def items(self) -> list[str]:
        """Item names to subscribe to: every channel plus the two time items."""
        return [*self.channels, TIME_ITEM, YEAR_ITEM]

    def handle_update(self, update: UpdateLike) -> Observation:
        """Turn one SDK update into a buffered reading (or count it as skipped)."""
        snapshot = bool(update.isSnapshot())
        observation = self._process(update, snapshot)
        self.counts[observation.outcome] += 1
        if (
            observation.outcome == "unmapped"
            and self._unmapped_logged < _UNMAPPED_LOG_LIMIT
        ):
            self._unmapped_logged += 1
            logger.warning("update for unconfigured item %s", observation.item)
        if self.observer is not None:
            self.observer(observation)
        return observation

    def _process(self, update: UpdateLike, snapshot: bool) -> Observation:
        item = update.getItemName()
        now = self._clock()
        self._note_activity()

        value_raw = update.getValue("Value")
        if item == YEAR_ITEM:
            self._set_year(value_raw)
            return Observation(item, "time", snapshot=snapshot)
        if item == TIME_ITEM:
            return Observation(item, "time", snapshot=snapshot)

        channel = self.channels.get(item)
        if channel is None:
            return Observation(item, "unmapped", snapshot=snapshot)

        try:
            hours = float(update.getValue("TimeStamp") or "")
        except ValueError:
            return Observation(item, "invalid", snapshot=snapshot)
        if not math.isfinite(hours) or value_raw is None:
            return Observation(item, "invalid", snapshot=snapshot)
        if hours <= 0:
            # The feed reports TimeStamp 0 for channels that have no data.
            return Observation(item, "nodata", snapshot=snapshot)
        t = resolve_source_time(hours, self._year, now)

        if self._last_seen.get(item) == (t, value_raw):
            return Observation(item, "duplicate", t, snapshot)
        self._last_seen[item] = (t, value_raw)

        number = self._to_float(value_raw)
        row: dict[str, Any] = {
            "source": SOURCE,
            "asset_id": ASSET_ID,
            "channel": channel.name,
            "t": t,
            "value": number,
            "value_text": value_raw
            if channel.kind == "state" or math.isnan(number)
            else None,
            "status": update.getValue("Status.Class"),
            "ingested_at": now,
        }
        with self._lock:
            self._buffer.append(row)
            full = len(self._buffer) >= self.max_buffer
        if full:
            self.flush()
        return Observation(item, "stored", t, snapshot)

    @staticmethod
    def _to_float(raw: str) -> float:
        try:
            return float(raw)
        except ValueError:
            return math.nan

    def _set_year(self, raw: str | None) -> None:
        try:
            year = int(float(raw or ""))
        except ValueError:
            return
        if 2000 <= year <= 2200:
            self._year = year

    def _note_activity(self) -> None:
        current = time.monotonic()
        gap = current - self._last_update
        if gap > GAP_SECONDS:
            logger.warning("ingest_gap: no updates for %.0f s (now resumed)", gap)
        self._last_update = current
        self._gap_logged = False

    def _check_gap(self) -> None:
        silent = time.monotonic() - self._last_update
        if silent > GAP_SECONDS and not self._gap_logged:
            self._gap_logged = True
            logger.warning("ingest_gap: no updates for %.0f s", silent)

    # -- writing -------------------------------------------------------------

    def flush(self) -> int:
        """Write buffered readings to the store. Returns the number written.

        If the write fails the readings stay buffered and the next flush retries.
        """
        with self._lock:
            rows, self._buffer = self._buffer, []
        if not rows:
            return 0
        frame = pd.DataFrame(rows, columns=[c.name for c in READINGS])
        for column in ("t", "ingested_at"):
            frame[column] = frame[column].astype("datetime64[ns, UTC]")
        frame["value"] = frame["value"].astype("float64")
        for column in ("value_text", "status"):
            frame[column] = frame[column].astype(object)
        try:
            self.store.write_readings(frame)
        except Exception:
            logger.exception(
                "flush failed; keeping %d readings in the buffer", len(rows)
            )
            with self._lock:
                self._buffer = rows + self._buffer
            return 0
        self.counts["flushed"] += len(rows)
        logger.info("flushed %d readings", len(rows))
        return len(rows)

    def _flush_loop(self) -> None:
        while not self._stop.wait(self.flush_seconds):
            self.flush()
            self._check_gap()

    # -- lifecycle -------------------------------------------------------------

    def run(self, duration: float | None = None) -> None:
        """Collect until `duration` seconds pass or SIGINT/SIGTERM arrives.

        Buffered readings are flushed on the way out, whatever the reason.
        """
        self._stop.clear()
        client = self._connect()
        self._flusher = threading.Thread(
            target=self._flush_loop, daemon=True, name="flush"
        )
        self._flusher.start()

        def request_stop(signum: int, _frame: FrameType | None) -> None:
            logger.info("received signal %d, stopping", signum)
            self._stop.set()

        previous = {
            sig: signal.signal(sig, request_stop)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            self._stop.wait(duration)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            self._stop.set()
            client.disconnect()
            if self._flusher is not None:
                self._flusher.join(timeout=5)
            self.flush()
            logger.info("collector stopped: %s", self.counts)

    def _connect(self) -> Any:
        return open_session(self.items, self._on_update, self._on_status)

    def _on_update(self, update: Any) -> None:
        self.handle_update(update)

    @staticmethod
    def _on_status(status: str) -> None:
        logger.info("client status: %s", status)


# -- Lightstreamer plumbing ------------------------------------------------------


class _Updates(SubscriptionListener):  # type: ignore[misc]
    def __init__(self, on_update: Callable[[Any], None]) -> None:
        self._on_update = on_update

    def onSubscription(self) -> None:
        logger.info("subscription active")

    def onSubscriptionError(self, code: int, message: str) -> None:
        logger.error("subscription error %s: %s", code, message)

    def onItemUpdate(self, update: Any) -> None:
        try:
            self._on_update(update)
        except Exception:
            logger.exception("failed to process an update")


class _Status(ClientListener):  # type: ignore[misc]
    def __init__(self, on_status: Callable[[str], None]) -> None:
        self._on_status = on_status

    def onStatusChange(self, status: str) -> None:
        self._on_status(status)

    def onServerError(self, code: int, message: str) -> None:
        logger.error("server error %s: %s", code, message)


def open_session(
    items: list[str],
    on_update: Callable[[Any], None],
    on_status: Callable[[str], None],
) -> Any:
    """Connect to ISSLive and subscribe to `items` in MERGE mode.

    The SDK handles reconnection. Returns the client; call `disconnect()` on it.
    """
    client = LightstreamerClient(SERVER, ADAPTER_SET)
    client.addListener(_Status(on_status))
    subscription = Subscription("MERGE", items, FIELDS)
    subscription.setDataAdapter(DATA_ADAPTER)
    subscription.addListener(_Updates(on_update))
    client.subscribe(subscription)
    client.connect()
    return client


# -- Probe -----------------------------------------------------------------------


def probe(
    items: list[str],
    seconds: float,
    out: Path,
    *,
    max_per_item: int = 25,
) -> Mapping[str, int]:
    """Record raw updates for `seconds` and save a trimmed copy as JSON lines.

    At most `max_per_item` updates are kept per item so the file stays small;
    the returned mapping holds the full counts per item.
    """
    records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    lock = threading.Lock()

    def record(update: Any) -> None:
        item = update.getItemName()
        with lock:
            counts[item] = counts.get(item, 0) + 1
            if counts[item] <= max_per_item:
                records.append(
                    {
                        "wall": datetime.now(UTC).isoformat(),
                        "item": item,
                        "snapshot": bool(update.isSnapshot()),
                        **{field: update.getValue(field) for field in FIELDS},
                    }
                )

    client = open_session(
        items, record, lambda status: logger.info("client status: %s", status)
    )
    try:
        time.sleep(seconds)
    finally:
        client.disconnect()

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for entry in records:
            handle.write(json.dumps(entry) + "\n")
    return dict(counts)
