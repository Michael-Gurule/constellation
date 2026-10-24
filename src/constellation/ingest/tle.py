"""Download orbital element sets from CelesTrak's GP service."""

import os
import time
import uuid
from pathlib import Path

import httpx

from constellation.core.errors import ConfigError
from constellation.orbit import Tle, load_tle

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php"
MAX_AGE_SECONDS = 12 * 3600


def tle_path(directory: Path, norad: int) -> Path:
    """Where the element set for `norad` is kept."""
    return directory / f"{norad}.tle"


def fetch_tle(
    norad: int,
    directory: Path,
    *,
    force: bool = False,
    max_age_seconds: float = MAX_AGE_SECONDS,
    client: httpx.Client | None = None,
) -> tuple[Path, bool]:
    """Download the current TLE for `norad` unless the saved copy is recent.

    Returns the path and whether a download happened. The file is written to a
    temporary name and renamed, so a failed download never damages the saved copy.
    """
    path = tle_path(directory, norad)
    if (
        not force
        and path.exists()
        and time.time() - path.stat().st_mtime < max_age_seconds
    ):
        return path, False

    owns_client = client is None
    http = client or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        response = http.get(CELESTRAK_URL, params={"CATNR": norad, "FORMAT": "tle"})
        response.raise_for_status()
    finally:
        if owns_client:
            http.close()

    tle = load_tle(response.text)
    if tle.norad_id != norad:
        raise ConfigError(f"asked for NORAD {norad} but received {tle.norad_id}")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{norad}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(tle.text(), encoding="utf-8")
    os.replace(temporary, path)
    return path, True


def read_tle(path: Path) -> Tle:
    """Load a saved element set."""
    return load_tle(path.read_text(encoding="utf-8"))
