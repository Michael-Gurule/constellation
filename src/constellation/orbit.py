"""Orbital geometry: SGP4 propagation, sun position, eclipses, beta angle, passes.

All positions are in the geocentric inertial frame of skyfield (GCRS, equal to
J2000 to well below the accuracy needed here), in kilometres. Times are UTC.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sgp4.api import WGS72, Satrec
from sgp4.exporter import export_tle
from skyfield.api import EarthSatellite, Timescale, load, wgs84

from constellation.core.errors import ConfigError
from constellation.core.timeutils import ensure_utc

EARTH_RADIUS_KM = 6378.137
EARTH_MU_KM3_S2 = 398600.8  # WGS-72, the constants SGP4 uses
_UNIX_EPOCH_JD = 2440587.5
_SGP4_EPOCH_JD = 2433281.5  # 1949-12-31 00:00 UT, the epoch of SGP4's time scale

Times = pd.DatetimeIndex | Sequence[pd.Timestamp | datetime | str]


@dataclass(frozen=True)
class Tle:
    """A two-line element set."""

    name: str
    line1: str
    line2: str

    @property
    def norad_id(self) -> int:
        return int(self.line1[2:7])

    @property
    def epoch(self) -> pd.Timestamp:
        year_2 = int(self.line1[18:20])
        year = 2000 + year_2 if year_2 < 57 else 1900 + year_2
        day_of_year = float(self.line1[20:32])
        start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        return start + pd.Timedelta(round((day_of_year - 1.0) * 86400e6), unit="us")

    def text(self) -> str:
        return f"{self.name}\n{self.line1}\n{self.line2}\n"


@dataclass(frozen=True)
class GroundStation:
    """A ground station position on the WGS-84 ellipsoid."""

    name: str
    lat_deg: float
    lon_deg: float
    elevation_m: float = 0.0


def _checksum_ok(line: str) -> bool:
    total = sum(int(c) if c.isdigit() else 1 if c == "-" else 0 for c in line[:68])
    return line[68:69].isdigit() and total % 10 == int(line[68])


def load_tle(text: str) -> Tle:
    """Parse a TLE with or without the name line, checking format and checksums."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) == 2 and lines[0].startswith("1 "):
        name, line1, line2 = "", lines[0], lines[1]
    elif len(lines) == 3:
        name, line1, line2 = lines[0].strip(), lines[1], lines[2]
    else:
        raise ConfigError("a TLE has two lines, optionally preceded by a name line")
    if not (line1.startswith("1 ") and line2.startswith("2 ")):
        raise ConfigError("TLE lines must start with '1 ' and '2 '")
    if len(line1) != 69 or len(line2) != 69:
        raise ConfigError("TLE lines must be 69 characters long")
    if not (_checksum_ok(line1) and _checksum_ok(line2)):
        raise ConfigError("TLE checksum mismatch")
    try:
        Satrec.twoline2rv(line1, line2)
    except ValueError as error:
        raise ConfigError(f"invalid TLE: {error}") from error
    return Tle(name=name, line1=line1, line2=line2)


@lru_cache(maxsize=1)
def _timescale() -> Timescale:
    return load.timescale()  # built-in leap-second data, no download


def _index(times: Times) -> pd.DatetimeIndex:
    return ensure_utc(
        times if isinstance(times, pd.DatetimeIndex) else pd.DatetimeIndex(list(times))
    )


def _skyfield_times(index: pd.DatetimeIndex) -> Any:
    seconds = (
        index.second.to_numpy()
        + index.microsecond.to_numpy() / 1e6
        + index.nanosecond.to_numpy() / 1e9
    )
    return _timescale().utc(
        index.year.to_numpy(),
        index.month.to_numpy(),
        index.day.to_numpy(),
        index.hour.to_numpy(),
        index.minute.to_numpy(),
        seconds,
    )


def _satellite(tle: Tle) -> EarthSatellite:
    return EarthSatellite(tle.line1, tle.line2, tle.name or "SAT", _timescale())


def propagate_state(tle: Tle, times: Times) -> tuple[np.ndarray, np.ndarray]:
    """Position (km) and velocity (km/s), each of shape (n, 3)."""
    geocentric = _satellite(tle).at(_skyfield_times(_index(times)))
    position = np.asarray(geocentric.position.km, dtype=float).T
    velocity = np.asarray(geocentric.velocity.km_per_s, dtype=float).T
    return position, velocity


def propagate(tle: Tle, times: Times) -> np.ndarray:
    """Position in kilometres, shape (n, 3)."""
    return propagate_state(tle, times)[0]


def sun_unit_vector(times: Times) -> np.ndarray:
    """Unit vector from the Earth to the Sun, shape (n, 3).

    Low-precision analytic solar position (the almanac series for the apparent
    ecliptic longitude), rotated into the J2000 equator. The error is about
    0.01 degrees over the decades around 2000, and no ephemeris is needed.
    """
    index = _index(times)
    jd = (
        index.to_numpy(dtype="datetime64[ns]").astype("int64") / 86400e9
        + _UNIX_EPOCH_JD
    )
    centuries = (jd - 2451545.0) / 36525.0
    mean_longitude = np.radians(280.460 + 36000.771 * centuries)
    anomaly = np.radians(357.5291092 + 35999.05034 * centuries)
    longitude = (
        mean_longitude
        + np.radians(1.914666471) * np.sin(anomaly)
        + np.radians(0.019994643) * np.sin(2.0 * anomaly)
    )
    longitude_j2000 = longitude - np.radians(1.3969713) * centuries  # precession
    obliquity = math.radians(23.4392911)
    return np.stack(
        [
            np.cos(longitude_j2000),
            np.sin(longitude_j2000) * math.cos(obliquity),
            np.sin(longitude_j2000) * math.sin(obliquity),
        ],
        axis=1,
    )


def in_shadow(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """True where the satellite is inside the Earth's cylindrical shadow.

    `r` holds positions in km and `s` the unit vectors to the Sun, both (n, 3).
    The umbra is modelled as a cylinder of Earth radius; the penumbra is ignored.
    """
    along = np.einsum("ij,ij->i", r, s)
    across = np.linalg.norm(r - along[:, None] * s, axis=1)
    shadow: np.ndarray = (along < 0.0) & (across < EARTH_RADIUS_KM)
    return shadow


def beta_angle(tle: Tle, times: Times) -> np.ndarray:
    """Angle in degrees between the orbit plane and the Sun direction."""
    index = _index(times)
    r, v = propagate_state(tle, index)
    normal = np.cross(r, v)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    sine = np.einsum("ij,ij->i", normal, sun_unit_vector(index))
    angle: np.ndarray = np.degrees(np.arcsin(np.clip(sine, -1.0, 1.0)))
    return angle


def analytic_eclipse_fraction(altitude_km: float, beta_deg: float) -> float:
    """Fraction of a circular orbit spent in shadow for a given beta angle."""
    radius = EARTH_RADIUS_KM + altitude_km
    limit = math.degrees(math.asin(EARTH_RADIUS_KM / radius))
    if abs(beta_deg) >= limit:
        return 0.0
    horizon = math.sqrt(radius**2 - EARTH_RADIUS_KM**2)
    cosine = horizon / (radius * math.cos(math.radians(beta_deg)))
    return math.acos(min(1.0, cosine)) / math.pi


def orbit_features(tle: Tle, times: Times) -> pd.DataFrame:
    """Sunlit state and eclipse phase on the given times.

    `sunlit` is 1.0 in sunlight and 0.0 in shadow. `eclipse_phase` is the
    position along the orbit as a fraction in [0, 1), measured from the point
    of the orbit that lies opposite the Sun (mid-eclipse is 0).
    """
    index = _index(times)
    r, v = propagate_state(tle, index)
    s = sun_unit_vector(index)
    shadow = in_shadow(r, s)

    normal = np.cross(r, v)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    anti_sun = -s
    in_plane = anti_sun - np.einsum("ij,ij->i", anti_sun, normal)[:, None] * normal
    norm = np.linalg.norm(in_plane, axis=1)
    reference = np.divide(
        in_plane, norm[:, None], out=np.zeros_like(in_plane), where=norm[:, None] > 0
    )
    x = np.einsum("ij,ij->i", r, reference)
    y = np.einsum("ij,ij->i", r, np.cross(normal, reference))
    phase = (np.arctan2(y, x) / (2.0 * math.pi)) % 1.0
    return pd.DataFrame(
        {"sunlit": (~shadow).astype(float), "eclipse_phase": phase}, index=index
    )


def contact_windows(
    tle: Tle,
    station: GroundStation,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    min_elev_deg: float = 10.0,
) -> pd.DataFrame:
    """Passes above `min_elev_deg` between `start` and `end`.

    Columns: `start`, `end` (UTC) and `max_elevation_deg`. A pass already in
    progress at `start`, or still up at `end`, is cut at that boundary.
    """
    first, last = ensure_utc(start), ensure_utc(end)
    satellite = _satellite(tle)
    site = wgs84.latlon(
        station.lat_deg, station.lon_deg, elevation_m=station.elevation_m
    )
    t0 = _skyfield_times(pd.DatetimeIndex([first]))[0]
    t1 = _skyfield_times(pd.DatetimeIndex([last]))[0]
    times, events = satellite.find_events(site, t0, t1, altitude_degrees=min_elev_deg)

    def elevation(moment: pd.Timestamp) -> float:
        topocentric = (satellite - site).at(_skyfield_times(pd.DatetimeIndex([moment])))
        return float(topocentric.altaz()[0].degrees[0])

    stamps = [pd.Timestamp(t.utc_datetime()) for t in times]
    windows: list[dict[str, Any]] = []
    rise: pd.Timestamp | None = None
    peak = -90.0
    if len(events) and events[0] != 0:
        rise, peak = first, elevation(first)  # already above the horizon at start
    for stamp, event in zip(stamps, events, strict=True):
        if event == 0:
            rise, peak = stamp, min_elev_deg
        elif event == 1:
            peak = max(peak, elevation(stamp))
        elif event == 2 and rise is not None:
            windows.append({"start": rise, "end": stamp, "max_elevation_deg": peak})
            rise = None
    if rise is not None:
        windows.append(
            {
                "start": rise,
                "end": last,
                "max_elevation_deg": max(peak, elevation(last)),
            }
        )
    frame = pd.DataFrame(windows, columns=["start", "end", "max_elevation_deg"])
    for column in ("start", "end"):
        frame[column] = pd.to_datetime(frame[column], utc=True).astype(
            "datetime64[ns, UTC]"
        )
    frame["max_elevation_deg"] = frame["max_elevation_deg"].astype(float)
    return frame


def synthetic_tle(
    alt_km: float,
    inc_deg: float,
    raan_deg: float,
    mean_anomaly_deg: float,
    epoch: pd.Timestamp | str,
    *,
    eccentricity: float = 1e-4,
    norad_id: int = 90001,
    name: str = "SYNTHETIC",
) -> Tle:
    """A near-circular orbit at `alt_km`, for the simulator's constellation."""
    when = ensure_utc(epoch)
    semi_major = EARTH_RADIUS_KM + alt_km
    mean_motion = math.sqrt(EARTH_MU_KM3_S2 / semi_major**3) * 60.0  # rad/min
    julian_date = when.value / 86400e9 + _UNIX_EPOCH_JD
    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        "i",
        norad_id,
        julian_date - _SGP4_EPOCH_JD,
        0.0,  # bstar
        0.0,  # ndot
        0.0,  # nddot
        eccentricity,
        0.0,  # argument of perigee
        math.radians(inc_deg),
        math.radians(mean_anomaly_deg),
        mean_motion,
        math.radians(raan_deg),
    )
    line1, line2 = export_tle(satrec)
    return Tle(name=name, line1=line1, line2=line2)
