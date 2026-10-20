import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from constellation import orbit
from constellation.core.errors import ConfigError

TLE_TEXT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "iss_25544.tle"
).read_text()
START = pd.Timestamp("2026-10-19T12:00:00", tz="UTC")


@pytest.fixture(scope="module")
def iss():
    return orbit.load_tle(TLE_TEXT)


def one_orbit(tle, step="10s"):
    period = 86400.0 / float(tle.line2[52:63])
    return pd.date_range(START, START + pd.Timedelta(period, unit="s"), freq=step)


# -- TLE parsing --------------------------------------------------------------


def test_load_tle_with_name(iss):
    assert iss.name == "ISS (ZARYA)"
    assert iss.norad_id == 25544
    assert iss.epoch == pd.Timestamp("2026-10-19T10:26:22.356672", tz="UTC")


def test_load_tle_without_name(iss):
    bare = orbit.load_tle(f"{iss.line1}\n{iss.line2}\n")

    assert bare.name == ""
    assert (bare.line1, bare.line2) == (iss.line1, iss.line2)
    assert orbit.load_tle(bare.text()) == bare


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a, b: (a[:-1] + str((int(a[-1]) + 1) % 10), b),  # checksum
        lambda a, b: (a[:-2], b),  # length
        lambda a, b: (b, a),  # order
        lambda a, b: (a, a),  # line 2 is not line 2
    ],
)
def test_load_tle_rejects_malformed_input(iss, mutate):
    line1, line2 = mutate(iss.line1, iss.line2)

    with pytest.raises(ConfigError):
        orbit.load_tle(f"{line1}\n{line2}\n")


def test_load_tle_rejects_wrong_line_count():
    with pytest.raises(ConfigError, match="two lines"):
        orbit.load_tle("1 only one line")


# -- propagation ------------------------------------------------------------------


def test_iss_orbit_has_the_expected_radius_and_period(iss):
    times = one_orbit(iss)

    r = orbit.propagate(iss, times)
    radius = np.linalg.norm(r, axis=1)

    assert r.shape == (len(times), 3)
    assert 6700 < radius.min() < radius.max() < 6850
    # After one period the satellite is back near where it started.
    assert np.linalg.norm(r[-1] - r[0]) < 100


def test_state_vector_is_consistent_with_a_finite_difference(iss):
    times = pd.DatetimeIndex([START, START + pd.Timedelta(1, unit="s")])

    r, v = orbit.propagate_state(iss, times)

    assert np.linalg.norm(r[1] - r[0] - v[0]) < 0.01  # km over one second
    assert 7.5 < np.linalg.norm(v[0]) < 7.8


def test_propagation_accepts_timestamp_lists(iss):
    as_list = orbit.propagate(iss, [START, START + pd.Timedelta(60, unit="s")])
    as_index = orbit.propagate(
        iss, pd.DatetimeIndex([START, START + pd.Timedelta(60, unit="s")])
    )

    np.testing.assert_allclose(as_list, as_index)


# -- sun and shadow ----------------------------------------------------------------


def _ra_dec(vector):
    ra = math.degrees(math.atan2(vector[1], vector[0])) % 360
    dec = math.degrees(math.asin(vector[2]))
    return ra, dec


def test_sun_position_matches_the_almanac_at_j2000():
    vector = orbit.sun_unit_vector(pd.DatetimeIndex(["2000-01-01T12:00:00"], tz="UTC"))[
        0
    ]

    ra, dec = _ra_dec(vector)

    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert ra == pytest.approx(281.28, abs=0.02)
    assert dec == pytest.approx(-23.03, abs=0.02)


def _precession_to_j2000(vector, centuries):
    """Rotate a vector from the mean equator of date to J2000 (Meeus, ch. 21)."""
    arcsec = math.radians(1 / 3600)
    zeta = (2306.2181 * centuries + 0.30188 * centuries**2) * arcsec
    z = (2306.2181 * centuries + 1.09468 * centuries**2) * arcsec
    theta = (2004.3109 * centuries - 0.42665 * centuries**2) * arcsec

    def rot_z(a):
        return np.array(
            [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        )

    def rot_y(a):
        return np.array(
            [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        )

    # Meeus' angles describe the passive rotation from J2000 to the date, which is
    # the active rotation built here from the date back to J2000.
    return rot_z(-z) @ rot_y(theta) @ rot_z(-zeta) @ vector


@pytest.mark.parametrize(
    ("when", "ecliptic_longitude_deg"),
    [
        ("2026-03-20T14:46:00", 0.0),  # March equinox
        ("2026-06-21T08:24:00", 90.0),  # June solstice
        ("2026-09-23T00:05:00", 180.0),  # September equinox
    ],
)
def test_sun_position_matches_the_almanac_events_of_2026(when, ecliptic_longitude_deg):
    julian_date = pd.Timestamp(when, tz="UTC").value / 86400e9 + 2440587.5
    centuries = (julian_date - 2451545.0) / 36525.0
    obliquity = math.radians(23.4392911 - 0.0130042 * centuries)
    lon = math.radians(ecliptic_longitude_deg)
    of_date = np.array(
        [
            math.cos(lon),
            math.sin(lon) * math.cos(obliquity),
            math.sin(lon) * math.sin(obliquity),
        ]
    )
    expected = _precession_to_j2000(of_date, centuries)

    vector = orbit.sun_unit_vector(pd.DatetimeIndex([when], tz="UTC"))[0]

    error = math.degrees(math.acos(min(1.0, float(vector @ expected))))
    assert error < 0.03


def test_sun_unit_vector_shape_and_norm():
    times = pd.date_range("2026-01-01", periods=50, freq="7D", tz="UTC")

    vectors = orbit.sun_unit_vector(times)

    assert vectors.shape == (50, 3)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_cylindrical_shadow_geometry():
    sun = np.array([[1.0, 0.0, 0.0]] * 5)
    positions = np.array(
        [
            [-7000.0, 0.0, 0.0],  # behind the Earth
            [7000.0, 0.0, 0.0],  # facing the Sun
            [-7000.0, 7000.0, 0.0],  # behind, but outside the cylinder
            [-7000.0, 6000.0, 0.0],  # behind, inside the cylinder
            [0.0, 0.0, 7000.0],  # terminator, above the pole
        ]
    )

    assert orbit.in_shadow(positions, sun).tolist() == [True, False, False, True, False]


def test_beta_angle_is_the_angle_between_sun_and_orbit_plane(iss):
    beta = orbit.beta_angle(iss, pd.DatetimeIndex([START]))[0]

    assert beta == pytest.approx(-22.4, abs=1.0)
    assert (
        abs(orbit.beta_angle(iss, pd.date_range(START, periods=200, freq="1min")).std())
        < 0.2
    )


def test_analytic_eclipse_fraction():
    assert orbit.analytic_eclipse_fraction(420, 0.0) == pytest.approx(0.388, abs=0.002)
    assert orbit.analytic_eclipse_fraction(420, 60.0) < orbit.analytic_eclipse_fraction(
        420, 20.0
    )
    assert orbit.analytic_eclipse_fraction(420, 75.0) == 0.0  # eclipse-free season
    assert orbit.analytic_eclipse_fraction(420, -75.0) == 0.0


def test_one_orbit_is_sunlit_for_about_sixty_percent_and_matches_the_analytic_fraction(
    iss,
):
    times = one_orbit(iss)[:-1]  # exactly one period
    features = orbit.orbit_features(iss, times)
    r = orbit.propagate(iss, times)
    altitude = float(np.linalg.norm(r, axis=1).mean() - orbit.EARTH_RADIUS_KM)
    beta = float(orbit.beta_angle(iss, times).mean())

    sunlit = features["sunlit"].mean()
    analytic = 1.0 - orbit.analytic_eclipse_fraction(altitude, beta)

    assert 0.55 <= sunlit <= 0.75
    assert abs(sunlit - analytic) < 0.03


def test_sunlit_takes_both_values_and_phase_is_centred_on_mid_eclipse(iss):
    features = orbit.orbit_features(iss, one_orbit(iss))

    assert set(features["sunlit"].unique()) == {0.0, 1.0}
    assert features["eclipse_phase"].between(0.0, 1.0, inclusive="left").all()
    shadow_phase = features.loc[features["sunlit"] == 0.0, "eclipse_phase"]
    # Mid-eclipse is phase 0, so the shadow arc straddles the wrap-around.
    assert ((shadow_phase < 0.25) | (shadow_phase > 0.75)).all()
    assert shadow_phase.min() < 0.05
    assert shadow_phase.max() > 0.95


# -- passes ----------------------------------------------------------------------


HELSINKI = orbit.GroundStation("Helsinki", 60.17, 24.94)
HOUSTON = orbit.GroundStation("Houston", 29.76, -95.37)
SVALBARD = orbit.GroundStation("Svalbard", 78.23, 15.4)


def test_a_high_latitude_station_has_passes_in_24_hours(iss):
    windows = orbit.contact_windows(iss, HELSINKI, "2026-10-20", "2026-10-21", 10.0)

    assert len(windows) >= 2
    assert (windows["end"] > windows["start"]).all()
    assert windows["start"].is_monotonic_increasing
    durations = (windows["end"] - windows["start"]).dt.total_seconds()
    assert durations.between(30, 900).all()
    assert (windows["max_elevation_deg"] >= 10.0).all()
    assert (windows["start"] >= pd.Timestamp("2026-10-20", tz="UTC")).all()
    assert str(windows["start"].dtype) == "datetime64[ns, UTC]"


def test_a_station_beyond_the_ground_track_sees_nothing(iss):
    windows = orbit.contact_windows(iss, SVALBARD, "2026-10-20", "2026-10-21", 10.0)

    assert windows.empty
    assert windows["max_elevation_deg"].dtype == float


def test_a_lower_elevation_mask_gives_more_and_longer_passes(iss):
    strict = orbit.contact_windows(iss, HOUSTON, "2026-10-20", "2026-10-21", 30.0)
    loose = orbit.contact_windows(iss, HOUSTON, "2026-10-20", "2026-10-21", 5.0)

    assert len(loose) >= len(strict)
    assert (loose["end"] - loose["start"]).sum() > (
        strict["end"] - strict["start"]
    ).sum()


def test_a_pass_in_progress_is_cut_at_the_requested_start(iss):
    full = orbit.contact_windows(iss, HOUSTON, "2026-10-20", "2026-10-21", 10.0)
    first = full.iloc[0]
    middle = first["start"] + (first["end"] - first["start"]) / 2

    cut = orbit.contact_windows(iss, HOUSTON, middle, "2026-10-21", 10.0)

    assert cut.iloc[0]["start"] == middle
    assert abs((cut.iloc[0]["end"] - first["end"]).total_seconds()) < 1.0

    until_middle = orbit.contact_windows(iss, HOUSTON, "2026-10-20", middle, 10.0)
    assert until_middle.iloc[-1]["end"] == middle


# -- synthetic orbits -------------------------------------------------------------


def test_synthetic_tle_is_a_valid_circular_orbit_at_the_requested_altitude():
    tle = orbit.synthetic_tle(550.0, 97.6, 40.0, 10.0, "2026-03-01T00:00:00")
    times = pd.date_range("2026-03-01", periods=100, freq="1min", tz="UTC")

    r, v = orbit.propagate_state(tle, times)
    altitude = np.linalg.norm(r, axis=1) - orbit.EARTH_RADIUS_KM
    normal = np.cross(r, v)
    inclination = np.degrees(np.arccos(normal[:, 2] / np.linalg.norm(normal, axis=1)))

    assert orbit.load_tle(tle.text()) == tle  # checksums are valid
    assert tle.epoch == pd.Timestamp("2026-03-01", tz="UTC")
    assert altitude.mean() == pytest.approx(550.0, abs=10.0)
    assert altitude.max() - altitude.min() < 20.0  # J2 short-period variation
    assert inclination.mean() == pytest.approx(97.6, abs=0.1)


def test_synthetic_tles_differ_by_node_and_anomaly():
    time = pd.DatetimeIndex(["2026-03-01T01:00:00"], tz="UTC")
    base = orbit.propagate(orbit.synthetic_tle(550, 97.6, 40, 10, "2026-03-01"), time)
    other_node = orbit.propagate(
        orbit.synthetic_tle(550, 97.6, 100, 10, "2026-03-01"), time
    )
    other_phase = orbit.propagate(
        orbit.synthetic_tle(550, 97.6, 40, 190, "2026-03-01"), time
    )

    assert np.linalg.norm(base - other_node) > 1000
    assert np.linalg.norm(base - other_phase) > 1000


def test_synthetic_tle_beta_angle_follows_the_node():
    time = pd.DatetimeIndex(["2026-03-01T00:00:00"], tz="UTC")
    betas = [
        orbit.beta_angle(orbit.synthetic_tle(550, 97.6, raan, 0, "2026-03-01"), time)[0]
        for raan in (0, 90, 180, 270)
    ]

    assert max(betas) - min(betas) > 40
