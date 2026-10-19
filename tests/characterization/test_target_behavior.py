"""Target behavior for the v2 improvements, stated against the current code.

Every test asserts the behavior the finished system must have. Until the
module that delivers it lands, each one is a strict expected failure. The
change that delivers the behavior rewrites the test against the new module and
removes the marker, so a test that starts passing early fails the run. Tests
without a marker have been delivered and now run against the new modules.
"""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pulp
import pytest
import torch

from constellation.core.store import ParquetStore
from constellation.ingest.channels import load_channels, load_dictionary
from constellation.ingest.isslive import IssLiveCollector
from constellation.legacy.features.domain_features import AerospaceFeatureExtractor
from constellation.legacy.features.time_series_features import (
    TimeSeriesFeatureExtractor,
)
from constellation.legacy.ingestion.data_simulator import TelemetrySimulator
from constellation.legacy.models import base_model, degradation_forecast
from constellation.legacy.models.anomaly_detection import (
    IsolationForestDetector,
    LSTMAutoencoderDetector,
)
from constellation.legacy.models.degradation_forecast import DegradationForecaster
from constellation.legacy.models.model_trainer import ModelTrainer
from constellation.legacy.optimization.alert_generator import AlertSeverity
from constellation.legacy.optimization.health_scoring import HealthScorer
from constellation.legacy.optimization.maintenance_scheduler import (
    MaintenanceScheduler,
    MaintenanceTask,
)
from constellation.legacy.optimization.system_integrator import ConstellationSystem

REPO_ROOT = Path(__file__).resolve().parents[2]


class FeatureMismatchError(Exception):
    """Stand-in for the error the v2 artifact loader raises on a schema mismatch."""


def _cbc_runs() -> bool:
    """True when the bundled CBC binary can be executed on this machine."""
    if not pulp.PULP_CBC_CMD().available():
        return False
    problem = pulp.LpProblem("cbc_probe", pulp.LpMinimize)
    problem += pulp.LpVariable("x", 0, 1)
    try:
        problem.solve(pulp.PULP_CBC_CMD(msg=False))
    except (OSError, pulp.PulpSolverError):
        return False
    return True


# --- ingestion and storage -------------------------------------------------


class _Update:
    """The slice of the Lightstreamer SDK's ItemUpdate that the collector reads."""

    def __init__(self, item, **fields):
        self._item = item
        self._fields = fields

    def getItemName(self):
        return self._item

    def getValue(self, field):
        return self._fields.get(field)

    def isSnapshot(self):
        return False


def test_lightstreamer_update_maps_to_subscribed_channel(tmp_path):
    channels = load_channels(REPO_ROOT / "configs/ingest/iss.yaml").channels
    store = ParquetStore(tmp_path)
    collector = IssLiveCollector(channels, store)
    # Z1000011 is the third wheel-speed item in the subscription.
    update = _Update(
        "Z1000011", Value="6601", TimeStamp="7031.5", **{"Status.Class": "24"}
    )

    collector.handle_update(update)
    collector.flush()

    (reading,) = store.read_readings().itertuples()
    assert reading.channel == "cmg3_wheel_speed"
    assert reading.value == 6601.0


# Words the dictionary description must contain for each kind of channel name.
KEYWORDS = [
    (r"cmg\d_online", "on-line"),
    (r"cmgs_online", "number of control moment gyroscope"),
    (r"cmg_torque_(roll|pitch|yaw)", "control torque - {0}"),
    (r"cmg_momentum_pct", "momentum percentage"),
    (r"cmg_momentum_capacity", "momentum capacity"),
    (r"cmg_momentum", "momentum (nms)"),
    (r"desat_request", "desaturation request"),
    (r"gnc_mode", "guidance, navigation and control (gnc) mode"),
    (r"loac_alarm", "loss of cmg attitude control"),
    (r"cmg\d_spin_bearing_temp", "spin motor spin bearing temperature"),
    (r"cmg\d_hall_bearing_temp", "hall resolver spin bearing temperature"),
    (r"cmg\d_vibration", "vibration"),
    (r"cmg\d_spin_current", "spin motor current"),
    (r"cmg\d_wheel_speed", "wheel speed"),
    (r"att_source", "attitude source"),
    (r"rate_source", "rate source"),
    (r"controller_type", "attitude controller type"),
    (r"ref_frame", "reference frame"),
    (r"q_lvlh_\d", "lvlh) attitude quaternion"),
    (r"att_err_(roll|pitch|yaw)", "attitude {0} error"),
    (r"att_rate_([xyz])", "attitude rate {0}"),
    (r"q_cmd_\d", "commanded attitude quaternion"),
    (r"maneuver_in_progress", "maneuver in progress"),
    (r"sa_(\d[ab])_voltage", "solar array - {0} - drive voltage"),
    (r"sa_(\d[ab])_current", "solar array - {0} - drive current"),
    (r"bga_(\d[ab])", "solar array - {0} - beta gimble"),
    (r"sarj_(stbd)", "starboard joint angle"),
    (r"sarj_(port)", "port joint angle"),
    (r"loop_([ab])_flow", "loop {0} pump flowrate"),
    (r"loop_([ab])_press", "loop {0} pm out press"),
    (r"loop_([ab])_temp", "loop {0} pm out temp"),
    (r"trrj_stbd", "starboard thermal radiator rotating joint"),
    (r"trrj_port", "port thermal radiator rotating joint"),
    (r"sband_rfg(\d)_az", "rfg {0}) azimuth"),
    (r"sband_rfg(\d)_el", "rfg {0}) elevation"),
    (r"sband_rfg(\d)_on", "rfg {0}), on-off"),
    (r"sband_active_string", "active string of s-band"),
    (r"ku_transmit", "ku-band transmit"),
    (r"ku_sgant_el", "sgant elevation"),
    (r"ku_sgant_xel", "sgant cross-elevation"),
]


def _expected_phrase(name):
    for pattern, template in KEYWORDS:
        match = re.fullmatch(pattern, name)
        if match:
            return template.format(*match.groups()).lower()
    return None


def test_configured_iss_channels_match_dictionary():
    dictionary = load_dictionary()
    channels = load_channels(REPO_ROOT / "configs/ingest/iss.yaml").channels

    mismatches = []
    for channel in channels:
        expected = _expected_phrase(channel.name)
        entry = dictionary.get(channel.id)
        if expected is None:
            mismatches.append(f"{channel.name}: no keyword rule")
        elif entry is None:
            mismatches.append(f"{channel.id} ({channel.name}): not in dictionary")
        elif expected not in entry.description.lower():
            mismatches.append(
                f"{channel.id} ({channel.name}): {entry.description!r} "
                f"does not contain {expected!r}"
            )

    assert not mismatches, "\n".join(mismatches)


def test_store_partitions_by_event_time(tmp_path):
    times = pd.to_datetime(
        ["2026-03-01T12:00:00", "2026-03-02T12:00:00"], utc=True
    ).as_unit("ns")
    readings = pd.DataFrame(
        {
            "source": "iss",
            "asset_id": "ISS",
            "channel": "cmg1_wheel_speed",
            "t": times,
            "value": [6600.0, 6601.0],
            "value_text": pd.Series([None, None], dtype=object),
            "status": pd.Series([None, None], dtype=object),
            "ingested_at": pd.Timestamp.now(tz="UTC").as_unit("ns"),
        }
    )

    ParquetStore(tmp_path).write_readings(readings)

    partitions = sorted(
        str(path.relative_to(tmp_path))
        for path in tmp_path.glob("source=*/asset=*/date=*")
    )
    assert partitions == [
        "source=iss/asset=ISS/date=2026-03-01",
        "source=iss/asset=ISS/date=2026-03-02",
    ]


# --- features and simulation -----------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="The sunlit flag follows the orbit: both states, sunlit about 60 percent",
)
def test_daylight_flag_varies_over_an_orbit():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-01", periods=185, freq="1min"),
            "value": 0.0,
        }
    )

    flag = AerospaceFeatureExtractor().extract_orbital_features(frame)["is_daylight"]

    assert set(flag.unique()) == {0, 1}
    assert 0.5 <= flag.mean() <= 0.8


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="A time window covers its duration, not a fixed number of rows",
)
def test_time_window_covers_its_duration():
    rng = np.random.default_rng(7)
    start = pd.Timestamp("2026-03-01")
    offsets = np.cumsum(rng.integers(30, 900, size=150))
    frame = pd.DataFrame(
        {
            "timestamp": start + pd.to_timedelta(offsets, unit="s"),
            "value": rng.normal(size=150),
        }
    )

    result = TimeSeriesFeatureExtractor(windows=[3600]).extract_rolling_features(frame)

    times = frame["timestamp"].to_numpy()
    values = frame["value"].to_numpy()
    one_hour = np.timedelta64(1, "h")
    expected = np.array(
        [values[(times > t - one_hour) & (times <= t)].mean() for t in times]
    )
    np.testing.assert_allclose(result["rolling_mean_1h"].to_numpy(), expected)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="An injected spike leaves the channel's normal range",
)
def test_simulated_spike_leaves_normal_range():
    simulator = TelemetrySimulator()
    low, high = simulator.params_config["USLAB000098"]["normal_range"]
    assert (low, high) == (-120, -60)

    records = simulator.generate_batch(
        pd.Timestamp("2026-03-01").to_pydatetime(),
        duration_seconds=60,
        interval_seconds=1,
    )
    records = simulator.inject_anomaly(records, "USLAB000098", "spike")

    values = np.array(
        [float(r["value"]) for r in records if r["parameter_id"] == "USLAB000098"]
    )
    assert ((values < low) | (values > high)).any()


# --- detection, forecasting, diagnosis -------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="A saved and reloaded detector reproduces its threshold and scores",
)
def test_detector_artifact_round_trip(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])

    detector = LSTMAutoencoderDetector(input_dim=3, sequence_length=10)
    detector.train(frame, epochs=2, batch_size=16)
    scores = detector.score_samples(frame)

    path = detector.save(tmp_path / "detector.pkl")
    reloaded = LSTMAutoencoderDetector(input_dim=3, sequence_length=10).load(path)

    assert reloaded.threshold == detector.threshold
    np.testing.assert_array_equal(reloaded.score_samples(frame), scores)


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="Scoring a frame that lacks trained features raises FeatureMismatchError",
)
def test_feature_mismatch_raises_clear_error():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    detector = IsolationForestDetector().train(frame)

    with pytest.raises(FeatureMismatchError):
        detector.score_samples(frame[["a", "b"]])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="The forecast target for prediction i is y[i + L + H - 1]",
)
def test_forecast_evaluation_aligns_to_horizon(monkeypatch):
    sequence_length, horizon, n = 5, 3, 40
    captured = {}

    def capture(y_true, y_pred):
        captured["y_true"] = np.asarray(y_true)
        return {}

    monkeypatch.setattr(degradation_forecast, "calculate_regression_metrics", capture)
    torch.manual_seed(0)
    rng = np.random.default_rng(11)
    features = pd.DataFrame(rng.normal(size=(n, 2)), columns=["f1", "f2"])
    target = pd.Series(np.arange(n, dtype=float))

    forecaster = DegradationForecaster(
        input_dim=2, sequence_length=sequence_length, forecast_horizon=horizon
    )
    forecaster.train(features, target, epochs=1, batch_size=8)
    forecaster.evaluate(features, target)

    valid = n - sequence_length - horizon + 1
    expected = target.to_numpy()[sequence_length + horizon - 1 :]
    np.testing.assert_array_equal(captured["y_true"][:valid], expected[:valid])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Diagnosis is trained on ground-truth labels, never on invented ones",
)
def test_fault_training_requires_ground_truth_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(base_model, "MODELS_DIR", tmp_path)
    rng = np.random.default_rng(13)
    features = pd.DataFrame(
        {
            "rolling_std_1h": rng.normal(size=400),
            "second_derivative": rng.normal(size=400),
            "value_numeric": rng.normal(size=400) + 10.0,
            "is_daylight": np.tile([0, 1], 200),
        }
    )

    raised = None
    try:
        ModelTrainer().train_fault_classifier(features)
    except ValueError as error:
        raised = error

    assert raised is not None, "training without a labels table must raise ValueError"


# --- decision layer --------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="A batch of pure noise raises no critical alert",
)
def test_noise_only_batch_raises_no_critical_alert():
    rng = np.random.default_rng(17)
    columns = [*"abcd", "value_numeric"]
    train = pd.DataFrame(rng.normal(size=(500, 5)), columns=columns)
    batch = pd.DataFrame(rng.normal(size=(500, 5)), columns=columns)
    batch["parameter_id"] = "NOISE"

    system = ConstellationSystem()
    system.anomaly_detector = IsolationForestDetector().train(train)
    scored = system.run_anomaly_detection(batch)
    health = system.calculate_health_scores(scored)
    alerts = system.generate_alerts(scored, health)

    critical = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
    assert len(critical) == 0


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="The health score is exactly 100.0 at best and exactly 0.0 at worst",
)
def test_health_score_bounds_are_exact():
    scorer = HealthScorer()

    best = scorer.calculate_component_health(
        anomaly_score=0.0,
        degradation_rate=0.0,
        fault_probability=0.0,
        days_since_maintenance=0,
    )
    worst = scorer.calculate_component_health(
        anomaly_score=1.0,
        degradation_rate=1.0,
        fault_probability=1.0,
        days_since_maintenance=180,
    )

    assert best == 100.0
    assert worst == 0.0


@pytest.mark.skipif(
    not _cbc_runs(),
    reason="CBC solver binary is not available on this platform",
)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="No maintenance task starts after its latest start",
)
def test_scheduler_respects_deadlines():
    now = pd.Timestamp.now().to_pydatetime()
    hour = pd.Timedelta(1, unit="h").to_pytimedelta()
    scheduler = MaintenanceScheduler()
    # One crew slot. The high-risk task has slack; the low-risk task must start
    # within the hour, so it has to go first.
    scheduler.add_task(
        MaintenanceTask("SLACK", "COMP_A", 0.9, 0.9, 4.0, now, now + 8 * hour)
    )
    scheduler.add_task(
        MaintenanceTask("URGENT", "COMP_B", 0.2, 0.2, 4.0, now, now + 1 * hour)
    )

    schedule = scheduler.optimize_schedule(max_concurrent_tasks=1, time_horizon_days=2)

    latest = {task.task_id: task.latest_start for task in scheduler.tasks}
    late = [
        row.task_id
        for row in schedule.itertuples()
        if row.start_time > latest[row.task_id]
    ]
    assert not late, f"tasks started after their latest start: {late}"


# --- dashboard -------------------------------------------------------------


def _numpy_random_uses(path: Path) -> list[int]:
    """Line numbers where a module uses numpy's random module."""
    tree = ast.parse(path.read_text())
    lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "random"
            and isinstance(node.value, ast.Name)
            and node.value.id in {"np", "numpy"}
        ) or (isinstance(node, ast.ImportFrom) and node.module == "numpy.random"):
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "numpy":
            lines.extend(node.lineno for a in node.names if a.name == "random")
        elif isinstance(node, ast.Import):
            lines.extend(
                node.lineno for a in node.names if a.name.startswith("numpy.random")
            )
    return lines


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="The dashboard shows measured or simulated data, never random numbers",
)
def test_dashboard_contains_no_random_data():
    offenders = {
        str(path.relative_to(REPO_ROOT)): _numpy_random_uses(path)
        for path in sorted((REPO_ROOT / "dashboard").rglob("*.py"))
    }
    offenders = {path: lines for path, lines in offenders.items() if lines}

    assert not offenders, f"np.random used in: {offenders}"
