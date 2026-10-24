import pandas as pd
import pytest

from constellation.core import errors, schemas
from constellation.core.schemas import (
    empty_labels,
    empty_readings,
    validate_labels,
    validate_readings,
)


def test_valid_readings_pass_through_unchanged(make_readings):
    frame = make_readings([("2026-03-01T00:00:00", "temp", 1.5)])

    assert validate_readings(frame) is frame


def test_empty_frames_satisfy_their_contracts():
    assert validate_readings(empty_readings()).empty
    assert validate_labels(empty_labels()).empty


def test_missing_and_unexpected_columns_are_reported(make_readings):
    frame = make_readings([("2026-03-01", "temp", 1.0)]).drop(columns="status")
    frame["extra"] = 1

    with pytest.raises(errors.SchemaError, match=r"missing columns.*status"):
        validate_readings(frame)

    frame["status"] = "ok"
    with pytest.raises(errors.SchemaError, match=r"unexpected columns.*extra"):
        validate_readings(frame)


def test_naive_timestamps_are_rejected(make_readings):
    frame = make_readings([("2026-03-01", "temp", 1.0)])
    frame["t"] = frame["t"].dt.tz_localize(None)

    with pytest.raises(errors.SchemaError, match="datetime64"):
        validate_readings(frame)


def test_non_utc_timezones_are_rejected(make_readings):
    frame = make_readings([("2026-03-01", "temp", 1.0)])
    frame["t"] = frame["t"].dt.tz_convert("Europe/Paris")

    with pytest.raises(errors.SchemaError, match="datetime64"):
        validate_readings(frame)


def test_value_must_be_float64(make_readings):
    frame = make_readings([("2026-03-01", "temp", 1.0)])
    frame["value"] = frame["value"].astype("int64")

    with pytest.raises(errors.SchemaError, match="float64"):
        validate_readings(frame)


def test_required_columns_may_not_hold_nulls(make_readings):
    frame = make_readings([("2026-03-01", "temp", 1.0), ("2026-03-02", "temp", 2.0)])
    frame["channel"] = pd.Series(["temp", None], dtype=object)

    with pytest.raises(errors.SchemaError, match=r"channel.*1 null"):
        validate_readings(frame)


def test_nullable_columns_accept_nulls(make_readings):
    frame = make_readings([("2026-03-01", "temp", float("nan"))])
    frame["status"] = pd.Series([None], dtype=object)

    validate_readings(frame)


def test_text_columns_must_hold_strings(make_readings):
    frame = make_readings([("2026-03-01", "temp", 1.0)])
    frame["value_text"] = pd.Series([3], dtype=object)

    with pytest.raises(errors.SchemaError, match="non-string"):
        validate_readings(frame)

    frame["value_text"] = 3.5
    with pytest.raises(errors.SchemaError, match="expected strings"):
        validate_readings(frame)


def test_unknown_source_and_empty_ids_are_rejected(make_readings):
    with pytest.raises(errors.SchemaError, match=r"unknown values.*radar"):
        validate_readings(make_readings([("2026-03-01", "temp", 1.0)], source="radar"))
    with pytest.raises(errors.SchemaError, match="must not be empty"):
        validate_readings(make_readings([("2026-03-01", "temp", 1.0)], asset_id=""))


def _labels(**overrides) -> pd.DataFrame:
    row = {
        "asset_id": "SIM-001",
        "channel": "bus_voltage",
        "start": pd.Timestamp("2026-03-01T00:00", tz="UTC").as_unit("ns"),
        "end": pd.Timestamp("2026-03-01T01:00", tz="UTC").as_unit("ns"),
        "kind": "anomaly",
        "fault_class": None,
        "severity": None,
        "source": "sim",
    }
    row.update(overrides)
    frame = pd.DataFrame([row])
    for column in ("channel", "fault_class", "severity"):
        frame[column] = frame[column].astype(object)
    return frame


def test_valid_labels_pass():
    validate_labels(_labels())
    validate_labels(_labels(channel=None, kind="failure", fault_class="bearing"))


def test_label_kind_and_interval_are_checked():
    with pytest.raises(errors.SchemaError, match="kind"):
        validate_labels(_labels(kind="glitch"))
    with pytest.raises(errors.SchemaError, match="end must not be earlier"):
        validate_labels(_labels(end=pd.Timestamp("2026-02-28", tz="UTC").as_unit("ns")))


def test_feature_mismatch_error_is_shared_with_the_errors_module():
    assert schemas.FeatureMismatchError is errors.FeatureMismatchError
    assert issubclass(errors.FeatureMismatchError, ValueError)
    assert issubclass(errors.SchemaError, errors.ConstellationError)
