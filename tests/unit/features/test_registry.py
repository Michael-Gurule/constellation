from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from constellation import orbit
from constellation.core.errors import ConfigError
from constellation.features import windows
from constellation.features.registry import FeatureSet, FeatureSpec, compute
from constellation.ingest.channels import load_channels

ROOT = Path(__file__).resolve().parents[3]
TLE = orbit.load_tle((ROOT / "tests/fixtures/iss_25544.tle").read_text())

TYPES = [
    {"match": r"cmg\d_wheel_speed", "type": "speed"},
    {"match": r"cmg\d_spin_current", "type": "current"},
    {"match": r"loop_a_temp", "type": "temperature"},
]


def feature_set(*features, types=TYPES):
    return FeatureSet.from_mapping({"channel_types": types, "features": list(features)})


def spec(name, function="roll_mean", applies_to=("speed",), **extra):
    params = extra.pop("params", {"window": "10min"})
    return {
        "name": name,
        "function": function,
        "params": params,
        "applies_to": list(applies_to),
        **extra,
    }


def frame(
    seed=0,
    n=300,
    channels=("cmg1_wheel_speed", "cmg1_spin_current", "cmg2_wheel_speed"),
):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-03-01", periods=n, freq="10s", tz="UTC", name="t")
    data = {c: 100.0 + rng.normal(size=n).cumsum() for c in channels}
    result = pd.DataFrame(data, index=index)
    result["valid_fraction"] = 1.0
    return result


# -- loading and validation ---------------------------------------------------------


def test_foreach_expands_one_spec_per_value():
    fs = FeatureSet.from_mapping(
        {
            "channel_types": TYPES,
            "features": [
                {
                    "name": "cmg{i}_k",
                    "function": "safe_ratio",
                    "inputs": ["cmg{i}_spin_current", "cmg{i}_wheel_speed"],
                    "applies_to": ["current", "speed"],
                    "foreach": {"i": [1, 2]},
                }
            ],
        }
    )

    assert [f.name for f in fs.features] == ["cmg1_k", "cmg2_k"]
    assert fs.features[1].inputs == ["cmg2_spin_current", "cmg2_wheel_speed"]


def test_foreach_supports_a_single_variable():
    with pytest.raises(ConfigError, match="one variable"):
        FeatureSet.from_mapping(
            {"features": [{**spec("x"), "foreach": {"i": [1], "j": [2]}}]}
        )


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (
            {"name": "x", "function": "nope", "applies_to": ["speed"]},
            "unknown function",
        ),
        (
            {"name": "x", "function": "roll_mean", "params": {"window": "1h"}},
            "needs applies_to",
        ),
        (
            {
                "name": "x",
                "function": "safe_ratio",
                "inputs": ["a"],
                "applies_to": ["speed"],
            },
            "two inputs",
        ),
        (
            {"name": "x", "function": "orbit_sunlit", "applies_to": ["speed"]},
            "takes no channels",
        ),
        (
            {
                "name": "x",
                "function": "roll_mean",
                "params": {"width": "1h"},
                "applies_to": ["speed"],
            },
            "bad params",
        ),
    ],
)
def test_malformed_specs_are_rejected(bad, message):
    with pytest.raises(ConfigError, match=message):
        FeatureSet.from_mapping({"features": [bad]})


def test_from_yaml_reports_missing_and_malformed_files(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        FeatureSet.from_yaml(tmp_path / "absent.yaml")
    path = tmp_path / "list.yaml"
    path.write_text("- a\n")
    with pytest.raises(ConfigError, match="mapping"):
        FeatureSet.from_yaml(path)


def test_channel_types_use_the_first_matching_rule():
    fs = feature_set(
        spec("m"),
        types=[{"match": "a.*", "type": "first"}, {"match": "ab", "type": "second"}],
    )

    assert fs.channel_type("ab") == "first"
    assert fs.channel_type("zz") is None


# -- planning -----------------------------------------------------------------------


def test_column_order_follows_the_spec_order_then_sorted_channels():
    fs = feature_set(spec("mean"), spec("std", "roll_std", ("speed", "current")))

    names = fs.feature_names(
        ["cmg2_wheel_speed", "cmg1_spin_current", "cmg1_wheel_speed"]
    )

    assert names == [
        "cmg1_wheel_speed__mean",
        "cmg2_wheel_speed__mean",
        "cmg1_spin_current__std",
        "cmg1_wheel_speed__std",
        "cmg2_wheel_speed__std",
    ]
    assert names == fs.feature_names(
        ["cmg1_wheel_speed", "cmg2_wheel_speed", "cmg1_spin_current"]
    )


def test_untyped_channels_get_no_features():
    fs = feature_set(spec("mean"))

    assert fs.feature_names(["mystery", "cmg1_wheel_speed"]) == [
        "cmg1_wheel_speed__mean"
    ]


def test_a_spec_cannot_be_applied_to_a_channel_of_another_type():
    fs = feature_set(spec("mean", inputs=["cmg1_spin_current"]))

    with pytest.raises(
        ConfigError,
        match=r"applies to channel types.*speed.*cmg1_spin_current.*current",
    ):
        fs.feature_names(["cmg1_spin_current", "cmg1_wheel_speed"])


def test_binary_specs_check_the_type_of_each_input():
    ratio = {
        "name": "k",
        "function": "safe_ratio",
        "inputs": ["cmg1_wheel_speed", "cmg1_spin_current"],  # swapped
        "applies_to": ["current", "speed"],
    }

    with pytest.raises(ConfigError, match="applies to channel types"):
        feature_set(ratio).feature_names(["cmg1_wheel_speed", "cmg1_spin_current"])


def test_missing_channels_fail_unless_the_spec_is_optional():
    inputs = ["cmg9_wheel_speed"]

    with pytest.raises(ConfigError, match="not available"):
        feature_set(spec("m", inputs=inputs)).feature_names(["cmg1_wheel_speed"])
    assert (
        feature_set(spec("m", inputs=inputs, optional=True)).feature_names(
            ["cmg1_wheel_speed"]
        )
        == []
    )

    ratio = {
        "name": "k",
        "function": "safe_ratio",
        "inputs": ["cmg1_spin_current", "cmg1_wheel_speed"],
        "applies_to": ["current", "speed"],
    }
    with pytest.raises(ConfigError, match="not available"):
        feature_set(ratio).feature_names(["cmg1_wheel_speed"])
    assert (
        feature_set({**ratio, "optional": True}).feature_names(["cmg1_wheel_speed"])
        == []
    )


def test_orbit_features_need_a_tle_unless_optional():
    required = {"name": "sunlit", "function": "orbit_sunlit"}

    with pytest.raises(ConfigError, match="needs an orbit"):
        feature_set(required).feature_names(["cmg1_wheel_speed"])
    assert feature_set({**required, "optional": True}).feature_names([]) == []
    assert feature_set(required).feature_names([], orbit_available=True) == ["sunlit"]


def test_duplicate_feature_names_are_rejected():
    with pytest.raises(ConfigError, match="duplicate feature names"):
        feature_set(spec("mean"), spec("mean")).feature_names(["cmg1_wheel_speed"])


# -- computing -------------------------------------------------------------------------


def test_compute_matches_the_window_functions_and_names():
    fs = feature_set(
        spec("mean", params={"window": "10min"}),
        spec("slope", "roll_slope", ("speed", "current"), params={"window": "1h"}),
        spec("rate", "rate_per_s", ("current",), params={}),
    )
    f = frame()

    result = compute({"A": f}, fs)

    names = fs.feature_names(
        ["cmg1_wheel_speed", "cmg1_spin_current", "cmg2_wheel_speed"]
    )
    assert list(result.columns) == names
    assert result.attrs["feature_names"] == names
    assert result.index.names == ["asset_id", "t"]
    assert (result.index.get_level_values("asset_id") == "A").all()
    pd.testing.assert_series_equal(
        result["cmg2_wheel_speed__mean"].droplevel("asset_id"),
        windows.roll_mean(f["cmg2_wheel_speed"], "10min"),
        check_names=False,
    )
    np.testing.assert_allclose(
        result["cmg1_spin_current__rate"].dropna(),
        windows.rate_per_s(f["cmg1_spin_current"]).dropna(),
    )


def test_compute_stacks_assets_in_a_fixed_column_order():
    fs = feature_set(spec("mean"))

    result = compute(
        {
            "B": frame(1, channels=("cmg1_wheel_speed",)),
            "A": frame(2, channels=("cmg1_wheel_speed", "cmg2_wheel_speed")),
        },
        fs,
    )

    assert list(result.columns) == ["cmg1_wheel_speed__mean", "cmg2_wheel_speed__mean"]
    assert result.index.get_level_values("asset_id").unique().tolist() == ["A", "B"]
    assert (
        result.loc["B", "cmg2_wheel_speed__mean"].isna().all()
    )  # channel absent for B


def test_physics_features_combine_channels():
    fs = feature_set(
        {
            "name": "k",
            "function": "safe_ratio",
            "params": {"eps": 1.0},
            "inputs": ["cmg1_spin_current", "cmg1_wheel_speed"],
            "applies_to": ["current", "speed"],
        },
        {
            "name": "gap",
            "function": "difference",
            "inputs": ["cmg1_wheel_speed", "cmg2_wheel_speed"],
            "applies_to": ["speed", "speed"],
        },
    )
    f = frame()

    result = compute({"A": f}, fs).droplevel("asset_id")

    np.testing.assert_allclose(
        result["k"], f["cmg1_spin_current"] / f["cmg1_wheel_speed"]
    )
    np.testing.assert_allclose(
        result["gap"], f["cmg1_wheel_speed"] - f["cmg2_wheel_speed"]
    )


def test_orbit_features_use_the_assets_tle():
    fs = feature_set(
        {"name": "sunlit", "function": "orbit_sunlit"},
        {"name": "eclipse_phase", "function": "orbit_eclipse_phase"},
    )
    index = pd.date_range(
        "2026-10-19T12:00", periods=600, freq="10s", tz="UTC", name="t"
    )
    f = pd.DataFrame({"cmg1_wheel_speed": 1.0}, index=index)

    result = compute({"ISS": f}, fs, {"ISS": TLE}).droplevel("asset_id")

    assert set(result["sunlit"].unique()) <= {0.0, 1.0}
    assert result["eclipse_phase"].between(0, 1).all()
    expected = orbit.orbit_features(TLE, index)
    pd.testing.assert_series_equal(
        result["sunlit"], expected["sunlit"], check_names=False
    )


def test_compute_with_no_frames_is_empty():
    result = compute({}, feature_set(spec("mean")))

    assert result.empty


# -- the shipped feature set -----------------------------------------------------------


def test_default_feature_set_covers_the_iss_channels_without_misuse():
    fs = FeatureSet.from_yaml(ROOT / "configs/features/default.yaml")
    channels = [
        c.name for c in load_channels(ROOT / "configs/ingest/iss.yaml").channels
    ]

    names = fs.feature_names(channels, orbit_available=True)

    assert len(names) == len(set(names))
    assert len(names) > 250
    assert {"sunlit", "eclipse_phase"} <= set(names)
    assert "cmg1_k_current_per_speed" in names
    assert "cmg3_bearing_minus_coolant" in names
    assert "cmg1_wheel_speed__robust_z_1h" in names
    assert "gnc_mode__code" in names
    assert not any(
        n.startswith("q_lvlh") for n in names
    )  # untyped channels get nothing


def test_default_feature_set_skips_optional_features_without_inputs_or_orbit():
    fs = FeatureSet.from_yaml(ROOT / "configs/features/default.yaml")

    names = fs.feature_names(["cmg1_wheel_speed"])

    assert "sunlit" not in names
    assert not any("k_current_per_speed" in n for n in names)


def test_feature_spec_model_is_importable_and_validates():
    assert (
        FeatureSpec(name="x", function="level", applies_to=["state"]).optional is False
    )
