import json
from datetime import UTC, datetime

import numpy as np
import pytest
import torch
from sklearn.linear_model import LinearRegression

from constellation.core import artifacts
from constellation.core.artifacts import (
    create_metadata,
    feature_set_hash,
    load_artifact,
    read_metadata,
    save_artifact,
)
from constellation.core.errors import ArtifactError, FeatureMismatchError

FEATURES = ["bus_voltage", "bus_current", "temp"]
CONFIG = {"alpha": 0.1, "window": 32}
CREATED = datetime(2026, 3, 1, 12, 30, 45, tzinfo=UTC)


def _metadata(**overrides):
    kwargs = {
        "config": CONFIG,
        "feature_names": FEATURES,
        "thresholds": {"alarm": 0.97},
        "metrics": {"f1": 0.9},
        "data_manifest_hashes": {"sim": "abc"},
        "created_at": CREATED,
    }
    kwargs.update(overrides)
    return create_metadata("detector", **kwargs)


def _model():
    x = np.arange(12, dtype=float).reshape(4, 3)
    return LinearRegression().fit(x, x @ np.array([1.0, 2.0, 3.0]))


def test_metadata_carries_the_documented_fields():
    meta = _metadata()

    assert meta.name == "detector"
    assert meta.version == f"20260301-123045-{meta.config_hash[:8]}"
    assert meta.feature_names == FEATURES
    assert meta.feature_set_hash == feature_set_hash(FEATURES)
    assert meta.thresholds == {"alarm": 0.97}
    assert meta.metrics == {"f1": 0.9}
    assert meta.data_manifest_hashes == {"sim": "abc"}
    assert "numpy" in meta.library_versions
    assert "python" in meta.library_versions
    assert meta.git_sha is None or len(meta.git_sha) == 40


def test_config_hash_and_feature_hash_are_stable_but_sensitive():
    assert _metadata().config_hash == _metadata().config_hash
    assert _metadata(config={"alpha": 0.2}).config_hash != _metadata().config_hash
    assert feature_set_hash(FEATURES) != feature_set_hash(FEATURES[::-1])


def test_artifact_dir_follows_the_name_version_layout(tmp_path):
    meta = _metadata()

    assert (
        artifacts.artifact_dir(tmp_path, meta) == tmp_path / "detector" / meta.version
    )


def test_joblib_round_trip(tmp_path):
    model = _model()
    meta = _metadata()
    directory = tmp_path / "detector" / meta.version

    save_artifact(model, directory, meta)
    loaded = load_artifact(directory, expected_features=FEATURES)

    assert sorted(p.name for p in directory.iterdir()) == [
        "metadata.json",
        "model.joblib",
    ]
    np.testing.assert_array_equal(loaded.model.coef_, model.coef_)
    assert loaded.metadata == meta
    assert json.loads((directory / "metadata.json").read_text())["name"] == "detector"


def test_torch_module_is_stored_as_a_state_dict(tmp_path):
    torch.manual_seed(0)
    module = torch.nn.Linear(3, 2)
    meta = _metadata()
    directory = tmp_path / "net"

    save_artifact(module, directory, meta)
    loaded = load_artifact(directory)

    assert sorted(p.name for p in directory.iterdir()) == ["metadata.json", "model.pt"]
    assert isinstance(loaded.model, dict)
    fresh = torch.nn.Linear(3, 2)
    fresh.load_state_dict(loaded.model)
    assert torch.equal(fresh.weight, module.weight)


@pytest.mark.parametrize(
    ("features", "detail"),
    [
        (["bus_voltage", "bus_current"], "missing=['temp']"),
        ([*FEATURES, "extra"], "unexpected=['extra']"),
        (["temp", "bus_voltage", "bus_current"], "different order"),
    ],
)
def test_feature_mismatch_raises_a_clear_error(tmp_path, features, detail):
    directory = tmp_path / "a"
    save_artifact(_model(), directory, _metadata())

    with pytest.raises(FeatureMismatchError, match="trained features") as info:
        load_artifact(directory, expected_features=features)

    assert detail in str(info.value)


def test_loading_without_expected_features_skips_the_check(tmp_path):
    directory = tmp_path / "a"
    save_artifact(_model(), directory, _metadata())

    assert load_artifact(directory).metadata.feature_names == FEATURES


def test_existing_directory_is_never_overwritten(tmp_path):
    directory = tmp_path / "a"
    save_artifact(_model(), directory, _metadata())

    with pytest.raises(ArtifactError, match="already exists"):
        save_artifact(_model(), directory, _metadata())


def test_failed_save_leaves_nothing_behind(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(artifacts.joblib, "dump", explode)

    with pytest.raises(OSError, match="disk full"):
        save_artifact(_model(), tmp_path / "models" / "a", _metadata())

    assert list((tmp_path / "models").iterdir()) == []


def test_missing_metadata_or_model_file_is_reported(tmp_path):
    with pytest.raises(ArtifactError, match="cannot read artifact metadata"):
        read_metadata(tmp_path / "absent")

    directory = tmp_path / "a"
    save_artifact(_model(), directory, _metadata())
    (directory / "model.joblib").unlink()
    with pytest.raises(ArtifactError, match="no model file"):
        load_artifact(directory)


def test_pydantic_config_is_accepted():
    from pydantic import BaseModel

    class Config(BaseModel):
        alpha: float = 0.1
        window: int = 32

    meta = _metadata(config=Config())

    assert meta.config == {"alpha": 0.1, "window": 32}
    assert meta.config_hash == _metadata().config_hash
