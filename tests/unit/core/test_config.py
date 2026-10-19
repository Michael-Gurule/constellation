import pytest
from pydantic import BaseModel

from constellation.core.config import canonical_json, config_hash, load_config
from constellation.core.errors import ConfigError


class Window(BaseModel):
    size: int
    stride: int = 1


class Experiment(BaseModel):
    name: str
    window: Window


def test_load_config_validates_into_the_model(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("name: e1\nwindow:\n  size: 32\n")

    config = load_config(path, Experiment)

    assert config == Experiment(name="e1", window=Window(size=32, stride=1))


def test_load_config_reports_validation_problems(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("name: e1\nwindow:\n  size: not-a-number\n")

    with pytest.raises(ConfigError, match="size"):
        load_config(path, Experiment)


def test_load_config_rejects_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml", Experiment)

    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n")
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(bad, Experiment)


def test_load_config_requires_a_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n")

    with pytest.raises(ConfigError, match="mapping"):
        load_config(path, Experiment)


def test_config_hash_is_stable_and_ignores_key_order():
    a = {"name": "e1", "window": {"size": 32, "stride": 1}}
    b = {"window": {"stride": 1, "size": 32}, "name": "e1"}

    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 64


def test_config_hash_changes_with_content():
    base = Experiment(name="e1", window=Window(size=32))
    changed = Experiment(name="e1", window=Window(size=64))

    assert config_hash(base) != config_hash(changed)


def test_config_hash_of_a_model_matches_its_dict_form():
    model = Experiment(name="e1", window=Window(size=32))

    assert config_hash(model) == config_hash(model.model_dump())
    assert canonical_json(model) == '{"name":"e1","window":{"size":32,"stride":1}}'
