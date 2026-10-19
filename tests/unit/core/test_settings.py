from pathlib import Path

import pytest
from pydantic import ValidationError

from constellation.core.settings import Settings, get_settings

VARIABLES = [
    "DATA_DIR",
    "MODELS_DIR",
    "RUNS_DIR",
    "REPORTS_DIR",
    "RESULTS_DIR",
    "LOG_LEVEL",
    "API_KEY",
    "API_URL",
    "DASHBOARD_ORIGIN",
    "SATNOGS_TOKEN",
    "ISS_FLUSH_SECONDS",
]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    for name in VARIABLES:
        monkeypatch.delenv(f"CONSTELLATION_{name}", raising=False)
    monkeypatch.chdir(tmp_path)  # no stray .env from the repository
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_match_the_contract():
    settings = Settings()

    assert settings.data_dir == Path("data")
    assert settings.results_dir == Path("data/results")
    assert settings.log_level == "INFO"
    assert settings.api_url == "http://127.0.0.1:8000"
    assert settings.dashboard_origin == "http://127.0.0.1:8501"
    assert settings.iss_flush_seconds == 60
    assert settings.api_key is None
    assert settings.satnogs_token is None


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("CONSTELLATION_DATA_DIR", "/srv/telemetry")
    monkeypatch.setenv("CONSTELLATION_ISS_FLUSH_SECONDS", "15")

    settings = Settings()

    assert settings.data_dir == Path("/srv/telemetry")
    assert settings.iss_flush_seconds == 15


def test_dotenv_file_is_read(tmp_path):
    (tmp_path / ".env").write_text("CONSTELLATION_LOG_LEVEL=DEBUG\n")

    assert Settings().log_level == "DEBUG"


def test_empty_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("CONSTELLATION_DATA_DIR", "")
    monkeypatch.setenv("CONSTELLATION_ISS_FLUSH_SECONDS", "")

    settings = Settings()

    assert settings.data_dir == Path("data")
    assert settings.iss_flush_seconds == 60


def test_secrets_are_not_printed(monkeypatch):
    monkeypatch.setenv("CONSTELLATION_API_KEY", "hunter2-secret")

    settings = Settings()

    assert "hunter2-secret" not in repr(settings)
    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == "hunter2-secret"


def test_flush_interval_must_be_positive(monkeypatch):
    monkeypatch.setenv("CONSTELLATION_ISS_FLUSH_SECONDS", "0")

    with pytest.raises(ValidationError):
        Settings()


def test_get_settings_is_cached_until_cleared(monkeypatch):
    first = get_settings()
    monkeypatch.setenv("CONSTELLATION_LOG_LEVEL", "WARNING")

    assert get_settings() is first
    assert get_settings().log_level == "INFO"

    get_settings.cache_clear()
    assert get_settings().log_level == "WARNING"
