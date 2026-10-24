"""Typed settings read from `CONSTELLATION_*` environment variables and `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Empty environment values fall back to the defaults."""

    model_config = SettingsConfigDict(
        env_prefix="CONSTELLATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    runs_dir: Path = Path("runs")
    reports_dir: Path = Path("reports")
    results_dir: Path = Path("data/results")
    log_level: str = "INFO"
    log_file: Path | None = None
    api_key: SecretStr | None = None
    api_url: str = "http://127.0.0.1:8000"
    dashboard_origin: str = "http://127.0.0.1:8501"
    satnogs_token: SecretStr | None = None
    iss_flush_seconds: int = Field(default=60, gt=0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings. Call `get_settings.cache_clear()` to reload."""
    return Settings()
