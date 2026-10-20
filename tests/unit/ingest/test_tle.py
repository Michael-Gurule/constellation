import os
import time
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from constellation.cli import app
from constellation.core.errors import ConfigError
from constellation.core.settings import get_settings
from constellation.ingest import tle as tle_module
from constellation.ingest.tle import fetch_tle, read_tle, tle_path

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "iss_25544.tle"
).read_text()


def client(body=FIXTURE, status=200):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, text=body)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_fetch_downloads_validates_and_saves(tmp_path):
    http, calls = client()

    path, downloaded = fetch_tle(25544, tmp_path / "tle", client=http)

    assert downloaded
    assert path == tle_path(tmp_path / "tle", 25544)
    assert read_tle(path).norad_id == 25544
    assert dict(calls[0].url.params) == {"CATNR": "25544", "FORMAT": "tle"}
    assert calls[0].url.host == "celestrak.org"
    assert [p.name for p in (tmp_path / "tle").iterdir()] == [
        "25544.tle"
    ]  # no temp files


def test_a_recent_copy_is_kept_without_a_request(tmp_path):
    http, calls = client()
    fetch_tle(25544, tmp_path, client=http)

    _, downloaded = fetch_tle(25544, tmp_path, client=http)

    assert not downloaded
    assert len(calls) == 1


def test_an_old_copy_is_refreshed_and_force_overrides_the_age_limit(tmp_path):
    http, calls = client()
    path, _ = fetch_tle(25544, tmp_path, client=http)
    old = time.time() - 13 * 3600
    os.utime(path, (old, old))

    assert fetch_tle(25544, tmp_path, client=http)[1]
    assert fetch_tle(25544, tmp_path, force=True, client=http)[1]
    assert len(calls) == 3


def test_a_bad_download_never_damages_the_saved_copy(tmp_path):
    good, _ = client()
    path, _ = fetch_tle(25544, tmp_path, client=good)
    before = path.read_text()

    for body, status in [("No GP data found", 200), (FIXTURE, 503)]:
        bad, _ = client(body, status)
        with pytest.raises((ConfigError, httpx.HTTPStatusError)):
            fetch_tle(25544, tmp_path, force=True, client=bad)

    assert path.read_text() == before


def test_the_wrong_satellite_is_rejected(tmp_path):
    http, _ = client()

    with pytest.raises(ConfigError, match="received 25544"):
        fetch_tle(43017, tmp_path, client=http)


def test_cli_fetch_uses_the_data_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSTELLATION_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    seen = {}

    def fake_fetch(norad, directory, *, force=False):
        seen.update(norad=norad, directory=directory, force=force)
        return directory / f"{norad}.tle", True

    monkeypatch.setattr(tle_module, "fetch_tle", fake_fetch)

    result = CliRunner().invoke(app, ["tle", "fetch", "--norad", "25544", "--force"])

    get_settings.cache_clear()
    assert result.exit_code == 0
    assert seen == {
        "norad": 25544,
        "directory": tmp_path / "data" / "external" / "tle",
        "force": True,
    }
    assert "downloaded" in result.stdout


def test_cli_fetch_reports_a_kept_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSTELLATION_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr(
        tle_module,
        "fetch_tle",
        lambda norad, directory, force=False: (directory / "x.tle", False),
    )

    result = CliRunner().invoke(app, ["tle", "fetch"])

    get_settings.cache_clear()
    assert "kept recent copy" in result.stdout
