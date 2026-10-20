import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from constellation.cli import app
from constellation.core.errors import StoreError
from constellation.core.settings import get_settings
from constellation.data.benchmarks.base import (
    BenchmarkFile,
    BenchmarkSource,
    ManualDownloadError,
    fetch,
    file_digest,
    load_benchmark_config,
    load_manifest,
    safe_extract,
    update_manifest,
    verify_file,
)

ROOT = Path(__file__).resolve().parents[3]
CONTENT = b"time,value\n1,2\n3,4\n"
MD5 = hashlib.md5(CONTENT).hexdigest()
SHA = hashlib.sha256(CONTENT).hexdigest()


def archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


def client(routes: dict[str, bytes | int]):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        body = routes.get(request.url.path)
        if body is None or isinstance(body, int):
            return httpx.Response(body or 404)
        return httpx.Response(200, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def source(*files: BenchmarkFile, manual: str = "") -> BenchmarkSource:
    return BenchmarkSource(
        title="Demo",
        doi="10.1/demo",
        citation="Demo et al.",
        licence="CC0",
        files=list(files),
        manual=manual,
    )


def demo_file(**overrides) -> BenchmarkFile:
    values = {
        "name": "data.csv",
        "url": "https://example.org/data.csv",
        "md5": MD5,
        "sha256": SHA,
        "size": len(CONTENT),
    }
    values.update(overrides)
    return BenchmarkFile(**values)


def test_fetch_downloads_verifies_and_reports_digests(tmp_path):
    http, calls = client({"/data.csv": CONTENT})
    seen: list[int] = []

    (result,) = fetch(
        "demo", source(demo_file()), tmp_path, client=http, progress=seen.append
    )

    assert result.downloaded
    assert (tmp_path / "data.csv").read_bytes() == CONTENT
    assert result.digests == {"size": len(CONTENT), "md5": MD5, "sha256": SHA}
    assert sum(seen) == len(CONTENT)
    assert calls == ["/data.csv"]
    assert [p.name for p in tmp_path.iterdir()] == ["data.csv"]  # no temporary files


def test_a_verified_file_is_not_downloaded_again(tmp_path):
    http, calls = client({"/data.csv": CONTENT})
    fetch("demo", source(demo_file()), tmp_path, client=http)

    (result,) = fetch("demo", source(demo_file()), tmp_path, client=http)

    assert not result.downloaded
    assert calls == ["/data.csv"]


@pytest.mark.parametrize("field", ["md5", "sha256"])
def test_a_download_that_fails_its_checksum_is_discarded(tmp_path, field):
    http, _ = client({"/data.csv": CONTENT})

    with pytest.raises(StoreError, match=f"{field} mismatch"):
        fetch(
            "demo",
            source(demo_file(**{field: "0" * (32 if field == "md5" else 64)})),
            tmp_path,
            client=http,
        )

    assert not (tmp_path / "data.csv").exists()


def test_a_size_mismatch_is_rejected(tmp_path):
    http, _ = client({"/data.csv": CONTENT})

    with pytest.raises(StoreError, match="expected 5 bytes"):
        fetch("demo", source(demo_file(size=5)), tmp_path, client=http)


def test_an_existing_file_that_fails_verification_is_kept_for_inspection(tmp_path):
    (tmp_path / "data.csv").write_bytes(b"something else")
    http, calls = client({"/data.csv": CONTENT})

    with pytest.raises(StoreError, match=r"mismatch|expected"):
        fetch("demo", source(demo_file()), tmp_path, client=http)

    assert (tmp_path / "data.csv").read_bytes() == b"something else"
    assert calls == []


def test_http_errors_leave_no_partial_files(tmp_path):
    http, _ = client({"/data.csv": 503})

    with pytest.raises(httpx.HTTPStatusError):
        fetch("demo", source(demo_file()), tmp_path, client=http)

    assert list(tmp_path.iterdir()) == []


def test_files_without_a_url_must_be_supplied_by_hand(tmp_path):
    spec = source(
        BenchmarkFile(name="data.zip", extract=True), manual="Get it from the mirror."
    )

    with pytest.raises(
        ManualDownloadError, match=r"saved to .*data\.zip.*Get it from the mirror"
    ):
        fetch("demo", spec, tmp_path)

    (tmp_path / "data.zip").write_bytes(archive({"train/a.npy": b"x"}))
    (result,) = fetch("demo", spec, tmp_path)
    assert not result.downloaded
    assert (tmp_path / "train" / "a.npy").read_bytes() == b"x"  # extracted


def test_extracted_directories_stand_in_for_a_missing_archive(tmp_path):
    for split in ("train", "test"):
        (tmp_path / split).mkdir()
        (tmp_path / split / "A-1.npy").write_bytes(split.encode())
    (tmp_path / "train" / ".DS_Store").write_bytes(b"ignored")
    spec = source(
        BenchmarkFile(name="data.zip", extract=True, provided_as=["train", "test"])
    )

    (result,) = fetch("demo", spec, tmp_path)

    assert result.tree
    assert result.name == "train+test"
    assert not result.downloaded
    assert result.digests["files"] == 2
    assert result.digests["size"] == len(b"train") + len(b"test")

    first = result.digests["sha256"]
    assert fetch("demo", spec, tmp_path)[0].digests["sha256"] == first  # stable
    (tmp_path / "test" / "A-1.npy").write_bytes(b"changed")
    assert (
        fetch("demo", spec, tmp_path)[0].digests["sha256"] != first
    )  # sensitive to content
    (tmp_path / "test" / "B-1.npy").write_bytes(b"x")
    assert fetch("demo", spec, tmp_path)[0].digests["files"] == 3  # and to new files


def test_the_archive_wins_when_both_it_and_extracted_folders_exist(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "data.zip").write_bytes(archive({"train/x.npy": b"x"}))
    spec = source(
        BenchmarkFile(name="data.zip", extract=True, provided_as=["train", "test"])
    )

    (result,) = fetch("demo", spec, tmp_path)

    assert not result.tree
    assert (tmp_path / "train" / "x.npy").exists()


def test_a_partial_extracted_layout_still_asks_for_a_manual_step(tmp_path):
    (tmp_path / "train").mkdir()  # no test/
    spec = source(
        BenchmarkFile(name="data.zip", provided_as=["train", "test"]),
        manual="See the notes.",
    )

    with pytest.raises(
        ManualDownloadError, match=r"extracted so that train, test sit beside it"
    ):
        fetch("demo", spec, tmp_path)


def test_manifest_entries_for_extracted_trees_carry_a_file_count(tmp_path):
    for split in ("train", "test"):
        (tmp_path / split).mkdir()
        (tmp_path / split / "A-1.npy").write_bytes(b"x")
    spec = source(BenchmarkFile(name="data.zip", provided_as=["train", "test"]))
    files = fetch("demo", spec, tmp_path)

    manifest = update_manifest(tmp_path / "m.json", "demo", files)

    assert set(manifest["demo"]["train+test"]) == {"sha256", "size", "files"}


def test_archives_are_extracted_after_verification(tmp_path):
    payload = archive(
        {"ESA-Mission1/labels.csv": b"ID\n", "ESA-Mission1/channels/c.zip": b"z"}
    )
    http, _ = client({"/a.zip": payload})
    spec = source(
        BenchmarkFile(
            name="a.zip",
            url="https://example.org/a.zip",
            extract=True,
            md5=hashlib.md5(payload).hexdigest(),
        )
    )

    fetch("demo", spec, tmp_path, client=http)

    assert (tmp_path / "ESA-Mission1" / "labels.csv").exists()
    assert (tmp_path / "ESA-Mission1" / "channels" / "c.zip").exists()


def test_archive_members_cannot_escape_the_target_directory(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(archive({"../escape.txt": b"x"}))

    with pytest.raises(StoreError, match="unsafe path"):
        safe_extract(bad, tmp_path / "out")

    assert not (tmp_path / "escape.txt").exists()


def test_verify_file_returns_observed_digests_when_nothing_is_pinned(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(CONTENT)

    observed = verify_file(path, BenchmarkFile(name="x.bin"))

    assert observed == {"size": len(CONTENT), "md5": MD5, "sha256": SHA}
    assert file_digest(path, "sha256") == SHA


def test_manifest_records_checksums_per_benchmark_and_merges(tmp_path):
    http, _ = client({"/data.csv": CONTENT})
    files = fetch("demo", source(demo_file()), tmp_path / "demo", client=http)
    manifest_path = tmp_path / "manifests" / "benchmarks.json"

    update_manifest(manifest_path, "zeta", files)
    update_manifest(manifest_path, "alpha", files)

    manifest = json.loads(manifest_path.read_text())
    assert list(manifest) == ["alpha", "zeta"]
    assert manifest["alpha"]["data.csv"] == {
        "md5": MD5,
        "sha256": SHA,
        "size": len(CONTENT),
    }
    assert load_manifest(manifest_path) == manifest
    assert load_manifest(tmp_path / "none.json") == {}


def test_the_shipped_config_describes_the_three_benchmarks():
    config = load_benchmark_config(ROOT / "configs/data/benchmarks.yaml").benchmarks

    assert set(config) == {"smap_msl", "esa_adb", "opssat"}
    assert config["opssat"].licence == "CC-BY-4.0"
    assert all(f.sha256 and len(f.sha256) == 64 for f in config["opssat"].files)
    mission = config["esa_adb"].files[0]
    assert (mission.name, mission.size, mission.extract) == (
        "ESA-Mission1.zip",
        3776246054,
        True,
    )
    assert mission.md5 is not None
    assert len(mission.md5) == 32
    assert config["esa_adb"].doi == "10.5281/zenodo.12528696"
    assert config["opssat"].doi == "10.5281/zenodo.12588359"
    assert config["smap_msl"].files[0].url is None  # manual download
    assert config["smap_msl"].files[0].provided_as == ["train", "test"]
    assert (
        config["smap_msl"].files[1].sha256
        == "057ce2d6c8875982bf4e5404aefea14efdcbce413d80826d2b737c95b59b7539"
    )
    assert "extracted" in config["smap_msl"].manual


# -- command line ----------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("CONSTELLATION_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    external = tmp_path / "data" / "external"
    (external / "demo").mkdir(parents=True)
    (external / "demo" / "data.csv").write_bytes(CONTENT)
    config = tmp_path / "benchmarks.yaml"
    config.write_text(
        f"""
benchmarks:
  demo:
    title: Demo set
    doi: 10.1/demo
    citation: Demo et al.
    licence: CC0
    files:
      - {{name: data.csv, md5: {MD5}, sha256: {SHA}, size: {len(CONTENT)}}}
  handmade:
    title: Hand made
    doi: 10.1/hand
    citation: Hand et al.
    licence: CC0
    manual: Download it somewhere.
    files:
      - {{name: data.zip}}
"""
    )
    yield tmp_path, config
    get_settings.cache_clear()


def test_cli_verifies_existing_files_and_writes_the_manifest(workspace):
    tmp_path, config = workspace

    result = CliRunner().invoke(app, ["data", "fetch", "demo", "--config", str(config)])

    assert result.exit_code == 0
    assert "data.csv: verified" in result.stdout
    manifest = json.loads(
        (tmp_path / "data" / "manifests" / "benchmarks.json").read_text()
    )
    assert manifest["demo"]["data.csv"]["sha256"] == SHA


def test_cli_reports_a_manual_step_with_exit_code_2(workspace):
    _, config = workspace

    result = CliRunner().invoke(
        app, ["data", "fetch", "handmade", "--config", str(config)]
    )

    assert result.exit_code == 2
    assert "needs a manual step" in result.output
    assert "Download it somewhere" in result.output


def test_cli_all_continues_past_a_manual_download(workspace):
    tmp_path, config = workspace

    result = CliRunner().invoke(
        app, ["data", "fetch", "--all", "--config", str(config)]
    )

    assert result.exit_code == 2
    assert "data.csv: verified" in result.stdout
    assert "demo" in json.loads(
        (tmp_path / "data" / "manifests" / "benchmarks.json").read_text()
    )


@pytest.mark.parametrize(
    "arguments",
    [["nope"], [], ["demo", "--all"]],
)
def test_cli_rejects_bad_arguments(workspace, arguments):
    _, config = workspace

    result = CliRunner().invoke(
        app, ["data", "fetch", *arguments, "--config", str(config)]
    )

    assert result.exit_code == 2
