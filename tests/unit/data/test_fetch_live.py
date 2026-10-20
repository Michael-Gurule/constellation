"""Download smoke test against Zenodo. Run with `pytest --run-live`."""

from pathlib import Path

import pytest

from constellation.data.benchmarks.base import (
    BenchmarkSource,
    fetch,
    load_benchmark_config,
)

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "data" / "benchmarks.yaml"


@pytest.mark.live
def test_the_smallest_opssat_file_downloads_and_matches_its_checksums(tmp_path):
    source = load_benchmark_config(CONFIG).benchmarks["opssat"]
    small = min(source.files, key=lambda f: f.size or 0)
    only_that = BenchmarkSource(**{**source.model_dump(), "files": [small]})

    (result,) = fetch("opssat", only_that, tmp_path)

    assert result.downloaded
    assert result.digests["sha256"] == small.sha256
    assert result.digests["md5"] == small.md5
