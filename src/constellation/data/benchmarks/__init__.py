"""Labeled spacecraft anomaly benchmarks in a common series format."""

from pathlib import Path

from constellation.core.errors import ConfigError
from constellation.data.benchmarks.base import (
    Benchmark,
    BenchmarkSeries,
    fetch,
    load_benchmark_config,
)
from constellation.data.benchmarks.esa_adb import EsaAdbBenchmark
from constellation.data.benchmarks.opssat import OpsSatBenchmark
from constellation.data.benchmarks.smap_msl import SmapMslBenchmark

__all__ = [
    "BENCHMARKS",
    "Benchmark",
    "BenchmarkSeries",
    "EsaAdbBenchmark",
    "OpsSatBenchmark",
    "SmapMslBenchmark",
    "fetch",
    "get_benchmark",
    "load_benchmark_config",
]

BENCHMARKS = {
    "smap_msl": SmapMslBenchmark,
    "esa_adb": EsaAdbBenchmark,
    "opssat": OpsSatBenchmark,
}


def get_benchmark(name: str, data_dir: Path, **options: object) -> Benchmark:
    """The benchmark `name`, reading from `<data_dir>/external/<name>/`."""
    if name not in BENCHMARKS:
        raise ConfigError(f"unknown benchmark {name!r}; known: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name](data_dir / "external" / name, **options)  # type: ignore[no-any-return]
