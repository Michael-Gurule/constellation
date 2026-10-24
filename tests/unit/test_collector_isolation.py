"""The collector must run without the heavy packages the rest of the project uses.

A machine that cannot install PyTorch (an Intel Mac, for example) runs the collector
from a minimal environment. This test blocks the heavy packages in a fresh
interpreter and makes sure the collector's import path and CLI still work.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

BLOCKED = [
    "torch", "lightgbm", "sklearn", "sksurv", "lifelines", "ortools", "stumpy", "shap",
    "statsmodels", "skyfield", "sgp4", "matplotlib", "streamlit", "fastapi", "scipy",
    "httpx", "joblib", "plotly",
]  # fmt: skip

SCRIPT = textwrap.dedent(
    """
    import sys

    for name in {blocked!r}:
        sys.modules[name] = None  # any import of these now raises ImportError

    import constellation.cli
    import constellation.ingest.isslive
    import constellation.core.compact
    import constellation.data.snapshot

    for name in {blocked!r}:
        assert sys.modules[name] is None, name + " was imported"

    try:
        constellation.cli.app(["collect", "status", "--help"])
    except SystemExit as stop:
        assert stop.code == 0, stop.code
    """
)


def run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )


def test_collector_and_maintenance_commands_need_only_the_light_dependencies():
    result = run(SCRIPT.format(blocked=BLOCKED))

    assert result.returncode == 0, result.stderr
    assert "Summarize what the store holds" in result.stdout


@pytest.mark.parametrize(
    "command", [["store", "compact", "--help"], ["data", "snapshot", "--help"]]
)
def test_store_maintenance_help_works_without_the_heavy_packages(command):
    code = SCRIPT.format(blocked=BLOCKED).replace(
        '["collect", "status", "--help"]', repr(command)
    )

    result = run(code)

    assert result.returncode == 0, result.stderr


def test_the_blocking_actually_blocks():
    result = run("import sys; sys.modules['torch'] = None; import torch")

    assert result.returncode != 0
    assert "import of torch halted" in result.stderr


def test_collector_requirements_are_the_eight_packages_at_the_pinned_versions():
    def pins(path):
        return {
            line.split("==")[0].lower(): line.strip()
            for line in (ROOT / path).read_text().splitlines()
            if "==" in line and not line.startswith("#")
        }

    collector = pins("requirements-collector.txt")
    full = pins("requirements.txt")

    assert set(collector) == {
        "numpy", "pandas", "pyarrow", "pydantic", "pydantic-settings", "pyyaml", "typer", "lightstreamer-client-lib",
    }  # fmt: skip
    assert all(collector[name] == full[name] for name in collector)
