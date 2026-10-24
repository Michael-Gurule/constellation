import os
import plistlib
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INSTALL = ROOT / "scripts" / "install_collector_service.sh"
SYNC = ROOT / "scripts" / "sync_readings.sh"
bash = shutil.which("bash")

pytestmark = pytest.mark.skipif(bash is None, reason="bash is not available")


def run(command, **kwargs):
    return subprocess.run(
        command, capture_output=True, text=True, check=False, **kwargs
    )


@pytest.mark.parametrize("script", [INSTALL, SYNC, ROOT / "scripts" / "collect_iss.sh"])
def test_scripts_parse_and_are_executable(script):
    assert run(["bash", "-n", str(script)]).returncode == 0
    assert script.stat().st_mode & stat.S_IXUSR


# -- installer ---------------------------------------------------------------------------


@pytest.fixture
def fake_env(tmp_path):
    prefix = tmp_path / "miniconda3" / "envs" / "constellation-collector"
    (prefix / "bin").mkdir(parents=True)
    tool = prefix / "bin" / "constellation"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    return prefix


def test_dry_run_renders_both_plists_and_the_log_rule(fake_env, tmp_path):
    out = tmp_path / "rendered"

    result = run(
        [
            "bash",
            str(INSTALL),
            "--dry-run",
            "--env",
            str(fake_env),
            "--render-to",
            str(out),
        ]
    )

    assert result.returncode == 0, result.stderr
    assert "dry run: nothing installed" in result.stdout
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "com.constellation.collect-iss.plist",
        "com.constellation.compact.plist",
        "constellation.conf",
    ]
    for path in out.iterdir():
        assert not re.search(r"@[A-Z]+@", path.read_text()), path.name


def test_rendered_plists_are_valid_and_carry_the_documented_settings(
    fake_env, tmp_path
):
    out = tmp_path / "rendered"
    run(
        [
            "bash",
            str(INSTALL),
            "--dry-run",
            "--env",
            str(fake_env),
            "--render-to",
            str(out),
        ]
    )

    collect = plistlib.loads((out / "com.constellation.collect-iss.plist").read_bytes())
    compact = plistlib.loads((out / "com.constellation.compact.plist").read_bytes())

    user = os.environ.get("SUDO_USER") or os.environ["USER"]
    home = collect["EnvironmentVariables"]["HOME"]
    assert collect["Label"] == "com.constellation.collect-iss"
    assert collect["UserName"] == user
    assert collect["WorkingDirectory"] == str(ROOT)
    assert collect["ProgramArguments"] == [
        f"{fake_env}/bin/constellation",
        "collect",
        "iss",
    ]
    assert collect["RunAtLoad"] is True
    assert collect["KeepAlive"] is True
    assert collect["ThrottleInterval"] == 30
    assert collect["ExitTimeOut"] >= 30  # time to flush after SIGTERM
    env = collect["EnvironmentVariables"]
    assert env["CONSTELLATION_DATA_DIR"] == f"{home}/constellation-data"
    assert env["CONSTELLATION_ISS_FLUSH_SECONDS"] == "300"
    assert env["CONSTELLATION_LOG_LEVEL"] == "INFO"
    assert (
        env["CONSTELLATION_LOG_FILE"]
        == f"{home}/constellation-data/logs/collect_iss.log"
    )
    assert collect["StandardOutPath"].endswith(
        "collect_iss.launchd.log"
    )  # not the rotated log
    assert compact["Label"] == "com.constellation.compact"
    assert compact["StartInterval"] == 3600
    assert compact["RunAtLoad"] is False
    assert compact["ProgramArguments"] == [
        f"{fake_env}/bin/constellation",
        "store",
        "compact",
    ]
    assert "CONSTELLATION_LOG_FILE" not in compact["EnvironmentVariables"]

    rule = (out / "constellation.conf").read_text()
    active = [
        line.split() for line in rule.splitlines() if line and not line.startswith("#")
    ]
    assert [row[0] for row in active] == [
        f"{home}/constellation-data/logs/collect_iss.log",
        f"{home}/constellation-data/logs/compact.log",
    ]
    assert active[0][1] == f"{user}:staff"
    assert active[0][3:] == ["14", "*", "$D0", "ZN"]  # daily, 14 compressed copies


@pytest.mark.skipif(shutil.which("plutil") is None, reason="plutil is only on macOS")
def test_rendered_plists_pass_plutil_lint(fake_env, tmp_path):
    out = tmp_path / "rendered"
    run(
        [
            "bash",
            str(INSTALL),
            "--dry-run",
            "--env",
            str(fake_env),
            "--render-to",
            str(out),
        ]
    )

    result = run(["plutil", "-lint", *map(str, out.glob("*.plist"))])

    assert result.returncode == 0, result.stdout + result.stderr


def test_installing_without_root_refuses_and_changes_nothing(fake_env):
    if os.geteuid() == 0:
        pytest.skip("running as root")

    result = run(["bash", str(INSTALL), "--env", str(fake_env)])

    assert result.returncode == 1
    assert "needs root" in result.stderr


def test_uninstall_dry_run_lists_what_it_would_do(fake_env):
    result = run(
        ["bash", str(INSTALL), "--uninstall", "--dry-run", "--env", str(fake_env)]
    )

    assert result.returncode == 0
    assert "launchctl bootout system/com.constellation.collect-iss" in result.stdout
    assert "/Library/LaunchDaemons/com.constellation.compact.plist" in result.stdout


def test_a_missing_env_is_reported(tmp_path):
    result = run(["bash", str(INSTALL), "--dry-run", "--env", str(tmp_path / "nope")])

    assert result.returncode == 1
    assert "not found" in result.stderr


def test_unknown_options_are_rejected(fake_env):
    assert run(["bash", str(INSTALL), "--bogus"]).returncode == 2


# -- sync --------------------------------------------------------------------------------


@pytest.fixture
def sync_env(tmp_path):
    if shutil.which("rsync") is None:
        pytest.skip("rsync is not available")
    source = tmp_path / "remote-copy"
    mine = tmp_path / "project-data"
    day = "readings/source=iss/asset=ISS/date=2026-03-01"
    (source / day).mkdir(parents=True)
    (source / day / "part-aaa.parquet").write_bytes(b"remote a")
    (source / day / "part-bbb.parquet").write_bytes(b"remote b")
    (source / day / ".part-ccc.parquet.tmp").write_bytes(b"in progress")
    (mine / day).mkdir(parents=True)
    (mine / day / "part-aaa.parquet").write_bytes(
        b"local a"
    )  # already here: must not be overwritten
    stubs = tmp_path / "bin"
    stubs.mkdir()
    log = tmp_path / "calls.log"
    stub = stubs / "constellation"
    stub.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\n')
    stub.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "CONSTELLATION_DATA_DIR": str(mine),
    }
    return source, mine / day, log, env


def test_sync_copies_only_new_files_then_compacts_and_reports(sync_env):
    source, local, log, env = sync_env

    result = run(["bash", str(SYNC), str(source)], env=env, cwd=ROOT)

    assert result.returncode == 0, result.stderr
    assert (local / "part-aaa.parquet").read_bytes() == b"local a"  # --ignore-existing
    assert (local / "part-bbb.parquet").read_bytes() == b"remote b"
    assert not (local / ".part-ccc.parquet.tmp").exists()  # temporary files stay behind
    assert log.read_text().splitlines() == ["store compact", "collect status"]
    assert (
        source / "readings/source=iss/asset=ISS/date=2026-03-01/part-aaa.parquet"
    ).read_bytes() == b"remote a"


def test_sync_dry_run_copies_nothing(sync_env):
    source, local, log, env = sync_env

    result = run(["bash", str(SYNC), str(source), "--dry-run"], env=env, cwd=ROOT)

    assert result.returncode == 0, result.stderr
    assert not (local / "part-bbb.parquet").exists()
    assert log.read_text().splitlines() == ["store compact --dry-run", "collect status"]


def test_sync_never_deletes_on_the_remote_side_and_checks_its_arguments(sync_env):
    source, _, _, env = sync_env
    script = SYNC.read_text()

    assert "--delete" not in script
    assert run(["bash", str(SYNC)], env=env).returncode == 2
    assert run(["bash", str(SYNC), str(source), "--bogus"], env=env).returncode == 2
