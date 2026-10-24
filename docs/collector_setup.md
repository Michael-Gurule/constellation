# Running the ISS collector on an always-on Mac

My daily laptop cannot stay awake for weeks, and the forecasting experiments need at least three weeks of continuous ISS data. So an old Intel MacBook Pro at home does one job: it runs the collector, around the clock, into its own small store. My main Mac pulls that store into the project when I need it. This page is how I set the old Mac up and keep it running.

The collector is light: about 90,000 readings an hour, roughly 50 MB a day before compaction, and very little CPU. The old Mac never trains a model.

## What runs where

| | Old Mac (collector) | Main Mac (project) |
|---|---|---|
| Install | collector-only environment (`make env-collector`) | full environment (`make env`) |
| Data | `~/constellation-data/readings` | `data/readings` |
| Jobs | `collect iss` (always), `store compact` (hourly) | everything else |

The full install does not run on an Intel Mac: PyTorch stopped publishing macOS x86_64 wheels after version 2.2, and the project pins a newer one. The collector only needs numpy, pandas, pyarrow, pydantic, pydantic-settings, PyYAML, typer and the Lightstreamer client, so `requirements-collector.txt` lists exactly those, at the same versions as `requirements.txt`.

## Prerequisites on the old Mac

1. **Check the macOS version** with `sw_vers`. Recent pyarrow wheels for Intel Macs may need macOS 12 or later. If `make env-collector` fails while installing pyarrow, install it from conda-forge instead:
   `conda install -n constellation-collector -c conda-forge pyarrow`, then re-run the `pip install` lines of the `env-collector` target without it.
2. **Never sleep**, including with the lid closed:
   ```bash
   sudo pmset -a sleep 0 disksleep 0 powernap 0 autorestart 1 womp 1
   sudo pmset -a disablesleep 1
   ```
   `autorestart` brings the machine back after a power cut; `womp` allows wake on network access.
3. **FileVault.** After an unattended restart, a FileVault-encrypted disk waits for someone to log in before the service can start. I either turn FileVault off on this machine (it holds nothing but public telemetry) or accept a manual login after each restart.
4. **Turn off automatic macOS update installs** (System Settings, General, Software Update, Automatic Updates), so the machine does not restart on its own. I install updates by hand.
5. **Battery.** It stays plugged in for weeks. I turn on optimized charging or cap the charge near 60% if this macOS version allows it, check the battery condition in System Information, and keep the machine on a hard, ventilated surface.
6. **Network.** Ethernet if I can, Wi-Fi otherwise.
7. **Remote Login** (System Settings, General, Sharing) with SSH keys only: copy my public key with `ssh-copy-id`, then set `PasswordAuthentication no` in `/etc/ssh/sshd_config`.
8. **conda.** Install Miniconda (the Intel build) for the user that will run the service.

## Getting the code

Either clone from GitHub:

```bash
git clone <repository url> ~/constellation && cd ~/constellation && git checkout v2
```

or clone straight from the main Mac over SSH:

```bash
git clone ssh://<main-mac>/<path to the repository> -b v2 ~/constellation
```

Then build the collector environment:

```bash
cd ~/constellation
make env-collector
```

That creates the conda environment `constellation-collector` (Python 3.12), installs the pinned packages, and installs the project itself without its other dependencies. Test it before installing the service:

```bash
export CONSTELLATION_DATA_DIR=/tmp/collector-test
~/miniconda3/envs/constellation-collector/bin/constellation collect iss --duration 2m
~/miniconda3/envs/constellation-collector/bin/constellation collect status
```

`status` should show `[ok]` and a last update a few seconds ago.

## Installing the service

The service runs as my user (not root) and starts at boot. `scripts/install_collector_service.sh` renders two launchd jobs and a log-rotation rule from the templates in `deploy/macos/`:

| File | What it does |
|---|---|
| `com.constellation.collect-iss.plist` | `constellation collect iss`, started at load and restarted if it stops, flushing to disk every 5 minutes |
| `com.constellation.compact.plist` | `constellation store compact` every hour |
| `/etc/newsyslog.d/constellation.conf` | rotates the logs daily and keeps 14 compressed copies |

Look at what it will do first. This changes nothing and needs no sudo:

```bash
scripts/install_collector_service.sh --dry-run
```

Then install:

```bash
sudo scripts/install_collector_service.sh
```

Everything lives under `~/constellation-data/`: `readings/` (the store) and `logs/`. The collector writes its own log to `logs/collect_iss.log` and reopens it after a rotation. launchd's own output files (`collect_iss.launchd.log`, `compact.log`) only hold what happens outside Python, such as a failure to start.

Check it after about ten minutes, and again the next day:

```bash
ssh <old-mac> '~/miniconda3/envs/constellation-collector/bin/constellation collect status'
ssh <old-mac> 'tail -n 20 ~/constellation-data/logs/collect_iss.log'
```

## Daily operations

- **Status:** `constellation collect status` over SSH, as above. `[STALE]` means no update for more than five minutes.
- **Stop and start:**
  ```bash
  sudo launchctl bootout system/com.constellation.collect-iss      # stops it; it flushes its buffer first
  sudo launchctl bootstrap system /Library/LaunchDaemons/com.constellation.collect-iss.plist
  ```
  A deliberate stop is not restarted. `sudo scripts/install_collector_service.sh --uninstall` removes both jobs and the log rule and leaves the data alone.
- **Updating the code:** `git pull`, then restart the service with `sudo launchctl kickstart -k system/com.constellation.collect-iss`.
- **Never run the laptop collector at the same time.** Two collectors would write the same readings twice. The store removes exact duplicates when it reads, but there is no reason to rely on that.

## Syncing and snapshots

From the main Mac, pull what the old Mac has collected:

```bash
scripts/sync_readings.sh <old-mac>
```

It copies only files I do not have yet, skips files still being written, never deletes anything on the old Mac, then compacts the local store (one deduplicated file per finished day) and shows the status. `--dry-run` shows what would be copied. I sync weekly, and before running the E1 and E2 experiments. If I have a copy of the old Mac's `constellation-data` folder on a disk, `scripts/sync_readings.sh /path/to/that/folder` does the same without SSH.

When an experiment needs a fixed set of data, I take a snapshot of the finished days it will use:

```bash
constellation store compact
constellation data snapshot iss --start 2026-10-21 --end 2026-11-11 --name e2
constellation data snapshot iss --start 2026-10-21 --end 2026-11-11 --name e2 --verify
```

The snapshot is a small manifest in `data/manifests/` (file hashes, row counts, first and last time per channel, and every gap longer than five minutes). `--verify` reports any file that has changed, gone missing or appeared since. The experiment records the snapshot name it used.

## Optional: a heartbeat

If the old Mac stops sending data I would like to hear about it. A free heartbeat service works by pinging a URL on a schedule and sending a message when the pings stop. Adding a `curl` to the hourly job would be enough, but whether to use an outside service at all is my call, so nothing in the repository does this.

## Troubleshooting

- **`collect status` says `[STALE]`.** Look at the end of `collect_iss.log`. An `ingest_gap` line means the feed went quiet and the collector is waiting for it; the client reconnects by itself. If the log is silent, check that the service is loaded with `sudo launchctl print system/com.constellation.collect-iss`.
- **The service does not start after a restart.** Check FileVault (a login may be needed), then `collect_iss.launchd.log` for a Python error.
- **A job keeps restarting.** launchd waits 30 seconds between starts. `collect_iss.launchd.log` has the reason, usually an environment path that does not exist; re-run the installer with `--env <path to the constellation-collector env>`.
- **`make env-collector` fails on pyarrow or numpy.** See the first prerequisite: older macOS versions need a different wheel; install pyarrow from conda-forge.
- **The store has many small files.** Compaction runs hourly and only touches finished days, so today always has many files. Run `constellation store compact --dry-run` to see what it would merge.
- **A compaction was interrupted.** It writes and verifies the new file before deleting the old ones, so an interruption leaves the old files in place; the next run does the work again.
- **I want to start over.** Stop the service, move `~/constellation-data` aside, and start it again. The old data is not touched by anything above.
