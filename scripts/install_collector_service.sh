#!/bin/bash
# Install the ISS collector as a launchd service on an always-on Mac.
#
#   scripts/install_collector_service.sh --dry-run        render and check, change nothing
#   sudo scripts/install_collector_service.sh             install and start both services
#   sudo scripts/install_collector_service.sh --uninstall stop and remove them (data is kept)
#
# Options:
#   --env PREFIX      the constellation-collector conda env (default: $CONDA_PREFIX or a search)
#   --render-to DIR   with --dry-run, also write the rendered files into DIR
#
# The service runs as the user who invokes this script (the user behind sudo).

set -euo pipefail

LABEL_COLLECT="com.constellation.collect-iss"
LABEL_COMPACT="com.constellation.compact"
DAEMONS="/Library/LaunchDaemons"
NEWSYSLOG_DIR="/etc/newsyslog.d"
NEWSYSLOG_FILE="$NEWSYSLOG_DIR/constellation.conf"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TEMPLATES="$REPO/deploy/macos"

DRY_RUN=0
UNINSTALL=0
ENV_PREFIX=""
RENDER_TO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --uninstall) UNINSTALL=1 ;;
        --env) ENV_PREFIX="${2:?--env needs a path}"; shift ;;
        --render-to) RENDER_TO="${2:?--render-to needs a directory}"; shift ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_HOME="$(dscl . -read "/Users/$RUN_USER" NFSHomeDirectory 2>/dev/null | awk '{print $2}')"
RUN_HOME="${RUN_HOME:-$(eval echo "~$RUN_USER")}"

find_env() {
    local candidate
    if [[ -n "$ENV_PREFIX" ]]; then
        echo "$ENV_PREFIX"
        return
    fi
    for candidate in \
        "${CONDA_PREFIX:-}" \
        "$RUN_HOME/miniconda3/envs/constellation-collector" \
        "$RUN_HOME/miniforge3/envs/constellation-collector" \
        "$RUN_HOME/anaconda3/envs/constellation-collector" \
        "/opt/miniconda3/envs/constellation-collector" \
        "/usr/local/miniconda3/envs/constellation-collector"; do
        if [[ -n "$candidate" && -x "$candidate/bin/constellation" && "$(basename "$candidate")" == "constellation-collector" ]]; then
            echo "$candidate"
            return
        fi
    done
}

ENV_PREFIX="$(find_env)"
if [[ -z "$ENV_PREFIX" ]]; then
    echo "cannot find the constellation-collector env; run 'make env-collector' or pass --env PREFIX" >&2
    exit 1
fi
if [[ ! -x "$ENV_PREFIX/bin/constellation" ]]; then
    echo "$ENV_PREFIX/bin/constellation not found or not executable" >&2
    exit 1
fi

escape() { printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'; }

render() {  # render <template> <output>
    sed -e "s|@USER@|$(escape "$RUN_USER")|g" \
        -e "s|@HOME@|$(escape "$RUN_HOME")|g" \
        -e "s|@REPO@|$(escape "$REPO")|g" \
        -e "s|@ENV@|$(escape "$ENV_PREFIX")|g" \
        "$1" > "$2"
}

if [[ $UNINSTALL -eq 1 ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "would run: launchctl bootout system/$LABEL_COLLECT and system/$LABEL_COMPACT"
        echo "would remove: $DAEMONS/$LABEL_COLLECT.plist $DAEMONS/$LABEL_COMPACT.plist $NEWSYSLOG_FILE"
        exit 0
    fi
    if [[ $EUID -ne 0 ]]; then echo "uninstall needs root: sudo $0 --uninstall" >&2; exit 1; fi
    for label in "$LABEL_COLLECT" "$LABEL_COMPACT"; do
        launchctl bootout "system/$label" 2>/dev/null || echo "$label was not loaded"
        rm -f "${DAEMONS:?}/${label:?}.plist"
    done
    rm -f "${NEWSYSLOG_FILE:?}"
    echo "services removed; the data in $RUN_HOME/constellation-data is untouched"
    exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK:?}"' EXIT
render "$TEMPLATES/$LABEL_COLLECT.plist.template" "$WORK/$LABEL_COLLECT.plist"
render "$TEMPLATES/$LABEL_COMPACT.plist.template" "$WORK/$LABEL_COMPACT.plist"
render "$TEMPLATES/constellation.newsyslog.conf" "$WORK/constellation.conf"

if grep -n '@[A-Z]*@' "$WORK"/*; then
    echo "unfilled placeholders in the rendered files" >&2
    exit 1
fi

if command -v plutil > /dev/null; then
    plutil -lint "$WORK/$LABEL_COLLECT.plist" "$WORK/$LABEL_COMPACT.plist"
else
    echo "plutil not found; skipping plist validation" >&2
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "user: $RUN_USER  home: $RUN_HOME  repo: $REPO  env: $ENV_PREFIX"
    for file in "$LABEL_COLLECT.plist" "$LABEL_COMPACT.plist" constellation.conf; do
        echo "----- $file"
        cat "$WORK/$file"
    done
    if [[ -n "$RENDER_TO" ]]; then
        mkdir -p "$RENDER_TO"
        cp "$WORK"/* "$RENDER_TO"/
        echo "rendered files copied to $RENDER_TO"
    fi
    echo "-----"
    echo "dry run: nothing installed. A real run would:"
    echo "  create $RUN_HOME/constellation-data/logs (owned by $RUN_USER)"
    echo "  install both plists to $DAEMONS (root:wheel, 644) and the log rule to $NEWSYSLOG_FILE"
    echo "  launchctl bootstrap system $DAEMONS/$LABEL_COLLECT.plist and $LABEL_COMPACT.plist"
    exit 0
fi

if [[ $EUID -ne 0 ]]; then
    echo "installing needs root: sudo $0" >&2
    exit 1
fi

sudo -u "$RUN_USER" mkdir -p "$RUN_HOME/constellation-data/logs"
mkdir -p "$NEWSYSLOG_DIR"
install -m 644 -o root -g wheel "$WORK/constellation.conf" "$NEWSYSLOG_FILE"
for label in "$LABEL_COLLECT" "$LABEL_COMPACT"; do
    launchctl bootout "system/$label" 2>/dev/null || true
    install -m 644 -o root -g wheel "$WORK/$label.plist" "$DAEMONS/$label.plist"
    launchctl bootstrap system "$DAEMONS/$label.plist"
done
launchctl print "system/$LABEL_COLLECT" | head -20
echo "installed. Check with: $ENV_PREFIX/bin/constellation collect status"
