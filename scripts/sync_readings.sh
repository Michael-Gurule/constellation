#!/bin/bash
# Pull ISS readings from the collector machine into this project's store.
#
#   scripts/sync_readings.sh <host> [--remote-dir '~/constellation-data'] [--dry-run]
#   scripts/sync_readings.sh /path/to/copy/of/constellation-data     (a local copy, no ssh)
#
# Only files that are not here yet are copied, hidden temporary files are skipped, and
# nothing is ever deleted on the remote side. Afterwards the local store is compacted,
# so files copied before and after the remote compaction become one deduplicated file
# per day, and the store's status is shown.

set -euo pipefail

if [[ $# -lt 1 || "$1" == -* ]]; then
    sed -n '2,10p' "$0"
    exit 2
fi
SOURCE="$1"
shift
REMOTE_DIR="~/constellation-data"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote-dir) REMOTE_DIR="${2:?--remote-dir needs a path}"; shift ;;
        --dry-run) DRY_RUN=1 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

DATA_DIR="${CONSTELLATION_DATA_DIR:-data}"
DEST="$DATA_DIR/readings/"
mkdir -p "$DEST"

if [[ "$SOURCE" == /* || "$SOURCE" == .* ]]; then
    ORIGIN="${SOURCE%/}/readings/"   # a local directory
else
    ORIGIN="$SOURCE:$REMOTE_DIR/readings/"
fi

OPTIONS=(-a --ignore-existing --exclude '.*.tmp')
if [[ $DRY_RUN -eq 1 ]]; then
    OPTIONS+=(--dry-run -v)
fi
echo "syncing $ORIGIN -> $DEST"
rsync "${OPTIONS[@]}" "$ORIGIN" "$DEST"

if [[ $DRY_RUN -eq 1 ]]; then
    constellation store compact --dry-run
else
    constellation store compact
fi
constellation collect status
