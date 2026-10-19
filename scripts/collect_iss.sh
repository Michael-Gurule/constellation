#!/bin/bash
# Run the ISS collector for the long term: keeps the machine awake, rotates the
# log and restarts the collector if it exits unexpectedly.
#
#   scripts/collect_iss.sh [extra arguments for `constellation collect iss`]
#
# Stop it with Ctrl+C (the collector flushes its buffer before exiting).

set -u

LOG_DIR="${CONSTELLATION_LOG_DIR:-logs}"
LOG_FILE="$LOG_DIR/collect_iss.log"
MAX_BYTES=$((10 * 1024 * 1024))
KEEP=5

if [[ -z "${CONDA_PREFIX:-${VIRTUAL_ENV:-}}" ]]; then
    echo "Python environment not activated. Run: conda activate constellation"
    exit 1
fi

mkdir -p "$LOG_DIR"

rotate() {
    [[ -f "$LOG_FILE" ]] || return 0
    local size
    size=$(wc -c < "$LOG_FILE")
    (( size > MAX_BYTES )) || return 0
    for ((i = KEEP - 1; i >= 1; i--)); do
        [[ -f "$LOG_FILE.$i" ]] && mv "$LOG_FILE.$i" "$LOG_FILE.$((i + 1))"
    done
    mv "$LOG_FILE" "$LOG_FILE.1"
}

if command -v caffeinate > /dev/null; then
    KEEP_AWAKE=(caffeinate -i)
else
    KEEP_AWAKE=()
fi

while true; do
    rotate
    "${KEEP_AWAKE[@]}" constellation collect iss "$@" 2>&1 | tee -a "$LOG_FILE"
    status=${PIPESTATUS[0]}
    # 0: finished, 130/143: stopped with SIGINT/SIGTERM
    if [[ $status -eq 0 || $status -eq 130 || $status -eq 143 ]]; then
        exit "$status"
    fi
    echo "collector exited with status $status; restarting in 15 s" | tee -a "$LOG_FILE"
    sleep 15
done
