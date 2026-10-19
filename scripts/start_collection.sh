#!/bin/bash

# CONSTELLATION - Start Telemetry Collection
# This script starts the telemetry collection process

echo "=========================================="
echo "CONSTELLATION Telemetry Collection"
echo "=========================================="
echo ""

# Check that the conda environment is activated
if [[ -z "${CONDA_PREFIX:-$VIRTUAL_ENV}" ]]; then
    echo "⚠️  Python environment not activated"
    echo "Run: conda activate constellation"
    exit 1
fi

echo "✓ Python environment active"
echo ""

# Create logs directory
mkdir -p logs

# Start collection with logging
echo "Starting telemetry collection..."
echo "Press Ctrl+C to stop"
echo ""

python -m constellation.legacy.ingestion.collect_telemetry 2>&1 | tee logs/collection_$(date +%Y%m%d_%H%M%S).log
