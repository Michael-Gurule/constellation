#!/bin/bash

# CONSTELLATION - Monitor Collection Status

echo "=========================================="
echo "CONSTELLATION Collection Monitor"
echo "=========================================="
echo ""

# Check that the conda environment is activated
if [[ -z "${CONDA_PREFIX:-$VIRTUAL_ENV}" ]]; then
    echo "⚠️  Python environment not activated"
    echo "Run: conda activate constellation"
    exit 1
fi
# Run monitor
python -m constellation.legacy.ingestion.collection_monitor
