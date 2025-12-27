#!/bin/bash

# CONSTELLATION - Start Telemetry Collection
# This script starts the telemetry collection process

echo "=========================================="
echo "CONSTELLATION Telemetry Collection"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "Run: source venv/bin/activate"
    exit 1
fi

echo "✓ Virtual environment active"
echo ""

# Create logs directory
mkdir -p logs

# Start collection with logging
echo "Starting telemetry collection..."
echo "Press Ctrl+C to stop"
echo ""

python -m src.ingestion.collect_telemetry 2>&1 | tee logs/collection_$(date +%Y%m%d_%H%M%S).log