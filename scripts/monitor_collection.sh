#!/bin/bash

# CONSTELLATION - Monitor Collection Status

echo "=========================================="
echo "CONSTELLATION Collection Monitor"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "Run: source venv/bin/activate"
    exit 1
fi

# Run monitor
python -m src.ingestion.collection_monitorch