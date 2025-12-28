#!/bin/bash
# Launch CONSTELLATION API

echo "Starting CONSTELLATION API..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload