# CONSTELLATION API

REST API for satellite health monitoring system.

## Quick Start
```bash
# From project root
python -m uvicorn api.main:app --reload
```

Or use the launch script:
```bash
./api/run.sh
```

API will be available at: http://localhost:8000

Interactive docs: http://localhost:8000/docs

## Endpoints

### Health Check
- `GET /` - Root health check
- `GET /health` - Detailed health status

### Analysis
- `POST /api/v1/detect-anomalies` - Run anomaly detection
- `POST /api/v1/run-analysis` - Run complete system analysis

### Data Retrieval
- `GET /api/v1/health-scores` - Get component health scores
- `GET /api/v1/maintenance-schedule` - Get maintenance schedule

### Model Management
- `POST /api/v1/models/load` - Load models into memory
- `GET /api/v1/models/status` - Check model status

## Example Usage

### Python
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Run anomaly detection
response = requests.post(
    "http://localhost:8000/api/v1/detect-anomalies",
    json={"subsystem": "attitude_control", "sample_size": 10000}
)
print(response.json())
```

### cURL
```bash
# Health check
curl http://localhost:8000/health

# Get health scores
curl http://localhost:8000/api/v1/health-scores

# Run analysis
curl -X POST http://localhost:8000/api/v1/run-analysis \
  -H "Content-Type: application/json" \
  -d '{"subsystem": "attitude_control", "export_results": true}'
```

