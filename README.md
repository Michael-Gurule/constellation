# CONSTELLATION

**Continuous Operations Network for Satellite Telemetry Evaluation, Life-cycle Analysis, Tracking, Intelligence, Operations, and Notification**

A production-grade satellite fleet health management platform using real ISS telemetry to demonstrate predictive maintenance, anomaly detection, and operational decision support capabilities.

## Overview

CONSTELLATION monitors the International Space Station's attitude control and communications subsystems using real-time telemetry from NASA's public Lightstreamer feed. The system provides:

- Real-time anomaly detection
- Degradation forecasting
- Survival analysis for component failures
- Fault diagnosis and classification
- Optimized maintenance scheduling
- Operations-style monitoring dashboard

## Architecture

- **Data Ingestion**: NASA Lightstreamer API → Validation → Local/S3 Storage
- **Feature Engineering**: Time series features + domain-specific aerospace calculations
- **ML Models**: Isolation Forest, LSTM Autoencoder, Temporal Fusion Transformer, Cox Proportional Hazards, XGBoost
- **Deployment**: AWS Lambda, SageMaker, DynamoDB, CloudWatch
- **Dashboard**: Streamlit with Plotly visualizations

## Quick Start

### Prerequisites

- Python 3.10+
- AWS Account (for cloud deployment)
- Git

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd constellation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Collect Telemetry Data
```bash
# Start telemetry collection
python -m src.ingestion.collect_telemetry
```

Data will be saved to `data/raw/` in date-partitioned Parquet files.

## Project Status

**Phase 1: Foundation** ✅ (In Progress)
- [x] Project structure
- [x] Data ingestion pipeline
- [ ] Historical data collection (30+ days)

**Phase 2: Feature Engineering** 

**Phase 3: Core ML Models** 

**Phase 4: Advanced Analytics** 

**Phase 5: Production Deployment** 

## Project Structure
```
constellation/
├── config/           # Configuration files
├── data/            # Data storage (raw, processed, models)
├── src/             # Source code
│   ├── ingestion/   # Data collection
│   ├── features/    # Feature engineering
│   ├── models/      # ML models
│   └── utils/       # Utilities
├── notebooks/       # Jupyter notebooks
├── dashboard/       # Streamlit dashboard
├── aws/            # AWS deployment
└── tests/          # Unit tests
```

## License

MIT

## Author

Michael Gurule