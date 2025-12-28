# CONSTELLATION

**Continuous Operations Network for Satellite Telemetry Evaluation, Life-cycle Analysis, Tracking, Intelligence, Operations, and Notification**

A production-grade satellite fleet health management platform using real ISS telemetry to demonstrate predictive maintenance, anomaly detection, and operational decision support capabilities for aerospace and defense applications.


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

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

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  NASA Lightstreamer → Lambda (Real-time) → DynamoDB             │
│  Historical Archive → S3 Data Lake                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│  • Time series windowing                                        │
│  • Statistical feature extraction                               │
│  • Subsystem correlation analysis                               │
│  • Degradation rate calculation                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      ML MODEL SUITE                             │
├─────────────────────────────────────────────────────────────────┤
│  Anomaly Detection    → Isolation Forest / Autoencoder          │
│  Degradation Forecast → LSTM / Temporal Fusion Transformer      │
│  Survival Analysis    → Cox Proportional Hazards / Weibull      │
│  Fault Classification → Random Forest / XGBoost                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   OPERATIONAL LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Health Scoring → Maintenance Scheduling → Alert Generation     │
│  Dashboard (Streamlit) → CloudWatch Monitoring                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

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


## License

MIT

## Author

**Michael Gurule**  
Data Scientist | ML Engineer  

- [![Email Me](https://img.shields.io/badge/EMAIL-8A2BE2)](michaelgurule1164@gmail.com)
- [![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff)](www.linkedin.com/in/michael-j-gurule-447aa2134)
- [![Medium](https://img.shields.io/badge/Medium-%23000000.svg?logo=medium&logoColor=white)](https://medium.com/@michaelgurule1164)