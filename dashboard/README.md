# CONSTELLATION Dashboard

Interactive web interface for satellite health monitoring.

## Quick Start
```bash
# From project root
cd dashboard
streamlit run app.py
```

Or use the launch script:
```bash
./dashboard/run.sh
```

## Pages

1. **Overview** - System status and quick metrics
2. **Telemetry** - Real-time telemetry visualization
3. **Anomalies** - Anomaly detection results
4. **Health** - Component health scores
5. **Maintenance** - Maintenance scheduling
6. **Settings** - Configuration and system info

## Requirements

- Streamlit >= 1.28.0
- Plotly >= 5.17.0
- Analysis results in `data/processed/analysis_results/`

## Usage

Run system analysis first to generate data:
```bash
python scripts/run_system_analysis.py --subsystem attitude_control --export
```

Then launch dashboard:
```bash
streamlit run dashboard/app.py
```

Dashboard will be available at: http://localhost:8501