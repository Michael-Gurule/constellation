"""
Settings and Configuration Page
System settings and model configuration.
"""

import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import MODELS_DIR, PROCESSED_DATA_DIR

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

st.title("Settings and Configuration")
st.markdown("System settings and model parameters")

# Model status
st.markdown("### Model Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Isolation Forest")
    
    if (MODELS_DIR / "isolation_forest.pkl").exists():
        st.success("Model loaded")
        st.caption("Status: Active")
    else:
        st.error("Model not found")
        st.caption("Status: Inactive")

with col2:
    st.markdown("#### XGBoost Classifier")
    
    if (MODELS_DIR / "fault_classifier.pkl").exists():
        st.success("Model loaded")
        st.caption("Status: Active")
    else:
        st.error("Model not found")
        st.caption("Status: Inactive")

with col3:
    st.markdown("#### LSTM Forecaster")
    
    if (MODELS_DIR / "degradation_forecaster_colab.pkl").exists():
        st.success("Model loaded")
        st.caption("Status: Active")
    else:
        st.error("Model not found")
        st.caption("Status: Inactive")

st.markdown("---")

# Alert thresholds
st.markdown("### Alert Thresholds")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Anomaly Detection")
    
    anomaly_warning = st.slider(
        "Warning Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.80,
        step=0.05,
        help="Anomaly score threshold for warning alerts"
    )
    
    anomaly_critical = st.slider(
        "Critical Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.05,
        help="Anomaly score threshold for critical alerts"
    )

with col2:
    st.markdown("#### Health Monitoring")
    
    health_warning = st.slider(
        "Warning Threshold",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        help="Health score threshold for warning"
    )
    
    health_critical = st.slider(
        "Critical Threshold",
        min_value=0,
        max_value=100,
        value=40,
        step=5,
        help="Health score threshold for critical alerts"
    )

st.markdown("---")

# Data refresh settings
st.markdown("### Data Refresh Settings")

col1, col2 = st.columns(2)

with col1:
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
    
    if auto_refresh:
        refresh_interval = st.number_input(
            "Refresh Interval (seconds)",
            min_value=30,
            max_value=600,
            value=60,
            step=30
        )

with col2:
    data_retention = st.number_input(
        "Data Retention (days)",
        min_value=7,
        max_value=365,
        value=30,
        step=7,
        help="How long to keep historical data"
    )

st.markdown("---")

# System information
st.markdown("### System Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Paths")
    st.code(f"Models: {MODELS_DIR}")
    st.code(f"Data: {PROCESSED_DATA_DIR}")

with col2:
    st.markdown("#### Storage")
    
    # Count files
    model_count = len(list(MODELS_DIR.glob("*.pkl"))) if MODELS_DIR.exists() else 0
    data_count = len(list(PROCESSED_DATA_DIR.glob("*.parquet"))) if PROCESSED_DATA_DIR.exists() else 0
    
    st.metric("Model Files", model_count)
    st.metric("Data Files", data_count)

st.markdown("---")

# Actions
st.markdown("### System Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Retrain Models", use_container_width=True):
        st.info("Model retraining would be triggered here")

with col2:
    if st.button("Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared")

with col3:
    if st.button("Export Configuration", use_container_width=True):
        config = {
            'anomaly_warning': anomaly_warning,
            'anomaly_critical': anomaly_critical,
            'health_warning': health_warning,
            'health_critical': health_critical,
            'auto_refresh': auto_refresh,
            'data_retention': data_retention
        }
        st.json(config)

st.markdown("---")

# About
st.markdown("### About CONSTELLATION")

st.markdown("""
**CONSTELLATION** is an advanced satellite fleet health management system.

**Features:**
- Real-time anomaly detection
- Predictive fault classification
- Health score monitoring
- Optimized maintenance scheduling

**Models:**
- Isolation Forest for anomaly detection
- XGBoost for fault classification
- LSTM for degradation forecasting

**Developer:** Michael Gurule

**Version:** 1.0.0
""")
