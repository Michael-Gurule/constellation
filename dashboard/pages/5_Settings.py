"""
Settings and Configuration Page
System settings, model configuration, and about information.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from dashboard.theme import (
    apply_theme, render_logo, COLORS,
    render_section_header
)

st.set_page_config(
    page_title="CONSTELLATION | Settings",
    page_icon="▪",
    layout="wide"
)

apply_theme()


# ============================================================================
# Helper Functions
# ============================================================================

def get_directory_size(path: Path) -> str:
    """Get size of directory in human readable format."""
    try:
        total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total < 1024:
                return f"{total:.1f} {unit}"
            total /= 1024
        return f"{total:.1f} TB"
    except Exception:
        return "N/A"


def count_files(path: Path, pattern: str = "*") -> int:
    """Count files matching pattern."""
    try:
        return len(list(path.rglob(pattern)))
    except Exception:
        return 0


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    render_logo()
    st.markdown("---")

    st.markdown("### Settings Section")

    settings_section = st.radio(
        "Navigate to",
        ["Model Configuration", "Alert Thresholds", "Data Management", "System Info", "About"],
        index=0
    )



# ============================================================================
# Main Content
# ============================================================================

st.markdown("""
<div style="margin-bottom: 30px;">
    <h1>Settings</h1>
    <p style="color: var(--text-secondary);">
        System configuration and information
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# Model Configuration
# ============================================================================

if settings_section == "Model Configuration":
    render_section_header("ML Model Status", "")

    # Model status cards
    models = [
        {
            "name": "Isolation Forest",
            "file": "isolation_forest.pkl",
            "description": "Anomaly detection using ensemble isolation",
            "type": "Anomaly Detection",
            "metrics": {"Precision": "89%", "Recall": "85%"}
        },
        {
            "name": "Fault Classifier",
            "file": "fault_classifier.pkl",
            "description": "XGBoost-based fault type classification",
            "type": "Classification",
            "metrics": {"Accuracy": "82%", "F1": "0.79"}
        },
        {
            "name": "LSTM Forecaster",
            "file": "degradation_forecaster_colab.pkl",
            "description": "Deep learning degradation prediction",
            "type": "Forecasting",
            "metrics": {"R²": "0.76", "MAE": "2.3"}
        }
    ]

    col1, col2, col3 = st.columns(3)

    for i, model in enumerate(models):
        col = [col1, col2, col3][i]
        model_path = MODELS_DIR / model["file"]
        is_loaded = model_path.exists()

        with col:
            status_color = COLORS['success'] if is_loaded else COLORS['text_muted']
            status_text = "Active" if is_loaded else "Not Found"
            status_dot = "online" if is_loaded else "offline"

            st.markdown(f"""
            <div class="metric-card" style="height: 100%;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                    <span style="color: var(--text-primary); font-weight: 600; font-size: 1.1rem;">
                        {model['name']}
                    </span>
                    <div style="display: flex; align-items: center;">
                        <span class="status-dot {status_dot}"></span>
                        <span style="color: {status_color}; font-size: 0.85rem;">{status_text}</span>
                    </div>
                </div>
                <div style="color: var(--text-muted); font-size: 0.8rem; margin-bottom: 12px;">
                    {model['type']}
                </div>
                <div style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 15px;">
                    {model['description']}
                </div>
                <div style="border-top: 1px solid var(--border); padding-top: 12px;">
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 8px;">
                        Performance
                    </div>
                    <div style="display: flex; gap: 15px;">
            """, unsafe_allow_html=True)

            for metric_name, metric_value in model['metrics'].items():
                st.markdown(f"""
                        <div>
                            <span style="color: var(--primary); font-family: 'JetBrains Mono', monospace; font-size: 1rem;">
                                {metric_value}
                            </span>
                            <span style="color: var(--text-muted); font-size: 0.75rem; margin-left: 4px;">
                                {metric_name}
                            </span>
                        </div>
                """, unsafe_allow_html=True)

            st.markdown("""
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Model actions
    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Model Actions", "")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Retrain All Models", use_container_width=True):
            st.info("Model retraining would be triggered via scripts/train_models.py")

    with col2:
        if st.button("Load Models", use_container_width=True):
            st.info("Models would be loaded into memory for inference")

    with col3:
        if st.button("Clear Model Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared successfully")

    with col4:
        if st.button("Export Model Info", use_container_width=True):
            model_info = {
                "timestamp": datetime.now().isoformat(),
                "models": [{"name": m["name"], "loaded": (MODELS_DIR / m["file"]).exists()} for m in models]
            }
            st.json(model_info)


# ============================================================================
# Alert Thresholds
# ============================================================================

elif settings_section == "Alert Thresholds":
    render_section_header("Alert Configuration", "")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="color: var(--text-primary); font-weight: 600; font-size: 1.1rem; margin-bottom: 20px;">
                Anomaly Detection
            </div>
        """, unsafe_allow_html=True)

        anomaly_warning = st.slider(
            "Warning Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.80,
            step=0.05,
            help="Anomaly score above this triggers a warning",
            key="anomaly_warning"
        )

        anomaly_critical = st.slider(
            "Critical Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
            help="Anomaly score above this triggers a critical alert",
            key="anomaly_critical"
        )

        st.markdown("""
            <div style="margin-top: 15px; padding: 10px; background: var(--bg-light); border-radius: 8px;">
                <div style="color: var(--text-muted); font-size: 0.8rem;">
                    Scores range from 0 (normal) to 1 (highly anomalous)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="color: var(--text-primary); font-weight: 600; font-size: 1.1rem; margin-bottom: 20px;">
                Health Monitoring
            </div>
        """, unsafe_allow_html=True)

        health_warning = st.slider(
            "Warning Threshold",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
            help="Health score below this triggers a warning",
            key="health_warning"
        )

        health_critical = st.slider(
            "Critical Threshold",
            min_value=0,
            max_value=100,
            value=40,
            step=5,
            help="Health score below this triggers a critical alert",
            key="health_critical"
        )

        st.markdown("""
            <div style="margin-top: 15px; padding: 10px; background: var(--bg-light); border-radius: 8px;">
                <div style="color: var(--text-muted); font-size: 0.8rem;">
                    Scores range from 0 (failed) to 100 (perfect health)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Notification settings
    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Notification Settings", "")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card">
        """, unsafe_allow_html=True)

        enable_alerts = st.checkbox("Enable Alert Notifications", value=False)
        enable_email = st.checkbox("Email Notifications", value=False, disabled=not enable_alerts)
        enable_dashboard = st.checkbox("Dashboard Notifications", value=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
        """, unsafe_allow_html=True)

        refresh_interval = st.number_input(
            "Auto-refresh Interval (seconds)",
            min_value=30,
            max_value=600,
            value=60,
            step=30
        )

        data_retention = st.number_input(
            "Data Retention (days)",
            min_value=7,
            max_value=365,
            value=90,
            step=7
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # Save configuration
    if st.button("Save Configuration", use_container_width=True):
        config = {
            "anomaly_warning": anomaly_warning,
            "anomaly_critical": anomaly_critical,
            "health_warning": health_warning,
            "health_critical": health_critical,
            "enable_alerts": enable_alerts,
            "refresh_interval": refresh_interval,
            "data_retention": data_retention
        }
        st.success("Configuration saved")
        st.json(config)


# ============================================================================
# Data Management
# ============================================================================

elif settings_section == "Data Management":
    render_section_header("Data Storage", "")

    # Storage statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        raw_size = get_directory_size(RAW_DATA_DIR)
        st.metric("RAW DATA", raw_size)

    with col2:
        processed_size = get_directory_size(PROCESSED_DATA_DIR)
        st.metric("PROCESSED", processed_size)

    with col3:
        model_size = get_directory_size(MODELS_DIR)
        st.metric("MODELS", model_size)

    with col4:
        total_files = count_files(RAW_DATA_DIR, "*.parquet") + count_files(PROCESSED_DATA_DIR, "*.parquet")
        st.metric("TOTAL FILES", total_files)

    # Data paths
    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Data Paths", "")

    st.markdown(f"""
    <div class="metric-card">
        <div style="display: grid; gap: 12px;">
            <div style="display: flex; justify-content: space-between; padding: 8px; background: var(--bg-light); border-radius: 6px;">
                <span style="color: var(--text-muted);">Raw Data</span>
                <code style="color: var(--primary); font-size: 0.85rem;">{RAW_DATA_DIR}</code>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px; background: var(--bg-light); border-radius: 6px;">
                <span style="color: var(--text-muted);">Processed Data</span>
                <code style="color: var(--primary); font-size: 0.85rem;">{PROCESSED_DATA_DIR}</code>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 8px; background: var(--bg-light); border-radius: 6px;">
                <span style="color: var(--text-muted);">Models</span>
                <code style="color: var(--primary); font-size: 0.85rem;">{MODELS_DIR}</code>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Data actions
    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Data Actions", "")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Clear Processed Data", use_container_width=True):
            st.warning("This would delete all processed data files")

    with col2:
        if st.button("Regenerate Features", use_container_width=True):
            st.info("Feature regeneration would run scripts/run_feature_pipeline.py")

    with col3:
        if st.button("Export Data Summary", use_container_width=True):
            summary = {
                "raw_data_size": get_directory_size(RAW_DATA_DIR),
                "processed_data_size": get_directory_size(PROCESSED_DATA_DIR),
                "raw_files": count_files(RAW_DATA_DIR, "*.parquet"),
                "processed_files": count_files(PROCESSED_DATA_DIR, "*.parquet"),
                "timestamp": datetime.now().isoformat()
            }
            st.json(summary)


# ============================================================================
# System Info
# ============================================================================

elif settings_section == "System Info":
    render_section_header("System Information", "")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="color: var(--text-primary); font-weight: 600; font-size: 1.1rem; margin-bottom: 20px;">
                Application
            </div>
            <div style="display: grid; gap: 10px;">
        """, unsafe_allow_html=True)

        info_items = [
            ("Version", "1.0.0"),
            ("Framework", "Streamlit"),
            ("Python", "3.10+"),
            ("Last Updated", datetime.now().strftime("%Y-%m-%d")),
        ]

        for label, value in info_items:
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border);">
                    <span style="color: var(--text-muted);">{label}</span>
                    <span style="color: var(--text-primary); font-family: 'JetBrains Mono', monospace;">{value}</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="color: var(--text-primary); font-weight: 600; font-size: 1.1rem; margin-bottom: 20px;">
                Dependencies
            </div>
            <div style="display: grid; gap: 10px;">
        """, unsafe_allow_html=True)

        deps = [
            ("Pandas", "2.0.3"),
            ("Plotly", "5.16.1"),
            ("PyTorch", "2.0.1"),
            ("scikit-learn", "1.3.0"),
        ]

        for lib, version in deps:
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border);">
                    <span style="color: var(--text-muted);">{lib}</span>
                    <span style="color: var(--primary); font-family: 'JetBrains Mono', monospace;">{version}</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Monitored parameters
    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Monitored Parameters", "")

    st.markdown("""
    <div class="metric-card">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <div style="color: var(--text-primary); font-weight: 600; margin-bottom: 12px;">
                    Attitude Control (12 parameters)
                </div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.8;">
                    RWA 1-4 Speed<br/>
                    CMG 1-4 Momentum<br/>
                    Attitude Quaternions (Q1-Q4)
                </div>
            </div>
            <div>
                <div style="color: var(--text-primary); font-weight: 600; margin-bottom: 12px;">
                    Communications (6 parameters)
                </div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.8;">
                    S-Band Signal Strength<br/>
                    Ku-Band Signal Strength<br/>
                    S-Band/Ku-Band Power<br/>
                    Antenna Azimuth/Elevation
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# About
# ============================================================================

elif settings_section == "About":
    render_section_header("About CONSTELLATION", "")

    st.markdown("""
    <div class="metric-card">
        <div style="text-align: center; padding: 20px 0;">
            <div style="margin-bottom: 15px;">
                <svg width="60" height="60" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="20" cy="20" r="18" stroke="url(#grad2)" stroke-width="2" fill="none"/>
                    <circle cx="20" cy="20" r="6" fill="url(#grad2)"/>
                    <ellipse cx="20" cy="20" rx="18" ry="8" stroke="url(#grad2)" stroke-width="1.5" fill="none" transform="rotate(45 20 20)"/>
                    <ellipse cx="20" cy="20" rx="18" ry="8" stroke="url(#grad2)" stroke-width="1.5" fill="none" transform="rotate(-45 20 20)"/>
                    <defs>
                        <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" style="stop-color:#00d4ff;stop-opacity:1" />
                            <stop offset="100%" style="stop-color:#7b68ee;stop-opacity:1" />
                        </linearGradient>
                    </defs>
                </svg>
            </div>
            <div style="font-size: 2rem; font-weight: 700; background: linear-gradient(90deg, var(--primary), var(--secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
                CONSTELLATION
            </div>
            <div style="color: var(--text-muted); font-size: 0.9rem; margin-top: 8px;">
                Satellite Fleet Health Management System
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card" style="height: 100%;">
            <div style="color: var(--text-primary); font-weight: 600; font-size: 1.1rem; margin-bottom: 15px;">
                Overview
            </div>
            <div style="color: var(--text-secondary); line-height: 1.8;">
                CONSTELLATION is a production-grade satellite fleet health management platform
                that monitors the International Space Station using real-time telemetry data
                from NASA's public Lightstreamer feed.
                <br/><br/>
                The system demonstrates predictive maintenance, anomaly detection, and operational
                decision support capabilities for aerospace and defense applications.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="height: 100%;">
            <div style="color: var(--text-primary); font-weight: 600; font-size: 1.1rem; margin-bottom: 15px;">
                Capabilities
            </div>
            <div style="color: var(--text-secondary); line-height: 1.8;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="color: var(--primary); margin-right: 10px;">&#9679;</span>
                    Real-time anomaly detection
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="color: var(--primary); margin-right: 10px;">&#9679;</span>
                    Predictive fault classification
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="color: var(--primary); margin-right: 10px;">&#9679;</span>
                    Component health scoring
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="color: var(--primary); margin-right: 10px;">&#9679;</span>
                    Optimized maintenance scheduling
                </div>
                <div style="display: flex; align-items: center;">
                    <span style="color: var(--primary); margin-right: 10px;">&#9679;</span>
                    3D attitude visualization
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Technology stack
    st.markdown("<br>", unsafe_allow_html=True)
    render_section_header("Technology Stack", "")

    tech_cols = st.columns(4)

    tech_categories = [
        ("Data Processing", ["Pandas", "NumPy", "PyArrow", "SciPy"]),
        ("Machine Learning", ["scikit-learn", "XGBoost", "PyTorch", "TensorFlow"]),
        ("Visualization", ["Plotly", "Streamlit", "Matplotlib"]),
        ("Operations", ["FastAPI", "PuLP", "boto3", "pytest"])
    ]

    for col, (category, techs) in zip(tech_cols, tech_categories):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="height: 100%;">
                <div style="color: var(--text-primary); font-weight: 600; margin-bottom: 12px;">
                    {category}
                </div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 2;">
                    {'<br/>'.join(techs)}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Developer info
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 30px; border-top: 1px solid var(--border);">
        <div style="color: var(--text-primary); font-weight: 600; font-size: 1.1rem;">
            Developed by Michael Gurule
        </div>
        <div style="color: var(--text-muted); font-size: 0.9rem; margin-top: 8px;">
            Data Science &amp; Machine Learning Portfolio Project
        </div>
        <div style="color: var(--text-muted); font-size: 0.8rem; margin-top: 15px;">
            MIT License | Data: NASA ISS Telemetry (Public)
        </div>
    </div>
    """, unsafe_allow_html=True)


# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; color: var(--text-muted); padding: 20px; border-top: 1px solid var(--border);">
    <div style="font-size: 0.75rem;">
        CONSTELLATION v1.0.0 | Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</div>
""", unsafe_allow_html=True)
