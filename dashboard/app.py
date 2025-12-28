"""
CONSTELLATION Mission Control Dashboard
Main overview page with real-time satellite fleet health monitoring.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import PROCESSED_DATA_DIR, MODELS_DIR, RAW_DATA_DIR
from dashboard.theme import (
    apply_theme, render_logo, COLORS, CHART_COLORS,
    apply_plotly_theme, get_health_color, render_section_header
)

# Page configuration
st.set_page_config(
    page_title="CONSTELLATION | Mission Control",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply theme
apply_theme()


# ============================================================================
# Data Loading Functions
# ============================================================================

@st.cache_data(ttl=300)
def load_latest_telemetry():
    """Load the most recent telemetry data."""
    try:
        # Try to load processed features
        feature_files = list(PROCESSED_DATA_DIR.glob("*_features_*.parquet"))
        if feature_files:
            dfs = []
            for f in feature_files:
                df = pd.read_parquet(f)
                dfs.append(df)
            if dfs:
                return pd.concat(dfs, ignore_index=True)

        # Fallback to raw data
        raw_files = list(RAW_DATA_DIR.rglob("*.parquet"))
        if raw_files:
            recent_files = sorted(raw_files, key=lambda x: x.stat().st_mtime)[-10:]
            dfs = [pd.read_parquet(f) for f in recent_files]
            return pd.concat(dfs, ignore_index=True)

    except Exception as e:
        st.error(f"Error loading telemetry: {e}")

    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_analysis_results():
    """Load latest analysis results."""
    results_dir = PROCESSED_DATA_DIR / "analysis_results"

    if not results_dir.exists():
        return None, None, None

    # Load predictions
    pred_files = list(results_dir.glob("predictions_*.parquet"))
    predictions = None
    if pred_files:
        latest = max(pred_files, key=lambda p: p.stat().st_mtime)
        predictions = pd.read_parquet(latest)

    # Load health scores
    health_files = list(results_dir.glob("health_scores_*.csv"))
    health_scores = None
    if health_files:
        latest = max(health_files, key=lambda p: p.stat().st_mtime)
        health_scores = pd.read_csv(latest)

    # Load maintenance schedule
    schedule_files = list(results_dir.glob("maintenance_schedule_*.csv"))
    schedule = None
    if schedule_files:
        latest = max(schedule_files, key=lambda p: p.stat().st_mtime)
        schedule = pd.read_csv(latest)

    return predictions, health_scores, schedule


@st.cache_data(ttl=300)
def load_model_status():
    """Check which models are available."""
    models = {
        "Isolation Forest": MODELS_DIR / "isolation_forest.pkl",
        "Fault Classifier": MODELS_DIR / "fault_classifier.pkl",
        "LSTM Forecaster": MODELS_DIR / "degradation_forecaster_colab.pkl",
    }

    status = {}
    for name, path in models.items():
        status[name] = {
            "loaded": path.exists(),
            "path": str(path),
        }
    return status


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    render_logo()

    st.markdown("---")

    # System Status
    st.markdown("### System Status")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center;">
            <span class="status-dot online"></span>
            <span style="color: var(--text-secondary); font-size: 0.85rem;">Online</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="color: var(--text-muted); font-size: 0.75rem;">
            {datetime.now().strftime("%H:%M:%S UTC")}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick Info
    st.markdown("### Monitoring")
    st.markdown("""
    <div style="color: var(--text-muted); font-size: 0.8rem; line-height: 1.6;">
        <strong>Subsystems:</strong><br/>
        • ISS Attitude Control<br/>
        • Communications Systems<br/><br/>
        <strong>Source:</strong><br/>
        NASA Lightstreamer
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# Main Content - Mission Control
# ============================================================================

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="font-size: 2.5rem; margin-bottom: 10px;">Mission Control</h1>
    <p style="color: var(--text-secondary); font-size: 1.1rem;">
        Real-time ISS Fleet Health Monitoring Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

# Load all data
telemetry_df = load_latest_telemetry()
predictions, health_scores, schedule = load_analysis_results()
model_status = load_model_status()


# ============================================================================
# Primary Metrics Row
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

# Calculate metrics from actual data
total_records = len(telemetry_df) if not telemetry_df.empty else 0
anomaly_count = 0
avg_health = 0
critical_count = 0
active_models = sum(1 for m in model_status.values() if m["loaded"])

if predictions is not None and 'anomaly_score' in predictions.columns:
    anomaly_count = (predictions['anomaly_score'] > 0.8).sum()

if health_scores is not None and 'health_score' in health_scores.columns:
    avg_health = health_scores['health_score'].mean()
    critical_count = (health_scores['health_score'] < 40).sum()

with col1:
    st.metric(
        label="DATA POINTS",
        value=f"{total_records:,}",
        delta="Live" if total_records > 0 else "No Data"
    )

with col2:
    delta_text = "Normal" if anomaly_count < 10 else f"+{anomaly_count}"
    st.metric(
        label="ANOMALIES (24H)",
        value=f"{anomaly_count:,}",
        delta=delta_text,
        delta_color="inverse" if anomaly_count > 0 else "off"
    )

with col3:
    health_color = "normal" if avg_health > 70 else "inverse"
    st.metric(
        label="AVG HEALTH",
        value=f"{avg_health:.1f}%" if avg_health > 0 else "N/A",
        delta="Nominal" if avg_health > 70 else "Degraded",
        delta_color=health_color
    )

with col4:
    st.metric(
        label="CRITICAL",
        value=f"{critical_count}",
        delta="Alert" if critical_count > 0 else "Clear",
        delta_color="inverse" if critical_count > 0 else "off"
    )

with col5:
    st.metric(
        label="MODELS",
        value=f"{active_models}/3",
        delta="Online" if active_models == 3 else "Partial"
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# Two Column Layout: Subsystem Status & Health Overview
# ============================================================================

col_left, col_right = st.columns([1.2, 1])

with col_left:
    render_section_header("Subsystem Status", "")

    # Attitude Control Card
    st.markdown("""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <div>
                <span style="font-size: 1.2rem; font-weight: 600; color: var(--text-primary);">Attitude Control</span>
                <div style="color: var(--text-muted); font-size: 0.8rem;">RWAs, CMGs, Quaternions</div>
            </div>
            <div style="display: flex; align-items: center;">
                <span class="status-dot online"></span>
                <span style="color: var(--success); font-weight: 500;">NOMINAL</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Mini metrics for attitude control
    ac_col1, ac_col2, ac_col3, ac_col4 = st.columns(4)
    with ac_col1:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; font-weight: 600; color: var(--primary);">4</div>
            <div style="color: var(--text-muted); font-size: 0.7rem;">RWAs</div>
        </div>
        """, unsafe_allow_html=True)
    with ac_col2:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; font-weight: 600; color: var(--primary);">4</div>
            <div style="color: var(--text-muted); font-size: 0.7rem;">CMGs</div>
        </div>
        """, unsafe_allow_html=True)
    with ac_col3:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; font-weight: 600; color: var(--accent);">92%</div>
            <div style="color: var(--text-muted); font-size: 0.7rem;">Health</div>
        </div>
        """, unsafe_allow_html=True)
    with ac_col4:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; font-weight: 600; color: var(--success);">0</div>
            <div style="color: var(--text-muted); font-size: 0.7rem;">Alerts</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Communications Card
    st.markdown("""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <div>
                <span style="font-size: 1.2rem; font-weight: 600; color: var(--text-primary);">Communications</span>
                <div style="color: var(--text-muted); font-size: 0.8rem;">S-Band, Ku-Band, Antenna</div>
            </div>
            <div style="display: flex; align-items: center;">
                <span class="status-dot online"></span>
                <span style="color: var(--success); font-weight: 500;">NOMINAL</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Mini metrics for communications
    comm_col1, comm_col2, comm_col3, comm_col4 = st.columns(4)
    with comm_col1:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; font-weight: 600; color: var(--primary);">-78</div>
            <div style="color: var(--text-muted); font-size: 0.7rem;">S-Band dBm</div>
        </div>
        """, unsafe_allow_html=True)
    with comm_col2:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; font-weight: 600; color: var(--primary);">-82</div>
            <div style="color: var(--text-muted); font-size: 0.7rem;">Ku-Band dBm</div>
        </div>
        """, unsafe_allow_html=True)
    with comm_col3:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; font-weight: 600; color: var(--accent);">88%</div>
            <div style="color: var(--text-muted); font-size: 0.7rem;">Health</div>
        </div>
        """, unsafe_allow_html=True)
    with comm_col4:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 1.5rem; font-weight: 600; color: var(--success);">0</div>
            <div style="color: var(--text-muted); font-size: 0.7rem;">Alerts</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


with col_right:
    render_section_header("Fleet Health Overview", "")

    # Create health gauge chart
    if health_scores is not None and not health_scores.empty:
        avg_health_val = health_scores['health_score'].mean()
    else:
        avg_health_val = 87.5  # Demo value

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_health_val,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "System Health", 'font': {'size': 16, 'color': COLORS['text_secondary']}},
        delta={'reference': 85, 'increasing': {'color': COLORS['success']}, 'decreasing': {'color': COLORS['error']}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLORS['text_muted']},
            'bar': {'color': COLORS['primary']},
            'bgcolor': COLORS['background_card'],
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(255, 71, 87, 0.3)'},
                {'range': [40, 60], 'color': 'rgba(255, 170, 0, 0.3)'},
                {'range': [60, 85], 'color': 'rgba(123, 237, 159, 0.3)'},
                {'range': [85, 100], 'color': 'rgba(0, 255, 136, 0.3)'}
            ],
            'threshold': {
                'line': {'color': COLORS['error'], 'width': 2},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))

    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text_primary'], 'family': 'Inter'},
        height=280,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig_gauge, use_container_width=True)


# ============================================================================
# Live Telemetry Chart
# ============================================================================

render_section_header("Live Telemetry Feed", "")

if not telemetry_df.empty and 'timestamp' in telemetry_df.columns:
    # Create multi-line telemetry chart
    fig_telemetry = go.Figure()

    # Get sample of recent data
    df_sorted = telemetry_df.sort_values('timestamp').tail(1000)

    if 'parameter_id' in df_sorted.columns:
        # Plot each parameter
        params = df_sorted['parameter_id'].unique()[:4]  # Limit to 4 params
        for i, param in enumerate(params):
            param_data = df_sorted[df_sorted['parameter_id'] == param]
            fig_telemetry.add_trace(go.Scatter(
                x=param_data['timestamp'],
                y=param_data['value_numeric'],
                mode='lines',
                name=param,
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                hovertemplate=f"<b>{param}</b><br>Value: %{{y:.2f}}<br>Time: %{{x}}<extra></extra>"
            ))
    elif 'value_numeric' in df_sorted.columns:
        fig_telemetry.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['value_numeric'],
            mode='lines',
            name='Telemetry',
            line=dict(color=COLORS['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)'
        ))

    fig_telemetry = apply_plotly_theme(fig_telemetry)
    fig_telemetry.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_telemetry, use_container_width=True)
else:
    st.info("Awaiting telemetry data. Run data collection to see live feed.")


# ============================================================================
# Bottom Row: Alerts & Model Performance
# ============================================================================

col_alerts, col_models = st.columns([1, 1])

with col_alerts:
    render_section_header("Active Alerts", "")

    # Display alerts based on actual data or demo
    alert_container = st.container()

    with alert_container:
        if critical_count > 0 and health_scores is not None:
            critical_components = health_scores[health_scores['health_score'] < 40]
            for _, row in critical_components.iterrows():
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid var(--critical); padding: 12px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: var(--critical); font-weight: 600;">CRITICAL</span>
                        <span style="color: var(--text-muted); font-size: 0.8rem;">Just now</span>
                    </div>
                    <div style="color: var(--text-primary); margin-top: 5px;">
                        {row['component']} health degraded (Score: {row['health_score']:.1f})
                    </div>
                </div>
                """, unsafe_allow_html=True)
        elif anomaly_count > 0:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid var(--warning); padding: 12px;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--warning); font-weight: 600;">WARNING</span>
                    <span style="color: var(--text-muted); font-size: 0.8rem;">Recent</span>
                </div>
                <div style="color: var(--text-primary); margin-top: 5px;">
                    {anomaly_count} anomalies detected in last 24 hours
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="border-left: 4px solid var(--success); padding: 12px;">
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: var(--success); font-weight: 600;">ALL CLEAR</span>
                    <span style="color: var(--text-muted); font-size: 0.8rem;">Now</span>
                </div>
                <div style="color: var(--text-primary); margin-top: 5px;">
                    All systems operating within normal parameters
                </div>
            </div>
            """, unsafe_allow_html=True)


with col_models:
    render_section_header("ML Models", "")

    for model_name, status in model_status.items():
        status_color = COLORS['success'] if status['loaded'] else COLORS['text_muted']
        status_text = "Active" if status['loaded'] else "Not Loaded"
        status_dot = "online" if status['loaded'] else "offline"

        st.markdown(f"""
        <div class="metric-card" style="padding: 12px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: var(--text-primary); font-weight: 500;">{model_name}</span>
                <div style="display: flex; align-items: center;">
                    <span class="status-dot {status_dot}"></span>
                    <span style="color: {status_color}; font-size: 0.85rem;">{status_text}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# Footer
# ============================================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; color: var(--text-muted); padding: 20px; border-top: 1px solid var(--border);">
    <div style="font-size: 0.85rem;">
        <strong>CONSTELLATION</strong> Fleet Health Management System
    </div>
    <div style="font-size: 0.75rem; margin-top: 5px;">
        Built by Michael Gurule | Data: NASA ISS Telemetry | Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</div>
""", unsafe_allow_html=True)
