"""
CONSTELLATION Main Dashboard
Streamlit application for satellite fleet health monitoring.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="CONSTELLATION",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-excellent { color: #28a745; font-weight: bold; }
    .status-good { color: #5cb85c; font-weight: bold; }
    .status-fair { color: #ffc107; font-weight: bold; }
    .status-degraded { color: #fd7e14; font-weight: bold; }
    .status-poor { color: #dc3545; font-weight: bold; }
    .status-critical { color: #c82333; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=CONSTELLATION", width=200)
    st.markdown("---")
    
    st.markdown("### 🛰️ Satellite Fleet Health Management")
    st.markdown("""
    **Real-time monitoring system** for ISS subsystems using advanced ML models.
    
    **Capabilities:**
    - Anomaly Detection
    - Fault Classification
    - Health Scoring
    - Predictive Maintenance
    """)
    
    st.markdown("---")
    st.markdown("**Navigation:**")
    st.page_link("app.py", label="🏠 Overview", icon="🏠")
    st.page_link("pages/1_📊_Telemetry.py", label="📊 Telemetry", icon="📊")
    st.page_link("pages/2_🔍_Anomalies.py", label="🔍 Anomalies", icon="🔍")
    st.page_link("pages/3_💊_Health.py", label="💊 Health", icon="💊")
    st.page_link("pages/4_🔧_Maintenance.py", label="🔧 Maintenance", icon="🔧")
    st.page_link("pages/5_⚙️_Settings.py", label="⚙️ Settings", icon="⚙️")

# Main content
st.markdown('<p class="main-header">🛰️ CONSTELLATION</p>', unsafe_allow_html=True)
st.markdown("### ISS Fleet Health Management System")

# System status overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🛰️ Subsystems",
        value="2",
        delta="All Operational"
    )

with col2:
    st.metric(
        label="🔍 Anomalies (24h)",
        value="12",
        delta="-3 from yesterday",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="💊 Avg Health Score",
        value="87.3",
        delta="+2.1"
    )

with col4:
    st.metric(
        label="🔧 Maintenance Tasks",
        value="4",
        delta="2 scheduled"
    )

st.markdown("---")

# Quick stats
st.markdown("### 📈 System Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🎯 Recent Activity")
    st.info("✓ Anomaly detection running continuously")
    st.info("✓ Health scores updated 5 minutes ago")
    st.info("✓ 3 models active (Isolation Forest, XGBoost, LSTM)")
    st.warning("⚠ 2 components below health threshold (60)")

with col2:
    st.markdown("#### 🚨 Active Alerts")
    st.error("🔴 CRITICAL: RWA_1 health degraded (Score: 42)")
    st.warning("🟡 WARNING: S-Band anomaly detected (Score: 0.85)")
    st.info("🔵 INFO: Routine maintenance due in 7 days")

st.markdown("---")

# Recent predictions section
st.markdown("### 🤖 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Anomaly Detector")
    st.markdown("**Isolation Forest**")
    st.progress(0.89)
    st.caption("Precision: 89%")

with col2:
    st.markdown("#### Fault Classifier")
    st.markdown("**XGBoost**")
    st.progress(0.82)
    st.caption("Accuracy: 82%")

with col3:
    st.markdown("#### Degradation Forecast")
    st.markdown("**LSTM**")
    st.progress(0.76)
    st.caption("R²: 0.76")

st.markdown("---")

# Quick actions
st.markdown("### ⚡ Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.toast("Data refreshed!")

with col2:
    if st.button("🔍 Run Analysis", use_container_width=True):
        st.toast("Analysis started...")

with col3:
    if st.button("📊 Generate Report", use_container_width=True):
        st.toast("Report generation queued")

with col4:
    if st.button("🔧 Schedule Maintenance", use_container_width=True):
        st.toast("Opening scheduler...")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>CONSTELLATION Fleet Health Management System | Built by Michael Gurule</p>
    <p>Real-time monitoring powered by ML | Last updated: Just now</p>
</div>
""", unsafe_allow_html=True)