"""
Anomaly Detection Page
View and analyze detected anomalies.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import PROCESSED_DATA_DIR

st.set_page_config(page_title="Anomaly Detection", page_icon="🔍", layout="wide")

st.title("🔍 Anomaly Detection")
st.markdown("Real-time anomaly detection and analysis")

# Load analysis results
@st.cache_data(ttl=300)
def load_analysis_results():
    """Load latest analysis results."""
    results_dir = PROCESSED_DATA_DIR / "analysis_results"
    
    if not results_dir.exists():
        return None
    
    pred_files = list(results_dir.glob("predictions_*.parquet"))
    
    if not pred_files:
        return None
    
    latest_file = max(pred_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_parquet(latest_file)
    
    # Sample if large
    if len(df) > 50000:
        df = df.sample(50000)
    
    return df

df = load_analysis_results()

if df is None:
    st.warning("⚠️ No analysis results available. Run system analysis first.")
    
    if st.button("🔄 Run Analysis Now"):
        st.info("Starting analysis... (this would trigger the analysis script)")
    
    st.stop()

st.success(f"✓ Loaded {len(df):,} analyzed records")

# Check for required columns
if 'anomaly_score' not in df.columns:
    st.error("Missing anomaly_score column. Please run complete analysis.")
    st.stop()

# Anomaly statistics
anomalies = df[df.get('is_anomaly', 0) == 1] if 'is_anomaly' in df.columns else df[df['anomaly_score'] > 0.8]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Anomalies Detected", f"{len(anomalies):,}")

with col3:
    anomaly_rate = len(anomalies) / len(df) * 100
    st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")

with col4:
    avg_score = df['anomaly_score'].mean()
    st.metric("Avg Anomaly Score", f"{avg_score:.3f}")

st.markdown("---")

# Anomaly score distribution
st.markdown("### 📊 Anomaly Score Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    fig = px.histogram(
        df,
        x='anomaly_score',
        nbins=50,
        title="Anomaly Score Distribution",
        labels={'anomaly_score': 'Anomaly Score', 'count': 'Frequency'}
    )
    
    # Add threshold line
    fig.add_vline(x=0.8, line_dash="dash", line_color="red", 
                  annotation_text="Critical Threshold")
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Score Ranges")
    
    score_ranges = pd.cut(df['anomaly_score'], 
                          bins=[0, 0.5, 0.8, 0.95, 1.0],
                          labels=['Low', 'Medium', 'High', 'Critical'])
    
    range_counts = score_ranges.value_counts().sort_index()
    
    for range_name, count in range_counts.items():
        pct = count / len(df) * 100
        st.metric(f"{range_name}", f"{count:,} ({pct:.1f}%)")

st.markdown("---")

# Anomalies over time
st.markdown("### ⏱️ Anomalies Over Time")

if 'timestamp' in df.columns:
    # Group by hour
    df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('H')
    hourly_anomalies = df.groupby('hour')['anomaly_score'].agg(['mean', 'count', 'max'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hourly_anomalies.index,
        y=hourly_anomalies['mean'],
        mode='lines+markers',
        name='Avg Score',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly_anomalies.index,
        y=hourly_anomalies['max'],
        mode='lines',
        name='Max Score',
        line=dict(color='#ff7f0e', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title="Anomaly Scores Over Time (Hourly)",
        xaxis_title="Time",
        yaxis_title="Anomaly Score",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Fault type distribution
st.markdown("### 🔬 Fault Classification")

if 'fault_type' in df.columns:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fault_counts = df['fault_type'].value_counts()
        
        fig = px.pie(
            values=fault_counts.values,
            names=fault_counts.index,
            title="Fault Type Distribution"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Fault Type Summary")
        
        for fault_type, count in fault_counts.items():
            pct = count / len(df) * 100
            st.markdown(f"**{fault_type}:** {count:,} ({pct:.1f}%)")
else:
    st.info("Fault classification data not available")

st.markdown("---")

# Critical anomalies table
st.markdown("### 🚨 Critical Anomalies")

critical_threshold = st.slider("Anomaly Score Threshold", 0.0, 1.0, 0.95, 0.05)

critical = df[df['anomaly_score'] >= critical_threshold].sort_values('anomaly_score', ascending=False)

if len(critical) > 0:
    st.warning(f"⚠️ {len(critical)} anomalies above threshold {critical_threshold}")
    
    display_cols = ['timestamp', 'anomaly_score']
    
    if 'fault_type' in critical.columns:
        display_cols.append('fault_type')
    if 'fault_confidence' in critical.columns:
        display_cols.append('fault_confidence')
    if 'component' in critical.columns:
        display_cols.insert(1, 'component')
    
    available_cols = [c for c in display_cols if c in critical.columns]
    
    st.dataframe(
        critical[available_cols].head(50),
        use_container_width=True,
        height=400
    )
else:
    st.success(f"✓ No anomalies above threshold {critical_threshold}")

# Export
st.markdown("### 💾 Export Anomalies")

csv = critical.to_csv(index=False)
st.download_button(
    label="📥 Download Critical Anomalies",
    data=csv,
    file_name=f"critical_anomalies_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)