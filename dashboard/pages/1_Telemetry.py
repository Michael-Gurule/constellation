"""
Telemetry Monitoring Page
Real-time telemetry data visualization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import PROCESSED_DATA_DIR

st.set_page_config(page_title="Telemetry Monitoring", page_icon="📊", layout="wide")

st.title("📊 Telemetry Monitoring")
st.markdown("Real-time satellite telemetry data visualization")

# Sidebar controls
st.sidebar.markdown("### 🎛️ Controls")

subsystem = st.sidebar.selectbox(
    "Subsystem",
    ["Attitude Control", "Communications"],
    index=0
)

time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
    index=2
)

auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)

if auto_refresh:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60000, key="telemetry_refresh")  # 60 seconds

# Load data
@st.cache_data(ttl=300)
def load_telemetry_data():
    """Load latest telemetry features."""
    pattern = "attitude_control_features_*.parquet"
    feature_files = list(PROCESSED_DATA_DIR.glob(pattern))
    
    if not feature_files:
        return pd.DataFrame()
    
    latest_file = max(feature_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_parquet(latest_file)
    
    # Sample for performance
    if len(df) > 10000:
        df = df.sample(10000).sort_values('timestamp')
    
    return df

df = load_telemetry_data()

if df.empty:
    st.error("No telemetry data available. Please run feature extraction first.")
    st.stop()

st.success(f"✓ Loaded {len(df):,} telemetry records")

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Data Points", f"{len(df):,}")

with col2:
    if 'parameter_id' in df.columns:
        st.metric("Parameters", df['parameter_id'].nunique())
    else:
        st.metric("Parameters", "N/A")

with col3:
    time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
    st.metric("Time Span", f"{time_span:.1f} hrs")

with col4:
    st.metric("Latest Update", df['timestamp'].max().strftime("%H:%M:%S"))

st.markdown("---")

# Time series plot
st.markdown("### 📈 Parameter Trends")

# Select parameter to visualize
if 'parameter_id' in df.columns:
    available_params = df['parameter_id'].unique()[:10]  # Limit to first 10
    selected_param = st.selectbox("Select Parameter", available_params)
    
    param_df = df[df['parameter_id'] == selected_param]
else:
    param_df = df
    selected_param = "value_numeric"

# Create time series plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=param_df['timestamp'],
    y=param_df['value_numeric'],
    mode='lines',
    name='Actual Value',
    line=dict(color='#1f77b4', width=2)
))

# Add rolling mean if available
if 'rolling_mean_1h' in param_df.columns:
    fig.add_trace(go.Scatter(
        x=param_df['timestamp'],
        y=param_df['rolling_mean_1h'],
        mode='lines',
        name='1-Hour Mean',
        line=dict(color='#ff7f0e', width=1, dash='dash')
    ))

fig.update_layout(
    title=f"Telemetry: {selected_param}",
    xaxis_title="Time",
    yaxis_title="Value",
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Statistics
st.markdown("### 📊 Statistical Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Recent Statistics")
    
    stats_df = param_df['value_numeric'].describe()
    stats_table = pd.DataFrame({
        'Metric': stats_df.index,
        'Value': stats_df.values
    })
    st.dataframe(stats_table, hide_index=True, use_container_width=True)

with col2:
    st.markdown("#### Feature Distribution")
    
    # Histogram
    fig_hist = px.histogram(
        param_df,
        x='value_numeric',
        nbins=50,
        title="Value Distribution"
    )
    fig_hist.update_layout(height=300)
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# Multi-parameter comparison
st.markdown("### 🔄 Multi-Parameter View")

if 'parameter_id' in df.columns:
    # Select up to 3 parameters
    params_to_compare = st.multiselect(
        "Select parameters to compare (max 3)",
        available_params,
        default=list(available_params[:2])
    )
    
    if params_to_compare:
        fig_multi = go.Figure()
        
        for param in params_to_compare[:3]:
            param_data = df[df['parameter_id'] == param]
            fig_multi.add_trace(go.Scatter(
                x=param_data['timestamp'],
                y=param_data['value_numeric'],
                mode='lines',
                name=param
            ))
        
        fig_multi.update_layout(
            title="Parameter Comparison",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_multi, use_container_width=True)

# Raw data table
st.markdown("### 📋 Raw Data")

with st.expander("View Raw Telemetry Data"):
    display_cols = ['timestamp', 'value_numeric']
    
    if 'parameter_id' in df.columns:
        display_cols.insert(1, 'parameter_id')
    
    available_display_cols = [c for c in display_cols if c in df.columns]
    
    st.dataframe(
        param_df[available_display_cols].tail(100),
        use_container_width=True,
        height=300
    )

# Download data
st.markdown("### 💾 Export Data")

col1, col2 = st.columns(2)

with col1:
    csv = param_df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"telemetry_{selected_param}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col2:
    if st.button("📊 Generate Report"):
        st.toast("Report generation not yet implemented")