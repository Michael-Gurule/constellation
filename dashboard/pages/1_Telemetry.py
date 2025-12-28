"""
Telemetry Monitoring Page
Real-time telemetry data visualization with advanced charting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from dashboard.theme import (
    apply_theme, render_logo, COLORS, CHART_COLORS,
    apply_plotly_theme, render_section_header, get_health_color
)

st.set_page_config(
    page_title="CONSTELLATION | Telemetry",
    page_icon="◇",
    layout="wide"
)

apply_theme()

# Load telemetry parameters
@st.cache_data
def load_param_definitions():
    """Load telemetry parameter definitions."""
    param_file = project_root / "config" / "telemetry_params.json"
    if param_file.exists():
        with open(param_file) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_telemetry_data(subsystem: str = None):
    """Load telemetry data for specified subsystem."""
    try:
        if subsystem:
            pattern = f"{subsystem}_features_*.parquet"
        else:
            pattern = "*_features_*.parquet"

        feature_files = list(PROCESSED_DATA_DIR.glob(pattern))

        if feature_files:
            dfs = []
            for f in feature_files:
                df = pd.read_parquet(f)
                dfs.append(df)
            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                if len(combined) > 50000:
                    combined = combined.sample(50000).sort_values('timestamp')
                return combined

        # Fallback to raw data
        if subsystem:
            raw_pattern = f"**/{subsystem}/*.parquet"
        else:
            raw_pattern = "**/*.parquet"

        raw_files = list(RAW_DATA_DIR.rglob(raw_pattern))
        if raw_files:
            recent = sorted(raw_files, key=lambda x: x.stat().st_mtime)[-20:]
            dfs = [pd.read_parquet(f) for f in recent]
            return pd.concat(dfs, ignore_index=True)

    except Exception as e:
        st.error(f"Error loading data: {e}")

    return pd.DataFrame()


PARAM_DEFINITIONS = load_param_definitions()


def get_param_display_name(param_id: str) -> str:
    """Get display name for parameter."""
    if param_id in PARAM_DEFINITIONS:
        return PARAM_DEFINITIONS[param_id].get('name', param_id)
    return param_id


def get_param_unit(param_id: str) -> str:
    """Get unit for parameter."""
    if param_id in PARAM_DEFINITIONS:
        return PARAM_DEFINITIONS[param_id].get('unit', '')
    return ''


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    render_logo()
    st.markdown("---")

    st.markdown("### Data Selection")

    subsystem = st.selectbox(
        "Subsystem",
        ["All", "attitude_control", "communications"],
        index=0,
        format_func=lambda x: x.replace('_', ' ').title() if x != "All" else "All Subsystems"
    )

    time_range = st.selectbox(
        "Time Range",
        ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days", "All Data"],
        index=2
    )

    st.markdown("---")

    st.markdown("### Display Options")

    chart_type = st.radio(
        "Chart Type",
        ["Line", "Area", "Candlestick"],
        index=0
    )

    show_anomalies = st.checkbox("Highlight Anomalies", value=True)
    show_bounds = st.checkbox("Show Normal Bounds", value=True)

    st.markdown("---")

    auto_refresh = st.checkbox("Auto Refresh (60s)", value=False)
    if auto_refresh:
        st.markdown("""
        <div style="color: var(--text-muted); font-size: 0.75rem;">
            Data refreshes every 60 seconds
        </div>
        """, unsafe_allow_html=True)



# Auto refresh
if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60000, key="telemetry_refresh")
    except ImportError:
        pass


# ============================================================================
# Main Content
# ============================================================================

st.markdown("""
<div style="margin-bottom: 30px;">
    <h1>Telemetry Monitoring</h1>
    <p style="color: var(--text-secondary);">
        Real-time satellite telemetry data visualization
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
subsystem_filter = None if subsystem == "All" else subsystem
df = load_telemetry_data(subsystem_filter)


if df.empty:
    st.warning("No telemetry data available. Run data collection to populate telemetry.")
    st.stop()


# ============================================================================
# Key Metrics
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("DATA POINTS", f"{len(df):,}")

with col2:
    if 'parameter_id' in df.columns:
        st.metric("PARAMETERS", df['parameter_id'].nunique())
    else:
        st.metric("PARAMETERS", "N/A")

with col3:
    if 'timestamp' in df.columns:
        time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        st.metric("TIME SPAN", f"{time_span:.1f} hrs")
    else:
        st.metric("TIME SPAN", "N/A")

with col4:
    if 'timestamp' in df.columns:
        st.metric("LATEST", df['timestamp'].max().strftime("%H:%M:%S"))
    else:
        st.metric("LATEST", "N/A")

with col5:
    status = "NOMINAL" if len(df) > 0 else "NO DATA"
    st.metric("STATUS", status)


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# Parameter Selection and Main Chart
# ============================================================================

render_section_header("Parameter Analysis", "")

# Get available parameters
if 'parameter_id' in df.columns:
    available_params = sorted(df['parameter_id'].unique())
    param_options = [(p, get_param_display_name(p)) for p in available_params]

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_params = st.multiselect(
            "Select Parameters to Visualize",
            options=available_params,
            default=available_params[:2] if len(available_params) >= 2 else available_params,
            format_func=get_param_display_name,
            max_selections=6
        )

    with col2:
        normalize = st.checkbox("Normalize Values", value=False,
                              help="Scale all parameters to 0-1 range for comparison")

    if selected_params:
        # Filter data
        plot_df = df[df['parameter_id'].isin(selected_params)].copy()
        plot_df = plot_df.sort_values('timestamp')

        # Create the main chart
        fig = go.Figure()

        for i, param in enumerate(selected_params):
            param_data = plot_df[plot_df['parameter_id'] == param]

            if param_data.empty:
                continue

            y_values = param_data['value_numeric'].values

            # Normalize if requested
            if normalize and len(y_values) > 0:
                min_val, max_val = y_values.min(), y_values.max()
                if max_val > min_val:
                    y_values = (y_values - min_val) / (max_val - min_val)

            display_name = get_param_display_name(param)
            unit = get_param_unit(param)
            color = CHART_COLORS[i % len(CHART_COLORS)]

            if chart_type == "Area":
                fig.add_trace(go.Scatter(
                    x=param_data['timestamp'],
                    y=y_values,
                    mode='lines',
                    name=display_name,
                    line=dict(color=color, width=2),
                    fill='tozeroy',
                    fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}",
                    hovertemplate=f"<b>{display_name}</b><br>Value: %{{y:.2f}} {unit}<br>Time: %{{x}}<extra></extra>"
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=param_data['timestamp'],
                    y=y_values,
                    mode='lines',
                    name=display_name,
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{display_name}</b><br>Value: %{{y:.2f}} {unit}<br>Time: %{{x}}<extra></extra>"
                ))

            # Add normal bounds if enabled
            if show_bounds and param in PARAM_DEFINITIONS:
                normal_range = PARAM_DEFINITIONS[param].get('normal_range', [])
                if len(normal_range) == 2:
                    min_bound, max_bound = normal_range
                    if normalize:
                        orig_min = param_data['value_numeric'].min()
                        orig_max = param_data['value_numeric'].max()
                        if orig_max > orig_min:
                            min_bound = (min_bound - orig_min) / (orig_max - orig_min)
                            max_bound = (max_bound - orig_min) / (orig_max - orig_min)

                    fig.add_hline(y=max_bound, line_dash="dot",
                                line_color=COLORS['warning'], opacity=0.5,
                                annotation_text=f"{display_name} max")
                    fig.add_hline(y=min_bound, line_dash="dot",
                                line_color=COLORS['warning'], opacity=0.5)

        fig = apply_plotly_theme(fig)
        fig.update_layout(
            height=450,
            margin=dict(l=0, r=0, t=20, b=0),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            yaxis_title="Normalized Value" if normalize else "Value"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Select parameters to visualize")

else:
    # No parameter_id column - plot raw values
    if 'value_numeric' in df.columns and 'timestamp' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['value_numeric'],
            mode='lines',
            name='Value',
            line=dict(color=COLORS['primary'], width=2)
        ))
        fig = apply_plotly_theme(fig)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Subsystem Comparison
# ============================================================================

st.markdown("<br>", unsafe_allow_html=True)
render_section_header("Subsystem Overview", "")

col1, col2 = st.columns(2)

with col1:
    # Attitude Control summary
    st.markdown("""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 1.1rem; font-weight: 600; color: var(--text-primary);">Attitude Control</span>
            <span style="color: var(--success); font-size: 0.85rem;">NOMINAL</span>
        </div>
    """, unsafe_allow_html=True)

    # Parameter mini-charts for attitude control
    ac_params = [p for p in PARAM_DEFINITIONS if PARAM_DEFINITIONS[p].get('subsystem') == 'attitude_control']

    if 'parameter_id' in df.columns:
        ac_data = df[df['parameter_id'].isin(ac_params[:4])]

        if not ac_data.empty:
            # Create sparklines
            fig_spark = make_subplots(rows=2, cols=2, subplot_titles=[
                get_param_display_name(p) for p in ac_params[:4]
            ] if len(ac_params) >= 4 else None,
            vertical_spacing=0.15, horizontal_spacing=0.1)

            for i, param in enumerate(ac_params[:4]):
                param_data = ac_data[ac_data['parameter_id'] == param].sort_values('timestamp').tail(100)
                if not param_data.empty:
                    row = (i // 2) + 1
                    col = (i % 2) + 1
                    fig_spark.add_trace(go.Scatter(
                        x=param_data['timestamp'],
                        y=param_data['value_numeric'],
                        mode='lines',
                        line=dict(color=CHART_COLORS[i], width=1.5),
                        showlegend=False,
                        hovertemplate="%{y:.2f}<extra></extra>"
                    ), row=row, col=col)

            fig_spark = apply_plotly_theme(fig_spark)
            fig_spark.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            fig_spark.update_xaxes(showticklabels=False)
            fig_spark.update_yaxes(showticklabels=False)

            st.plotly_chart(fig_spark, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Communications summary
    st.markdown("""
    <div class="metric-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 1.1rem; font-weight: 600; color: var(--text-primary);">Communications</span>
            <span style="color: var(--success); font-size: 0.85rem;">NOMINAL</span>
        </div>
    """, unsafe_allow_html=True)

    comm_params = [p for p in PARAM_DEFINITIONS if PARAM_DEFINITIONS[p].get('subsystem') == 'communications']

    if 'parameter_id' in df.columns:
        comm_data = df[df['parameter_id'].isin(comm_params[:4])]

        if not comm_data.empty:
            fig_spark = make_subplots(rows=2, cols=2, subplot_titles=[
                get_param_display_name(p) for p in comm_params[:4]
            ] if len(comm_params) >= 4 else None,
            vertical_spacing=0.15, horizontal_spacing=0.1)

            for i, param in enumerate(comm_params[:4]):
                param_data = comm_data[comm_data['parameter_id'] == param].sort_values('timestamp').tail(100)
                if not param_data.empty:
                    row = (i // 2) + 1
                    col = (i % 2) + 1
                    fig_spark.add_trace(go.Scatter(
                        x=param_data['timestamp'],
                        y=param_data['value_numeric'],
                        mode='lines',
                        line=dict(color=CHART_COLORS[i + 2], width=1.5),
                        showlegend=False,
                        hovertemplate="%{y:.2f}<extra></extra>"
                    ), row=row, col=col)

            fig_spark = apply_plotly_theme(fig_spark)
            fig_spark.update_layout(
                height=250,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            fig_spark.update_xaxes(showticklabels=False)
            fig_spark.update_yaxes(showticklabels=False)

            st.plotly_chart(fig_spark, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# Statistical Summary
# ============================================================================

st.markdown("<br>", unsafe_allow_html=True)
render_section_header("Statistical Summary", "")

if 'parameter_id' in df.columns and 'value_numeric' in df.columns:
    # Calculate stats per parameter
    stats_data = []

    for param in df['parameter_id'].unique():
        param_data = df[df['parameter_id'] == param]['value_numeric']
        if param_data.empty:
            continue

        stats_data.append({
            'Parameter': get_param_display_name(param),
            'Unit': get_param_unit(param),
            'Count': len(param_data),
            'Mean': param_data.mean(),
            'Std': param_data.std(),
            'Min': param_data.min(),
            'Max': param_data.max(),
            'Current': param_data.iloc[-1] if len(param_data) > 0 else None
        })

    if stats_data:
        stats_df = pd.DataFrame(stats_data)

        # Format numeric columns
        st.dataframe(
            stats_df,
            column_config={
                "Parameter": st.column_config.TextColumn("Parameter", width="medium"),
                "Unit": st.column_config.TextColumn("Unit", width="small"),
                "Count": st.column_config.NumberColumn("Count", format="%d"),
                "Mean": st.column_config.NumberColumn("Mean", format="%.2f"),
                "Std": st.column_config.NumberColumn("Std Dev", format="%.2f"),
                "Min": st.column_config.NumberColumn("Min", format="%.2f"),
                "Max": st.column_config.NumberColumn("Max", format="%.2f"),
                "Current": st.column_config.NumberColumn("Current", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )


# ============================================================================
# Export Section
# ============================================================================

st.markdown("<br>", unsafe_allow_html=True)
render_section_header("Data Export", "")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Export CSV", use_container_width=True):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("Export Statistics", use_container_width=True):
        if 'stats_df' in dir():
            csv = stats_df.to_csv(index=False)
            st.download_button(
                label="Download Stats",
                data=csv,
                file_name=f"telemetry_stats_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

with col3:
    if st.button("Generate Report", use_container_width=True):
        st.info("Report generation feature coming soon")


# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; color: var(--text-muted); padding: 20px; border-top: 1px solid var(--border);">
    <div style="font-size: 0.75rem;">
        Telemetry data from NASA ISS Lightstreamer | Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</div>
""", unsafe_allow_html=True)
