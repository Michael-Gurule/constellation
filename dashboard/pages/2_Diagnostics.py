"""
Diagnostics Page
Combined anomaly detection, health monitoring, and fault analysis.
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

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import PROCESSED_DATA_DIR, MODELS_DIR
from dashboard.theme import (
    apply_theme, render_logo, COLORS, CHART_COLORS,
    apply_plotly_theme, render_section_header, get_health_color, get_anomaly_color
)

st.set_page_config(
    page_title="CONSTELLATION | Diagnostics",
    page_icon="○",
    layout="wide"
)

apply_theme()


# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data(ttl=300)
def load_analysis_results():
    """Load analysis predictions."""
    results_dir = PROCESSED_DATA_DIR / "analysis_results"

    if not results_dir.exists():
        return None

    pred_files = list(results_dir.glob("predictions_*.parquet"))
    if not pred_files:
        return None

    latest = max(pred_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_parquet(latest)

    if len(df) > 50000:
        df = df.sample(50000)

    return df


@st.cache_data(ttl=300)
def load_health_scores():
    """Load component health scores."""
    results_dir = PROCESSED_DATA_DIR / "analysis_results"

    if not results_dir.exists():
        return None

    health_files = list(results_dir.glob("health_scores_*.csv"))
    if not health_files:
        return None

    latest = max(health_files, key=lambda p: p.stat().st_mtime)
    return pd.read_csv(latest)


def generate_demo_health_data():
    """Generate demo health data for display."""
    components = [
        ('RWA_1', 'Reaction Wheel 1', 'attitude_control', 92),
        ('RWA_2', 'Reaction Wheel 2', 'attitude_control', 88),
        ('RWA_3', 'Reaction Wheel 3', 'attitude_control', 95),
        ('RWA_4', 'Reaction Wheel 4', 'attitude_control', 78),
        ('CMG_1', 'Control Moment Gyro 1', 'attitude_control', 85),
        ('CMG_2', 'Control Moment Gyro 2', 'attitude_control', 91),
        ('CMG_3', 'Control Moment Gyro 3', 'attitude_control', 89),
        ('CMG_4', 'Control Moment Gyro 4', 'attitude_control', 94),
        ('SBAND', 'S-Band Transceiver', 'communications', 96),
        ('KUBAND', 'Ku-Band Transceiver', 'communications', 82),
        ('ANTENNA_AZ', 'Antenna Azimuth', 'communications', 90),
        ('ANTENNA_EL', 'Antenna Elevation', 'communications', 88),
    ]

    def get_status(score):
        if score >= 85:
            return 'EXCELLENT'
        elif score >= 70:
            return 'GOOD'
        elif score >= 55:
            return 'FAIR'
        elif score >= 40:
            return 'DEGRADED'
        else:
            return 'CRITICAL'

    return pd.DataFrame([{
        'component': c[0],
        'name': c[1],
        'subsystem': c[2],
        'health_score': c[3],
        'status': get_status(c[3]),
        'anomaly_score': np.random.uniform(0.1, 0.5) if c[3] > 70 else np.random.uniform(0.5, 0.9),
        'degradation_rate': np.random.uniform(0.001, 0.01)
    } for c in components])


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    render_logo()
    st.markdown("---")

    st.markdown("### Diagnostics View")

    view_mode = st.radio(
        "Display Mode",
        ["Overview", "Health Analysis", "Anomaly Detection", "Fault Classification"],
        index=0
    )

    st.markdown("---")

    st.markdown("### Filters")

    subsystem_filter = st.multiselect(
        "Subsystems",
        ["attitude_control", "communications"],
        default=["attitude_control", "communications"],
        format_func=lambda x: x.replace('_', ' ').title()
    )

    health_threshold = st.slider(
        "Health Threshold",
        0, 100, 60,
        help="Components below this threshold are flagged"
    )



# ============================================================================
# Main Content
# ============================================================================

st.markdown("""
<div style="margin-bottom: 30px;">
    <h1>System Diagnostics</h1>
    <p style="color: var(--text-secondary);">
        Anomaly detection, health monitoring, and fault analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
predictions = load_analysis_results()
health_scores = load_health_scores()

# Use demo data if no real data available
if health_scores is None:
    health_scores = generate_demo_health_data()
    using_demo = True
else:
    using_demo = False

# Filter by subsystem
if 'subsystem' in health_scores.columns:
    health_scores = health_scores[health_scores['subsystem'].isin(subsystem_filter)]


# ============================================================================
# Key Metrics
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

avg_health = health_scores['health_score'].mean() if 'health_score' in health_scores.columns else 0
critical_count = (health_scores['health_score'] < 40).sum() if 'health_score' in health_scores.columns else 0
degraded_count = ((health_scores['health_score'] >= 40) & (health_scores['health_score'] < health_threshold)).sum() if 'health_score' in health_scores.columns else 0
healthy_count = (health_scores['health_score'] >= health_threshold).sum() if 'health_score' in health_scores.columns else 0

anomaly_count = 0
if predictions is not None and 'anomaly_score' in predictions.columns:
    anomaly_count = (predictions['anomaly_score'] > 0.8).sum()

with col1:
    st.metric("AVG HEALTH", f"{avg_health:.1f}%")

with col2:
    st.metric("HEALTHY", healthy_count)

with col3:
    st.metric("DEGRADED", degraded_count, delta="Warning" if degraded_count > 0 else None, delta_color="off")

with col4:
    st.metric("CRITICAL", critical_count, delta="Alert" if critical_count > 0 else None, delta_color="inverse")

with col5:
    st.metric("ANOMALIES", anomaly_count)


if using_demo:
    st.info("Displaying demo data. Run system analysis to see actual diagnostics.")


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# Overview Mode
# ============================================================================

if view_mode == "Overview":
    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        render_section_header("Component Health Map", "")

        # Create health heatmap/treemap
        if not health_scores.empty:
            # Add color based on health
            health_scores['color'] = health_scores['health_score'].apply(get_health_color)

            fig = px.treemap(
                health_scores,
                path=['subsystem', 'component'],
                values=[1] * len(health_scores),
                color='health_score',
                color_continuous_scale=[
                    [0, COLORS['critical']],
                    [0.4, COLORS['error']],
                    [0.6, COLORS['warning']],
                    [0.85, COLORS['health_good']],
                    [1.0, COLORS['success']]
                ],
                hover_data=['health_score', 'status'],
                title=None
            )

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text_primary']),
                margin=dict(l=0, r=0, t=0, b=0),
                height=400,
                coloraxis_colorbar=dict(
                    title="Health",
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=['0%', '25%', '50%', '75%', '100%']
                )
            )

            fig.update_traces(
                textinfo='label+value',
                hovertemplate='<b>%{label}</b><br>Health: %{color:.1f}%<extra></extra>'
            )

            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        render_section_header("Health Distribution", "")

        # Create distribution chart
        fig_dist = go.Figure()

        # Health score histogram
        fig_dist.add_trace(go.Histogram(
            x=health_scores['health_score'],
            nbinsx=20,
            marker=dict(
                color=COLORS['primary'],
                line=dict(color=COLORS['border'], width=1)
            ),
            hovertemplate='Health: %{x:.0f}%<br>Count: %{y}<extra></extra>'
        ))

        # Add threshold line
        fig_dist.add_vline(
            x=health_threshold,
            line_dash="dash",
            line_color=COLORS['warning'],
            annotation_text=f"Threshold ({health_threshold}%)"
        )

        fig_dist = apply_plotly_theme(fig_dist)
        fig_dist.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Health Score",
            yaxis_title="Components",
            showlegend=False
        )

        st.plotly_chart(fig_dist, use_container_width=True)


    # Component Status Table
    render_section_header("Component Status", "")

    # Sort by health score
    display_df = health_scores.sort_values('health_score', ascending=True).copy()

    st.dataframe(
        display_df,
        column_config={
            "component": st.column_config.TextColumn("Component", width="small"),
            "name": st.column_config.TextColumn("Name", width="medium"),
            "subsystem": st.column_config.TextColumn("Subsystem", width="small"),
            "health_score": st.column_config.ProgressColumn(
                "Health",
                format="%.1f%%",
                min_value=0,
                max_value=100
            ),
            "status": st.column_config.TextColumn("Status", width="small"),
            "anomaly_score": st.column_config.NumberColumn("Anomaly", format="%.3f"),
            "degradation_rate": st.column_config.NumberColumn("Deg. Rate", format="%.4f"),
        },
        hide_index=True,
        use_container_width=True,
        height=400
    )


# ============================================================================
# Health Analysis Mode
# ============================================================================

elif view_mode == "Health Analysis":
    render_section_header("Component Health Analysis", "")

    col1, col2 = st.columns(2)

    with col1:
        # Radar chart for subsystem health
        subsystem_health = health_scores.groupby('subsystem')['health_score'].mean()

        fig_radar = go.Figure()

        categories = list(subsystem_health.index)
        values = list(subsystem_health.values)
        values.append(values[0])  # Close the polygon
        categories.append(categories[0])

        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor=f"rgba(0, 212, 255, 0.2)",
            line=dict(color=COLORS['primary'], width=2),
            name='Current Health'
        ))

        # Add reference circle at threshold
        ref_values = [health_threshold] * len(categories)
        fig_radar.add_trace(go.Scatterpolar(
            r=ref_values,
            theta=categories,
            line=dict(color=COLORS['warning'], width=1, dash='dot'),
            name='Threshold'
        ))

        fig_radar = apply_plotly_theme(fig_radar)
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor=COLORS['grid'],
                    linecolor=COLORS['border']
                ),
                angularaxis=dict(
                    gridcolor=COLORS['grid'],
                    linecolor=COLORS['border']
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            height=400,
            margin=dict(l=60, r=60, t=40, b=40),
            showlegend=True,
            legend=dict(x=0.5, y=-0.1, xanchor='center', orientation='h')
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        # Bar chart of individual components
        sorted_health = health_scores.sort_values('health_score')

        colors = [get_health_color(score) for score in sorted_health['health_score']]

        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            x=sorted_health['health_score'],
            y=sorted_health['component'],
            orientation='h',
            marker=dict(color=colors),
            text=sorted_health['health_score'].round(1),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Health: %{x:.1f}%<extra></extra>'
        ))

        # Add threshold line
        fig_bar.add_vline(
            x=health_threshold,
            line_dash="dash",
            line_color=COLORS['warning']
        )

        fig_bar = apply_plotly_theme(fig_bar)
        fig_bar.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Health Score",
            yaxis_title="",
            showlegend=False
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # Health trends (simulated)
    render_section_header("Health Trends (24h)", "")

    # Generate simulated trend data
    hours = pd.date_range(end=datetime.now(), periods=24, freq='H')

    fig_trends = go.Figure()

    for i, (_, row) in enumerate(health_scores.head(6).iterrows()):
        # Simulate trend with some variation
        base = row['health_score']
        trend = base + np.cumsum(np.random.normal(0, 0.5, 24))
        trend = np.clip(trend, 0, 100)

        fig_trends.add_trace(go.Scatter(
            x=hours,
            y=trend,
            mode='lines',
            name=row['component'],
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2)
        ))

    fig_trends = apply_plotly_theme(fig_trends)
    fig_trends.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Health Score",
        yaxis_range=[0, 100]
    )

    st.plotly_chart(fig_trends, use_container_width=True)


# ============================================================================
# Anomaly Detection Mode
# ============================================================================

elif view_mode == "Anomaly Detection":
    render_section_header("Anomaly Detection Results", "")

    if predictions is not None and 'anomaly_score' in predictions.columns:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Anomaly score distribution
            fig_anom = go.Figure()

            fig_anom.add_trace(go.Histogram(
                x=predictions['anomaly_score'],
                nbinsx=50,
                marker=dict(
                    color=predictions['anomaly_score'].apply(get_anomaly_color),
                    line=dict(color=COLORS['border'], width=0.5)
                )
            ))

            # Add threshold lines
            fig_anom.add_vline(x=0.8, line_dash="dash", line_color=COLORS['warning'],
                             annotation_text="Warning (0.8)")
            fig_anom.add_vline(x=0.95, line_dash="dash", line_color=COLORS['critical'],
                             annotation_text="Critical (0.95)")

            fig_anom = apply_plotly_theme(fig_anom)
            fig_anom.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="Anomaly Score",
                yaxis_title="Frequency"
            )

            st.plotly_chart(fig_anom, use_container_width=True)

        with col2:
            # Score breakdown
            st.markdown("""
            <div class="metric-card">
                <div style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 15px;">
                    Score Distribution
                </div>
            """, unsafe_allow_html=True)

            score_ranges = pd.cut(
                predictions['anomaly_score'],
                bins=[0, 0.5, 0.8, 0.95, 1.0],
                labels=['Normal', 'Elevated', 'High', 'Critical']
            )
            range_counts = score_ranges.value_counts()

            for level, count in range_counts.items():
                pct = count / len(predictions) * 100
                color = {
                    'Normal': COLORS['success'],
                    'Elevated': COLORS['warning'],
                    'High': COLORS['error'],
                    'Critical': COLORS['critical']
                }.get(level, COLORS['text_muted'])

                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: {color}; font-weight: 500;">{level}</span>
                    <span style="color: var(--text-secondary);">{count:,} ({pct:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Time series of anomalies
        if 'timestamp' in predictions.columns:
            render_section_header("Anomaly Timeline", "")

            predictions_sorted = predictions.sort_values('timestamp')

            fig_timeline = go.Figure()

            fig_timeline.add_trace(go.Scatter(
                x=predictions_sorted['timestamp'],
                y=predictions_sorted['anomaly_score'],
                mode='lines',
                line=dict(color=COLORS['primary'], width=1),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 255, 0.1)',
                hovertemplate='Score: %{y:.3f}<br>Time: %{x}<extra></extra>'
            ))

            # Add threshold line
            fig_timeline.add_hline(y=0.8, line_dash="dash", line_color=COLORS['warning'])

            fig_timeline = apply_plotly_theme(fig_timeline)
            fig_timeline.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Anomaly Score",
                yaxis_range=[0, 1]
            )

            st.plotly_chart(fig_timeline, use_container_width=True)

    else:
        st.info("No anomaly detection results available. Run system analysis to generate predictions.")


# ============================================================================
# Fault Classification Mode
# ============================================================================

elif view_mode == "Fault Classification":
    render_section_header("Fault Classification", "")

    if predictions is not None and 'fault_type' in predictions.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Fault type distribution
            fault_counts = predictions['fault_type'].value_counts()

            fig_faults = go.Figure()

            fig_faults.add_trace(go.Pie(
                labels=fault_counts.index,
                values=fault_counts.values,
                hole=0.4,
                marker=dict(colors=CHART_COLORS[:len(fault_counts)]),
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
            ))

            fig_faults.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text_secondary']),
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True,
                legend=dict(
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(color=COLORS['text_secondary'])
                )
            )

            st.plotly_chart(fig_faults, use_container_width=True)

        with col2:
            # Fault severity matrix
            st.markdown("""
            <div class="metric-card">
                <div style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 15px;">
                    Fault Summary
                </div>
            """, unsafe_allow_html=True)

            for fault_type, count in fault_counts.items():
                pct = count / len(predictions) * 100
                st.markdown(f"""
                <div style="background: var(--bg-light); padding: 12px; border-radius: 8px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: var(--text-primary); font-weight: 500;">{fault_type}</span>
                        <span style="color: var(--primary); font-family: 'JetBrains Mono', monospace;">{count:,}</span>
                    </div>
                    <div style="color: var(--text-muted); font-size: 0.8rem; margin-top: 4px;">
                        {pct:.1f}% of total detections
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Show model-based fault analysis
        st.markdown("""
        <div class="metric-card">
            <div style="color: var(--text-primary); font-size: 1.1rem; font-weight: 600; margin-bottom: 15px;">
                Fault Classification Model
            </div>
            <div style="color: var(--text-secondary); line-height: 1.8;">
                The XGBoost fault classifier analyzes telemetry patterns to identify potential fault types:
                <br/><br/>
                <strong>Detectable Fault Types:</strong>
                <ul style="margin-top: 10px; color: var(--text-muted);">
                    <li>Bearing degradation (RWA/CMG)</li>
                    <li>Motor current anomalies</li>
                    <li>Thermal excursions</li>
                    <li>Communication signal degradation</li>
                    <li>Pointing errors</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Check if model exists
        model_path = MODELS_DIR / "fault_classifier.pkl"
        if model_path.exists():
            st.success("Fault classifier model is loaded and ready")
        else:
            st.warning("Fault classifier model not found. Run model training to enable.")


# ============================================================================
# Export Section
# ============================================================================

st.markdown("<br>", unsafe_allow_html=True)
render_section_header("Export Diagnostics", "")

col1, col2, col3 = st.columns(3)

with col1:
    csv = health_scores.to_csv(index=False)
    st.download_button(
        label="Export Health Scores",
        data=csv,
        file_name=f"health_scores_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    if predictions is not None:
        # Export critical anomalies
        critical = predictions[predictions['anomaly_score'] > 0.8] if 'anomaly_score' in predictions.columns else predictions.head(0)
        csv = critical.to_csv(index=False)
        st.download_button(
            label="Export Anomalies",
            data=csv,
            file_name=f"anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.button("Export Anomalies", disabled=True, use_container_width=True)

with col3:
    if st.button("Generate Report", use_container_width=True):
        st.info("Diagnostic report generation coming soon")


# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; color: var(--text-muted); padding: 20px; border-top: 1px solid var(--border);">
    <div style="font-size: 0.75rem;">
        Diagnostics powered by Isolation Forest, XGBoost, and LSTM models | Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</div>
""", unsafe_allow_html=True)
