"""
Health Monitoring Page
Component and subsystem health scores.
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

st.set_page_config(page_title="Health Monitoring", page_icon="💊", layout="wide")

st.title("Health Monitoring")
st.markdown("Component and subsystem health status")

# Load health scores
@st.cache_data(ttl=300)
def load_health_scores():
    """Load latest health scores."""
    results_dir = PROCESSED_DATA_DIR / "analysis_results"
    
    if not results_dir.exists():
        return None
    
    health_files = list(results_dir.glob("health_scores_*.csv"))
    
    if not health_files:
        return None
    
    latest_file = max(health_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    return df

health_df = load_health_scores()

if health_df is None:
    st.warning("No health score data available. Run system analysis first.")
    
    if st.button("Run Analysis Now"):
        st.info("Starting analysis...")
    
    st.stop()

st.success(f"Loaded health scores for {len(health_df)} components")

# Overall health metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_health = health_df['health_score'].mean()
    st.metric("Average Health", f"{avg_health:.1f}")

with col2:
    min_health = health_df['health_score'].min()
    st.metric("Minimum Health", f"{min_health:.1f}")

with col3:
    critical_count = (health_df['health_score'] < 40).sum()
    st.metric("Critical Components", critical_count)

with col4:
    degraded_count = (health_df['health_score'] < 60).sum()
    st.metric("Degraded Components", degraded_count)

st.markdown("---")

# Health score distribution
st.markdown("### Health Score Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    # Horizontal bar chart
    health_sorted = health_df.sort_values('health_score')
    
    # Color by status
    colors = health_sorted['health_score'].apply(
        lambda x: '#28a745' if x >= 75 else '#ffc107' if x >= 60 else '#dc3545'
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=health_sorted['component'],
        x=health_sorted['health_score'],
        orientation='h',
        marker=dict(color=colors),
        text=health_sorted['health_score'].round(1),
        textposition='auto'
    ))
    
    # Add threshold lines
    fig.add_vline(x=60, line_dash="dash", line_color="orange",
                  annotation_text="Warning")
    fig.add_vline(x=40, line_dash="dash", line_color="red",
                  annotation_text="Critical")
    
    fig.update_layout(
        title="Component Health Scores",
        xaxis_title="Health Score",
        yaxis_title="Component",
        height=max(400, len(health_df) * 30),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Status Breakdown")
    
    status_counts = health_df['status'].value_counts()
    
    for status, count in status_counts.items():
        pct = count / len(health_df) * 100
        
        if status == "CRITICAL":
            st.error(f"{status}: {count} ({pct:.1f}%)")
        elif status == "POOR" or status == "DEGRADED":
            st.warning(f"{status}: {count} ({pct:.1f}%)")
        else:
            st.success(f"{status}: {count} ({pct:.1f}%)")

st.markdown("---")

# Component details
st.markdown("### Component Details")

# Filter options
status_filter = st.multiselect(
    "Filter by Status",
    options=health_df['status'].unique(),
    default=health_df['status'].unique()
)

filtered_df = health_df[health_df['status'].isin(status_filter)]

# Sort options
sort_by = st.selectbox(
    "Sort by",
    ["Health Score (Low to High)", "Health Score (High to Low)", "Component Name"]
)

if sort_by == "Health Score (Low to High)":
    filtered_df = filtered_df.sort_values('health_score', ascending=True)
elif sort_by == "Health Score (High to Low)":
    filtered_df = filtered_df.sort_values('health_score', ascending=False)
else:
    filtered_df = filtered_df.sort_values('component')

# Display table
st.dataframe(
    filtered_df,
    column_config={
        "component": "Component",
        "health_score": st.column_config.ProgressColumn(
            "Health Score",
            format="%.1f",
            min_value=0,
            max_value=100
        ),
        "status": "Status",
        "anomaly_score": st.column_config.NumberColumn(
            "Anomaly Score",
            format="%.3f"
        ),
        "degradation_rate": st.column_config.NumberColumn(
            "Degradation Rate",
            format="%.3f"
        )
    },
    hide_index=True,
    use_container_width=True
)

st.markdown("---")

# Health trends over time
st.markdown("### Health Trends")

st.info("Historical health trend visualization would appear here. Requires storing health scores over time.")

# Individual component inspection
st.markdown("### Component Inspector")

selected_component = st.selectbox(
    "Select Component",
    health_df['component'].tolist()
)

component_data = health_df[health_df['component'] == selected_component].iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Health Score", f"{component_data['health_score']:.1f}")
    st.metric("Status", component_data['status'])

with col2:
    st.metric("Anomaly Score", f"{component_data['anomaly_score']:.3f}")
    if 'degradation_rate' in component_data:
        st.metric("Degradation Rate", f"{component_data['degradation_rate']:.3f}")

with col3:
    if 'last_updated' in component_data:
        st.metric("Last Updated", component_data['last_updated'])

# Recommendations
st.markdown("#### Recommendations")

if component_data['health_score'] < 40:
    st.error("CRITICAL: Immediate maintenance required")
elif component_data['health_score'] < 60:
    st.warning("WARNING: Schedule maintenance soon")
elif component_data['health_score'] < 75:
    st.info("Monitor closely, maintenance recommended")
else:
    st.success("Component operating normally")

# Export
st.markdown("---")
st.markdown("### Export Health Data")

csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Health Scores",
    data=csv,
    file_name=f"health_scores_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)