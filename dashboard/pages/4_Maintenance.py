"""
Maintenance Scheduling Page
View and manage maintenance tasks.
"""

import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import PROCESSED_DATA_DIR

st.set_page_config(page_title="Maintenance Scheduling", page_icon="🔧", layout="wide")

st.title("Maintenance Scheduling")
st.markdown("Optimized maintenance task scheduling")

# Load maintenance schedule
@st.cache_data(ttl=300)
def load_maintenance_schedule():
    """Load latest maintenance schedule."""
    results_dir = PROCESSED_DATA_DIR / "analysis_results"
    
    if not results_dir.exists():
        return None
    
    schedule_files = list(results_dir.glob("maintenance_schedule_*.csv"))
    
    if not schedule_files:
        return None
    
    latest_file = max(schedule_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    # Convert datetime columns
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    return df

schedule_df = load_maintenance_schedule()

if schedule_df is None or len(schedule_df) == 0:
    st.info("No maintenance tasks scheduled. All components operating normally.")
    
    if st.button("Generate Schedule"):
        st.info("Schedule generation would be triggered here")
    
    st.stop()

st.success(f"Loaded {len(schedule_df)} scheduled maintenance tasks")

# Schedule summary
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Tasks", len(schedule_df))

with col2:
    total_hours = schedule_df['duration_hours'].sum()
    st.metric("Total Duration", f"{total_hours:.1f} hrs")

with col3:
    avg_urgency = schedule_df['urgency'].mean()
    st.metric("Avg Urgency", f"{avg_urgency:.2f}")

with col4:
    span_days = (schedule_df['end_time'].max() - schedule_df['start_time'].min()).days
    st.metric("Schedule Span", f"{span_days} days")

st.markdown("---")

# Gantt chart
st.markdown("### Schedule Timeline")

# Prepare data for Gantt chart
gantt_data = []

for idx, row in schedule_df.iterrows():
    gantt_data.append(dict(
        Task=row['task_id'],
        Start=row['start_time'],
        Finish=row['end_time'],
        Resource=row['component']
    ))

if gantt_data:
    fig = ff.create_gantt(
        gantt_data,
        index_col='Resource',
        show_colorbar=True,
        group_tasks=True,
        showgrid_x=True,
        showgrid_y=True,
        height=max(400, len(schedule_df) * 40)
    )
    
    fig.update_layout(
        title="Maintenance Schedule (Gantt Chart)",
        xaxis_title="Time"
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Task priority analysis
st.markdown("### Task Priority Analysis")

col1, col2 = st.columns(2)

with col1:
    # Risk score distribution
    fig = px.histogram(
        schedule_df,
        x='risk_score',
        nbins=20,
        title="Risk Score Distribution",
        labels={'risk_score': 'Risk Score', 'count': 'Number of Tasks'}
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Urgency vs Impact scatter
    fig = px.scatter(
        schedule_df,
        x='urgency',
        y='impact',
        size='duration_hours',
        color='risk_score',
        hover_data=['task_id', 'component'],
        title="Urgency vs Impact",
        labels={'urgency': 'Urgency', 'impact': 'Impact'}
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Task list
st.markdown("### Task Details")

# Filter options
priority_filter = st.select_slider(
    "Filter by Risk Score",
    options=["All", "Low (< 0.3)", "Medium (0.3-0.6)", "High (> 0.6)"],
    value="All"
)

filtered_schedule = schedule_df.copy()

if priority_filter == "Low (< 0.3)":
    filtered_schedule = filtered_schedule[filtered_schedule['risk_score'] < 0.3]
elif priority_filter == "Medium (0.3-0.6)":
    filtered_schedule = filtered_schedule[
        (filtered_schedule['risk_score'] >= 0.3) & 
        (filtered_schedule['risk_score'] <= 0.6)
    ]
elif priority_filter == "High (> 0.6)":
    filtered_schedule = filtered_schedule[filtered_schedule['risk_score'] > 0.6]

# Display table
st.dataframe(
    filtered_schedule,
    column_config={
        "task_id": "Task ID",
        "component": "Component",
        "start_time": st.column_config.DatetimeColumn(
            "Start Time",
            format="MM/DD/YY HH:mm"
        ),
        "end_time": st.column_config.DatetimeColumn(
            "End Time",
            format="MM/DD/YY HH:mm"
        ),
        "duration_hours": st.column_config.NumberColumn(
            "Duration (hrs)",
            format="%.1f"
        ),
        "urgency": st.column_config.ProgressColumn(
            "Urgency",
            format="%.2f",
            min_value=0,
            max_value=1
        ),
        "impact": st.column_config.ProgressColumn(
            "Impact",
            format="%.2f",
            min_value=0,
            max_value=1
        ),
        "risk_score": st.column_config.ProgressColumn(
            "Risk Score",
            format="%.2f",
            min_value=0,
            max_value=1
        )
    },
    hide_index=True,
    use_container_width=True
)

st.markdown("---")

# Task actions
st.markdown("### Task Management")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Optimize Schedule", use_container_width=True):
        st.info("Re-optimization would be triggered here")

with col2:
    if st.button("Add Manual Task", use_container_width=True):
        st.info("Task creation form would appear here")

with col3:
    if st.button("Export Schedule", use_container_width=True):
        csv = filtered_schedule.to_csv(index=False)
        st.download_button(
            label="Download Schedule",
            data=csv,
            file_name=f"maintenance_schedule_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Calendar view
st.markdown("---")
st.markdown("### Calendar View")

# Group tasks by day
schedule_df['date'] = schedule_df['start_time'].dt.date
daily_tasks = schedule_df.groupby('date').size()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### Tasks per Day")
    for date, count in daily_tasks.items():
        st.write(f"{date}: {count} task(s)")

with col2:
    fig = px.bar(
        x=daily_tasks.index,
        y=daily_tasks.values,
        title="Daily Task Distribution",
        labels={'x': 'Date', 'y': 'Number of Tasks'}
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)