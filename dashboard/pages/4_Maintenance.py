"""
Maintenance Scheduling Page
Optimized maintenance task scheduling and management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import PROCESSED_DATA_DIR
from dashboard.theme import (
    apply_theme, render_logo, COLORS, CHART_COLORS,
    apply_plotly_theme, render_section_header, get_health_color
)

st.set_page_config(
    page_title="CONSTELLATION | Maintenance",
    page_icon="□",
    layout="wide"
)

apply_theme()


# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data(ttl=300)
def load_maintenance_schedule():
    """Load maintenance schedule data."""
    results_dir = PROCESSED_DATA_DIR / "analysis_results"

    if not results_dir.exists():
        return None

    schedule_files = list(results_dir.glob("maintenance_schedule_*.csv"))
    if not schedule_files:
        return None

    latest = max(schedule_files, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(latest)

    # Convert datetime columns
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'])
    if 'end_time' in df.columns:
        df['end_time'] = pd.to_datetime(df['end_time'])

    return df


def generate_demo_schedule():
    """Generate demo maintenance schedule."""
    now = datetime.now()

    tasks = [
        {
            'task_id': 'MAINT-001',
            'component': 'RWA_4',
            'task_type': 'Preventive',
            'description': 'Bearing lubrication check',
            'start_time': now + timedelta(days=1, hours=2),
            'end_time': now + timedelta(days=1, hours=4),
            'duration_hours': 2.0,
            'urgency': 0.75,
            'impact': 0.6,
            'risk_score': 0.68,
            'status': 'Scheduled',
            'priority': 'High'
        },
        {
            'task_id': 'MAINT-002',
            'component': 'KUBAND',
            'task_type': 'Corrective',
            'description': 'Signal calibration',
            'start_time': now + timedelta(days=2, hours=6),
            'end_time': now + timedelta(days=2, hours=9),
            'duration_hours': 3.0,
            'urgency': 0.65,
            'impact': 0.7,
            'risk_score': 0.67,
            'status': 'Scheduled',
            'priority': 'High'
        },
        {
            'task_id': 'MAINT-003',
            'component': 'CMG_1',
            'task_type': 'Preventive',
            'description': 'Momentum wheel inspection',
            'start_time': now + timedelta(days=3, hours=8),
            'end_time': now + timedelta(days=3, hours=12),
            'duration_hours': 4.0,
            'urgency': 0.45,
            'impact': 0.8,
            'risk_score': 0.54,
            'status': 'Scheduled',
            'priority': 'Medium'
        },
        {
            'task_id': 'MAINT-004',
            'component': 'ANTENNA_AZ',
            'task_type': 'Predictive',
            'description': 'Azimuth motor service',
            'start_time': now + timedelta(days=5, hours=10),
            'end_time': now + timedelta(days=5, hours=14),
            'duration_hours': 4.0,
            'urgency': 0.35,
            'impact': 0.5,
            'risk_score': 0.42,
            'status': 'Scheduled',
            'priority': 'Medium'
        },
        {
            'task_id': 'MAINT-005',
            'component': 'RWA_2',
            'task_type': 'Preventive',
            'description': 'Speed sensor calibration',
            'start_time': now + timedelta(days=7, hours=4),
            'end_time': now + timedelta(days=7, hours=6),
            'duration_hours': 2.0,
            'urgency': 0.25,
            'impact': 0.4,
            'risk_score': 0.32,
            'status': 'Planned',
            'priority': 'Low'
        },
        {
            'task_id': 'MAINT-006',
            'component': 'CMG_3',
            'task_type': 'Predictive',
            'description': 'Thermal analysis review',
            'start_time': now + timedelta(days=10, hours=9),
            'end_time': now + timedelta(days=10, hours=11),
            'duration_hours': 2.0,
            'urgency': 0.2,
            'impact': 0.3,
            'risk_score': 0.25,
            'status': 'Planned',
            'priority': 'Low'
        }
    ]

    return pd.DataFrame(tasks)


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    render_logo()
    st.markdown("---")

    st.markdown("### Schedule View")

    view_mode = st.radio(
        "Display",
        ["Timeline", "Calendar", "Task List"],
        index=0
    )

    st.markdown("---")

    st.markdown("### Filters")

    priority_filter = st.multiselect(
        "Priority",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )

    task_type_filter = st.multiselect(
        "Task Type",
        ["Preventive", "Corrective", "Predictive"],
        default=["Preventive", "Corrective", "Predictive"]
    )



# ============================================================================
# Main Content
# ============================================================================

st.markdown("""
<div style="margin-bottom: 30px;">
    <h1>Maintenance Scheduling</h1>
    <p style="color: var(--text-secondary);">
        Optimized maintenance task scheduling and resource management
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
schedule_df = load_maintenance_schedule()

if schedule_df is None:
    schedule_df = generate_demo_schedule()
    using_demo = True
else:
    using_demo = False

# Apply filters
if 'priority' in schedule_df.columns:
    schedule_df = schedule_df[schedule_df['priority'].isin(priority_filter)]
if 'task_type' in schedule_df.columns:
    schedule_df = schedule_df[schedule_df['task_type'].isin(task_type_filter)]


# ============================================================================
# Key Metrics
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

total_tasks = len(schedule_df)
total_hours = schedule_df['duration_hours'].sum() if 'duration_hours' in schedule_df.columns else 0
high_priority = (schedule_df['priority'] == 'High').sum() if 'priority' in schedule_df.columns else 0
avg_risk = schedule_df['risk_score'].mean() if 'risk_score' in schedule_df.columns else 0

# Calculate schedule span
if 'start_time' in schedule_df.columns and len(schedule_df) > 0:
    span_days = (schedule_df['end_time'].max() - schedule_df['start_time'].min()).days
else:
    span_days = 0

with col1:
    st.metric("TOTAL TASKS", total_tasks)

with col2:
    st.metric("TOTAL HOURS", f"{total_hours:.1f}")

with col3:
    st.metric("HIGH PRIORITY", high_priority, delta="Urgent" if high_priority > 0 else None, delta_color="off")

with col4:
    st.metric("AVG RISK", f"{avg_risk:.2f}")

with col5:
    st.metric("SCHEDULE SPAN", f"{span_days} days")


if using_demo:
    st.info("Displaying demo schedule. Run system analysis to generate optimized maintenance schedule.")


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# Timeline View
# ============================================================================

if view_mode == "Timeline":
    render_section_header("Maintenance Timeline", "")

    if len(schedule_df) > 0 and 'start_time' in schedule_df.columns:
        # Create Gantt chart data
        gantt_data = []

        for _, row in schedule_df.iterrows():
            # Color by priority
            color = {
                'High': COLORS['error'],
                'Medium': COLORS['warning'],
                'Low': COLORS['primary']
            }.get(row.get('priority', 'Medium'), COLORS['primary'])

            gantt_data.append(dict(
                Task=row['task_id'],
                Start=row['start_time'],
                Finish=row['end_time'],
                Resource=row.get('component', 'Unknown'),
                Description=row.get('description', ''),
                Priority=row.get('priority', 'Medium')
            ))

        # Create figure
        gantt_df = pd.DataFrame(gantt_data)

        fig = px.timeline(
            gantt_df,
            x_start="Start",
            x_end="Finish",
            y="Resource",
            color="Priority",
            color_discrete_map={
                'High': COLORS['error'],
                'Medium': COLORS['warning'],
                'Low': COLORS['success']
            },
            hover_data=['Task', 'Description'],
            title=None
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text_secondary']),
            height=max(350, len(schedule_df) * 50),
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(
                gridcolor=COLORS['grid'],
                linecolor=COLORS['border'],
            ),
            yaxis=dict(
                gridcolor=COLORS['grid'],
                linecolor=COLORS['border'],
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0)'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No scheduled tasks to display")


# ============================================================================
# Calendar View
# ============================================================================

elif view_mode == "Calendar":
    render_section_header("Maintenance Calendar", "")

    if len(schedule_df) > 0 and 'start_time' in schedule_df.columns:
        # Group by date
        schedule_df['date'] = schedule_df['start_time'].dt.date
        daily_tasks = schedule_df.groupby('date').agg({
            'task_id': 'count',
            'duration_hours': 'sum',
            'risk_score': 'mean'
        }).reset_index()
        daily_tasks.columns = ['date', 'task_count', 'total_hours', 'avg_risk']

        # Create calendar heatmap
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=daily_tasks['date'],
            y=daily_tasks['task_count'],
            marker=dict(
                color=daily_tasks['avg_risk'],
                colorscale=[
                    [0, COLORS['success']],
                    [0.5, COLORS['warning']],
                    [1, COLORS['error']]
                ],
                colorbar=dict(title="Risk")
            ),
            text=daily_tasks['task_count'],
            textposition='auto',
            hovertemplate='Date: %{x}<br>Tasks: %{y}<br>Risk: %{marker.color:.2f}<extra></extra>'
        ))

        fig = apply_plotly_theme(fig)
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Date",
            yaxis_title="Number of Tasks"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Daily breakdown
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <div style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 15px;">
                    Daily Breakdown
                </div>
            """, unsafe_allow_html=True)

            for _, row in daily_tasks.iterrows():
                risk_color = get_health_color(100 - row['avg_risk'] * 100)
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--border);">
                    <span style="color: var(--text-primary);">{row['date']}</span>
                    <div>
                        <span style="color: var(--text-secondary); margin-right: 15px;">{row['task_count']} tasks</span>
                        <span style="color: {risk_color};">{row['total_hours']:.1f} hrs</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # Task type breakdown
            if 'task_type' in schedule_df.columns:
                type_counts = schedule_df['task_type'].value_counts()

                fig_pie = go.Figure()

                fig_pie.add_trace(go.Pie(
                    labels=type_counts.index,
                    values=type_counts.values,
                    hole=0.4,
                    marker=dict(colors=[COLORS['primary'], COLORS['secondary'], COLORS['accent']]),
                    textinfo='label+percent'
                ))

                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=COLORS['text_secondary']),
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False
                )

                st.plotly_chart(fig_pie, use_container_width=True)


# ============================================================================
# Task List View
# ============================================================================

elif view_mode == "Task List":
    render_section_header("Maintenance Tasks", "")

    if len(schedule_df) > 0:
        # Sort options
        sort_col, filter_col = st.columns([1, 1])

        with sort_col:
            sort_by = st.selectbox(
                "Sort by",
                ["Start Time", "Risk Score (High to Low)", "Priority", "Duration"],
                index=0
            )

        with filter_col:
            status_filter = st.selectbox(
                "Status",
                ["All", "Scheduled", "Planned", "Completed"],
                index=0
            )

        # Apply sorting
        if sort_by == "Start Time":
            display_df = schedule_df.sort_values('start_time')
        elif sort_by == "Risk Score (High to Low)":
            display_df = schedule_df.sort_values('risk_score', ascending=False)
        elif sort_by == "Priority":
            priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
            display_df = schedule_df.copy()
            display_df['priority_order'] = display_df['priority'].map(priority_order)
            display_df = display_df.sort_values('priority_order').drop('priority_order', axis=1)
        else:
            display_df = schedule_df.sort_values('duration_hours', ascending=False)

        # Apply status filter
        if status_filter != "All" and 'status' in display_df.columns:
            display_df = display_df[display_df['status'] == status_filter]

        # Display as cards
        for _, task in display_df.iterrows():
            priority_color = {
                'High': COLORS['error'],
                'Medium': COLORS['warning'],
                'Low': COLORS['success']
            }.get(task.get('priority', 'Medium'), COLORS['primary'])

            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {priority_color}; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="color: var(--text-primary); font-weight: 600; font-size: 1.1rem;">
                                {task['task_id']}
                            </span>
                            <span style="background: {priority_color}; color: var(--bg-dark); padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                                {task.get('priority', 'Medium')}
                            </span>
                            <span style="color: var(--text-muted); font-size: 0.8rem;">
                                {task.get('task_type', 'Maintenance')}
                            </span>
                        </div>
                        <div style="color: var(--text-secondary); margin-top: 8px;">
                            {task.get('description', 'No description')}
                        </div>
                        <div style="color: var(--text-muted); font-size: 0.85rem; margin-top: 8px;">
                            Component: <strong>{task.get('component', 'N/A')}</strong>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: var(--primary); font-family: 'JetBrains Mono', monospace;">
                            {task['start_time'].strftime('%b %d, %H:%M') if pd.notna(task.get('start_time')) else 'TBD'}
                        </div>
                        <div style="color: var(--text-muted); font-size: 0.8rem; margin-top: 4px;">
                            Duration: {task.get('duration_hours', 0):.1f} hrs
                        </div>
                        <div style="color: var(--text-muted); font-size: 0.8rem; margin-top: 4px;">
                            Risk: {task.get('risk_score', 0):.2f}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# Risk Analysis Section
# ============================================================================

st.markdown("<br>", unsafe_allow_html=True)
render_section_header("Risk Analysis", "")

col1, col2 = st.columns(2)

with col1:
    # Urgency vs Impact scatter
    if 'urgency' in schedule_df.columns and 'impact' in schedule_df.columns:
        fig_scatter = go.Figure()

        fig_scatter.add_trace(go.Scatter(
            x=schedule_df['urgency'],
            y=schedule_df['impact'],
            mode='markers+text',
            marker=dict(
                size=schedule_df['duration_hours'] * 8,
                color=schedule_df['risk_score'],
                colorscale=[
                    [0, COLORS['success']],
                    [0.5, COLORS['warning']],
                    [1, COLORS['error']]
                ],
                colorbar=dict(title="Risk"),
                line=dict(color=COLORS['border'], width=1)
            ),
            text=schedule_df['task_id'],
            textposition='top center',
            textfont=dict(size=10, color=COLORS['text_muted']),
            hovertemplate='<b>%{text}</b><br>Urgency: %{x:.2f}<br>Impact: %{y:.2f}<extra></extra>'
        ))

        # Add quadrant lines
        fig_scatter.add_hline(y=0.5, line_dash="dot", line_color=COLORS['border'])
        fig_scatter.add_vline(x=0.5, line_dash="dot", line_color=COLORS['border'])

        fig_scatter = apply_plotly_theme(fig_scatter)
        fig_scatter.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Urgency",
            yaxis_title="Impact",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    # Risk distribution
    if 'risk_score' in schedule_df.columns:
        fig_risk = go.Figure()

        fig_risk.add_trace(go.Histogram(
            x=schedule_df['risk_score'],
            nbinsx=20,
            marker=dict(
                color=COLORS['primary'],
                line=dict(color=COLORS['border'], width=1)
            )
        ))

        # Add threshold lines
        fig_risk.add_vline(x=0.3, line_dash="dot", line_color=COLORS['success'],
                         annotation_text="Low")
        fig_risk.add_vline(x=0.6, line_dash="dot", line_color=COLORS['warning'],
                         annotation_text="Medium")

        fig_risk = apply_plotly_theme(fig_risk)
        fig_risk.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Risk Score",
            yaxis_title="Task Count"
        )

        st.plotly_chart(fig_risk, use_container_width=True)


# ============================================================================
# Actions Section
# ============================================================================

st.markdown("<br>", unsafe_allow_html=True)
render_section_header("Schedule Actions", "")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Optimize Schedule", use_container_width=True):
        st.info("Schedule optimization would run the maintenance scheduler")

with col2:
    if st.button("Add Task", use_container_width=True):
        st.info("Task creation form would appear here")

with col3:
    csv = schedule_df.to_csv(index=False)
    st.download_button(
        label="Export Schedule",
        data=csv,
        file_name=f"maintenance_schedule_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col4:
    if st.button("Generate Report", use_container_width=True):
        st.info("Maintenance report generation coming soon")


# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; color: var(--text-muted); padding: 20px; border-top: 1px solid var(--border);">
    <div style="font-size: 0.75rem;">
        Schedule optimization powered by PuLP linear programming | Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
</div>
""", unsafe_allow_html=True)
