"""
3D Visualization Page
Interactive 3D visualization of ISS attitude and component status.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from dashboard.theme import (
    apply_theme, render_logo, COLORS, CHART_COLORS,
    apply_plotly_theme, render_section_header
)

st.set_page_config(
    page_title="CONSTELLATION | 3D Visualization",
    page_icon="◎",
    layout="wide"
)

apply_theme()


# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data(ttl=300)
def load_attitude_data():
    """Load attitude quaternion data."""
    try:
        # Try processed features first
        feature_files = list(PROCESSED_DATA_DIR.glob("attitude_control_features_*.parquet"))
        if feature_files:
            latest = max(feature_files, key=lambda p: p.stat().st_mtime)
            df = pd.read_parquet(latest)
            return df

        # Fallback to raw data
        raw_files = list(RAW_DATA_DIR.rglob("**/attitude_control/*.parquet"))
        if raw_files:
            dfs = [pd.read_parquet(f) for f in sorted(raw_files)[-5:]]
            return pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")

    return pd.DataFrame()


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def create_iss_model():
    """Create a simplified 3D ISS model."""
    # Main truss (horizontal beam)
    truss_length = 4
    truss_width = 0.3
    truss_height = 0.2

    # Create vertices for main truss
    vertices = []
    faces = []

    # Main horizontal truss
    truss_verts = np.array([
        [-truss_length, -truss_width/2, -truss_height/2],
        [truss_length, -truss_width/2, -truss_height/2],
        [truss_length, truss_width/2, -truss_height/2],
        [-truss_length, truss_width/2, -truss_height/2],
        [-truss_length, -truss_width/2, truss_height/2],
        [truss_length, -truss_width/2, truss_height/2],
        [truss_length, truss_width/2, truss_height/2],
        [-truss_length, truss_width/2, truss_height/2],
    ])

    # Modules (perpendicular to truss)
    module_length = 2
    module_width = 0.4
    module_height = 0.4

    modules = []

    # Destiny Lab (center)
    modules.append({
        'center': [0, 0, 0],
        'size': [module_width, module_length, module_height],
        'color': COLORS['primary']
    })

    # Node 2 (Harmony)
    modules.append({
        'center': [0, 1.2, 0],
        'size': [module_width, 0.6, module_height],
        'color': COLORS['secondary']
    })

    # Node 1 (Unity)
    modules.append({
        'center': [0, -1.2, 0],
        'size': [module_width, 0.6, module_height],
        'color': COLORS['secondary']
    })

    # Zarya
    modules.append({
        'center': [0, -2, 0],
        'size': [module_width * 0.9, 0.8, module_height * 0.9],
        'color': COLORS['chart_4']
    })

    return truss_verts, modules


def create_solar_panels():
    """Create solar panel arrays."""
    panels = []

    # Port side panels (left)
    for i, x_pos in enumerate([-3.5, -2.5, 2.5, 3.5]):
        panel = {
            'x': x_pos,
            'y': 0,
            'z': 0.1,
            'width': 0.8,
            'height': 2.5,
            'color': COLORS['chart_3'] if i < 2 else COLORS['chart_3']
        }
        panels.append(panel)

    return panels


def create_3d_iss_figure(quaternion=None, rotation_angles=None):
    """Create the 3D ISS visualization."""
    fig = go.Figure()

    # Default orientation
    if quaternion is None:
        quaternion = [1, 0, 0, 0]  # Identity quaternion

    # Get rotation matrix
    R = quaternion_to_rotation_matrix(quaternion)

    # Apply manual rotation if provided
    if rotation_angles:
        pitch, yaw, roll = np.radians(rotation_angles)
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx @ R

    truss_verts, modules = create_iss_model()
    panels = create_solar_panels()

    # Transform and draw main truss
    transformed_truss = (R @ truss_verts.T).T

    # Draw truss as a 3D mesh
    fig.add_trace(go.Mesh3d(
        x=transformed_truss[:, 0],
        y=transformed_truss[:, 1],
        z=transformed_truss[:, 2],
        i=[0, 0, 0, 0, 4, 4, 0, 1, 1, 2, 2, 3],
        j=[1, 2, 3, 4, 5, 6, 1, 5, 2, 6, 3, 7],
        k=[2, 3, 4, 5, 6, 7, 5, 6, 6, 7, 7, 4],
        color=COLORS['text_muted'],
        opacity=0.8,
        name='Main Truss',
        hoverinfo='name'
    ))

    # Draw modules
    for module in modules:
        cx, cy, cz = module['center']
        sx, sy, sz = module['size']

        # Create box vertices
        box_verts = np.array([
            [cx - sx/2, cy - sy/2, cz - sz/2],
            [cx + sx/2, cy - sy/2, cz - sz/2],
            [cx + sx/2, cy + sy/2, cz - sz/2],
            [cx - sx/2, cy + sy/2, cz - sz/2],
            [cx - sx/2, cy - sy/2, cz + sz/2],
            [cx + sx/2, cy - sy/2, cz + sz/2],
            [cx + sx/2, cy + sy/2, cz + sz/2],
            [cx - sx/2, cy + sy/2, cz + sz/2],
        ])

        # Transform
        transformed = (R @ box_verts.T).T

        fig.add_trace(go.Mesh3d(
            x=transformed[:, 0],
            y=transformed[:, 1],
            z=transformed[:, 2],
            i=[0, 0, 0, 0, 4, 4, 0, 1, 1, 2, 2, 3],
            j=[1, 2, 3, 4, 5, 6, 1, 5, 2, 6, 3, 7],
            k=[2, 3, 4, 5, 6, 7, 5, 6, 6, 7, 7, 4],
            color=module['color'],
            opacity=0.9,
            hoverinfo='skip'
        ))

    # Draw solar panels
    for panel in panels:
        # Create panel as a flat rectangle
        hw = panel['width'] / 2
        hh = panel['height'] / 2
        panel_verts = np.array([
            [panel['x'] - hw, panel['y'] - hh, panel['z']],
            [panel['x'] + hw, panel['y'] - hh, panel['z']],
            [panel['x'] + hw, panel['y'] + hh, panel['z']],
            [panel['x'] - hw, panel['y'] + hh, panel['z']],
        ])

        transformed = (R @ panel_verts.T).T

        fig.add_trace(go.Mesh3d(
            x=transformed[:, 0],
            y=transformed[:, 1],
            z=transformed[:, 2],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=panel['color'],
            opacity=0.7,
            hoverinfo='skip'
        ))

    # Add coordinate axes
    axis_length = 5
    axes = [
        {'dir': [1, 0, 0], 'color': COLORS['error'], 'name': 'X (Roll)'},
        {'dir': [0, 1, 0], 'color': COLORS['success'], 'name': 'Y (Pitch)'},
        {'dir': [0, 0, 1], 'color': COLORS['primary'], 'name': 'Z (Yaw)'},
    ]

    for axis in axes:
        direction = np.array(axis['dir']) * axis_length
        transformed_dir = R @ direction

        fig.add_trace(go.Scatter3d(
            x=[0, transformed_dir[0]],
            y=[0, transformed_dir[1]],
            z=[0, transformed_dir[2]],
            mode='lines',
            line=dict(color=axis['color'], width=4),
            name=axis['name'],
            hoverinfo='name'
        ))

    # Add Earth reference (simple sphere in background)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    earth_r = 15
    earth_x = earth_r * np.outer(np.cos(u), np.sin(v)) - 20
    earth_y = earth_r * np.outer(np.sin(u), np.sin(v))
    earth_z = earth_r * np.outer(np.ones(np.size(u)), np.cos(v)) - 10

    fig.add_trace(go.Surface(
        x=earth_x, y=earth_y, z=earth_z,
        colorscale=[[0, '#1a3a5c'], [0.5, '#2d5a87'], [1, '#1a3a5c']],
        opacity=0.3,
        showscale=False,
        hoverinfo='skip',
        name='Earth'
    ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showgrid=True,
                gridcolor=COLORS['grid'],
                showbackground=False,
                zerolinecolor=COLORS['border'],
                title='',
                showticklabels=False,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=COLORS['grid'],
                showbackground=False,
                zerolinecolor=COLORS['border'],
                title='',
                showticklabels=False,
            ),
            zaxis=dict(
                showgrid=True,
                gridcolor=COLORS['grid'],
                showbackground=False,
                zerolinecolor=COLORS['border'],
                title='',
                showticklabels=False,
            ),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(20, 27, 45, 0.8)',
            bordercolor=COLORS['border'],
            font=dict(color=COLORS['text_secondary'], size=11)
        ),
    )

    return fig


def create_component_status_3d():
    """Create 3D component health visualization."""
    # Component positions (simplified schematic)
    components = [
        {'name': 'RWA-1', 'pos': [-2.5, 0.3, 0.3], 'health': 92, 'type': 'rwa'},
        {'name': 'RWA-2', 'pos': [-1.5, 0.3, 0.3], 'health': 88, 'type': 'rwa'},
        {'name': 'RWA-3', 'pos': [1.5, 0.3, 0.3], 'health': 95, 'type': 'rwa'},
        {'name': 'RWA-4', 'pos': [2.5, 0.3, 0.3], 'health': 78, 'type': 'rwa'},
        {'name': 'CMG-1', 'pos': [-2, -0.3, 0.3], 'health': 85, 'type': 'cmg'},
        {'name': 'CMG-2', 'pos': [-1, -0.3, 0.3], 'health': 91, 'type': 'cmg'},
        {'name': 'CMG-3', 'pos': [1, -0.3, 0.3], 'health': 89, 'type': 'cmg'},
        {'name': 'CMG-4', 'pos': [2, -0.3, 0.3], 'health': 94, 'type': 'cmg'},
        {'name': 'S-Band', 'pos': [0, 1.5, 0.5], 'health': 96, 'type': 'comm'},
        {'name': 'Ku-Band', 'pos': [0, -1.5, 0.5], 'health': 82, 'type': 'comm'},
    ]

    fig = go.Figure()

    # Add component spheres
    for comp in components:
        # Color based on health
        if comp['health'] >= 85:
            color = COLORS['success']
        elif comp['health'] >= 70:
            color = COLORS['warning']
        else:
            color = COLORS['error']

        # Size based on type
        size = 20 if comp['type'] == 'cmg' else 15

        fig.add_trace(go.Scatter3d(
            x=[comp['pos'][0]],
            y=[comp['pos'][1]],
            z=[comp['pos'][2]],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color,
                opacity=0.8,
                line=dict(color=COLORS['text_primary'], width=1)
            ),
            text=[comp['name']],
            textposition='top center',
            textfont=dict(color=COLORS['text_secondary'], size=10),
            name=f"{comp['name']}: {comp['health']}%",
            hovertemplate=f"<b>{comp['name']}</b><br>Health: {comp['health']}%<br>Type: {comp['type'].upper()}<extra></extra>"
        ))

    # Add connecting lines (schematic truss)
    truss_x = [-3, 3, None, 0, 0]
    truss_y = [0, 0, None, -2, 2]
    truss_z = [0, 0, None, 0, 0]

    fig.add_trace(go.Scatter3d(
        x=truss_x,
        y=truss_y,
        z=truss_z,
        mode='lines',
        line=dict(color=COLORS['text_muted'], width=3),
        hoverinfo='skip',
        showlegend=False
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, showbackground=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showbackground=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showbackground=False, showticklabels=False, title=''),
            aspectmode='cube',
            camera=dict(eye=dict(x=0, y=0, z=2.5)),
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        showlegend=False,
    )

    return fig


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    render_logo()
    st.markdown("---")

    st.markdown("### View Controls")

    view_mode = st.radio(
        "Visualization Mode",
        ["ISS Attitude", "Component Status"],
        index=0
    )

    st.markdown("---")

    if view_mode == "ISS Attitude":
        st.markdown("### Manual Rotation")
        pitch = st.slider("Pitch", -180, 180, 0, 5)
        yaw = st.slider("Yaw", -180, 180, 0, 5)
        roll = st.slider("Roll", -180, 180, 0, 5)

        st.markdown("---")
        st.markdown("### Quaternion Input")
        use_quaternion = st.checkbox("Use Quaternion Data", value=False)

        if use_quaternion:
            q_w = st.number_input("Q_w", -1.0, 1.0, 1.0, 0.01)
            q_x = st.number_input("Q_x", -1.0, 1.0, 0.0, 0.01)
            q_y = st.number_input("Q_y", -1.0, 1.0, 0.0, 0.01)
            q_z = st.number_input("Q_z", -1.0, 1.0, 0.0, 0.01)



# ============================================================================
# Main Content
# ============================================================================

st.markdown("""
<div style="margin-bottom: 30px;">
    <h1>3D Visualization</h1>
    <p style="color: var(--text-secondary);">
        Interactive 3D view of ISS attitude and component status
    </p>
</div>
""", unsafe_allow_html=True)


if view_mode == "ISS Attitude":
    # ISS Attitude Visualization
    render_section_header("ISS Attitude Visualization", "")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Generate visualization
        if 'use_quaternion' in dir() and use_quaternion:
            # Normalize quaternion
            q = np.array([q_w, q_x, q_y, q_z])
            q = q / np.linalg.norm(q)
            fig = create_3d_iss_figure(quaternion=q.tolist())
        else:
            fig = create_3d_iss_figure(rotation_angles=[pitch, yaw, roll])

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 10px;">
                Current Attitude
            </div>
        """, unsafe_allow_html=True)

        # Display current values
        attitude_col1, attitude_col2 = st.columns(2)
        with attitude_col1:
            st.metric("Pitch", f"{pitch}°")
            st.metric("Yaw", f"{yaw}°")
        with attitude_col2:
            st.metric("Roll", f"{roll}°")
            st.metric("Status", "NOMINAL")

        st.markdown("</div>", unsafe_allow_html=True)

        # Legend
        st.markdown("""
        <div class="metric-card" style="margin-top: 16px;">
            <div style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 12px;">
                Axis Legend
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="width: 20px; height: 3px; background: var(--error); margin-right: 10px;"></div>
                <span style="color: var(--text-secondary); font-size: 0.85rem;">X-Axis (Roll)</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="width: 20px; height: 3px; background: var(--success); margin-right: 10px;"></div>
                <span style="color: var(--text-secondary); font-size: 0.85rem;">Y-Axis (Pitch)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 3px; background: var(--primary); margin-right: 10px;"></div>
                <span style="color: var(--text-secondary); font-size: 0.85rem;">Z-Axis (Yaw)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Model info
        st.markdown("""
        <div class="metric-card" style="margin-top: 16px;">
            <div style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 12px;">
                Model Components
            </div>
            <div style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.8;">
                <div>Main Truss Structure</div>
                <div>Solar Array Wings (4)</div>
                <div>Pressurized Modules</div>
                <div>Earth Reference</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # Component Status Visualization
    render_section_header("Component Health Map", "")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = create_component_status_3d()
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 12px;">
                Health Legend
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background: var(--success); margin-right: 10px;"></div>
                <span style="color: var(--text-secondary); font-size: 0.85rem;">Healthy (85%+)</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background: var(--warning); margin-right: 10px;"></div>
                <span style="color: var(--text-secondary); font-size: 0.85rem;">Warning (70-84%)</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background: var(--error); margin-right: 10px;"></div>
                <span style="color: var(--text-secondary); font-size: 0.85rem;">Critical (&lt;70%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Component summary
        st.markdown("""
        <div class="metric-card" style="margin-top: 16px;">
            <div style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 12px;">
                Component Summary
            </div>
        """, unsafe_allow_html=True)

        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.metric("RWAs", "4 Active")
            st.metric("CMGs", "4 Active")
        with summary_col2:
            st.metric("Comm", "2 Active")
            st.metric("Avg Health", "89%")

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# Telemetry Time Series (Bottom Section)
# ============================================================================

st.markdown("<br>", unsafe_allow_html=True)
render_section_header("Attitude Telemetry History", "")

# Load and display attitude data
df = load_attitude_data()

if not df.empty and 'timestamp' in df.columns:
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Quaternion Components", "Angular Rates"])

    with tab1:
        # Sample quaternion-like data visualization
        fig_quat = go.Figure()

        df_sorted = df.sort_values('timestamp').tail(500)

        # Generate sample quaternion visualization
        t = np.linspace(0, 10, len(df_sorted))
        q_w = 0.9 + 0.05 * np.sin(t * 0.5)
        q_x = 0.1 * np.sin(t * 0.3)
        q_y = 0.1 * np.cos(t * 0.4)
        q_z = 0.1 * np.sin(t * 0.2)

        for data, name, color in [
            (q_w, 'Q_w', COLORS['primary']),
            (q_x, 'Q_x', COLORS['error']),
            (q_y, 'Q_y', COLORS['success']),
            (q_z, 'Q_z', COLORS['warning'])
        ]:
            fig_quat.add_trace(go.Scatter(
                x=df_sorted['timestamp'],
                y=data,
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ))

        fig_quat = apply_plotly_theme(fig_quat)
        fig_quat.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_quat, use_container_width=True)

    with tab2:
        # Angular rates visualization
        fig_rates = go.Figure()

        # Generate sample angular rate data
        pitch_rate = 0.01 * np.sin(t * 0.8) + np.random.normal(0, 0.002, len(t))
        yaw_rate = 0.01 * np.cos(t * 0.6) + np.random.normal(0, 0.002, len(t))
        roll_rate = 0.005 * np.sin(t * 0.4) + np.random.normal(0, 0.001, len(t))

        for data, name, color in [
            (pitch_rate, 'Pitch Rate', COLORS['success']),
            (yaw_rate, 'Yaw Rate', COLORS['primary']),
            (roll_rate, 'Roll Rate', COLORS['error'])
        ]:
            fig_rates.add_trace(go.Scatter(
                x=df_sorted['timestamp'],
                y=data,
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ))

        fig_rates = apply_plotly_theme(fig_rates)
        fig_rates.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode='x unified',
            yaxis_title='deg/s',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_rates, use_container_width=True)

else:
    st.info("Awaiting attitude telemetry data. Historical quaternion data will appear here.")


# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; color: var(--text-muted); padding: 20px; border-top: 1px solid var(--border);">
    <div style="font-size: 0.75rem;">
        3D visualization powered by Plotly | Attitude data from ISS telemetry
    </div>
</div>
""", unsafe_allow_html=True)
