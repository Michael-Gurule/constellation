"""
CONSTELLATION Dashboard Theme
Dark space-themed styling for the satellite monitoring dashboard.
"""

import streamlit as st

# Color palette - Space theme
COLORS = {
    # Primary colors
    "background_dark": "#0a0e17",
    "background_medium": "#0f1629",
    "background_light": "#1a2035",
    "background_card": "#141b2d",

    # Accent colors
    "primary": "#00d4ff",      # Cyan - primary actions
    "secondary": "#7b68ee",    # Purple - secondary elements
    "accent": "#00ff88",       # Green - success/positive

    # Status colors
    "success": "#00ff88",
    "warning": "#ffaa00",
    "error": "#ff4757",
    "critical": "#ff0055",
    "info": "#00d4ff",

    # Health score colors
    "health_excellent": "#00ff88",
    "health_good": "#7bed9f",
    "health_fair": "#ffaa00",
    "health_poor": "#ff6b6b",
    "health_critical": "#ff0055",

    # Text colors
    "text_primary": "#ffffff",
    "text_secondary": "#8892b0",
    "text_muted": "#5a6a8a",

    # Chart colors
    "chart_1": "#00d4ff",
    "chart_2": "#7b68ee",
    "chart_3": "#00ff88",
    "chart_4": "#ffaa00",
    "chart_5": "#ff6b6b",
    "chart_6": "#ff00ff",

    # Grid/border
    "border": "#2a3f5f",
    "grid": "#1e3a5f",
}

# Chart color sequence for Plotly
CHART_COLORS = [
    COLORS["chart_1"],
    COLORS["chart_2"],
    COLORS["chart_3"],
    COLORS["chart_4"],
    COLORS["chart_5"],
    COLORS["chart_6"],
]

def get_health_color(score: float) -> str:
    """Get color based on health score (0-100)."""
    if score >= 85:
        return COLORS["health_excellent"]
    elif score >= 70:
        return COLORS["health_good"]
    elif score >= 55:
        return COLORS["health_fair"]
    elif score >= 40:
        return COLORS["health_poor"]
    else:
        return COLORS["health_critical"]

def get_status_color(status: str) -> str:
    """Get color based on status string."""
    status_lower = status.lower()
    if status_lower in ["excellent", "nominal", "active", "online"]:
        return COLORS["success"]
    elif status_lower in ["good", "normal"]:
        return COLORS["health_good"]
    elif status_lower in ["fair", "warning", "degraded"]:
        return COLORS["warning"]
    elif status_lower in ["poor", "error"]:
        return COLORS["error"]
    elif status_lower in ["critical", "offline", "failed"]:
        return COLORS["critical"]
    return COLORS["text_secondary"]

def get_anomaly_color(score: float) -> str:
    """Get color based on anomaly score (0-1)."""
    if score < 0.5:
        return COLORS["success"]
    elif score < 0.8:
        return COLORS["warning"]
    elif score < 0.95:
        return COLORS["error"]
    else:
        return COLORS["critical"]

# Plotly theme template
PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "Inter, SF Pro Display, -apple-system, sans-serif",
            "color": COLORS["text_secondary"],
            "size": 12,
        },
        "title": {
            "font": {
                "color": COLORS["text_primary"],
                "size": 16,
            }
        },
        "xaxis": {
            "gridcolor": COLORS["grid"],
            "linecolor": COLORS["border"],
            "tickcolor": COLORS["text_muted"],
            "zerolinecolor": COLORS["border"],
        },
        "yaxis": {
            "gridcolor": COLORS["grid"],
            "linecolor": COLORS["border"],
            "tickcolor": COLORS["text_muted"],
            "zerolinecolor": COLORS["border"],
        },
        "colorway": CHART_COLORS,
        "hoverlabel": {
            "bgcolor": COLORS["background_card"],
            "bordercolor": COLORS["border"],
            "font": {"color": COLORS["text_primary"]},
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": COLORS["text_secondary"]},
        },
    }
}

def apply_theme():
    """Apply the dark space theme to Streamlit."""
    st.markdown(f"""
    <style>
        /* Import fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Root variables */
        :root {{
            --bg-dark: {COLORS["background_dark"]};
            --bg-medium: {COLORS["background_medium"]};
            --bg-light: {COLORS["background_light"]};
            --bg-card: {COLORS["background_card"]};
            --primary: {COLORS["primary"]};
            --secondary: {COLORS["secondary"]};
            --accent: {COLORS["accent"]};
            --success: {COLORS["success"]};
            --warning: {COLORS["warning"]};
            --error: {COLORS["error"]};
            --critical: {COLORS["critical"]};
            --text-primary: {COLORS["text_primary"]};
            --text-secondary: {COLORS["text_secondary"]};
            --text-muted: {COLORS["text_muted"]};
            --border: {COLORS["border"]};
        }}

        /* Main app background */
        .stApp {{
            background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-medium) 50%, var(--bg-dark) 100%);
            background-attachment: fixed;
        }}

        /* Add subtle stars effect */
        .stApp::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image:
                radial-gradient(2px 2px at 20px 30px, rgba(255,255,255,0.3), transparent),
                radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.2), transparent),
                radial-gradient(1px 1px at 90px 40px, rgba(255,255,255,0.4), transparent),
                radial-gradient(2px 2px at 160px 120px, rgba(255,255,255,0.2), transparent),
                radial-gradient(1px 1px at 230px 80px, rgba(255,255,255,0.3), transparent),
                radial-gradient(2px 2px at 300px 150px, rgba(255,255,255,0.15), transparent),
                radial-gradient(1px 1px at 400px 60px, rgba(255,255,255,0.25), transparent);
            background-size: 500px 200px;
            animation: twinkle 8s ease-in-out infinite;
            pointer-events: none;
            z-index: 0;
        }}

        @keyframes twinkle {{
            0%, 100% {{ opacity: 0.5; }}
            50% {{ opacity: 1; }}
        }}

        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, var(--bg-dark) 0%, var(--bg-medium) 100%);
            border-right: 1px solid var(--border);
        }}

        [data-testid="stSidebar"] .stMarkdown {{
            color: var(--text-secondary);
        }}

        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Inter', sans-serif !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }}

        h1 {{
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        /* Text */
        p, span, label, .stMarkdown {{
            font-family: 'Inter', sans-serif !important;
            color: var(--text-secondary);
        }}

        /* Metric cards */
        [data-testid="stMetric"] {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}

        [data-testid="stMetricLabel"] {{
            color: var(--text-muted) !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        [data-testid="stMetricValue"] {{
            color: var(--text-primary) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.8rem !important;
            font-weight: 600 !important;
        }}

        [data-testid="stMetricDelta"] {{
            font-family: 'JetBrains Mono', monospace !important;
        }}

        /* Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: var(--bg-dark);
            border: none;
            border-radius: 8px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(0, 212, 255, 0.5);
        }}

        /* Secondary buttons */
        .stButton > button[kind="secondary"] {{
            background: transparent;
            border: 1px solid var(--primary);
            color: var(--primary);
        }}

        /* Selectbox and inputs */
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stTextInput > div > div > input {{
            background-color: var(--bg-card) !important;
            border-color: var(--border) !important;
            color: var(--text-primary) !important;
            border-radius: 8px;
        }}

        /* Dataframes */
        .stDataFrame {{
            background: var(--bg-card);
            border-radius: 12px;
            border: 1px solid var(--border);
        }}

        [data-testid="stDataFrame"] > div {{
            background: transparent;
        }}

        /* Info/Warning/Error boxes */
        .stAlert {{
            border-radius: 8px;
            border-left-width: 4px;
        }}

        [data-baseweb="notification"] {{
            background: var(--bg-card) !important;
        }}

        /* Expanders */
        .streamlit-expanderHeader {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary) !important;
        }}

        .streamlit-expanderContent {{
            background: var(--bg-light);
            border: 1px solid var(--border);
            border-top: none;
            border-radius: 0 0 8px 8px;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background: var(--bg-card);
            border-radius: 12px;
            padding: 4px;
            gap: 4px;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border-radius: 8px;
            color: var(--text-secondary);
            font-weight: 500;
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: var(--bg-dark) !important;
        }}

        /* Progress bars */
        .stProgress > div > div {{
            background: linear-gradient(90deg, var(--primary), var(--accent));
            border-radius: 4px;
        }}

        /* Slider */
        .stSlider > div > div > div {{
            background: var(--primary) !important;
        }}

        /* Custom card class */
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            margin: 8px 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}

        .metric-card:hover {{
            border-color: var(--primary);
            box-shadow: 0 4px 25px rgba(0, 212, 255, 0.15);
        }}

        /* Status indicators */
        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }}

        .status-dot.online {{
            background: var(--success);
            box-shadow: 0 0 10px var(--success);
        }}

        .status-dot.warning {{
            background: var(--warning);
            box-shadow: 0 0 10px var(--warning);
        }}

        .status-dot.offline {{
            background: var(--error);
            box-shadow: 0 0 10px var(--error);
        }}

        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}

        /* Glowing text effect */
        .glow-text {{
            color: var(--primary);
            text-shadow: 0 0 10px var(--primary), 0 0 20px var(--primary), 0 0 30px var(--primary);
        }}

        /* Logo styling */
        .logo-container {{
            text-align: center;
            padding: 20px 0;
        }}

        .logo-text {{
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: 2px;
        }}

        /* Hide Streamlit branding and navigation arrows */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        /* Hide the expand/collapse arrow in sidebar */
        [data-testid="collapsedControl"] {{
            display: none !important;
        }}

        /* Hide the sidebar collapse/expand button completely */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="baseButton-headerNoPadding"],
        button[kind="headerNoPadding"],
        [data-testid="stSidebar"] > div > div > div > button,
        .stSidebar button[aria-label*="Collapse"],
        .stSidebar button[aria-label*="Expand"],
        [data-testid="stSidebar"] button:has(span:contains("keyboard")) {{
            display: none !important;
            visibility: hidden !important;
        }}

        /* Target the collapse button by its position */
        [data-testid="stSidebar"] > div:first-child > button {{
            display: none !important;
        }}

        /* Hide ALL sidebar toggle buttons globally */
        button[aria-expanded][aria-label],
        [data-testid="stSidebarCollapsedControl"],
        div[data-testid="collapsedControl"] {{
            display: none !important;
            pointer-events: none !important;
        }}

        /* Prevent the hover zone from triggering */
        [data-testid="stSidebar"]::before {{
            display: none !important;
        }}

        /* Style the built-in page navigation */
        [data-testid="stSidebarNav"] {{
            padding-top: 0;
        }}

        [data-testid="stSidebarNav"] ul {{
            padding-top: 10px;
        }}

        [data-testid="stSidebarNav"] li {{
            margin-bottom: 4px;
        }}

        [data-testid="stSidebarNav"] a {{
            color: var(--text-secondary) !important;
            padding: 8px 12px;
            border-radius: 8px;
            transition: all 0.2s ease;
        }}

        [data-testid="stSidebarNav"] a:hover {{
            background: var(--bg-light);
            color: var(--text-primary) !important;
        }}

        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(123, 104, 238, 0.2));
            color: var(--primary) !important;
            border-left: 3px solid var(--primary);
        }}

        /* Hide the keyboard arrow icon that appears on pages */
        [data-testid="stSidebarNavLink"] span[data-testid="stIconMaterial"],
        [data-testid="stSidebarNav"] span[data-testid="stIconMaterial"],
        [data-testid="stSidebarNavItems"] span[data-testid="stIconMaterial"] {{
            display: none !important;
        }}

        /* Hide any element containing keyboard_arrow text */
        [data-testid="stSidebarNav"] span:has(> svg),
        [data-testid="stSidebar"] [data-testid="stMarkdown"] span[class*="icon"],
        span[data-testid="stIconMaterial"] {{
            display: none !important;
        }}

        /* Hide the expander header arrow icons in sidebar nav */
        [data-testid="stSidebarNav"] summary svg,
        [data-testid="stSidebarNav"] summary::marker,
        [data-testid="stSidebarNav"] summary::-webkit-details-marker,
        [data-testid="stSidebarNavItems"] svg {{
            display: none !important;
        }}

        /* Force hide any keyboard_double_arrow elements */
        [data-testid="stSidebar"] *[class*="keyboard"],
        [data-testid="stSidebar"] *[data-icon*="keyboard"] {{
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            overflow: hidden !important;
        }}

        /* Hide the sidebar nav expander toggle completely */
        [data-testid="stSidebarNavSeparator"],
        [data-testid="stSidebarNavExpander"],
        [data-testid="stSidebarNav"] details summary {{
            display: none !important;
        }}

        /* Ensure nav items are always visible (not collapsed) */
        [data-testid="stSidebarNav"] details {{
            display: block !important;
        }}

        [data-testid="stSidebarNav"] details[open] {{
            display: block !important;
        }}

        /* Hide material icon spans - target the icon wrapper specifically */
        [data-testid="stSidebarNavLink"] > div:first-child,
        [data-testid="stSidebarNavLink"] > span:first-child:not(:last-child) {{
            display: none !important;
        }}

        /* Alternative: use clip to hide icon text while preserving layout */
        [data-testid="stSidebarNav"] [data-testid="stIconMaterial"],
        [data-testid="stSidebarNav"] .material-icons,
        [data-testid="stSidebarNav"] [class*="material"] {{
            clip: rect(0, 0, 0, 0) !important;
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            overflow: hidden !important;
        }}

        /* Scrollbar styling */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: var(--bg-dark);
        }}

        ::-webkit-scrollbar-thumb {{
            background: var(--border);
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: var(--primary);
        }}
    </style>
    """, unsafe_allow_html=True)


def render_logo():
    """Render the CONSTELLATION logo."""
    st.markdown("""
    <div class="logo-container">
        <div style="margin-bottom: 8px;">
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="20" cy="20" r="18" stroke="url(#grad1)" stroke-width="2" fill="none"/>
                <circle cx="20" cy="20" r="6" fill="url(#grad1)"/>
                <ellipse cx="20" cy="20" rx="18" ry="8" stroke="url(#grad1)" stroke-width="1.5" fill="none" transform="rotate(45 20 20)"/>
                <ellipse cx="20" cy="20" rx="18" ry="8" stroke="url(#grad1)" stroke-width="1.5" fill="none" transform="rotate(-45 20 20)"/>
                <defs>
                    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#00d4ff;stop-opacity:1" />
                        <stop offset="100%" style="stop-color:#7b68ee;stop-opacity:1" />
                    </linearGradient>
                </defs>
            </svg>
        </div>
        <div class="logo-text">CONSTELLATION</div>
        <div style="color: var(--text-muted); font-size: 0.7rem; letter-spacing: 1.5px; text-transform: uppercase;">
            Fleet Health Management
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status_indicator(status: str, label: str = ""):
    """Render a status indicator dot with optional label."""
    status_class = "online" if status.lower() in ["online", "active", "nominal"] else \
                   "warning" if status.lower() in ["warning", "degraded"] else "offline"

    html = f'<span class="status-dot {status_class}"></span>'
    if label:
        html += f'<span style="color: var(--text-secondary);">{label}</span>'

    st.markdown(html, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = "", color: str = None):
    """Render a custom metric card."""
    color_style = f"color: {color};" if color else ""
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px;">
            {title}
        </div>
        <div style="font-size: 2rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; {color_style}">
            {value}
        </div>
        <div style="color: var(--text-muted); font-size: 0.85rem;">
            {subtitle}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, icon: str = ""):
    """Render a styled section header."""
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin: 30px 0 20px 0; padding-bottom: 10px; border-bottom: 1px solid var(--border);">
        <span style="font-size: 1.5rem; margin-right: 12px;">{icon}</span>
        <span style="font-size: 1.25rem; font-weight: 600; color: var(--text-primary);">{title}</span>
    </div>
    """, unsafe_allow_html=True)


def apply_plotly_theme(fig):
    """Apply the dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color=COLORS["text_secondary"],
            size=12,
        ),
        title=dict(
            font=dict(
                color=COLORS["text_primary"],
                size=16,
            )
        ),
        xaxis=dict(
            gridcolor=COLORS["grid"],
            linecolor=COLORS["border"],
            tickcolor=COLORS["text_muted"],
            zerolinecolor=COLORS["border"],
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"],
            linecolor=COLORS["border"],
            tickcolor=COLORS["text_muted"],
            zerolinecolor=COLORS["border"],
        ),
        hoverlabel=dict(
            bgcolor=COLORS["background_card"],
            bordercolor=COLORS["border"],
            font=dict(color=COLORS["text_primary"]),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text_secondary"]),
        ),
    )
    return fig
