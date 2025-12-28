# CONSTELLATION Dashboard

<p align="center">
  <img src="../assets/logo.svg" alt="CONSTELLATION Logo" width="80">
</p>

Interactive web interface for satellite fleet health monitoring with a dark space-themed design.

> **See the full showcase with screenshots:** [Dashboard Showcase](../docs/DASHBOARD_SHOWCASE.md)

## Quick Start

```bash
# From project root
streamlit run dashboard/app.py
```

Or use the launch script:
```bash
./dashboard/run.sh
```

Dashboard will be available at: **http://localhost:8501**

## Pages

| Page | Description |
|------|-------------|
| **Mission Control** | Main overview with fleet health gauge, subsystem status, live telemetry, and alerts |
| **Telemetry** | Multi-parameter visualization with statistical analysis and export |
| **Diagnostics** | Anomaly detection, health monitoring, and fault classification |
| **Visualization** | 3D ISS attitude model with quaternion controls |
| **Maintenance** | Task scheduling with timeline, calendar, and risk analysis |
| **Settings** | Model configuration, alert thresholds, and system info |

## Architecture

```
dashboard/
├── app.py                 # Mission Control (main page)
├── theme.py               # Shared theme, colors, and styling
├── pages/
│   ├── 1_Telemetry.py     # Telemetry monitoring
│   ├── 2_Diagnostics.py   # Anomaly & health analysis
│   ├── 3_Visualization.py # 3D visualization
│   ├── 4_Maintenance.py   # Maintenance scheduling
│   └── 5_Settings.py      # Configuration
└── README.md
```

## Requirements

- Python 3.10+
- Streamlit >= 1.28.0
- Plotly >= 5.17.0
- Pandas, NumPy

Install dependencies:
```bash
pip install streamlit plotly pandas numpy
```

## Data Requirements

For full functionality, run system analysis first:

```bash
# Generate analysis results
python scripts/run_system_analysis.py --subsystem attitude_control --export

# Or collect live telemetry
python scripts/collect_telemetry.py --duration 3600
```

The dashboard will display demo data if no analysis results are available.

## Theme

The dashboard uses a custom dark space theme with:

- **Primary**: Cyan (`#00d4ff`)
- **Secondary**: Purple (`#7b68ee`)
- **Accent**: Green (`#00ff88`)
- **Background**: Deep space gradient with star animation

See [theme.py](theme.py) for the complete color palette and styling.

## Customization

### Modify Colors

Edit the `COLORS` dictionary in `theme.py`:

```python
COLORS = {
    "primary": "#00d4ff",
    "secondary": "#7b68ee",
    # ...
}
```

### Add New Pages

1. Create a new file in `pages/` with numeric prefix (e.g., `6_NewPage.py`)
2. Import and apply the theme:
   ```python
   from dashboard.theme import apply_theme, render_logo, COLORS
   apply_theme()
   ```
3. Use shared components like `render_section_header()` and `apply_plotly_theme()`

## License

MIT License - See project root for details.
