

<p align="center">
  <img src="../assets/logo-full.svg" alt="CONSTELLATION Logo" width="300">
</p>

<p align="center">
  <strong>Satellite Fleet Health Management System</strong><br>
  Real-time ISS telemetry monitoring with ML-powered diagnostics
</p>

---

## Overview

The CONSTELLATION dashboard provides a comprehensive interface for monitoring the International Space Station's subsystems using real-time telemetry data from NASA's public Lightstreamer feed.

### Key Features

- Real-time anomaly detection with Isolation Forest
- Predictive fault classification using XGBoost
- Component health scoring and degradation forecasting
- Optimized maintenance scheduling with PuLP
- Interactive 3D attitude visualization

---

## Dashboard Pages
  
### Mission Control
<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/ef464686-12bb-4fbb-a9fd-cd5cc5017f60" alt="Mission Control" width="800">
</p>
<p align="center">
The main overview page displaying fleet-wide health status, active alerts, and live telemetry feeds.
</p>

<br>
**Features:**
- Fleet health gauge with real-time scoring
- Subsystem status cards (Attitude Control, Communications)
- Live telemetry chart with multi-parameter visualization
- Active alerts panel with severity indicators
- ML model status monitoring
<br>

---

### Telemetry Monitoring
<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/89133d07-693a-4714-96d8-cfb1141c059a" alt="Telemetry Monitoring" width="800">
</p>
<p align="center">
Detailed telemetry visualization with multi-parameter selection and statistical analysis.
</p>
<br>
**Features:**
- Multi-parameter selection (up to 6 simultaneous)
- Line, area, and candlestick chart modes
- Normal bounds visualization
- Subsystem comparison with sparklines
- Statistical summary table with export options
<br>

---

### System Diagnostics
<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/f652bbe1-d2fd-490c-8480-4e85887cad5e" alt="System Diagnostics" width="800">
</p>
<p align="center">
Consolidated anomaly detection, health monitoring, and fault analysis.
</p>
<br>
**Features:**
- Component health treemap visualization
- Radar chart for subsystem health comparison
- Anomaly score distribution analysis
- Fault classification breakdown
- Health trend visualization (24h)
<br>

---

### 3D Visualization
<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/4f9a0f7b-25d4-49c3-9fc6-72191c319bd7" alt="3D Visualization" width="800">
</p>
<p align="center">
Interactive 3D visualization of ISS attitude and component status.
</p>
<br>
**Features:**
- 3D ISS model with solar arrays and modules
- Quaternion-based attitude control
- Manual rotation controls (pitch, yaw, roll)
- Component health 3D map view
- Earth reference sphere
<br>

---

### Maintenance Scheduling
<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/c68958ba-a651-43ed-830f-9318c9ba1840" alt="Maintenance Scheduling" width="800">
</p>
<p align="center">
Optimized maintenance task scheduling and resource management.
</p>
<br>
**Features:**
- Gantt chart timeline view
- Calendar heatmap with risk scoring
- Task list with priority indicators
- Urgency vs Impact scatter analysis
- Risk distribution visualization
<br>

---

### Settings
<br>
<p align="center">
  <img src="https://github.com/user-attachments/assets/ee2e256a-1a50-44fd-8179-a2be99446a53" alt="Settings" width="800">
</p>
<p align="center">
System configuration, model management, and about information.
</p>
<br>
**Features:**
- ML model status and performance metrics
- Alert threshold configuration
- Data storage management
- System information display
- Technology stack overview
<br>

---

## Design System

### Color Palette

| Color | Hex | Usage |
|-------|-----|-------|
| Primary | `#00d4ff` | Primary actions, highlights |
| Secondary | `#7b68ee` | Secondary elements, accents |
| Success | `#00ff88` | Healthy status, positive indicators |
| Warning | `#ffaa00` | Degraded status, warnings |
| Error | `#ff4757` | Critical alerts, errors |
| Background | `#0a0e17` | Main background |
| Card | `#141b2d` | Card backgrounds |

### Typography

- **Headers**: Inter (600-700 weight)
- **Body**: Inter (400-500 weight)
- **Monospace**: JetBrains Mono (metrics, code)

### Visual Effects

- Subtle star field animation in background
- Gradient text effects on headers
- Pulsing status indicators
- Hover effects on interactive elements
- Smooth transitions throughout

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run data collection (optional - for live data)
python scripts/collect_telemetry.py --duration 3600

# Launch dashboard
streamlit run dashboard/app.py
```

Dashboard will be available at: **http://localhost:8501**

---

<p align="center">
  <img src="../assets/logo-icon.svg" alt="CONSTELLATION" width="40">
  <br>
  <sub>Built by Michael Gurule | Data: NASA ISS Telemetry (Public)</sub>
</p>
