
<p align="center">
  <img src="https://github.com/user-attachments/assets/a675358c-19d6-4bf6-b332-2cc65d36e2dc" alt="Alt text description">
<p align="center">
  <strong>Satellite Fleet Health Management System</strong><br>
  Real-time ISS telemetry monitoring with ML-powered diagnostics
</p>  
<br>

*Continuous Operations Network for Satellite Telemetry Evaluation, Life-cycle Analysis, Tracking, Intelligence, Operations, and Notification*
<br>
<br>
<br>

> **Status:** v2 is under active development on the `v2` branch; see [CHANGELOG.md](CHANGELOG.md).

## Project Overview

**CONSTELLATION monitors the International Space Station's attitude control and communications subsystems using real-time telemetry from NASA's public Lightstreamer feed. The system provides:**

### 1. Data Ingestion Layer

**Real-Time Telemetry Processor**

- Collector subscribing to NASA's public ISS Live Lightstreamer feed
- Filters for attitude control and communications subsystems
- Writes date-partitioned Parquet files under `data/raw/`

**Parameters Monitored:**

**Attitude Control (Reaction Wheels & CMGs):**

- `USLAB000084`: Reaction Wheel Assembly (RWA) speed
- `USLAB000085`: RWA bearing temperature
- `USLAB000086`: RWA current draw
- `USLAB000087`: CMG (Control Moment Gyroscope) momentum
- Attitude quaternions (pitch, roll, yaw)
- Rate gyro outputs

**Communications:**

- S-band transponder power levels
- Ku-band signal strength
- Antenna pointing accuracy
- Data throughput metrics
- Ground station contact windows
- Communication link quality indicators

**Cross-Cutting:**

- Power system voltage/current (affects both subsystems)
- Thermal readings (reaction wheel bearings, transmitter temps)
- Time-on-orbit (cumulative degradation tracking)

### 2. Feature Engineering Pipeline

**Time Series Features:**

- Rolling statistics (mean, std, min, max) over multiple windows (1hr, 6hr, 24hr, 7day)
- Rate of change calculations
- Autocorrelation features
- Fourier transform for periodic patterns
- Lag features (t-1, t-6, t-24 for hourly data)

**Domain-Specific Features:**

- Reaction wheel friction coefficient (derived from speed vs. current)
- Thermal cycling count (number of orbital day/night transitions)
- Momentum accumulation rate
- Communication link budget margin
- Signal degradation trends
- Anomaly persistence scores

**Engineering Calculations:**

- Power efficiency ratios
- Thermal dissipation rates
- Bearing wear indicators
- Transmitter efficiency
- Pointing error accumulation

### 3. ML Model Suite

**Model 1: Anomaly Detection (Isolation Forest)**

*Purpose:* Real-time detection of unusual telemetry patterns

*Approach:*

- Isolation Forest for fast, lightweight anomaly flagging

*Training Data:*

- Nominal operational periods (healthy operation)

*Metrics:* Precision, Recall, F1-Score, False Positive Rate

**Model 2: Degradation Forecasting**

*Purpose:* Predict subsystem performance degradation over time

*Targets:*

- Reaction wheel bearing temperature trend
- Solar panel output decline
- Battery capacity fade
- Communication signal strength degradation

*Features:* Time series telemetry + orbital mechanics (radiation exposure, thermal cycling)

*Output:* Forecasted parameter values with confidence intervals

**Model 3: Survival Analysis (Cox Proportional Hazards)**

*Purpose:* Estimate time-to-failure for critical components

*Approach:*

- Cox model for component-level survival curves
- Censored data handling for components still operational
- Hazard ratios for risk factors (high temps, usage patterns)

*Output:* Probability of failure within time windows (30d, 60d, 90d, 180d)

**Model 4: Fault Classification (XGBoost)**

*Purpose:* Diagnose root cause when anomalies occur

*Classes:*

- Thermal stress
- Mechanical wear (bearings, gimbals)
- Electrical fault
- Software/command error
- External disturbance (debris impact, space weather)
- Normal operational variation

*Features:* Anomaly signatures, subsystem interactions, environmental context

*Output:* Ranked list of probable causes with confidence scores

### 4. Maintenance Optimization Engine

**Constraint Satisfaction Problem:**

*Variables:*

- Maintenance task list (derived from predictions)
- Available maintenance windows
- Crew availability (for ISS; ground station access for unmanned satellites)
- Orbital position constraints
- Mission priority levels

*Constraints:*

- Ground station contact requirements
- Crew schedule conflicts
- Tool/equipment availability
- Task dependencies (some maintenance requires others first)
- Safety margins (don't defer critical items)

*Objective Function:*

- Minimize risk-weighted maintenance delay
- Balance urgency vs. operational disruption
- Optimize crew time utilization

*Algorithm:* Mixed Integer Programming (MIP) using PuLP or Google OR-Tools

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  NASA Lightstreamer → Collector → Parquet (data/raw)            │
│  Date-partitioned Parquet archive                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│  • Time series windowing                                        │
│  • Statistical feature extraction                               │
│  • Subsystem correlation analysis                               │
│  • Degradation rate calculation                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      ML MODEL SUITE                             │
├─────────────────────────────────────────────────────────────────┤
│  Anomaly Detection    → Isolation Forest                        │
│  Degradation Forecast → LSTM                                    │
│  Survival Analysis    → Cox Proportional Hazards / Weibull      │
│  Fault Classification → Random Forest / XGBoost                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   OPERATIONAL LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Health Scoring → Maintenance Scheduling → Alert Generation     │
│  Dashboard (Streamlit)                                          │
└─────────────────────────────────────────────────────────────────┘
```

<br>
### Model Training Strategy

### Training Workflow

**Local Development:**

- Jupyter notebooks for experimentation
- Local training for model development
- Small data samples for rapid iteration

### Evaluation Metrics

**Anomaly Detection:**

- Precision, Recall, F1-Score
- False Positive Rate (critical for operational systems)
- Detection latency
- ROC-AUC, PR-AUC

**Degradation Forecasting:**

- RMSE, MAE, MAPE
- Prediction interval coverage
- Directional accuracy (did we predict the trend correctly?)
- Performance by forecast horizon

**Survival Analysis:**

- Concordance index (C-index)
- Brier score
- Calibration plots (predicted vs observed survival)
- Time-dependent AUC

**Fault Classification:**

- Accuracy, Precision, Recall per class
- Confusion matrix
- Top-k accuracy (are correct diagnoses in top 3 predictions?)

## Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd constellation

# Create and activate the conda environment (Python 3.12)
conda env create -f environment.yml && conda activate constellation
```

### Collect Telemetry Data

```bash
# Start telemetry collection
python -m constellation.legacy.ingestion.collect_telemetry
```

Data will be saved to `data/raw/` in date-partitioned Parquet files.

<br>

<h1 align="center">LET'S CONNECT!</h1>

<p align="center">
Constellation is a portfolio project on spacecraft telemetry health monitoring. All code and documentation are available for technical review.
</p>

<h3 align="center">Michael Gurule</h3>

<p align="center">
  <strong>Data Science | ML Engineering</strong>
</p>
<br>

<div align="center">
  <a href="mailto:michaelgurule1164@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"></a>
  
  <a href="https://michaelgurule.com">
    <img src="https://custom-icon-badges.demolab.com/badge/MICHAELGURULE.COM-150458?style=for-the-badge&logo=browser&logoColor=white"></a>
  
  <a href="https://www.linkedin.com/in/michael-gurule-447aa2134">
    <img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin-white&logoColor=fff"></a>
  
  <a href="https://medium.com/@michaelgurule1164">
    <img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white"></a>
</div>
<br>

---

<p align="center">
<img  width="450" alt="Designed By" src="https://github.com/user-attachments/assets/12ddff9c-b9b6-4e69-ace0-5cbc94f1a3ad">
</p>
<p align="center">
  <sub> Constellation: spacecraft telemetry health monitoring </sub>
</p>
