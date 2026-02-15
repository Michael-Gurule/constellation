

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
**Mission:** Production-grade satellite fleet health management platform using real ISS telemetry to demonstrate predictive maintenance, anomaly detection, and operational decision support capabilities for aerospace and defense applications.
<br>


## Project Overview

**CONSTELLATION monitors the International Space Station's attitude control and communications subsystems using real-time telemetry from NASA's public Lightstreamer feed. The system provides:**

### 1. Data Ingestion Layer

**Real-Time Telemetry Processor**

- AWS Lambda function subscribing to NASA Lightstreamer
- Filters for attitude control and communications subsystems
- Writes to DynamoDB for real-time access
- Archives to S3 for historical analysis

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

**Model 1: Anomaly Detection (Isolation Forest + LSTM Autoencoder)**

_Purpose:_ Real-time detection of unusual telemetry patterns

_Approach:_

- Isolation Forest for fast, lightweight anomaly flagging
- LSTM Autoencoder for complex temporal anomaly detection
- Ensemble voting for final anomaly score

_Training Data:_

- Nominal operational periods (confirmed healthy operation)
- Labeled anomalies from NASA incident reports

_Metrics:_ Precision, Recall, F1-Score, False Positive Rate

**Model 2: Degradation Forecasting (Temporal Fusion Transformer)**

_Purpose:_ Predict subsystem performance degradation over time

_Targets:_

- Reaction wheel bearing temperature trend
- Solar panel output decline
- Battery capacity fade
- Communication signal strength degradation

_Features:_ Time series telemetry + orbital mechanics (radiation exposure, thermal cycling)

_Output:_ Forecasted parameter values with confidence intervals (7, 30, 90 days ahead)

**Model 3: Survival Analysis (Cox Proportional Hazards)**

_Purpose:_ Estimate time-to-failure for critical components

_Approach:_

- Cox model for component-level survival curves
- Censored data handling for components still operational
- Hazard ratios for risk factors (high temps, usage patterns)

_Output:_ Probability of failure within time windows (30d, 60d, 90d, 180d)

**Model 4: Fault Classification (XGBoost)**

_Purpose:_ Diagnose root cause when anomalies occur

_Classes:_

- Thermal stress
- Mechanical wear (bearings, gimbals)
- Electrical fault
- Software/command error
- External disturbance (debris impact, space weather)
- Normal operational variation

_Features:_ Anomaly signatures, subsystem interactions, environmental context

_Output:_ Ranked list of probable causes with confidence scores

### 4. Maintenance Optimization Engine

**Constraint Satisfaction Problem:**

_Variables:_

- Maintenance task list (derived from predictions)
- Available maintenance windows
- Crew availability (for ISS; ground station access for unmanned satellites)
- Orbital position constraints
- Mission priority levels

_Constraints:_

- Ground station contact requirements
- Crew schedule conflicts
- Tool/equipment availability
- Task dependencies (some maintenance requires others first)
- Safety margins (don't defer critical items)

_Objective Function:_

- Minimize risk-weighted maintenance delay
- Balance urgency vs. operational disruption
- Optimize crew time utilization

_Algorithm:_ Mixed Integer Programming (MIP) using PuLP or Google OR-Tools

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  NASA Lightstreamer → Lambda (Real-time) → DynamoDB             │
│  Historical Archive → S3 Data Lake                              │
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
│  Anomaly Detection    → Isolation Forest / Autoencoder          │
│  Degradation Forecast → LSTM / Temporal Fusion Transformer      │
│  Survival Analysis    → Cox Proportional Hazards / Weibull      │
│  Fault Classification → Random Forest / XGBoost                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   OPERATIONAL LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Health Scoring → Maintenance Scheduling → Alert Generation     │
│  Dashboard (Streamlit) → CloudWatch Monitoring                  │
└─────────────────────────────────────────────────────────────────┘
```
<br>
### Model Training Strategy

### Training Infrastructure

**Local Development:**

- Jupyter notebooks for experimentation
- GPU-enabled local training for initial model development
- Small data samples for rapid iteration

**Production Training:**

- AWS SageMaker for full dataset training
- Hyperparameter tuning with SageMaker Automatic Model Tuning
- Distributed training for large models
- Model versioning with MLflow

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
- Forecast horizon performance (7d vs 30d vs 90d)

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

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Collect Telemetry Data
```bash
# Start telemetry collection
python -m src.ingestion.collect_telemetry
```

Data will be saved to `data/raw/` in date-partitioned Parquet files.

<br>

<h1 align="center">LET'S CONNECT!</h1>

<p align="center">
This project demonstrates production-grade ML engineering capabilities including distributed training infrastructure, experiment management, and systematic research methodology. All code and documentation available for technical review.
</p>


<h3 align="center">Michael Gurule</h3>

<p align="center">
  <strong>Data Science | ML Engineering</strong>
</p>
<br>

  
<div align="center">
  <a href="mailto:michaelgurule1164@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"></a>
  
  <a href="michaelgurule.com">
    <img src="https://custom-icon-badges.demolab.com/badge/MICHAELGURULE.COM-150458?style=for-the-badge&logo=browser&logoColor=white"></a>
  
  <a href="www.linkedin.com/in/michael-gurule-447aa2134">
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
  <sub> Demonstrating institutional-quality quantitative finance capabilities through production-grade portfolio optimization </sub>
</p>
