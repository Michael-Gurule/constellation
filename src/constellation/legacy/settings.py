import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TELEMETRY_PARAMS_FILE = Path(__file__).resolve().with_name("telemetry_params.json")

# Data directories (created by the code that writes into them):
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
SIMULATED_DATA_DIR = DATA_DIR / "simulated"

# NASA Lightstreamer Configuration
LIGHTSTREAMER_URL = "https://push.lightstreamer.com"
LIGHTSTREAMER_ADAPTER_SET = "ISSLIVE"

# ISS Telemetry Parameters
# Parameter IDs from NASA
ATTITUDE_CONTROL_PARAMS = [
    "USLAB000084",  # RWA 1 Speed
    "USLAB000085",  # RWA 2 Speed
    "USLAB000086",  # RWA 3 Speed
    "USLAB000087",  # RWA 4 Speed
    "NODE3000001",  # CMG 1 Momentum
    "NODE3000002",  # CMG 2 Momentum
    "NODE3000003",  # CMG 3 Momentum
    "NODE3000004",  # CMG 4 Momentum
    "USLAB000032",  # Attitude Quaternion Q1
    "USLAB000033",  # Attitude Quaternion Q2
    "USLAB000034",  # Attitude Quaternion Q3
    "USLAB000035",  # Attitude Quaternion Q4
]

COMMUNICATIONS_PARAMS = [
    "USLAB000098",  # S-band Signal Strength
    "USLAB000099",  # Ku-band Signal Strength
    "USLAB000100",  # S-band Power
    "USLAB000101",  # Ku-band Power
    "NODE2000015",  # Antenna Pointing Azimuth
    "NODE2000016",  # Antenna Pointing Elevation
]

# Combine all monitored parameters
MONITORED_PARAMS = ATTITUDE_CONTROL_PARAMS + COMMUNICATIONS_PARAMS

# Sampling configuration
SAMPLING_INTERVAL_SECONDS = 1  # How often to collect telemetry
DATA_RETENTION_DAYS = 90  # How long to keep raw data

# Model Configuration
ANOMALY_DETECTION_THRESHOLD = 0.85  # Confidence threshold for anomaly alerts
FORECAST_HORIZONS = [7, 30, 90]  # Days ahead to forecast
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test

# Alert Configuration
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")
ALERT_ENABLED = os.getenv("ALERT_ENABLED", "false").lower() == "true"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Dashboard Configuration
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))
DASHBOARD_REFRESH_SECONDS = int(os.getenv("DASHBOARD_REFRESH_SECONDS", "5"))
