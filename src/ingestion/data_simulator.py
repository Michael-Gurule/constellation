"""
Simulates ISS telemetry data for testing when Lightstreamer is unavailable.
Generates realistic telemetry patterns based on known ISS characteristics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import json

from config.settings import MONITORED_PARAMS, PROJECT_ROOT
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class TelemetrySimulator:
    """
    Generates simulated ISS telemetry data.
    """
    
    def __init__(self):
        # Load parameter definitions
        params_file = PROJECT_ROOT / "config" / "telemetry_params.json"
        with open(params_file, 'r') as f:
            self.params_config = json.load(f)
        
        # Orbital period (ISS completes orbit every ~90 minutes)
        self.orbital_period = 90 * 60  # seconds
        
        logger.info("Telemetry simulator initialized")
    
    def generate_value(
        self, 
        param_id: str, 
        timestamp: datetime,
        add_noise: bool = True
    ) -> float:
        """
        Generate a realistic value for a parameter.
        
        Args:
            param_id: Parameter identifier
            timestamp: Timestamp for this value
            add_noise: Whether to add random noise
        
        Returns:
            Simulated value
        """
        if param_id not in self.params_config:
            return 0.0
        
        param_info = self.params_config[param_id]
        min_val, max_val = param_info['normal_range']
        
        # Base value (mid-range)
        base_value = (min_val + max_val) / 2
        
        # Add orbital variation (sinusoidal pattern)
        orbital_phase = (timestamp.timestamp() % self.orbital_period) / self.orbital_period
        orbital_variation = np.sin(2 * np.pi * orbital_phase) * (max_val - min_val) * 0.1
        
        value = base_value + orbital_variation
        
        # Add random noise
        if add_noise:
            noise = np.random.normal(0, (max_val - min_val) * 0.02)
            value += noise
        
        # Clamp to range
        value = np.clip(value, min_val, max_val)
        
        return value
    
    def generate_batch(
        self, 
        start_time: datetime,
        duration_seconds: int = 3600,
        interval_seconds: int = 1
    ) -> List[Dict]:
        """
        Generate a batch of telemetry records.
        
        Args:
            start_time: Start timestamp
            duration_seconds: How long to simulate
            interval_seconds: Time between samples
        
        Returns:
            List of telemetry record dictionaries
        """
        records = []
        num_samples = duration_seconds // interval_seconds
        
        for i in range(num_samples):
            timestamp = start_time + timedelta(seconds=i * interval_seconds)
            
            # Generate value for each monitored parameter
            for param_id in MONITORED_PARAMS:
                value = self.generate_value(param_id, timestamp)
                
                record = {
                    'timestamp': timestamp.isoformat(),
                    'parameter_id': param_id,
                    'value': str(value),
                    'raw_message': f'SIMULATED_{param_id}'
                }
                
                records.append(record)
        
        logger.info(f"Generated {len(records)} simulated records")
        return records
    
    def inject_anomaly(
        self,
        records: List[Dict],
        param_id: str,
        anomaly_type: str = 'spike'
    ) -> List[Dict]:
        """
        Inject an anomaly into the dataset.
        
        Args:
            records: List of telemetry records
            param_id: Parameter to inject anomaly into
            anomaly_type: Type of anomaly ('spike', 'drift', 'dropout')
        
        Returns:
            Modified records list
        """
        param_records = [r for r in records if r['parameter_id'] == param_id]
        
        if not param_records:
            logger.warning(f"No records found for {param_id}")
            return records
        
        # Select anomaly location (middle 50% of data)
        start_idx = len(param_records) // 4
        end_idx = 3 * len(param_records) // 4
        anomaly_idx = np.random.randint(start_idx, end_idx)
        
        param_info = self.params_config[param_id]
        min_val, max_val = param_info['normal_range']
        
        if anomaly_type == 'spike':
            # Sudden spike in value
            spike_value = max_val * 1.2  # 20% above max
            param_records[anomaly_idx]['value'] = str(spike_value)
            logger.info(f"Injected spike anomaly in {param_id} at index {anomaly_idx}")
        
        elif anomaly_type == 'drift':
            # Gradual drift upward
            drift_length = min(100, len(param_records) - anomaly_idx)
            for i in range(drift_length):
                idx = anomaly_idx + i
                if idx < len(param_records):
                    current_val = float(param_records[idx]['value'])
                    drift = (i / drift_length) * (max_val * 0.3)
                    param_records[idx]['value'] = str(current_val + drift)
            logger.info(f"Injected drift anomaly in {param_id} starting at index {anomaly_idx}")
        
        elif anomaly_type == 'dropout':
            # Data dropout (missing values)
            dropout_length = min(50, len(param_records) - anomaly_idx)
            for i in range(dropout_length):
                idx = anomaly_idx + i
                if idx < len(param_records):
                    param_records[idx]['value'] = 'NaN'
            logger.info(f"Injected dropout anomaly in {param_id} at index {anomaly_idx}")
        
        return records


def main():
    """
    Generate sample simulated data for testing.
    """
    from src.ingestion.storage_handler import LocalStorageHandler
    
    print("Generating simulated telemetry data...")
    
    simulator = TelemetrySimulator()
    storage = LocalStorageHandler()
    
    # Generate 1 hour of data
    start_time = datetime.now() - timedelta(hours=1)
    records = simulator.generate_batch(
        start_time=start_time,
        duration_seconds=3600,
        interval_seconds=1
    )
    
    # Inject some anomalies
    records = simulator.inject_anomaly(records, 'USLAB000084', 'spike')
    records = simulator.inject_anomaly(records, 'USLAB000098', 'drift')
    
    # Save to storage
    storage.save_batch(records, subsystem='simulated')
    
    print(f"✓ Generated and saved {len(records)} simulated records")
    print(f"  Location: {storage.base_dir}")


if __name__ == "__main__":
    main()
