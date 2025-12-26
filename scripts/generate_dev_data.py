"""
Generate a comprehensive simulated dataset for development and testing.
Creates multiple days of data with various anomaly patterns.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.data_simulator import TelemetrySimulator
from src.ingestion.storage_handler import LocalStorageHandler
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def generate_development_dataset(days: int = 7):
    """
    Generate multi-day dataset with various patterns.
    
    Args:
        days: Number of days of data to generate
    """
    simulator = TelemetrySimulator()
    storage = LocalStorageHandler()
    
    print("=" * 60)
    print("Generating Development Dataset")
    print("=" * 60)
    print(f"Duration: {days} days")
    print(f"Sampling: 1 second intervals")
    print()
    
    # Calculate total records
    total_seconds = days * 24 * 60 * 60
    total_records = total_seconds * len(simulator.params_config)
    print(f"Expected records: ~{total_records:,}")
    print()
    
    # Generate data day by day (to avoid memory issues)
    start_date = datetime.now() - timedelta(days=days)
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        print(f"Generating day {day + 1}/{days}: {current_date.date()}")
        
        # Generate 24 hours of data
        records = simulator.generate_batch(
            start_time=current_date,
            duration_seconds=24 * 60 * 60,  # 24 hours
            interval_seconds=1
        )
        
        # Inject anomalies on some days
        if day % 3 == 0:  # Every 3rd day
            print("  → Injecting spike anomaly in RWA_1_Speed")
            records = simulator.inject_anomaly(records, 'USLAB000084', 'spike')
        
        if day % 4 == 1:  # Different pattern
            print("  → Injecting drift anomaly in S-band Signal")
            records = simulator.inject_anomaly(records, 'USLAB000098', 'drift')
        
        if day % 5 == 2:  # Another pattern
            print("  → Injecting dropout in CMG_1_Momentum")
            records = simulator.inject_anomaly(records, 'NODE3000001', 'dropout')
        
        # Separate by subsystem and save
        attitude_records = []
        comm_records = []
        
        for record in records:
            param_id = record['parameter_id']
            if param_id in simulator.params_config:
                subsystem = simulator.params_config[param_id]['subsystem']
                if subsystem == 'attitude_control':
                    attitude_records.append(record)
                elif subsystem == 'communications':
                    comm_records.append(record)
        
        # Save
        if attitude_records:
            storage.save_batch(attitude_records, subsystem='attitude_control')
        if comm_records:
            storage.save_batch(comm_records, subsystem='communications')
        
        print(f"  ✓ Saved {len(records):,} records")
    
    print()
    print("=" * 60)
    print("✓ Development dataset generated successfully!")
    print("=" * 60)
    print()
    
    # Run collection monitor to verify
    from src.ingestion.collection_monitor import CollectionMonitor
    monitor = CollectionMonitor()
    monitor.print_status(hours=days * 24)


if __name__ == "__main__":
    # Generate 7 days of data by default
    generate_development_dataset(days=7)