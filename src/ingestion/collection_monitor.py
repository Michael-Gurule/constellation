"""
Monitor for tracking telemetry collection progress.
Provides real-time statistics and health checks.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from config.settings import RAW_DATA_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class CollectionMonitor:
    """
    Monitors telemetry collection progress and data quality.
    """
    
    def __init__(self):
        self.data_dir = RAW_DATA_DIR
    
    def get_collection_stats(self, hours: int = 24) -> Dict:
        """
        Get collection statistics for recent time period.
        
        Args:
            hours: Number of hours to look back
        
        Returns:
            Dictionary of collection statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        total_records = 0
        total_files = 0
        subsystems = {}
        earliest_timestamp = None
        latest_timestamp = None
        
        # Scan all Parquet files
        for parquet_file in self.data_dir.rglob("*.parquet"):
            # Check file modification time
            file_mtime = datetime.fromtimestamp(parquet_file.stat().st_mtime)
            
            if file_mtime < cutoff_time:
                continue
            
            total_files += 1
            
            # Load and analyze
            df = pd.read_parquet(parquet_file)
            total_records += len(df)
            
            # Track subsystem (from file path)
            if "subsystem=" in str(parquet_file):
                subsystem = str(parquet_file).split("subsystem=")[1].split("/")[0]
                subsystems[subsystem] = subsystems.get(subsystem, 0) + len(df)
            
            # Track timestamps
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                if earliest_timestamp is None or df['timestamp'].min() < earliest_timestamp:
                    earliest_timestamp = df['timestamp'].min()
                
                if latest_timestamp is None or df['timestamp'].max() > latest_timestamp:
                    latest_timestamp = df['timestamp'].max()
        
        # Calculate collection duration
        duration_hours = 0
        if earliest_timestamp and latest_timestamp:
            duration = latest_timestamp - earliest_timestamp
            duration_hours = duration.total_seconds() / 3600
        
        # Calculate rate
        records_per_hour = total_records / duration_hours if duration_hours > 0 else 0
        
        stats = {
            'total_records': total_records,
            'total_files': total_files,
            'subsystems': subsystems,
            'earliest_timestamp': earliest_timestamp,
            'latest_timestamp': latest_timestamp,
            'duration_hours': duration_hours,
            'records_per_hour': records_per_hour,
            'collection_active': (datetime.now() - latest_timestamp).total_seconds() < 600 if latest_timestamp else False
        }
        
        return stats
    
    def print_status(self, hours: int = 24):
        """
        Print collection status to console.
        
        Args:
            hours: Number of hours to look back
        """
        stats = self.get_collection_stats(hours)
        
        print("\n" + "=" * 60)
        print("CONSTELLATION - Collection Status")
        print("=" * 60)
        print(f"\nTime Window: Last {hours} hours")
        print(f"Total Records: {stats['total_records']:,}")
        print(f"Total Files: {stats['total_files']}")
        
        if stats['subsystems']:
            print("\nRecords by Subsystem:")
            for subsystem, count in stats['subsystems'].items():
                print(f"  {subsystem}: {count:,}")
        
        if stats['earliest_timestamp']:
            print(f"\nEarliest Record: {stats['earliest_timestamp']}")
            print(f"Latest Record: {stats['latest_timestamp']}")
            print(f"Collection Duration: {stats['duration_hours']:.2f} hours")
            print(f"Collection Rate: {stats['records_per_hour']:.0f} records/hour")
        
        status = "ACTIVE ✓" if stats['collection_active'] else "INACTIVE ✗"
        print(f"\nCollection Status: {status}")
        print("=" * 60 + "\n")
    
    def get_parameter_coverage(self) -> Dict:
        """
        Check which parameters have been collected.
        
        Returns:
            Dictionary mapping parameter IDs to record counts
        """
        param_counts = {}
        
        for parquet_file in self.data_dir.rglob("*.parquet"):
            df = pd.read_parquet(parquet_file)
            
            if 'parameter_id' in df.columns:
                counts = df['parameter_id'].value_counts()
                for param_id, count in counts.items():
                    param_counts[param_id] = param_counts.get(param_id, 0) + count
        
        return param_counts
    
    def check_data_quality(self) -> Dict:
        """
        Perform data quality checks.
        
        Returns:
            Dictionary of data quality metrics
        """
        total_records = 0
        null_values = 0
        duplicate_timestamps = 0
        
        for parquet_file in self.data_dir.rglob("*.parquet"):
            df = pd.read_parquet(parquet_file)
            total_records += len(df)
            
            # Check for nulls
            null_values += df.isnull().sum().sum()
            
            # Check for duplicate timestamps (per parameter)
            if 'timestamp' in df.columns and 'parameter_id' in df.columns:
                dups = df.duplicated(subset=['timestamp', 'parameter_id']).sum()
                duplicate_timestamps += dups
        
        quality_metrics = {
            'total_records': total_records,
            'null_values': null_values,
            'null_percentage': (null_values / total_records * 100) if total_records > 0 else 0,
            'duplicate_timestamps': duplicate_timestamps,
            'duplicate_percentage': (duplicate_timestamps / total_records * 100) if total_records > 0 else 0
        }
        
        return quality_metrics


def main():
    """
    Run collection monitor.
    """
    monitor = CollectionMonitor()
    monitor.print_status(hours=24)
    
    print("\nParameter Coverage:")
    param_coverage = monitor.get_parameter_coverage()
    for param_id, count in sorted(param_coverage.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param_id}: {count:,} records")
    
    print("\nData Quality Metrics:")
    quality = monitor.check_data_quality()
    print(f"  Total Records: {quality['total_records']:,}")
    print(f"  Null Values: {quality['null_values']} ({quality['null_percentage']:.2f}%)")
    print(f"  Duplicates: {quality['duplicate_timestamps']} ({quality['duplicate_percentage']:.2f}%)")


if __name__ == "__main__":
    main()