"""
Storage handler for persisting telemetry data.
Handles both local file storage and cloud storage (S3).
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import json

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class LocalStorageHandler:
    """
    Handles local file storage for telemetry data.
    Uses Parquet format for efficient storage and retrieval.
    """
    
    def __init__(self, base_dir: Path = None):
        """
        Initialize storage handler.
        
        Args:
            base_dir: Base directory for data storage.
                     If None, uses RAW_DATA_DIR from settings.
        """
        self.base_dir = base_dir or RAW_DATA_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Storage handler initialized: {self.base_dir}")
    
    def save_batch(
        self, 
        records: List[Dict], 
        subsystem: str = "telemetry"
    ) -> Path:
        """
        Save a batch of telemetry records to Parquet file.
        
        Args:
            records: List of telemetry dictionaries
            subsystem: Subsystem name for organizing files
        
        Returns:
            Path to saved file
        """
        if not records:
            logger.warning("No records to save")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create date-partitioned path
        now = datetime.now()
        year = now.strftime('%Y')
        month = now.strftime('%m')
        day = now.strftime('%d')
        
        partition_dir = self.base_dir / f"year={year}" / f"month={month}" / f"day={day}" / f"subsystem={subsystem}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp_str = now.strftime('%Y%m%d_%H%M%S')
        filename = f"telemetry_{timestamp_str}.parquet"
        filepath = partition_dir / filename
        
        # Save to Parquet
        df.to_parquet(filepath, index=False, compression='snappy')
        
        logger.info(f"Saved {len(records)} records to {filepath}")
        return filepath
    
    def load_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime,
        subsystem: str = None
    ) -> pd.DataFrame:
        """
        Load telemetry data for a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            subsystem: Optional subsystem filter
        
        Returns:
            DataFrame with telemetry data
        """
        dfs = []
        
        # Iterate through date range
        current_date = start_date
        while current_date <= end_date:
            year = current_date.strftime('%Y')
            month = current_date.strftime('%m')
            day = current_date.strftime('%d')
            
            # Build search path
            if subsystem:
                search_path = self.base_dir / f"year={year}" / f"month={month}" / f"day={day}" / f"subsystem={subsystem}"
            else:
                search_path = self.base_dir / f"year={year}" / f"month={month}" / f"day={day}"
            
            # Load all Parquet files in this partition
            if search_path.exists():
                for parquet_file in search_path.rglob("*.parquet"):
                    df = pd.read_parquet(parquet_file)
                    dfs.append(df)
                    logger.debug(f"Loaded {len(df)} records from {parquet_file}")
            
            # Move to next day
            current_date = current_date + pd.Timedelta(days=1)
        
        if not dfs:
            logger.warning(f"No data found for date range {start_date} to {end_date}")
            return pd.DataFrame()
        
        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} total records")
        
        return combined_df
    
    def get_latest_records(self, n: int = 100) -> pd.DataFrame:
        """
        Get the most recent n telemetry records.
        
        Args:
            n: Number of records to retrieve
        
        Returns:
            DataFrame with latest records
        """
        # Get most recent Parquet file
        parquet_files = sorted(
            self.base_dir.rglob("*.parquet"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if not parquet_files:
            logger.warning("No telemetry data found")
            return pd.DataFrame()
        
        # Load most recent file
        df = pd.read_parquet(parquet_files[0])
        
        # Sort by timestamp and take last n records
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp', ascending=False).head(n)
        else:
            df = df.tail(n)
        
        logger.info(f"Retrieved {len(df)} latest records")
        return df
    
    def save_dataframe(
        self, 
        df: pd.DataFrame, 
        filename: str,
        directory: Path = None
    ) -> Path:
        """
        Save a DataFrame to Parquet file.
        
        Args:
            df: DataFrame to save
            filename: Output filename (without extension)
            directory: Target directory (default: PROCESSED_DATA_DIR)
        
        Returns:
            Path to saved file
        """
        if directory is None:
            directory = PROCESSED_DATA_DIR
        
        directory.mkdir(parents=True, exist_ok=True)
        
        filepath = directory / f"{filename}.parquet"
        df.to_parquet(filepath, index=False, compression='snappy')
        
        logger.info(f"Saved DataFrame ({len(df)} rows) to {filepath}")
        return filepath