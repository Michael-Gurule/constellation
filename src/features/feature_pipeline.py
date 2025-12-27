"""
End-to-end feature engineering pipeline.
Orchestrates time series and domain feature extraction.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from src.features.time_series_features import TimeSeriesFeatureExtractor
from src.features.domain_features import AerospaceFeatureExtractor
from src.ingestion.storage_handler import LocalStorageHandler
from src.ingestion.data_validator import TelemetryValidator
from config.settings import PROCESSED_DATA_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class FeaturePipeline:
    """
    Complete feature engineering pipeline for telemetry data.
    """
    
    def __init__(self):
        self.ts_extractor = TimeSeriesFeatureExtractor()
        self.domain_extractor = AerospaceFeatureExtractor()
        self.storage = LocalStorageHandler()
        self.validator = TelemetryValidator()
        
        logger.info("Feature pipeline initialized")
    
    def process_parameter(
        self,
        df: pd.DataFrame,
        parameter_id: str
    ) -> pd.DataFrame:
        """
        Process a single parameter through the complete pipeline.
        
        Args:
            df: DataFrame with telemetry for one parameter
            parameter_id: Parameter identifier
        
        Returns:
            DataFrame with all features
        """
        logger.info(f"Processing parameter {parameter_id}: {len(df)} records")
        
        # Get parameter metadata
        param_info = self.validator.get_parameter_info(parameter_id)
        if not param_info:
            logger.warning(f"No metadata for {parameter_id}")
            return df
        
        subsystem = param_info['subsystem']
        param_name = param_info['name']
        
        # Add metadata columns
        df['parameter_id'] = parameter_id
        df['parameter_name'] = param_name
        df['subsystem'] = subsystem
        
        # Convert value to numeric
        df['value_numeric'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Extract time series features
        df = self.ts_extractor.extract_all_features(
            df, 
            value_col='value_numeric',
            time_col='timestamp'
        )
        
        # Extract domain features
        df = self.domain_extractor.extract_all_domain_features(
            df,
            subsystem=subsystem,
            value_col='value_numeric',
            time_col='timestamp'
        )
        
        logger.info(f"Completed processing {parameter_id}: {len(df.columns)} columns")
        return df
    
    def process_subsystem(
        self,
        subsystem: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Process all parameters for a subsystem.
        
        Args:
            subsystem: Subsystem name ('attitude_control' or 'communications')
            start_date: Start of date range
            end_date: End of date range
        
        Returns:
            DataFrame with features for all parameters in subsystem
        """
        logger.info(f"Processing subsystem: {subsystem}")
        
        # Load raw data
        df_raw = self.storage.load_date_range(
            start_date, 
            end_date,
            subsystem=subsystem
        )
        
        if df_raw.empty:
            logger.warning(f"No data found for {subsystem}")
            return pd.DataFrame()
        
        # Get unique parameters
        parameters = df_raw['parameter_id'].unique()
        logger.info(f"Found {len(parameters)} parameters in {subsystem}")
        
        # Process each parameter separately
        processed_dfs = []
        for param_id in parameters:
            param_df = df_raw[df_raw['parameter_id'] == param_id].copy()
            param_df = param_df.sort_values('timestamp')
            
            # Process through pipeline
            try:
                processed = self.process_parameter(param_df, param_id)
                processed_dfs.append(processed)
            except Exception as e:
                logger.error(f"Error processing {param_id}: {e}")
                continue
        
        # Combine all parameters
        if not processed_dfs:
            return pd.DataFrame()
        
        result = pd.concat(processed_dfs, ignore_index=True)
        logger.info(f"Completed {subsystem}: {len(result)} records, {len(result.columns)} features")
        
        return result
    
    def process_all_data(
        self,
        start_date: datetime,
        end_date: datetime,
        save: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Process all data through feature pipeline.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            save: Whether to save processed data
        
        Returns:
            Dictionary with processed DataFrames by subsystem
        """
        logger.info(f"Processing all data from {start_date} to {end_date}")
        
        results = {}
        
        # Process attitude control
        logger.info("=" * 60)
        logger.info("Processing Attitude Control Subsystem")
        logger.info("=" * 60)
        attitude_df = self.process_subsystem('attitude_control', start_date, end_date)
        results['attitude_control'] = attitude_df
        
        # Process communications
        logger.info("=" * 60)
        logger.info("Processing Communications Subsystem")
        logger.info("=" * 60)
        comm_df = self.process_subsystem('communications', start_date, end_date)
        results['communications'] = comm_df
        
        # Save if requested
        if save:
            logger.info("Saving processed features...")
            for subsystem, df in results.items():
                if not df.empty:
                    filepath = self.storage.save_dataframe(
                        df,
                        filename=f"{subsystem}_features_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
                        directory=PROCESSED_DATA_DIR
                    )
                    logger.info(f"Saved {subsystem} to {filepath}")
        
        logger.info("=" * 60)
        logger.info("Feature pipeline complete!")
        logger.info("=" * 60)
        
        return results
    
    def create_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary statistics for all features.
        
        Args:
            df: DataFrame with features
        
        Returns:
            DataFrame with feature statistics
        """
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        summary = df[numeric_cols].describe().T
        summary['missing_pct'] = (df[numeric_cols].isnull().sum() / len(df) * 100)
        summary['unique_values'] = df[numeric_cols].nunique()
        
        return summary


def main():
    """
    Run feature pipeline on simulated data.
    """
    from datetime import timedelta
    
    print("=" * 60)
    print("CONSTELLATION - Feature Engineering Pipeline")
    print("=" * 60)
    print()
    
    pipeline = FeaturePipeline()
    
    # Process last 7 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"Processing data from {start_date.date()} to {end_date.date()}")
    print()
    
    # Run pipeline
    results = pipeline.process_all_data(start_date, end_date, save=True)
    
    # Print summary
    print("\nProcessing Summary:")
    print("=" * 60)
    for subsystem, df in results.items():
        if not df.empty:
            print(f"\n{subsystem.upper()}:")
            print(f"  Records: {len(df):,}")
            print(f"  Features: {len(df.columns)}")
            print(f"  Parameters: {df['parameter_id'].nunique()}")
            print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == "__main__":
    main()