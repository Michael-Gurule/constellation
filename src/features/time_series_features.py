"""
Time series feature extraction for telemetry data.
Generates rolling statistics, lag features, and temporal patterns.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from scipy import stats

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class TimeSeriesFeatureExtractor:
    """
    Extracts time series features from telemetry data.
    """
    
    def __init__(
        self, 
        windows: List[int] = None,
        lag_periods: List[int] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            windows: List of window sizes (in seconds) for rolling features
            lag_periods: List of lag periods (in seconds) for lag features
        """
        # Default windows: 1min, 5min, 15min, 1hr, 6hr, 24hr
        self.windows = windows or [60, 300, 900, 3600, 21600, 86400]
        
        # Default lags: 1s, 10s, 1min, 5min, 1hr
        self.lag_periods = lag_periods or [1, 10, 60, 300, 3600]
        
        logger.info(
            f"TimeSeriesFeatureExtractor initialized: "
            f"{len(self.windows)} windows, {len(self.lag_periods)} lags"
        )
    
    def extract_rolling_features(
        self, 
        df: pd.DataFrame,
        value_col: str = 'value',
        time_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Extract rolling window features.
        
        Args:
            df: DataFrame with telemetry data (single parameter)
            value_col: Column name for values
            time_col: Column name for timestamps
        
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        # Ensure sorted by time
        df = df.sort_values(time_col)
        
        # Convert to numeric if needed
        if df[value_col].dtype == 'object':
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        
        # Calculate rolling features for each window
        for window in self.windows:
            window_label = self._format_window_label(window)
            
            # Rolling mean
            df[f'rolling_mean_{window_label}'] = df[value_col].rolling(
                window=window, 
                min_periods=1
            ).mean()
            
            # Rolling std
            df[f'rolling_std_{window_label}'] = df[value_col].rolling(
                window=window,
                min_periods=1
            ).std()
            
            # Rolling min/max
            df[f'rolling_min_{window_label}'] = df[value_col].rolling(
                window=window,
                min_periods=1
            ).min()
            
            df[f'rolling_max_{window_label}'] = df[value_col].rolling(
                window=window,
                min_periods=1
            ).max()
            
            # Rolling range
            df[f'rolling_range_{window_label}'] = (
                df[f'rolling_max_{window_label}'] - df[f'rolling_min_{window_label}']
            )
            
            # Rolling coefficient of variation
            df[f'rolling_cv_{window_label}'] = (
                df[f'rolling_std_{window_label}'] / df[f'rolling_mean_{window_label}']
            )
        
        logger.debug(f"Extracted rolling features for {len(self.windows)} windows")
        return df
    
    def extract_lag_features(
        self,
        df: pd.DataFrame,
        value_col: str = 'value',
        time_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Extract lag features.
        
        Args:
            df: DataFrame with telemetry data
            value_col: Column name for values
            time_col: Column name for timestamps
        
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        df = df.sort_values(time_col)
        
        # Convert to numeric if needed
        if df[value_col].dtype == 'object':
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        
        # Create lag features
        for lag in self.lag_periods:
            lag_label = self._format_window_label(lag)
            df[f'lag_{lag_label}'] = df[value_col].shift(lag)
            
            # Rate of change from lag period
            df[f'roc_{lag_label}'] = (
                df[value_col] - df[f'lag_{lag_label}']
            ) / df[f'lag_{lag_label}']
        
        logger.debug(f"Extracted lag features for {len(self.lag_periods)} periods")
        return df
    
    def extract_rate_of_change(
        self,
        df: pd.DataFrame,
        value_col: str = 'value',
        time_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Extract rate of change features.
        
        Args:
            df: DataFrame with telemetry data
            value_col: Column name for values
            time_col: Column name for timestamps
        
        Returns:
            DataFrame with rate of change features
        """
        df = df.copy()
        df = df.sort_values(time_col)
        
        # Convert to numeric if needed
        if df[value_col].dtype == 'object':
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        
        # First derivative (instantaneous rate of change)
        df['first_derivative'] = df[value_col].diff()
        
        # Second derivative (acceleration)
        df['second_derivative'] = df['first_derivative'].diff()
        
        # Absolute rate of change
        df['abs_rate_of_change'] = df['first_derivative'].abs()
        
        # Velocity (rate over time)
        if time_col in df.columns:
            df['time_diff'] = df[time_col].diff().dt.total_seconds()
            df['velocity'] = df['first_derivative'] / df['time_diff']
        
        logger.debug("Extracted rate of change features")
        return df
    
    def extract_statistical_features(
        self,
        df: pd.DataFrame,
        value_col: str = 'value',
        window: int = 3600
    ) -> pd.DataFrame:
        """
        Extract statistical features over rolling windows.
        
        Args:
            df: DataFrame with telemetry data
            value_col: Column name for values
            window: Window size for statistics
        
        Returns:
            DataFrame with statistical features
        """
        df = df.copy()
        
        # Convert to numeric if needed
        if df[value_col].dtype == 'object':
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        
        window_label = self._format_window_label(window)
        
        # Skewness
        df[f'skew_{window_label}'] = df[value_col].rolling(
            window=window,
            min_periods=10
        ).skew()
        
        # Kurtosis
        df[f'kurtosis_{window_label}'] = df[value_col].rolling(
            window=window,
            min_periods=10
        ).kurt()
        
        # Median
        df[f'median_{window_label}'] = df[value_col].rolling(
            window=window,
            min_periods=1
        ).median()
        
        # Quantiles
        df[f'q25_{window_label}'] = df[value_col].rolling(
            window=window,
            min_periods=1
        ).quantile(0.25)
        
        df[f'q75_{window_label}'] = df[value_col].rolling(
            window=window,
            min_periods=1
        ).quantile(0.75)
        
        # IQR
        df[f'iqr_{window_label}'] = (
            df[f'q75_{window_label}'] - df[f'q25_{window_label}']
        )
        
        logger.debug(f"Extracted statistical features for window {window_label}")
        return df
    
    def extract_trend_features(
        self,
        df: pd.DataFrame,
        value_col: str = 'value',
        window: int = 3600
    ) -> pd.DataFrame:
        """
        Extract trend-based features.
        
        Args:
            df: DataFrame with telemetry data
            value_col: Column name for values
            window: Window size for trend analysis
        
        Returns:
            DataFrame with trend features
        """
        df = df.copy()
        
        # Convert to numeric if needed
        if df[value_col].dtype == 'object':
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        
        window_label = self._format_window_label(window)
        
        # Linear trend slope (using rolling regression)
        def calculate_slope(window_data):
            if len(window_data) < 2:
                return np.nan
            x = np.arange(len(window_data))
            y = window_data.values
            # Remove NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return np.nan
            slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
            return slope
        
        df[f'trend_slope_{window_label}'] = df[value_col].rolling(
            window=window,
            min_periods=10
        ).apply(calculate_slope, raw=False)
        
        # Is trending up/down
        df[f'is_trending_up_{window_label}'] = (
            df[f'trend_slope_{window_label}'] > 0
        ).astype(int)
        
        logger.debug(f"Extracted trend features for window {window_label}")
        return df
    
    def extract_all_features(
        self,
        df: pd.DataFrame,
        value_col: str = 'value',
        time_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Extract all time series features.
        
        Args:
            df: DataFrame with telemetry data
            value_col: Column name for values
            time_col: Column name for timestamps
        
        Returns:
            DataFrame with all features
        """
        logger.info(f"Extracting all features for {len(df)} records")
        
        # Extract all feature types
        df = self.extract_rolling_features(df, value_col, time_col)
        df = self.extract_lag_features(df, value_col, time_col)
        df = self.extract_rate_of_change(df, value_col, time_col)
        df = self.extract_statistical_features(df, value_col, window=3600)
        df = self.extract_trend_features(df, value_col, window=3600)
        
        feature_count = len(df.columns) - len([time_col, value_col, 'parameter_id'])
        logger.info(f"Extracted {feature_count} features")
        
        return df
    
    @staticmethod
    def _format_window_label(seconds: int) -> str:
        """
        Convert seconds to human-readable label.
        
        Args:
            seconds: Number of seconds
        
        Returns:
            Formatted label (e.g., '1h', '5m', '60s')
        """
        if seconds >= 86400:
            return f"{seconds // 86400}d"
        elif seconds >= 3600:
            return f"{seconds // 3600}h"
        elif seconds >= 60:
            return f"{seconds // 60}m"
        else:
            return f"{seconds}s"