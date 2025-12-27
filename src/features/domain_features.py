"""
Domain-specific feature engineering for aerospace/satellite systems.
Implements aerospace calculations and physics-based features.
"""

import pandas as pd
import numpy as np
from typing import Dict

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class AerospaceFeatureExtractor:
    """
    Extracts aerospace and satellite-specific features.
    """
    
    def __init__(self):
        # ISS orbital parameters
        self.orbital_period_seconds = 92.68 * 60  # ~92.68 minutes
        self.orbital_altitude_km = 408  # Average ISS altitude
        
        logger.info("AerospaceFeatureExtractor initialized")
    
    def extract_orbital_features(
        self,
        df: pd.DataFrame,
        time_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Extract orbital position and phase features.
        
        Args:
            df: DataFrame with telemetry data
            time_col: Timestamp column
        
        Returns:
            DataFrame with orbital features
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Calculate orbital phase (0 to 1 over one orbit)
        epoch = df[time_col].iloc[0]
        seconds_since_epoch = (df[time_col] - epoch).dt.total_seconds()
        df['orbital_phase'] = (
            seconds_since_epoch % self.orbital_period_seconds
        ) / self.orbital_period_seconds
        
        # Sine/cosine encoding of orbital phase (for ML models)
        df['orbital_phase_sin'] = np.sin(2 * np.pi * df['orbital_phase'])
        df['orbital_phase_cos'] = np.cos(2 * np.pi * df['orbital_phase'])
        
        # Approximate day/night cycle (simplified)
        # ISS experiences ~45 minutes daylight, ~45 minutes darkness
        df['is_daylight'] = (df['orbital_phase'] % 0.5 < 0.5).astype(int)
        
        # Number of complete orbits
        df['orbit_count'] = (
            seconds_since_epoch / self.orbital_period_seconds
        ).astype(int)
        
        logger.debug("Extracted orbital features")
        return df
    
    def extract_thermal_features(
        self,
        df: pd.DataFrame,
        temp_params: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Extract thermal-related features.
        
        Args:
            df: DataFrame with telemetry including temperature data
            temp_params: Dictionary mapping temp parameter names to columns
        
        Returns:
            DataFrame with thermal features
        """
        df = df.copy()
        
        # Count thermal cycles (day/night transitions)
        if 'is_daylight' in df.columns:
            df['thermal_cycle_count'] = (
                df['is_daylight'].diff().abs().fillna(0).cumsum()
            )
        
        logger.debug("Extracted thermal features")
        return df
    
    def extract_attitude_control_features(
        self,
        df: pd.DataFrame,
        speed_col: str = 'value',
        current_col: str = None
    ) -> pd.DataFrame:
        """
        Extract features specific to reaction wheels and attitude control.
        
        Args:
            df: DataFrame with RWA telemetry
            speed_col: Column with wheel speed (RPM)
            current_col: Optional column with current draw
        
        Returns:
            DataFrame with attitude control features
        """
        df = df.copy()
        
        # Convert to numeric
        if df[speed_col].dtype == 'object':
            df[speed_col] = pd.to_numeric(df[speed_col], errors='coerce')
        
        # Speed stability (std over rolling window)
        df['speed_stability'] = df[speed_col].rolling(
            window=300,  # 5 minutes
            min_periods=1
        ).std()
        
        # Speed variability coefficient
        rolling_mean = df[speed_col].rolling(window=300, min_periods=1).mean()
        df['speed_variability'] = df['speed_stability'] / rolling_mean
        
        # Friction coefficient estimate (if current available)
        if current_col and current_col in df.columns:
            # Higher current at same speed indicates more friction
            df['friction_indicator'] = df[current_col] / (df[speed_col] + 1)
        
        # Speed change rate (acceleration/deceleration)
        df['speed_change_rate'] = df[speed_col].diff()
        df['abs_speed_change'] = df['speed_change_rate'].abs()
        
        # Momentum accumulation (integral of speed)
        df['cumulative_momentum'] = df[speed_col].cumsum()
        
        logger.debug("Extracted attitude control features")
        return df
    
    def extract_communications_features(
        self,
        df: pd.DataFrame,
        signal_col: str = 'value',
        power_col: str = None
    ) -> pd.DataFrame:
        """
        Extract features for communications systems.
        
        Args:
            df: DataFrame with communications telemetry
            signal_col: Column with signal strength
            power_col: Optional column with transmitter power
        
        Returns:
            DataFrame with communications features
        """
        df = df.copy()
        
        # Convert to numeric
        if df[signal_col].dtype == 'object':
            df[signal_col] = pd.to_numeric(df[signal_col], errors='coerce')
        
        # Signal stability
        df['signal_stability'] = df[signal_col].rolling(
            window=300,
            min_periods=1
        ).std()
        
        # Link margin (difference from minimum acceptable)
        # Typical minimum S-band: -120 dBm
        min_acceptable_dbm = -120
        df['link_margin'] = df[signal_col] - min_acceptable_dbm
        
        # Signal degradation rate
        df['signal_degradation_rate'] = -df[signal_col].diff()
        
        # Efficiency (if power available)
        if power_col and power_col in df.columns:
            # Higher efficiency = more signal per watt
            df['transmission_efficiency'] = df[signal_col] / (df[power_col] + 0.1)
        
        # Signal dropout indicator (very low signal)
        df['signal_dropout'] = (df[signal_col] < -115).astype(int)
        
        logger.debug("Extracted communications features")
        return df
    
    def extract_degradation_indicators(
        self,
        df: pd.DataFrame,
        value_col: str = 'value',
        baseline_period: int = 86400
    ) -> pd.DataFrame:
        """
        Extract features indicating component degradation.
        
        Args:
            df: DataFrame with telemetry
            value_col: Column with parameter values
            baseline_period: Period (seconds) to use as baseline
        
        Returns:
            DataFrame with degradation indicators
        """
        df = df.copy()
        
        # Convert to numeric
        if df[value_col].dtype == 'object':
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        
        # Baseline (initial performance)
        baseline_mean = df[value_col].iloc[:baseline_period].mean()
        baseline_std = df[value_col].iloc[:baseline_period].std()
        
        # Deviation from baseline
        df['deviation_from_baseline'] = df[value_col] - baseline_mean
        df['normalized_deviation'] = (
            df['deviation_from_baseline'] / baseline_std
            if baseline_std > 0 else 0
        )
        
        # Trend away from baseline (long-term drift)
        df['drift_from_baseline'] = df[value_col].rolling(
            window=86400,  # 24-hour rolling average
            min_periods=1
        ).mean() - baseline_mean
        
        # Cumulative degradation (how much worse over time)
        df['cumulative_degradation'] = df['drift_from_baseline'].cumsum()
        
        logger.debug("Extracted degradation indicators")
        return df
    
    def extract_all_domain_features(
        self,
        df: pd.DataFrame,
        subsystem: str,
        value_col: str = 'value',
        time_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Extract all domain-specific features based on subsystem.
        
        Args:
            df: DataFrame with telemetry
            subsystem: Subsystem type ('attitude_control' or 'communications')
            value_col: Column with parameter values
            time_col: Timestamp column
        
        Returns:
            DataFrame with domain features
        """
        logger.info(f"Extracting domain features for {subsystem}")
        
        # Always extract orbital and thermal features
        df = self.extract_orbital_features(df, time_col)
        df = self.extract_thermal_features(df)
        df = self.extract_degradation_indicators(df, value_col)
        
        # Subsystem-specific features
        if subsystem == 'attitude_control':
            df = self.extract_attitude_control_features(df, value_col)
        elif subsystem == 'communications':
            df = self.extract_communications_features(df, value_col)
        
        feature_count = len(df.columns) - len([time_col, value_col, 'parameter_id'])
        logger.info(f"Extracted {feature_count} domain features")
        
        return df