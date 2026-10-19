"""
Unit tests for feature engineering modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from constellation.legacy.features.time_series_features import TimeSeriesFeatureExtractor
from constellation.legacy.features.domain_features import AerospaceFeatureExtractor
from constellation.legacy.features.feature_pipeline import FeaturePipeline


class TestTimeSeriesFeatures:
    """Test time series feature extraction."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample telemetry data."""
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1s')
        values = np.sin(np.linspace(0, 4*np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'value': values,
            'parameter_id': 'TEST001'
        })
        return df
    
    def test_rolling_features(self, sample_data):
        """Test rolling feature extraction."""
        extractor = TimeSeriesFeatureExtractor(windows=[60, 300])
        result = extractor.extract_rolling_features(sample_data)
        
        assert 'rolling_mean_1m' in result.columns
        assert 'rolling_std_1m' in result.columns
        assert 'rolling_mean_5m' in result.columns
        assert len(result) == len(sample_data)
    
    def test_lag_features(self, sample_data):
        """Test lag feature extraction."""
        extractor = TimeSeriesFeatureExtractor(lag_periods=[1, 10])
        result = extractor.extract_lag_features(sample_data)
        
        assert 'lag_1s' in result.columns
        assert 'lag_10s' in result.columns
        assert 'roc_1s' in result.columns
        assert len(result) == len(sample_data)
    
    def test_rate_of_change(self, sample_data):
        """Test rate of change features."""
        extractor = TimeSeriesFeatureExtractor()
        result = extractor.extract_rate_of_change(sample_data)
        
        assert 'first_derivative' in result.columns
        assert 'second_derivative' in result.columns
        assert len(result) == len(sample_data)


class TestDomainFeatures:
    """Test domain-specific feature extraction."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample telemetry data."""
        dates = pd.date_range(start='2024-01-01', periods=10000, freq='1s')
        values = np.random.uniform(1000, 5000, 10000)  # RWA speed
        
        df = pd.DataFrame({
            'timestamp': dates,
            'value_numeric': values,
            'parameter_id': 'USLAB000084'
        })
        return df
    
    def test_orbital_features(self, sample_data):
        """Test orbital feature extraction."""
        extractor = AerospaceFeatureExtractor()
        result = extractor.extract_orbital_features(sample_data)
        
        assert 'orbital_phase' in result.columns
        assert 'orbital_phase_sin' in result.columns
        assert 'orbital_phase_cos' in result.columns
        assert 'is_daylight' in result.columns
        
        # Check orbital phase is between 0 and 1
        assert result['orbital_phase'].min() >= 0
        assert result['orbital_phase'].max() <= 1
    
    def test_attitude_control_features(self, sample_data):
        """Test attitude control features."""
        extractor = AerospaceFeatureExtractor()
        result = extractor.extract_attitude_control_features(
            sample_data,
            speed_col='value_numeric'
        )
        
        assert 'speed_stability' in result.columns
        assert 'speed_variability' in result.columns
        assert 'cumulative_momentum' in result.columns


class TestFeaturePipeline:
    """Test complete feature pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = FeaturePipeline()
        
        assert pipeline.ts_extractor is not None
        assert pipeline.domain_extractor is not None
        assert pipeline.storage is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])