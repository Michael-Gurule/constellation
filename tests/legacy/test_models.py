
"""
Unit tests for ML models.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from constellation.legacy.models.anomaly_detection import IsolationForestDetector
from constellation.legacy.models.degradation_forecast import DegradationForecaster
from constellation.legacy.models.fault_classifier import FaultClassifier, create_fault_labels


class TestAnomalyDetection:
    """Test anomaly detection models."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Mostly normal data
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        
        # Inject some anomalies
        anomaly_indices = np.random.choice(n_samples, 100, replace=False)
        X.iloc[anomaly_indices] = X.iloc[anomaly_indices] * 3
        
        return X
    
    def test_isolation_forest_train(self, sample_data):
        """Test Isolation Forest training."""
        model = IsolationForestDetector(contamination=0.1)
        model.train(sample_data)
        
        assert model.trained == True
        assert model.model is not None
    
    def test_isolation_forest_predict(self, sample_data):
        """Test Isolation Forest predictions."""
        model = IsolationForestDetector(contamination=0.1)
        model.train(sample_data)
        
        predictions = model.predict(sample_data)
        
        assert len(predictions) == len(sample_data)
        assert set(predictions).issubset({-1, 1})


class TestDegradationForecaster:
    """Test degradation forecasting."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series."""
        n_samples = 1000
        n_features = 10
        
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = pd.Series(np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.randn(n_samples) * 0.1)
        
        return X, y
    
    def test_forecaster_train(self, sample_data):
        """Test forecaster training."""
        X, y = sample_data
        
        model = DegradationForecaster(
            input_dim=X.shape[1],
            sequence_length=50,
            forecast_horizon=10
        )
        
        model.train(X, y, epochs=5, batch_size=32)
        
        assert model.trained == True


class TestFaultClassifier:
    """Test fault classifier."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        n_samples = 1000
        n_features = 10
        
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        y = pd.Series(np.random.choice(['fault_a', 'fault_b', 'normal'], n_samples))
        
        return X, y
    
    def test_classifier_train(self, sample_data):
        """Test classifier training."""
        X, y = sample_data
        
        model = FaultClassifier(n_estimators=50)
        model.train(X, y)
        
        assert model.trained == True
    
    def test_create_fault_labels(self):
        """Test fault label creation."""
        df = pd.DataFrame({
            'rolling_std_1h': np.random.randn(100),
            'is_daylight': np.random.randint(0, 2, 100),
            'value_numeric': np.random.randn(100)
        })
        
        labels = create_fault_labels(df)
        
        assert len(labels) == len(df)
        assert labels.dtype == 'object'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])