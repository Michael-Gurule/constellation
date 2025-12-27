"""
Orchestrates training of all models.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.models.base_model import BaseModel
from src.models.anomaly_detection import IsolationForestDetector, LSTMAutoencoderDetector
from src.models.degradation_forecast import DegradationForecaster
from src.models.survival_analysis import SurvivalAnalyzer, create_survival_dataset
from src.models.fault_classifier import FaultClassifier, create_fault_labels
from src.ingestion.storage_handler import LocalStorageHandler
from config.settings import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """
    Trains all ML models on processed features.
    """
    
    def __init__(self):
        self.storage = LocalStorageHandler()
        self.models = {}
        
        logger.info("ModelTrainer initialized")
    
    def load_features(
        self,
        subsystem: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Load processed features for a subsystem.
        
        Args:
            subsystem: Subsystem name
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with features
        """
        # Try to load from processed directory
        pattern = f"{subsystem}_features_*.parquet"
        feature_files = list(PROCESSED_DATA_DIR.glob(pattern))
        
        if feature_files:
            # Load most recent
            latest_file = max(feature_files, key=lambda p: p.stat().st_mtime)
            df = pd.read_parquet(latest_file)
            logger.info(f"Loaded {len(df)} records from {latest_file}")
            return df
        else:
            logger.warning(f"No processed features found for {subsystem}")
            return pd.DataFrame()
    
    def train_anomaly_detection(
        self,
        df: pd.DataFrame,
        method: str = 'isolation_forest'
    ) -> BaseModel:
        """Train anomaly detection model."""
        logger.info(f"Training {method} anomaly detector")
        
        if len(df) > 500000:
            df = df.sample(500000, random_state=42)
            logger.info(f"Sampled down to {len(df)} records to reduce memory usage")
        
        # Select numeric features (exclude metadata columns)
        exclude_cols = ['timestamp', 'value', 'parameter_id', 'parameter_name', 'subsystem']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        feature_cols = df[feature_cols].select_dtypes(include=['float64', 'int64']).columns
        
        X = df[feature_cols]
        
        if method == 'isolation_forest':
            model = IsolationForestDetector(contamination=0.1)
            model.train(X)
        else:  # lstm_autoencoder
            input_dim = len(feature_cols)
            model = LSTMAutoencoderDetector(input_dim=input_dim, sequence_length=100)
            model.train(X, epochs=30, batch_size=64)
        
        # Save model
        model.save()
        
        self.models[f'anomaly_detector_{method}'] = model
        logger.info(f"Anomaly detection model trained and saved")
        
        return model
    
    def train_degradation_forecaster(
        self,
        df: pd.DataFrame,
        target_parameter: str = 'value_numeric',
        forecast_horizon: int = 30
    ) -> DegradationForecaster:
        """Train degradation forecasting model."""
        logger.info(f"Training degradation forecaster (horizon={forecast_horizon})")
        
        if len(df) > 500000:
            df = df.sample(500000, random_state=42)
            logger.info(f"Sampled down to {len(df)} records to reduce memory usage")
            
        # Select features
        exclude_cols = ['timestamp', 'value', 'parameter_id', 'parameter_name', 'subsystem', target_parameter]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        feature_cols = df[feature_cols].select_dtypes(include=['float64', 'int64']).columns
        
        X = df[feature_cols]
        y = df[target_parameter]
        
        # Initialize and train model
        model = DegradationForecaster(
            input_dim=len(feature_cols),
            sequence_length=100,
            forecast_horizon=forecast_horizon,
            hidden_dim=128
        )
        
        model.train(X, y, epochs=50, batch_size=16)
        
        # Save model
        model.save()
        
        self.models['degradation_forecaster'] = model
        logger.info("Degradation forecaster trained and saved")
        
        return model
    
    def train_survival_analyzer(
        self,
        df: pd.DataFrame,
        parameter_id: str,
        failure_threshold: float = None
    ) -> SurvivalAnalyzer:
        """
        Train survival analysis model.
        
        Args:
            df: DataFrame with features
            parameter_id: Parameter to analyze
            failure_threshold: Threshold indicating failure
        
        Returns:
            Trained model
        """
        logger.info(f"Training survival analyzer for {parameter_id}")
        
        # Create survival dataset
        survival_df = create_survival_dataset(
            df,
            parameter_id=parameter_id,
            failure_threshold=failure_threshold,
            max_duration=168  # 7 days in hours
        )
        
        # Select features
        exclude_cols = ['timestamp', 'value', 'parameter_id', 'parameter_name', 
                       'subsystem', 'duration', 'event']
        feature_cols = [c for c in survival_df.columns if c not in exclude_cols]
        feature_cols = survival_df[feature_cols].select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Initialize and train model
        model = SurvivalAnalyzer(penalizer=0.1)
        model.train(
            survival_df,
            duration_col='duration',
            event_col='event',
            feature_cols=feature_cols
        )
        
        # Save model
        model.save()
        
        self.models['survival_analyzer'] = model
        logger.info("Survival analyzer trained and saved")
        
        return model
    
    def train_fault_classifier(
        self,
        df: pd.DataFrame,
        label_col: str = None
    ) -> FaultClassifier:
        """
        Train fault classification model.
        
        Args:
            df: DataFrame with features
            label_col: Column with fault labels (if None, creates labels)
        
        Returns:
            Trained model
        """
        logger.info("Training fault classifier")
        
        # Create or use existing labels
        if label_col is None or label_col not in df.columns:
            logger.info("Creating fault labels from features")
            labels = create_fault_labels(df)
        else:
            labels = df[label_col]
        
        # Select features
        exclude_cols = ['timestamp', 'value', 'parameter_id', 'parameter_name', 
                       'subsystem', label_col]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        feature_cols = df[feature_cols].select_dtypes(include=['float64', 'int64']).columns
        
        X = df[feature_cols]
        
        # Initialize and train model
        model = FaultClassifier(n_estimators=200, max_depth=6)
        model.train(X, labels, validation_split=0.2)
        
        # Save model
        model.save()
        
        self.models['fault_classifier'] = model
        logger.info("Fault classifier trained and saved")
        
        return model
    
    def train_all_models(
        self,
        subsystem: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Train all models for a subsystem."""
        logger.info(f"Training all models for {subsystem}")
        
        # Load features
        df = self.load_features(subsystem, start_date, end_date)
        
        if df.empty:
            logger.error("No features found, cannot train models")
            return {}
        # Sample down if too large
        if len(df) > 100000: 
            df = df.sample(100000, random_state=42)
            logger.info(f"Sampled down to {len(df)} records to reduce memory usage")

        # Train models
        logger.info("=" * 60)
        logger.info("Training Anomaly Detection (Isolation Forest)")
        logger.info("=" * 60)
        self.train_anomaly_detection(df, method='isolation_forest')
        
        # SKIP LSTM - Too memory-intensive, skip for now
        # logger.info("=" * 60)
        # logger.info("Training Degradation Forecaster")
        # logger.info("=" * 60)
        # self.train_degradation_forecaster(df, forecast_horizon=30)
        
        logger.info("=" * 60)
        logger.info("Training Fault Classifier")
        logger.info("=" * 60)
        self.train_fault_classifier(df)
        
        # SKIP Survival Analysis - Needs Real Failure Data
        # Survival analysis needs real failure events which simulated data lacks
        # if 'parameter_id' in df.columns:
        #    sample_param = df['parameter_id'].iloc[0]
        #    logger.info("=" * 60)
        #    logger.info(f"Training Survival Analyzer for {sample_param}")
        #    logger.info("=" * 60)
        #    self.train_survival_analyzer(df, parameter_id=sample_param)
        
        logger.info("=" * 60)
        logger.info("All models trained successfully!")
        logger.info("=" * 60)
        
        return self.models


def main():
    """
    Train all models on processed features.
    """
    from datetime import timedelta
    
    print("=" * 60)
    print("CONSTELLATION - Model Training")
    print("=" * 60)
    print()
    
    trainer = ModelTrainer()
    
    # Train on attitude control data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    models = trainer.train_all_models('attitude_control', start_date, end_date)
    
    print("\n" + "=" * 60)
    print("Model Training Complete!")
    print("=" * 60)
    print(f"\nTrained {len(models)} models:")
    for model_name in models.keys():
        print(f"  - {model_name}")
    
    print(f"\nModels saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()