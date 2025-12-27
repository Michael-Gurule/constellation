"""
System integrator that combines all models and components.
Orchestrates the complete health management workflow.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.models.anomaly_detection import IsolationForestDetector
from src.models.fault_classifier import FaultClassifier
from src.optimization.health_scoring import HealthScorer
from src.optimization.alert_generator import AlertGenerator, AlertSeverity
from src.optimization.maintenance_scheduler import MaintenanceScheduler, MaintenanceTask
from src.ingestion.storage_handler import LocalStorageHandler
from config.settings import MODELS_DIR, PROCESSED_DATA_DIR
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ConstellationSystem:
    """
    Integrated satellite health management system.
    Combines all models and components into complete workflow.
    """
    
    def __init__(self):
        # Load trained models
        self.anomaly_detector = None
        self.fault_classifier = None
        
        # Initialize components
        self.health_scorer = HealthScorer()
        self.alert_generator = AlertGenerator()
        self.scheduler = MaintenanceScheduler()
        self.storage = LocalStorageHandler()
        
        self.results = {}
        
        logger.info("CONSTELLATION system initialized")
    
    def load_models(self):
        """Load trained ML models."""
        logger.info("Loading trained models...")
        
        # Load anomaly detector
        anomaly_path = MODELS_DIR / "isolation_forest.pkl"
        if anomaly_path.exists():
            self.anomaly_detector = IsolationForestDetector()
            self.anomaly_detector.load(anomaly_path)
            logger.info("✓ Loaded anomaly detector")
        else:
            logger.warning("Anomaly detector not found")
        
        # Load fault classifier
        classifier_path = MODELS_DIR / "fault_classifier.pkl"
        if classifier_path.exists():
            self.fault_classifier = FaultClassifier()
            self.fault_classifier.load(classifier_path)
            logger.info("✓ Loaded fault classifier")
        else:
            logger.warning("Fault classifier not found")
    
    def load_latest_features(self, subsystem: str = 'attitude_control') -> pd.DataFrame:
        """
        Load most recent processed features.
        
        Args:
            subsystem: Subsystem name
        
        Returns:
            DataFrame with features
        """
        pattern = f"{subsystem}_features_*.parquet"
        feature_files = list(PROCESSED_DATA_DIR.glob(pattern))
        
        if not feature_files:
            logger.error(f"No feature files found for {subsystem}")
            return pd.DataFrame()
        
        latest_file = max(feature_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_parquet(latest_file)
        
        logger.info(f"Loaded {len(df)} records from {latest_file.name}")
        return df
    
    def run_anomaly_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run anomaly detection on features.
        
        Args:
            df: DataFrame with features
        
        Returns:
            DataFrame with anomaly scores
        """
        if self.anomaly_detector is None:
            logger.error("Anomaly detector not loaded")
            return df
        
        logger.info("Running anomaly detection...")
        
        # Select features
        exclude_cols = ['timestamp', 'value', 'parameter_id', 'parameter_name', 'subsystem']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        feature_cols = df[feature_cols].select_dtypes(include=['float64', 'int64']).columns
        
        X = df[feature_cols]
        
        # Get predictions
        anomaly_scores = self.anomaly_detector.score_samples(X)
        predictions = self.anomaly_detector.predict(X)
        
        # Normalize scores to 0-1 range
        normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        normalized_scores = 1 - normalized_scores  # Invert so high = anomalous
        
        df['anomaly_score'] = normalized_scores
        df['is_anomaly'] = (predictions == -1).astype(int)
        
        anomaly_count = df['is_anomaly'].sum()
        logger.info(f"Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.1f}%)")
        
        return df
    
    def run_fault_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run fault classification on anomalies.
        
        Args:
            df: DataFrame with features
        
        Returns:
            DataFrame with fault predictions
        """
        if self.fault_classifier is None:
            logger.error("Fault classifier not loaded")
            return df
        
        logger.info("Running fault classification...")
        
        # Select features
        exclude_cols = ['timestamp', 'value', 'parameter_id', 'parameter_name', 
                       'subsystem', 'anomaly_score', 'is_anomaly']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        feature_cols = df[feature_cols].select_dtypes(include=['float64', 'int64']).columns
        
        X = df[feature_cols]
        
        # Get predictions
        fault_predictions = self.fault_classifier.predict(X)
        fault_probabilities = self.fault_classifier.predict_proba(X)
        
        df['fault_type'] = fault_predictions
        df['fault_confidence'] = fault_probabilities.max(axis=1)
        
        logger.info(f"Classified {len(df)} samples")
        logger.info(f"Fault distribution:\n{pd.Series(fault_predictions).value_counts()}")
        
        return df
    
    def calculate_health_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate health scores for all components.
        
        Args:
            df: DataFrame with predictions
        
        Returns:
            DataFrame with health scores
        """
        logger.info("Calculating health scores...")
        
        # Add component identifier if missing
        if 'component' not in df.columns:
            df['component'] = df['parameter_id']
        
        health_report = self.health_scorer.generate_health_report(
            df,
            anomaly_col='anomaly_score',
            component_col='component'
        )
        
        self.results['health_scores'] = health_report
        
        logger.info("\nHealth Summary:")
        logger.info(health_report.to_string())
        
        return health_report
    
    def generate_alerts(self, df: pd.DataFrame, health_scores: pd.DataFrame) -> List:
        """
        Generate alerts based on predictions and health scores.
        
        Args:
            df: DataFrame with predictions
            health_scores: DataFrame with health scores
        
        Returns:
            List of alerts
        """
        logger.info("Generating alerts...")
        
        # Check anomaly alerts
        anomaly_alerts = self.alert_generator.check_anomaly_alerts(
            df[df['is_anomaly'] == 1],
            component_col='component',
            score_col='anomaly_score'
        )
        
        # Check health alerts
        health_alerts = self.alert_generator.check_health_alerts(
            health_scores,
            component_col='component',
            score_col='health_score'
        )
        
        all_alerts = anomaly_alerts + health_alerts
        
        logger.info(f"Generated {len(all_alerts)} alerts:")
        logger.info(f"  Critical: {sum(1 for a in all_alerts if a.severity == AlertSeverity.CRITICAL)}")
        logger.info(f"  Warning: {sum(1 for a in all_alerts if a.severity == AlertSeverity.WARNING)}")
        
        self.results['alerts'] = all_alerts
        
        return all_alerts
    
    def schedule_maintenance(self, health_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Create optimized maintenance schedule.
        
        Args:
            health_scores: DataFrame with health scores
        
        Returns:
            DataFrame with maintenance schedule
        """
        logger.info("Creating maintenance schedule...")
        
        # Create maintenance tasks for components below threshold
        threshold = 70
        needs_maintenance = health_scores[health_scores['health_score'] < threshold]
        
        for idx, row in needs_maintenance.iterrows():
            urgency = 1.0 - (row['health_score'] / 100.0)
            impact = 0.8  # High impact assumption
            
            task = MaintenanceTask(
                task_id=f"MAINT_{row['component']}",
                component=row['component'],
                urgency=urgency,
                impact=impact,
                duration_hours=2.0,
                earliest_start=datetime.now(),
                latest_start=datetime.now() + timedelta(days=30)
            )
            self.scheduler.add_task(task)
        
        if self.scheduler.tasks:
            schedule = self.scheduler.optimize_schedule(
                max_concurrent_tasks=2,
                time_horizon_days=30
            )
            
            self.results['maintenance_schedule'] = schedule
            
            logger.info(f"Scheduled {len(schedule)} maintenance tasks")
            return schedule
        else:
            logger.info("No maintenance tasks required")
            return pd.DataFrame()
    
    def run_complete_analysis(self, subsystem: str = 'attitude_control') -> Dict:
        """
        Run complete end-to-end analysis.
        
        Args:
            subsystem: Subsystem to analyze
        
        Returns:
            Dictionary with all results
        """
        logger.info("=" * 60)
        logger.info("CONSTELLATION - Complete System Analysis")
        logger.info("=" * 60)
        
        # Load models
        self.load_models()
        
        # Load data
        logger.info(f"\nStep 1: Loading {subsystem} data...")
        df = self.load_latest_features(subsystem)
        
        if df.empty:
            logger.error("No data loaded, aborting analysis")
            return {}
        
        # Sample if too large
        if len(df) > 50000:
            df = df.sample(50000, random_state=42)
            logger.info(f"Sampled to {len(df)} records")
        
        # Run anomaly detection
        logger.info("\nStep 2: Anomaly Detection...")
        df = self.run_anomaly_detection(df)
        
        # Run fault classification
        logger.info("\nStep 3: Fault Classification...")
        df = self.run_fault_classification(df)
        
        # Calculate health scores
        logger.info("\nStep 4: Health Scoring...")
        health_scores = self.calculate_health_scores(df)
        
        # Generate alerts
        logger.info("\nStep 5: Alert Generation...")
        alerts = self.generate_alerts(df, health_scores)
        
        # Schedule maintenance
        logger.info("\nStep 6: Maintenance Scheduling...")
        schedule = self.schedule_maintenance(health_scores)
        
        # Store results
        self.results['predictions'] = df
        self.results['timestamp'] = datetime.now()
        
        logger.info("\n" + "=" * 60)
        logger.info("Analysis Complete!")
        logger.info("=" * 60)
        
        return self.results
    
    def export_results(self, output_dir: Path = None):
        """
        Export all results to files.
        
        Args:
            output_dir: Directory to save results
        """
        if output_dir is None:
            output_dir = PROCESSED_DATA_DIR / "analysis_results"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export predictions
        if 'predictions' in self.results:
            pred_path = output_dir / f"predictions_{timestamp}.parquet"
            self.results['predictions'].to_parquet(pred_path)
            logger.info(f"Saved predictions to {pred_path}")
        
        # Export health scores
        if 'health_scores' in self.results:
            health_path = output_dir / f"health_scores_{timestamp}.csv"
            self.results['health_scores'].to_csv(health_path, index=False)
            logger.info(f"Saved health scores to {health_path}")
        
        # Export alerts
        if 'alerts' in self.results:
            alerts_path = output_dir / f"alerts_{timestamp}.json"
            self.alert_generator.export_alerts(str(alerts_path))
            logger.info(f"Saved alerts to {alerts_path}")
        
        # Export schedule
        if 'maintenance_schedule' in self.results and not self.results['maintenance_schedule'].empty:
            schedule_path = output_dir / f"maintenance_schedule_{timestamp}.csv"
            self.results['maintenance_schedule'].to_csv(schedule_path, index=False)
            logger.info(f"Saved maintenance schedule to {schedule_path}")
        
        logger.info(f"\n✓ All results exported to {output_dir}")


def main():
    """
    Run integrated system analysis.
    """
    system = ConstellationSystem()
    results = system.run_complete_analysis(subsystem='attitude_control')
    system.export_results()
    
    print("\n" + "=" * 60)
    print("System Analysis Summary")
    print("=" * 60)
    
    if 'health_scores' in results:
        print(f"\nComponents Analyzed: {len(results['health_scores'])}")
        print(f"Average Health Score: {results['health_scores']['health_score'].mean():.1f}")
    
    if 'alerts' in results:
        print(f"\nTotal Alerts: {len(results['alerts'])}")
    
    if 'maintenance_schedule' in results:
        print(f"\nMaintenance Tasks Scheduled: {len(results['maintenance_schedule'])}")


if __name__ == "__main__":
    main()