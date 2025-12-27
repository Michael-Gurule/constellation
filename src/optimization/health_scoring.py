"""
Health scoring system for subsystems and components.
Aggregates model predictions into actionable health metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class HealthScorer:
    """
    Calculates health scores for satellite subsystems.
    """
    
    def __init__(self):
        self.weights = {
            'anomaly_score': 0.4,
            'degradation_trend': 0.3,
            'fault_risk': 0.2,
            'time_since_maintenance': 0.1
        }
        
        logger.info("HealthScorer initialized")
    
    def calculate_component_health(
        self,
        anomaly_score: float = 0.0,
        degradation_rate: float = 0.0,
        fault_probability: float = 0.0,
        days_since_maintenance: int = 0
    ) -> float:
        """
        Calculate health score for a single component.
        
        Args:
            anomaly_score: Anomaly detection score (0-1, higher = more anomalous)
            degradation_rate: Rate of performance degradation (0-1)
            fault_probability: Probability of fault (0-1)
            days_since_maintenance: Days since last maintenance
        
        Returns:
            Health score (0-100, higher = healthier)
        """
        # Normalize time since maintenance (assume 180 days = full cycle)
        time_factor = min(days_since_maintenance / 180.0, 1.0)
        
        # Calculate weighted risk score
        risk_score = (
            self.weights['anomaly_score'] * anomaly_score +
            self.weights['degradation_trend'] * degradation_rate +
            self.weights['fault_risk'] * fault_probability +
            self.weights['time_since_maintenance'] * time_factor
        )
        
        # Convert to health score (invert risk)
        health_score = (1.0 - risk_score) * 100
        
        return max(0, min(100, health_score))
    
    def calculate_subsystem_health(
        self,
        component_scores: Dict[str, float]
    ) -> float:
        """
        Calculate health score for entire subsystem.
        
        Args:
            component_scores: Dictionary of component_id -> health_score
        
        Returns:
            Subsystem health score (0-100)
        """
        if not component_scores:
            return 100.0  # No data = assume healthy
        
        scores = list(component_scores.values())
        
        # Use minimum score (weakest link)
        # Weight toward lower scores to be conservative
        subsystem_health = min(scores) * 0.7 + np.mean(scores) * 0.3
        
        return subsystem_health
    
    def categorize_health(self, score: float) -> str:
        """
        Categorize health score into status levels.
        
        Args:
            score: Health score (0-100)
        
        Returns:
            Status category
        """
        if score >= 90:
            return "EXCELLENT"
        elif score >= 75:
            return "GOOD"
        elif score >= 60:
            return "FAIR"
        elif score >= 40:
            return "DEGRADED"
        elif score >= 20:
            return "POOR"
        else:
            return "CRITICAL"
    
    def generate_health_report(
        self,
        df: pd.DataFrame,
        anomaly_col: str = 'anomaly_score',
        component_col: str = 'component'
    ) -> pd.DataFrame:
        """
        Generate health report for all components.
        
        Args:
            df: DataFrame with component data and scores
            anomaly_col: Column with anomaly scores
            component_col: Column with component IDs
        
        Returns:
            DataFrame with health scores
        """
        logger.info("Generating health report")
        
        health_data = []
        
        for component in df[component_col].unique():
            component_df = df[df[component_col] == component]
            
            # Get latest scores
            latest = component_df.iloc[-1]
            
            anomaly_score = latest.get(anomaly_col, 0.0) if anomaly_col in component_df.columns else 0.0
            
            # Calculate degradation rate (simplified)
            if len(component_df) > 10:
                recent_values = component_df['value_numeric'].tail(100)
                degradation_rate = abs(recent_values.diff().mean()) / recent_values.std()
                degradation_rate = min(degradation_rate, 1.0)
            else:
                degradation_rate = 0.0
            
            # Calculate health score
            health_score = self.calculate_component_health(
                anomaly_score=anomaly_score,
                degradation_rate=degradation_rate,
                fault_probability=0.0,  # Would come from fault classifier
                days_since_maintenance=0  # Would come from maintenance records
            )
            
            health_data.append({
                'component': component,
                'health_score': health_score,
                'status': self.categorize_health(health_score),
                'anomaly_score': anomaly_score,
                'degradation_rate': degradation_rate,
                'last_updated': datetime.now()
            })
        
        health_df = pd.DataFrame(health_data)
        health_df = health_df.sort_values('health_score')
        
        logger.info(f"Generated health report for {len(health_df)} components")
        return health_df