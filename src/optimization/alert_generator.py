"""
Alert generation system.
Creates prioritized alerts based on model predictions and health scores.
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict
from enum import Enum

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4


class Alert:
    """
    Represents a system alert.
    """
    
    def __init__(
        self,
        alert_id: str,
        severity: AlertSeverity,
        component: str,
        message: str,
        timestamp: datetime = None,
        metadata: Dict = None
    ):
        self.alert_id = alert_id
        self.severity = severity
        self.component = component
        self.message = message
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Alert({self.severity.name}, {self.component}, {self.message[:50]}...)"
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.name,
            'component': self.component,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class AlertGenerator:
    """
    Generates alerts from model predictions and health scores.
    """
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_counter = 0
        
        # Alert thresholds
        self.thresholds = {
            'anomaly_critical': 0.95,
            'anomaly_warning': 0.80,
            'health_critical': 40,
            'health_warning': 60,
            'degradation_critical': 0.8,
            'degradation_warning': 0.5
        }
        
        logger.info("AlertGenerator initialized")
    
    def generate_alert(
        self,
        severity: AlertSeverity,
        component: str,
        message: str,
        metadata: Dict = None
    ) -> Alert:
        """
        Create a new alert.
        
        Args:
            severity: Alert severity
            component: Component identifier
            message: Alert message
            metadata: Additional metadata
        
        Returns:
            Alert object
        """
        self.alert_counter += 1
        alert_id = f"ALERT_{self.alert_counter:06d}"
        
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            component=component,
            message=message,
            metadata=metadata
        )
        
        self.alerts.append(alert)
        logger.info(f"Generated {severity.name} alert: {alert_id}")
        
        return alert
    
    def check_anomaly_alerts(
        self,
        anomaly_scores: pd.DataFrame,
        component_col: str = 'component',
        score_col: str = 'anomaly_score'
    ) -> List[Alert]:
        """
        Generate alerts based on anomaly scores.
        
        Args:
            anomaly_scores: DataFrame with anomaly scores
            component_col: Column with component IDs
            score_col: Column with anomaly scores
        
        Returns:
            List of generated alerts
        """
        alerts = []
        
        for idx, row in anomaly_scores.iterrows():
            score = row[score_col]
            component = row[component_col]
            
            if score >= self.thresholds['anomaly_critical']:
                alert = self.generate_alert(
                    severity=AlertSeverity.CRITICAL,
                    component=component,
                    message=f"Critical anomaly detected (score: {score:.2f}). Immediate investigation required.",
                    metadata={'anomaly_score': score, 'row_index': idx}
                )
                alerts.append(alert)
            
            elif score >= self.thresholds['anomaly_warning']:
                alert = self.generate_alert(
                    severity=AlertSeverity.WARNING,
                    component=component,
                    message=f"Anomaly detected (score: {score:.2f}). Monitoring recommended.",
                    metadata={'anomaly_score': score, 'row_index': idx}
                )
                alerts.append(alert)
        
        return alerts
    
    def check_health_alerts(
        self,
        health_scores: pd.DataFrame,
        component_col: str = 'component',
        score_col: str = 'health_score'
    ) -> List[Alert]:
        """
        Generate alerts based on health scores.
        
        Args:
            health_scores: DataFrame with health scores
            component_col: Column with component IDs
            score_col: Column with health scores
        
        Returns:
            List of generated alerts
        """
        alerts = []
        
        for idx, row in health_scores.iterrows():
            score = row[score_col]
            component = row[component_col]
            status = row.get('status', 'UNKNOWN')
            
            if score <= self.thresholds['health_critical']:
                alert = self.generate_alert(
                    severity=AlertSeverity.CRITICAL,
                    component=component,
                    message=f"Component health critical ({status}, score: {score:.1f}). Maintenance required.",
                    metadata={'health_score': score, 'status': status}
                )
                alerts.append(alert)
            
            elif score <= self.thresholds['health_warning']:
                alert = self.generate_alert(
                    severity=AlertSeverity.WARNING,
                    component=component,
                    message=f"Component health degraded ({status}, score: {score:.1f}). Schedule maintenance.",
                    metadata={'health_score': score, 'status': status}
                )
                alerts.append(alert)
        
        return alerts
    
    def get_active_alerts(
        self,
        min_severity: AlertSeverity = AlertSeverity.INFO
    ) -> List[Alert]:
        """
        Get all active alerts above minimum severity.
        
        Args:
            min_severity: Minimum severity level
        
        Returns:
            List of alerts
        """
        return [
            alert for alert in self.alerts
            if alert.severity.value >= min_severity.value
        ]
    
    def get_alerts_dataframe(self) -> pd.DataFrame:
        """
        Convert alerts to DataFrame.
        
        Returns:
            DataFrame with all alerts
        """
        if not self.alerts:
            return pd.DataFrame()
        
        return pd.DataFrame([alert.to_dict() for alert in self.alerts])
    
    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []
        self.alert_counter = 0
        logger.info("Cleared all alerts")
    
    def export_alerts(self, filepath: str):
        """
        Export alerts to JSON file.
        
        Args:
            filepath: Path to output file
        """
        import json
        
        alerts_data = [alert.to_dict() for alert in self.alerts]
        
        with open(filepath, 'w') as f:
            json.dump(alerts_data, f, indent=2)
        
        logger.info(f"Exported {len(alerts_data)} alerts to {filepath}")