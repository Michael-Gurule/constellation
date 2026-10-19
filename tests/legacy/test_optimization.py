"""
Unit tests for optimization modules.
"""

import pytest
import pandas as pd
import numpy as np
import pulp
from datetime import datetime, timedelta

from constellation.legacy.optimization.health_scoring import HealthScorer
from constellation.legacy.optimization.alert_generator import AlertGenerator, AlertSeverity
from constellation.legacy.optimization.maintenance_scheduler import MaintenanceScheduler, MaintenanceTask


def _cbc_runs() -> bool:
    """True when the bundled CBC binary can be executed on this machine."""
    if not pulp.PULP_CBC_CMD().available():
        return False
    problem = pulp.LpProblem("cbc_probe", pulp.LpMinimize)
    problem += pulp.LpVariable("x", 0, 1)
    try:
        problem.solve(pulp.PULP_CBC_CMD(msg=False))
    except (OSError, pulp.PulpSolverError):
        return False
    return True


class TestHealthScorer:
    """Test health scoring."""
    
    def test_calculate_component_health_best_case(self):
        """Test component health calculation at the best case."""
        scorer = HealthScorer()
        
        score = scorer.calculate_component_health(
            anomaly_score=0.0,
            degradation_rate=0.0,
            fault_probability=0.0,
            days_since_maintenance=0
        )
        assert score == 100.0

    @pytest.mark.xfail(
        strict=True,
        reason="Health score bounds are reworked in the decision layer",
    )
    def test_calculate_component_health_worst_case(self):
        """Test component health calculation at the worst case."""
        scorer = HealthScorer()
        
        score = scorer.calculate_component_health(
            anomaly_score=1.0,
            degradation_rate=1.0,
            fault_probability=1.0,
            days_since_maintenance=180
        )
        assert score == 0.0
    
    def test_categorize_health(self):
        """Test health categorization."""
        scorer = HealthScorer()
        
        assert scorer.categorize_health(95) == "EXCELLENT"
        assert scorer.categorize_health(80) == "GOOD"
        assert scorer.categorize_health(65) == "FAIR"
        assert scorer.categorize_health(50) == "DEGRADED"
        assert scorer.categorize_health(30) == "POOR"
        assert scorer.categorize_health(10) == "CRITICAL"


class TestAlertGenerator:
    """Test alert generation."""
    
    def test_generate_alert(self):
        """Test alert creation."""
        generator = AlertGenerator()
        
        alert = generator.generate_alert(
            severity=AlertSeverity.WARNING,
            component="TEST_COMP",
            message="Test alert"
        )
        
        assert alert.severity == AlertSeverity.WARNING
        assert alert.component == "TEST_COMP"
        assert "ALERT_" in alert.alert_id
    
    def test_check_anomaly_alerts(self):
        """Test anomaly alert generation."""
        generator = AlertGenerator()
        
        df = pd.DataFrame({
            'component': ['COMP1', 'COMP2', 'COMP3'],
            'anomaly_score': [0.99, 0.85, 0.5]
        })
        
        alerts = generator.check_anomaly_alerts(df)
        
        assert len(alerts) == 2  # Two above threshold
        assert any(a.severity == AlertSeverity.CRITICAL for a in alerts)


class TestMaintenanceScheduler:
    """Test maintenance scheduling."""
    
    def test_add_task(self):
        """Test adding tasks."""
        scheduler = MaintenanceScheduler()
        
        task = MaintenanceTask(
            task_id="TEST",
            component="COMP1",
            urgency=0.8,
            impact=0.7,
            duration_hours=2.0,
            earliest_start=datetime.now(),
            latest_start=datetime.now() + timedelta(days=7)
        )
        
        scheduler.add_task(task)
        
        assert len(scheduler.tasks) == 1
        assert scheduler.tasks[0].task_id == "TEST"
    
    @pytest.mark.skipif(
        not _cbc_runs(),
        reason="CBC solver binary is not available on this platform",
    )
    def test_optimize_schedule(self):
        """Test schedule optimization."""
        scheduler = MaintenanceScheduler()
        
        # Add multiple tasks
        for i in range(3):
            task = MaintenanceTask(
                task_id=f"TASK_{i}",
                component=f"COMP_{i}",
                urgency=0.5 + i * 0.2,
                impact=0.6,
                duration_hours=2.0,
                earliest_start=datetime.now(),
                latest_start=datetime.now() + timedelta(days=7)
            )
            scheduler.add_task(task)
        
        schedule = scheduler.optimize_schedule(max_concurrent_tasks=2, time_horizon_days=7)
        
        assert len(schedule) == 3
        assert 'start_time' in schedule.columns
        assert 'end_time' in schedule.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])