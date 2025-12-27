"""
Maintenance scheduling optimizer.
Uses constraint satisfaction to optimize maintenance timing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class MaintenanceTask:
    """
    Represents a maintenance task.
    """
    
    def __init__(
        self,
        task_id: str,
        component: str,
        urgency: float,  # 0-1, higher = more urgent
        impact: float,   # 0-1, higher = more impact if delayed
        duration_hours: float,
        earliest_start: datetime,
        latest_start: datetime
    ):
        self.task_id = task_id
        self.component = component
        self.urgency = urgency
        self.impact = impact
        self.duration_hours = duration_hours
        self.earliest_start = earliest_start
        self.latest_start = latest_start
        
        # Risk score (urgency × impact)
        self.risk_score = urgency * impact
    
    def __repr__(self):
        return f"Task({self.task_id}, {self.component}, risk={self.risk_score:.2f})"


class MaintenanceScheduler:
    """
    Optimizes maintenance task scheduling under constraints.
    """
    
    def __init__(self):
        self.tasks: List[MaintenanceTask] = []
        self.schedule: pd.DataFrame = None
        
        logger.info("MaintenanceScheduler initialized")
    
    def add_task(self, task: MaintenanceTask):
        """Add a maintenance task to the schedule."""
        self.tasks.append(task)
        logger.debug(f"Added task: {task}")
    
    def add_tasks_from_predictions(
        self,
        predictions_df: pd.DataFrame,
        anomaly_threshold: float = 0.8,
        risk_threshold: float = 0.5
    ):
        """
        Create maintenance tasks from model predictions.
        
        Args:
            predictions_df: DataFrame with anomaly scores and risk predictions
            anomaly_threshold: Score above which to create task
            risk_threshold: Risk score threshold
        """
        # Filter to high-risk items
        high_risk = predictions_df[
            (predictions_df.get('anomaly_score', 0) > anomaly_threshold) |
            (predictions_df.get('risk_score', 0) > risk_threshold)
        ]
        
        for idx, row in high_risk.iterrows():
            task = MaintenanceTask(
                task_id=f"MAINT_{idx}",
                component=row.get('component', f'COMP_{idx}'),
                urgency=row.get('urgency', 0.5),
                impact=row.get('impact', 0.5),
                duration_hours=row.get('duration_hours', 2.0),
                earliest_start=datetime.now(),
                latest_start=datetime.now() + timedelta(days=30)
            )
            self.add_task(task)
        
        logger.info(f"Created {len(self.tasks)} maintenance tasks from predictions")
    
    def optimize_schedule(
        self,
        max_concurrent_tasks: int = 2,
        time_horizon_days: int = 30
    ) -> pd.DataFrame:
        """
        Optimize maintenance schedule using Mixed Integer Programming.
        
        Args:
            max_concurrent_tasks: Maximum tasks that can run simultaneously
            time_horizon_days: Planning horizon in days
        
        Returns:
            DataFrame with optimized schedule
        """
        if not self.tasks:
            logger.warning("No tasks to schedule")
            return pd.DataFrame()
        
        logger.info(f"Optimizing schedule for {len(self.tasks)} tasks")
        
        # Create optimization problem
        prob = LpProblem("Maintenance_Scheduling", LpMinimize)
        
        # Time slots (hours)
        time_slots = list(range(time_horizon_days * 24))
        
        # Decision variables: task i starts at time t
        start_vars = {}
        for task in self.tasks:
            for t in time_slots:
                var_name = f"start_{task.task_id}_t{t}"
                start_vars[(task.task_id, t)] = LpVariable(var_name, cat='Binary')
        
        # Objective: Minimize weighted delay penalty
        # Penalty increases with urgency and delay time
        delay_penalty = []
        for task in self.tasks:
            earliest_hour = 0  # Simplified: earliest is now
            for t in time_slots:
                delay = max(0, t - earliest_hour)
                penalty = task.risk_score * delay
                delay_penalty.append(penalty * start_vars[(task.task_id, t)])
        
        prob += lpSum(delay_penalty), "Minimize_Risk_Weighted_Delay"
        
        # Constraint 1: Each task starts exactly once
        for task in self.tasks:
            prob += (
                lpSum([start_vars[(task.task_id, t)] for t in time_slots]) == 1,
                f"Start_Once_{task.task_id}"
            )
        
        # Constraint 2: No more than max_concurrent_tasks at any time
        for t in time_slots:
            active_tasks = []
            for task in self.tasks:
                for start_t in range(max(0, t - int(task.duration_hours) + 1), t + 1):
                    if start_t in time_slots:
                        active_tasks.append(start_vars[(task.task_id, start_t)])
            
            if active_tasks:
                prob += (
                    lpSum(active_tasks) <= max_concurrent_tasks,
                    f"Max_Concurrent_t{t}"
                )
        
        # Solve
        prob.solve()
        
        status = LpStatus[prob.status]
        logger.info(f"Optimization status: {status}")
        
        # Extract solution
        scheduled_tasks = []
        for task in self.tasks:
            for t in time_slots:
                if value(start_vars[(task.task_id, t)]) == 1:
                    start_time = datetime.now() + timedelta(hours=t)
                    end_time = start_time + timedelta(hours=task.duration_hours)
                    
                    scheduled_tasks.append({
                        'task_id': task.task_id,
                        'component': task.component,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration_hours': task.duration_hours,
                        'urgency': task.urgency,
                        'impact': task.impact,
                        'risk_score': task.risk_score
                    })
                    break
        
        self.schedule = pd.DataFrame(scheduled_tasks)
        self.schedule = self.schedule.sort_values('start_time')
        
        logger.info(f"Scheduled {len(self.schedule)} tasks")
        return self.schedule
    
    def get_schedule_summary(self) -> Dict:
        """
        Get summary statistics of the schedule.
        
        Returns:
            Dictionary with summary metrics
        """
        if self.schedule is None or self.schedule.empty:
            return {}
        
        summary = {
            'total_tasks': len(self.schedule),
            'total_duration_hours': self.schedule['duration_hours'].sum(),
            'avg_urgency': self.schedule['urgency'].mean(),
            'avg_impact': self.schedule['impact'].mean(),
            'avg_risk_score': self.schedule['risk_score'].mean(),
            'schedule_span_days': (
                (self.schedule['end_time'].max() - self.schedule['start_time'].min()).days
            ),
            'earliest_task': self.schedule['start_time'].min(),
            'latest_task': self.schedule['end_time'].max()
        }
        
        return summary
    
    def plot_schedule(self, save_path: Optional[str] = None):
        """
        Plot Gantt chart of maintenance schedule.
        
        Args:
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        if self.schedule is None or self.schedule.empty:
            logger.warning("No schedule to plot")
            return
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(self.schedule) * 0.3)))
        
        # Plot each task as a horizontal bar
        for idx, row in self.schedule.iterrows():
            start = row['start_time']
            duration = timedelta(hours=row['duration_hours'])
            
            # Color by risk score
            color = plt.cm.RdYlGn_r(row['risk_score'])
            
            ax.barh(
                y=row['task_id'],
                width=duration,
                left=start,
                height=0.5,
                color=color,
                alpha=0.8,
                edgecolor='black'
            )
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
        plt.xticks(rotation=45)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Task ID')
        ax.set_title('Maintenance Schedule (Gantt Chart)')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Schedule plot saved to {save_path}")
        else:
            plt.show()


def create_sample_tasks() -> List[MaintenanceTask]:
    """
    Create sample maintenance tasks for testing.
    
    Returns:
        List of sample tasks
    """
    tasks = [
        MaintenanceTask(
            task_id="TASK_001",
            component="RWA_1",
            urgency=0.9,
            impact=0.8,
            duration_hours=3.0,
            earliest_start=datetime.now(),
            latest_start=datetime.now() + timedelta(days=7)
        ),
        MaintenanceTask(
            task_id="TASK_002",
            component="S_Band_Transmitter",
            urgency=0.6,
            impact=0.7,
            duration_hours=2.0,
            earliest_start=datetime.now() + timedelta(days=1),
            latest_start=datetime.now() + timedelta(days=14)
        ),
        MaintenanceTask(
            task_id="TASK_003",
            component="CMG_2",
            urgency=0.4,
            impact=0.5,
            duration_hours=4.0,
            earliest_start=datetime.now() + timedelta(days=2),
            latest_start=datetime.now() + timedelta(days=30)
        ),
        MaintenanceTask(
            task_id="TASK_004",
            component="RWA_3",
            urgency=0.8,
            impact=0.9,
            duration_hours=3.5,
            earliest_start=datetime.now(),
            latest_start=datetime.now() + timedelta(days=3)
        ),
    ]
    
    return tasks