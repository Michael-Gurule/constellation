"""
Survival analysis for component time-to-failure prediction.
Implements Cox Proportional Hazards model.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from typing import Dict, Tuple

from src.models.base_model import BaseModel
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class SurvivalAnalyzer(BaseModel):
    """
    Cox Proportional Hazards model for survival analysis.
    Predicts time-to-failure for satellite components.
    """
    
    def __init__(self, penalizer: float = 0.1):
        """
        Initialize survival analyzer.
        
        Args:
            penalizer: L2 penalty for regularization
        """
        super().__init__("survival_analyzer")
        
        self.penalizer = penalizer
        self.model = CoxPHFitter(penalizer=penalizer)
        self.km_fitter = KaplanMeierFitter()
    
    def train(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str,
        feature_cols: list = None
    ) -> 'SurvivalAnalyzer':
        """
        Train Cox model.
        
        Args:
            df: DataFrame with survival data
            duration_col: Column with time durations
            event_col: Column with event indicator (1 = failed, 0 = censored)
            feature_cols: List of feature columns (if None, uses all numeric columns)
        
        Returns:
            Self
        """
        logger.info(f"Training survival model on {len(df)} samples")
        
        # Select features
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c not in [duration_col, event_col]]
        
        # Prepare data
        survival_df = df[[duration_col, event_col] + feature_cols].copy()
        survival_df = survival_df.fillna(survival_df.median())
        
        # Fit model
        self.model.fit(
            survival_df,
            duration_col=duration_col,
            event_col=event_col
        )
        
        self.trained = True
        logger.info("Training complete")
        logger.info(f"Concordance index: {self.model.concordance_index_:.3f}")
        
        return self
    
    def predict_survival(
        self,
        X: pd.DataFrame,
        times: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Predict survival probabilities.
        
        Args:
            X: Feature matrix
            times: Time points for predictions (if None, uses training times)
        
        Returns:
            DataFrame with survival probabilities
        """
        X_clean = X.fillna(X.median())
        
        survival_probs = self.model.predict_survival_function(X_clean, times=times)
        
        return survival_probs
    
    def predict_median_survival(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict median survival time.
        
        Args:
            X: Feature matrix
        
        Returns:
            Series with median survival times
        """
        X_clean = X.fillna(X.median())
        
        median_times = self.model.predict_median(X_clean)
        
        return median_times
    
    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk scores (higher = higher risk of failure).
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of risk scores
        """
        X_clean = X.fillna(X.median())
        
        # Partial hazard (log-risk)
        risk_scores = self.model.predict_log_partial_hazard(X_clean)
        
        return risk_scores.values
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (hazard ratios).
        
        Returns:
            DataFrame with feature coefficients and hazard ratios
        """
        summary = self.model.summary.copy()
        summary['hazard_ratio'] = np.exp(summary['coef'])
        
        # Sort by absolute coefficient
        summary['abs_coef'] = summary['coef'].abs()
        summary = summary.sort_values('abs_coef', ascending=False)
        
        return summary[['coef', 'hazard_ratio', 'p']]
    
    def plot_survival_curves(self, X: pd.DataFrame, save_path: str = None):
        """
        Plot survival curves for samples.
        
        Args:
            X: Feature matrix (small sample for visualization)
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt
        
        survival_funcs = self.predict_survival(X)
        
        plt.figure(figsize=(10, 6))
        for idx in survival_funcs.columns[:10]:  # Plot first 10 samples
            plt.plot(survival_funcs.index, survival_funcs[idx], alpha=0.5)
        
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title('Survival Curves')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Survival curves saved to {save_path}")
        else:
            plt.show()
    
    def evaluate(
        self,
        df: pd.DataFrame,
        duration_col: str,
        event_col: str
    ) -> Dict:
        """
        Evaluate survival model.
        
        Args:
            df: DataFrame with survival data
            duration_col: Duration column
            event_col: Event indicator column
        
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        feature_cols = [c for c in df.columns if c not in [duration_col, event_col]]
        X = df[feature_cols]
        
        risk_scores = self.predict_risk_score(X)
        
        # Calculate concordance index
        c_index = concordance_index(
            df[duration_col],
            -risk_scores,  # Negative because higher risk = lower survival
            df[event_col]
        )
        
        self.metrics = {
            'concordance_index': c_index,
            'model_log_likelihood': self.model.log_likelihood_,
            'AIC': self.model.AIC_,
            'num_events': df[event_col].sum(),
            'num_censored': (df[event_col] == 0).sum()
        }
        
        logger.info(f"Evaluation metrics: {self.metrics}")
        return self.metrics


def create_survival_dataset(
    df: pd.DataFrame,
    parameter_id: str,
    failure_threshold: float = None,
    max_duration: int = None
) -> pd.DataFrame:
    """
    Create survival analysis dataset from telemetry data.
    
    Args:
        df: Telemetry DataFrame with features
        parameter_id: Parameter to analyze
        failure_threshold: Value threshold indicating failure
        max_duration: Maximum observation time (censoring point)
    
    Returns:
        DataFrame formatted for survival analysis
    """
    # Filter to one parameter
    param_df = df[df['parameter_id'] == parameter_id].copy()
    param_df = param_df.sort_values('timestamp')
    
    # Create duration (time since start)
    param_df['duration'] = (
        (param_df['timestamp'] - param_df['timestamp'].iloc[0]).dt.total_seconds() / 3600
    )  # Convert to hours
    
    # Define failure event
    if failure_threshold:
        param_df['event'] = (param_df['value_numeric'] > failure_threshold).astype(int)
    else:
        # Use anomaly indicators if available
        if 'is_anomaly' in param_df.columns:
            param_df['event'] = param_df['is_anomaly']
        else:
            param_df['event'] = 0
    
    # Apply censoring
    if max_duration:
        param_df.loc[param_df['duration'] > max_duration, 'duration'] = max_duration
        param_df.loc[param_df['duration'] == max_duration, 'event'] = 0  # Censored
    
    return param_df