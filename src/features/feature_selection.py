"""
Feature selection and importance analysis.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from typing import List, Dict, Tuple

from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class FeatureSelector:
    """
    Feature selection utilities.
    """
    
    def __init__(self):
        logger.info("FeatureSelector initialized")
    
    def calculate_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str = 'classification'
    ) -> pd.DataFrame:
        """
        Calculate feature importance using Random Forest.
        
        Args:
            X: Feature matrix
            y: Target variable
            task: 'classification' or 'regression'
        
        Returns:
            DataFrame with feature importance scores
        """
        logger.info(f"Calculating feature importance ({task})")
        
        # Train Random Forest
        if task == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Handle NaN values
        X_clean = X.fillna(X.median())
        
        model.fit(X_clean, y)
        
        # Get importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        logger.info(f"Top 5 features: {list(importance_df.head()['feature'])}")
        return importance_df
    
    def calculate_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str = 'classification'
    ) -> pd.DataFrame:
        """
        Calculate mutual information between features and target.
        
        Args:
            X: Feature matrix
            y: Target variable
            task: 'classification' or 'regression'
        
        Returns:
            DataFrame with MI scores
        """
        logger.info(f"Calculating mutual information ({task})")
        
        # Handle NaN values
        X_clean = X.fillna(X.median())
        
        # Calculate MI
        if task == 'classification':
            mi_scores = mutual_info_classif(X_clean, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X_clean, y, random_state=42)
        
        mi_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        })
        
        mi_df = mi_df.sort_values('mutual_info', ascending=False)
        
        return mi_df
    
    def select_top_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 50,
        method: str = 'importance'
    ) -> List[str]:
        """
        Select top N features.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_features: Number of features to select
            method: 'importance' or 'mutual_info'
        
        Returns:
            List of selected feature names
        """
        if method == 'importance':
            scores = self.calculate_feature_importance(X, y)
        else:
            scores = self.calculate_mutual_information(X, y)
        
        top_features = list(scores.head(n_features)['feature'])
        
        logger.info(f"Selected {len(top_features)} features using {method}")
        return top_features
    
    def remove_correlated_features(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95
    ) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            df: DataFrame with features
            threshold: Correlation threshold
        
        Returns:
            List of features to keep
        """
        logger.info(f"Removing features with correlation > {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = df.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > threshold)
        ]
        
        to_keep = [col for col in df.columns if col not in to_drop]
        
        logger.info(f"Removed {len(to_drop)} correlated features, kept {len(to_keep)}")
        return to_keep
    
    def get_low_variance_features(
        self,
        df: pd.DataFrame,
        threshold: float = 0.01
    ) -> List[str]:
        """
        Identify low variance features.
        
        Args:
            df: DataFrame with features
            threshold: Variance threshold
        
        Returns:
            List of low variance features
        """
        variances = df.var()
        low_variance = variances[variances < threshold].index.tolist()
        
        logger.info(f"Found {len(low_variance)} low variance features")
        return low_variance