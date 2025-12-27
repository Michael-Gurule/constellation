"""
Fault classification model.
XGBoost classifier for diagnosing root causes of anomalies.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple

from src.models.base_model import BaseModel
from src.utils.logging_config import setup_logger
from src.utils.metrics import calculate_classification_metrics

logger = setup_logger(__name__)


class FaultClassifier(BaseModel):
    """
    XGBoost classifier for fault diagnosis.
    Classifies anomalies into fault categories.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1
    ):
        """
        Initialize fault classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
        """
        super().__init__("fault_classifier")
        
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        self.label_encoder = LabelEncoder()
        self.fault_categories = [
            'thermal_stress',
            'mechanical_wear',
            'electrical_fault',
            'software_error',
            'external_disturbance',
            'normal_variation'
        ]
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> 'FaultClassifier':
        """
        Train fault classifier.
        
        Args:
            X: Feature matrix
            y: Fault labels
            validation_split: Validation fraction
        
        Returns:
            Self
        """
        logger.info(f"Training fault classifier on {len(X)} samples")
        
        # Handle NaN values
        X_clean = X.fillna(X.median())
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_clean, y_encoded,
            test_size=validation_split,
            random_state=42,
            stratify=y_encoded
        )
        
        # Train model with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.trained = True
        
        # Get training accuracy
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        
        logger.info(f"Training complete: Train Acc = {train_acc:.3f}, Val Acc = {val_acc:.3f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fault categories.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of predicted fault labels
        """
        X_clean = X.fillna(X.median())
        
        predictions_encoded = self.model.predict(X_clean)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fault probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of class probabilities
        """
        X_clean = X.fillna(X.median())
        
        probabilities = self.model.predict_proba(X_clean)
        
        return probabilities
    
    def get_top_k_predictions(
        self,
        X: pd.DataFrame,
        k: int = 3
    ) -> List[List[Tuple[str, float]]]:
        """
        Get top-k most likely fault categories.
        
        Args:
            X: Feature matrix
            k: Number of top predictions to return
        
        Returns:
            List of lists containing (fault_category, probability) tuples
        """
        probabilities = self.predict_proba(X)
        classes = self.label_encoder.classes_
        
        results = []
        for prob_row in probabilities:
            # Get top-k indices
            top_k_idx = np.argsort(prob_row)[-k:][::-1]
            
            top_k_predictions = [
                (classes[idx], prob_row[idx])
                for idx in top_k_idx
            ]
            
            results.append(top_k_predictions)
        
        return results
    
    def get_feature_importance(self, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            feature_names: Optional list of feature names
        
        Returns:
            DataFrame with feature importance scores
        """
        importance = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate classifier performance.
        
        Args:
            X: Feature matrix
            y: True labels
        
        Returns:
            Dictionary of metrics
        """
        X_clean = X.fillna(X.median())
        
        # Get predictions
        y_encoded = self.label_encoder.transform(y)
        predictions_encoded = self.model.predict(X_clean)
        probabilities = self.model.predict_proba(X_clean)
        
        # Calculate metrics
        self.metrics = calculate_classification_metrics(
            y_encoded,
            predictions_encoded,
            probabilities
        )
        
        # Add per-class metrics
        from sklearn.metrics import classification_report
        report = classification_report(
            y_encoded,
            predictions_encoded,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        self.metrics['per_class'] = report
        
        logger.info(f"Evaluation metrics: Accuracy = {self.metrics['Precision']:.3f}")
        
        return self.metrics


def create_fault_labels(
    df: pd.DataFrame,
    rules: Dict[str, callable] = None
) -> pd.Series:
    """
    Create fault labels from telemetry features using rules.
    
    Args:
        df: DataFrame with features
        rules: Dictionary of fault_category -> rule_function
               rule_function takes df and returns boolean mask
    
    Returns:
        Series with fault labels
    """
    if rules is None:
        # Default rules based on feature patterns
        rules = {
            'thermal_stress': lambda df: (
                (df.get('rolling_std_1h', 0) > df.get('rolling_std_1h', 0).quantile(0.95)) &
                (df.get('is_daylight', 1) == 1)
            ),
            'mechanical_wear': lambda df: (
                (df.get('speed_stability', 0) > df.get('speed_stability', 0).quantile(0.9)) |
                (df.get('friction_indicator', 0) > df.get('friction_indicator', 0).quantile(0.9))
            ),
            'electrical_fault': lambda df: (
                (df.get('value_numeric', 0) == 0) |
                (df.get('signal_dropout', 0) == 1)
            ),
            'external_disturbance': lambda df: (
                (df.get('second_derivative', 0).abs() > df.get('second_derivative', 0).abs().quantile(0.99))
            ),
            'normal_variation': lambda df: pd.Series([True] * len(df), index=df.index)
        }
    
    # Apply rules in priority order
    labels = pd.Series(['normal_variation'] * len(df), index=df.index)
    
    for fault_type, rule in rules.items():
        if fault_type == 'normal_variation':
            continue
        
        try:
            mask = rule(df)
            labels[mask] = fault_type
        except Exception as e:
            logger.warning(f"Error applying rule for {fault_type}: {e}")
    
    return labels