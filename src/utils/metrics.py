"""
Performance metrics and evaluation utilities.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, mean_absolute_error, 
    mean_squared_error, r2_score
)
from typing import Dict, Tuple


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression performance metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metric names and values
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def calculate_classification_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate classification performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for ROC-AUC)
    
    Returns:
        Dictionary of metric names and values
    """
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
    
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
            metrics['ROC_AUC'] = roc_auc
        except ValueError:
            # Handle case where ROC-AUC can't be calculated
            pass
    
    return metrics


def calculate_anomaly_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    positive_label: int = 1
) -> Dict[str, float]:
    """
    Calculate metrics specifically for anomaly detection.
    
    Args:
        y_true: True labels (0=normal, 1=anomaly)
        y_pred: Predicted labels
        positive_label: Label indicating anomaly class
    
    Returns:
        Dictionary of metric names and values
    """
    precision = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    
    # False Positive Rate (critical for operations)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'False_Positive_Rate': fpr
    }