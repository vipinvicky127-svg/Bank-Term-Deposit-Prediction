"""
Evaluation Module
=================
Computes classification metrics, confusion matrices, ROC-AUC scores,
and formats evaluation summaries.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics for classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_pred : array-like
        Predicted binary class labels (0 or 1).
    y_prob : Optional array-like
        Predicted probabilities for the positive class (class 1).

    Returns
    -------
    Dict[str, Any]
        Dictionary of calculated metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_class1': precision_score(y_true, y_pred, zero_division=0),
        'recall_class1': recall_score(y_true, y_pred, zero_division=0),
        'f1_class1': f1_score(y_true, y_pred, zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': cm,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'classification_report': classification_report(
            y_true, y_pred, target_names=['No (0)', 'Yes (1)'], zero_division=0
        )
    }

    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['roc_auc'] = None

    return metrics


def print_evaluation_report(model_name: str, metrics: Dict[str, Any]) -> None:
    """
    Display a clean, formatted evaluation report.
    """
    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS: {model_name.upper()}")
    print("=" * 65)
    print(f"Accuracy Score  : {metrics['accuracy']:.4%}")
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC Score   : {metrics['roc_auc']:.4f}")
    print(f"Precision (Yes) : {metrics['precision_class1']:.4%}")
    print(f"Recall (Yes)    : {metrics['recall_class1']:.4%}")
    print(f"F1-Score (Yes)  : {metrics['f1_class1']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    print("-" * 65)
    print("Confusion Matrix:")
    print(f"  TN: {metrics['true_negatives']:<6} | FP: {metrics['false_positives']:<6}")
    print(f"  FN: {metrics['false_negatives']:<6} | TP: {metrics['true_positives']:<6}")
    print("-" * 65)
    print("Classification Report:")
    print(metrics['classification_report'])
    print("=" * 65 + "\n")
