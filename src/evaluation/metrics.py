"""
src/evaluation/metrics.py

Full suite of evaluation metrics for clinical binary classification.

KEY UPGRADE over original evaluate.py:
- Adds Brier Score (calibration quality)
- Adds Sensitivity and Specificity separately
- Accuracy alone is insufficient for medical decisions
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss,
    classification_report,
)
from typing import Dict


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Computes:
      - AUC-ROC      : discrimination ability
      - Accuracy     : overall correctness
      - F1 Score     : balance of precision and recall
      - Sensitivity  : survivors correctly identified
      - Specificity  : non-survivors correctly identified
      - Brier Score  : calibration quality (lower is better)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "test_auc": roc_auc_score(y_true, y_prob),
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_f1": f1_score(y_true, y_pred),
        "test_sensitivity": sensitivity,
        "test_specificity": specificity,
        "test_brier_score": brier_score_loss(y_true, y_prob),
    }


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> None:
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print("\n" + "=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"  {name:<25s}: {value:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred,
            target_names=["Did Not Survive", "Survived"],
        )
    )
    print("=" * 50)