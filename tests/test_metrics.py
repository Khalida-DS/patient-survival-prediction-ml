"""
tests/test_metrics.py

Validates metric computation on synthetic data with known ground truth.
"""

import pytest
import numpy as np
from src.evaluation.metrics import compute_metrics


def test_perfect_classifier():
    """A perfect classifier: AUC=1.0, F1=1.0, Brier=0.0."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    m = compute_metrics(y_true, y_pred, y_prob)
    assert m["test_auc"] == pytest.approx(1.0)
    assert m["test_f1"] == pytest.approx(1.0)
    assert m["test_brier_score"] == pytest.approx(0.0)


def test_random_classifier():
    """A random classifier should have AUC near 0.5."""
    np.random.seed(42)
    n = 1000
    y_true = np.random.randint(0, 2, n)
    y_prob = np.random.uniform(0, 1, n)
    y_pred = (y_prob >= 0.5).astype(int)
    m = compute_metrics(y_true, y_pred, y_prob)
    assert 0.4 < m["test_auc"] < 0.6


def test_sensitivity_and_specificity():
    """Perfect predictions: sensitivity=1, specificity=1."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])
    m = compute_metrics(y_true, y_pred, y_prob)
    assert m["test_sensitivity"] == pytest.approx(1.0)
    assert m["test_specificity"] == pytest.approx(1.0)


def test_output_keys():
    """All required metric keys must be present."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_prob = np.array([0.2, 0.8, 0.6, 0.4])
    m = compute_metrics(y_true, y_pred, y_prob)
    required = {
        "test_auc", "test_accuracy", "test_f1",
        "test_sensitivity", "test_specificity",
        "test_brier_score",
    }
    assert required.issubset(set(m.keys()))


def test_all_metrics_in_range():
    """All metric values should be in [0, 1]."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.3, 0.4, 0.85])
    m = compute_metrics(y_true, y_pred, y_prob)
    for key, val in m.items():
        assert 0.0 <= val <= 1.0, f"{key} = {val} out of range"