# src/evaluate.py
# --------------------------------------------------
# Patient Survival Prediction - Model Evaluation
# --------------------------------------------------
# WHY THIS FILE EXISTS:
# We will train 3 different models (GradientBoosting,
# XGBoost, LightGBM). Every single one needs the same
# evaluation logic — metrics, plots, reports.
#
# Instead of copy-pasting evaluation code 3 times,
# we write it once here and call it from train.py.
# This is the DRY principle: Don't Repeat Yourself.
# --------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)


# --------------------------------------------------
# 1. COMPUTE METRICS
# --------------------------------------------------
# Takes the real labels (y_test) and the model's
# predictions, returns a clean dictionary of scores.
#
# WHY A DICTIONARY?
# So we can easily log it to MLflow with one call,
# and also print it or save it without reformatting.

def compute_metrics(y_test, y_pred, y_prob):
    """
    Compute all evaluation metrics for a binary classifier.

    Args:
        y_test  : True labels
        y_pred  : Predicted labels (0 or 1)
        y_prob  : Predicted probabilities for class 1

    Returns:
        dict: All metrics in one dictionary
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred),
        "roc_auc":   auc(fpr, tpr),
    }

    return metrics


# --------------------------------------------------
# 2. PRINT REPORT
# --------------------------------------------------
# A clean console summary so you can read results
# in the terminal while the script is running.

def print_report(model_name, metrics, y_test, y_pred):
    """
    Print a formatted evaluation report to the console.

    Args:
        model_name (str): Name of the model being evaluated
        metrics (dict)  : Output from compute_metrics()
        y_test          : True labels
        y_pred          : Predicted labels
    """
    print(f"\n{'='*50}")
    print(f"  {model_name} — Model Performance")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1_score']:.4f}")
    print(f"  ROC AUC   : {metrics['roc_auc']:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 3. PLOT ROC CURVE
# --------------------------------------------------
# The ROC curve shows the tradeoff between catching
# true positives (recall) vs. false alarms (false
# positive rate) at different decision thresholds.
#
# A perfect model hugs the top-left corner.
# A random model follows the diagonal line.
# We save this as a PNG and log it to MLflow so you
# can see it directly in the UI under Artifacts.

def plot_roc_curve(model_name, y_test, y_prob, output_dir):
    """
    Generate and save a ROC curve plot.

    Args:
        model_name (str) : Used in the plot title and filename
        y_test           : True labels
        y_prob           : Predicted probabilities for class 1
        output_dir (str) : Folder to save the PNG

    Returns:
        str: Path to the saved PNG file
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--',
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {model_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"roc_curve_{model_name}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ ROC curve saved → {plot_path}")
    return plot_path


# --------------------------------------------------
# 4. PLOT FEATURE IMPORTANCE
# --------------------------------------------------
# Feature importance tells you which input columns
# the model relied on most when making predictions.
#
# WHY THIS MATTERS:
# - Features with near-zero importance can be removed
#   (simplifies the model, reduces noise)
# - High-importance features are worth understanding
#   deeply — they drive your predictions
# - In clinical ML, this is also important for
#   explainability to doctors or stakeholders

def plot_feature_importance(model_name, model, feature_names, output_dir, top_n=20):
    """
    Generate and save a feature importance bar chart.

    Args:
        model_name (str)    : Used in title and filename
        model               : Trained model with feature_importances_ attribute
        feature_names (list): Column names from preprocessing
        output_dir (str)    : Folder to save the PNG
        top_n (int)         : How many top features to show (default 20)

    Returns:
        str: Path to the saved PNG file
    """
    # Get importance scores from the model
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(top_features[::-1], top_importances[::-1], color='steelblue')
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importances — {model_name}')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"feature_importance_{model_name}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ Feature importance saved → {plot_path}")
    return plot_path


# --------------------------------------------------
# 5. FULL EVALUATION PIPELINE
# --------------------------------------------------
# This is the single function that train.py calls.
# It runs ALL of the above in the correct order and
# logs everything to the active MLflow run.
#
# train.py only needs one line:
#   evaluate_and_log(model_name, model, X_test, y_test, ...)

def evaluate_and_log(model_name, model, X_test, y_test, feature_names, output_dir):
    """
    Run full evaluation and log results to MLflow.

    Args:
        model_name (str)    : e.g. 'GradientBoosting', 'XGBoost'
        model               : Trained sklearn-compatible model
        X_test              : Test features
        y_test              : True labels
        feature_names (list): Feature column names
        output_dir (str)    : Folder to save plots
    """
    # 1. Get predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 2. Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_prob)

    # 3. Print to console
    print_report(model_name, metrics, y_test, y_pred)

    # 4. Log metrics to MLflow
    mlflow.log_metrics(metrics)

    # 5. Generate and log ROC curve
    roc_path = plot_roc_curve(model_name, y_test, y_prob, output_dir)
    mlflow.log_artifact(roc_path)

    # 6. Generate and log feature importance
    importance_path = plot_feature_importance(
        model_name, model, feature_names, output_dir
    )
    mlflow.log_artifact(importance_path)

    return metrics
