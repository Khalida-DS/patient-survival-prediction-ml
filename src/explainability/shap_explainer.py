"""
src/explainability/shap_explainer.py

SHAP-based explainability for the trained model.

KEY UPGRADE over original feature_importances_:
- Original used model.feature_importances_ (global average, no direction)
- SHAP shows direction of effect (does smoking increase or decrease survival?)
- SHAP waterfall shows per-patient explanation (why THIS patient got THIS score)
- This is what clinical reviewers need for auditability
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
from typing import Any, Optional


def compute_shap_values(
    model: Any,
    X_transformed: np.ndarray,
    background_samples: int = 100,
) -> shap.Explanation:
    """
    Computes SHAP values using TreeExplainer.
    Fast and exact for tree-based models like GradientBoosting.
    """
    try:
        explainer = shap.TreeExplainer(
            model,
            data=X_transformed[:background_samples],
            feature_perturbation="interventional",
        )
        shap_values = explainer(X_transformed)
    except Exception:
        # Fallback for calibrated wrappers
        explainer = shap.KernelExplainer(
            model.predict_proba,
            shap.sample(X_transformed, background_samples),
        )
        raw = explainer.shap_values(X_transformed)
        shap_values = shap.Explanation(
            values=raw[1],
            base_values=explainer.expected_value[1],
            data=X_transformed,
        )
    return shap_values


def plot_global_summary(
    shap_values: shap.Explanation,
    feature_names: list,
    output_dir: str,
    max_display: int = 15,
    log_to_mlflow: bool = True,
) -> str:
    """
    Beeswarm summary plot showing global feature importance
    with direction of effect.
    Red = high feature value, Blue = low feature value.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / "shap_summary.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values.values
        if hasattr(shap_values, "values")
        else shap_values,
        features=shap_values.data,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Global Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    if log_to_mlflow:
        mlflow.log_artifact(output_path)

    return output_path


def plot_patient_waterfall(
    shap_values: shap.Explanation,
    patient_index: int,
    feature_names: list,
    output_dir: str,
    patient_id: Optional[str] = None,
) -> str:
    """
    Waterfall plot for a single patient.
    Shows which features pushed the prediction toward
    or away from survival — the per-patient audit trail.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    label = patient_id or f"patient_{patient_index}"
    output_path = str(Path(output_dir) / f"shap_waterfall_{label}.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values.values[patient_index],
            base_values=shap_values.base_values[patient_index]
            if hasattr(shap_values.base_values, "__len__")
            else shap_values.base_values,
            data=shap_values.data[patient_index],
            feature_names=feature_names,
        ),
        show=False,
        max_display=15,
    )
    plt.title(f"SHAP Explanation — {label}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path