"""
src/explainability/shap_explainer.py

SHAP-based explainability for the trained model.

KEY UPGRADE over original feature_importances_:
- Original used model.feature_importances_ (global average, no direction)
- SHAP shows direction of effect (does smoking increase or decrease survival?)
- SHAP waterfall shows per-patient explanation (why THIS patient got THIS score)
- This is what clinical reviewers need for auditability
"""

import logging
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Human-readable labels for the raw dataset columns, used in the
# per-patient "Why this prediction?" chart so a clinician sees
# "Smoker = Yes" instead of "Patient_Smoker" / an encoded number.
_INDICATOR_LETTERS = {"A", "B", "C", "D", "E", "F", "Z"}


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
    import matplotlib.pyplot as plt

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
        import mlflow
        mlflow.log_artifact(output_path)

    return output_path


def _readable_label(feature: str, value: Any) -> str:
    """Turns a raw column name + raw value into a clinician-readable label."""
    if feature == "Patient_Age":
        return f"Age = {int(value)}"
    if feature == "Patient_Body_Mass_Index":
        return f"BMI = {float(value):.1f}"
    if feature == "Number_of_prev_cond":
        return f"Previous conditions = {int(value)}"
    if feature in _INDICATOR_LETTERS:
        return f"Indicator {feature} = {'Yes' if float(value) == 1.0 else 'No'}"
    if feature == "Treated_with_drugs":
        return f"Treatment = {value}"
    if feature == "Patient_Smoker":
        return f"Smoker = {'Yes' if str(value).upper() == 'YES' else 'No'}"
    if feature == "Patient_Rural_Urban":
        return f"Location = {str(value).title()}"
    return f"{feature} = {value}"


def explain_patient(
    model: Any,
    input_df: pd.DataFrame,
    feature_names: List[str],
    background_df: pd.DataFrame,
    calibrated_probability: float,
) -> Dict[str, Any]:
    """
    Explains ONE patient's calibrated survival prediction.

    `model` is a fitted sklearn CalibratedClassifierCV wrapping a
    Pipeline(preprocessor, classifier) — `model.calibrated_classifiers_`
    holds one fold per cross-validation split used during Platt scaling
    (5 in this project). To explain "the whole calibrated model" rather
    than an arbitrary single fold, this averages SHAP attributions
    across every fold.

    SHAP's TreeExplainer with model_output="probability" gives exact,
    additive attributions in probability space for each fold's raw
    (pre-calibration) classifier — they sum exactly to that fold's own
    predicted probability. Averaging across folds gives a probability
    estimate close to, but not identical to, the final Platt-calibrated
    probability (calibration is a separate, non-linear correction on
    top). To make the chart arithmetic match the calibrated probability
    shown elsewhere in the app exactly, the averaged contributions are
    rescaled by a single constant so baseline + sum(contributions) ==
    `calibrated_probability`. This preserves each feature's relative
    size and direction; only the overall scale changes.

    Returns a dict:
      baseline_probability   — avg. predicted survival across background_df
      calibrated_probability — the number passed in, echoed back
      num_folds_averaged     — how many calibrated_classifiers_ were used
      features                — list of {label, value} dicts, `value` in
                                 probability points, sorted by |value| desc
    """
    fold_base_values = []
    fold_contributions = []

    for calibrated_classifier in model.calibrated_classifiers_:
        inner_pipeline = calibrated_classifier.estimator
        preprocessor = inner_pipeline.named_steps["preprocessor"]
        classifier = inner_pipeline.named_steps["classifier"]

        background_transformed = preprocessor.transform(background_df)
        patient_transformed = preprocessor.transform(input_df)

        explainer = shap.TreeExplainer(
            classifier,
            data=background_transformed,
            model_output="probability",
            feature_perturbation="interventional",
        )
        explanation = explainer(patient_transformed)

        fold_base_values.append(float(np.asarray(explanation.base_values).ravel()[0]))
        fold_contributions.append(np.asarray(explanation.values)[0])

    baseline_probability = float(np.mean(fold_base_values))
    mean_contributions = np.mean(fold_contributions, axis=0)
    raw_model_probability = baseline_probability + float(mean_contributions.sum())

    span = raw_model_probability - baseline_probability
    if abs(span) > 1e-6:
        scale = (calibrated_probability - baseline_probability) / span
    else:
        scale = 1.0
    if scale < 0:
        # Calibration flipped the direction relative to baseline — an
        # edge case. Fall back to the unscaled raw estimate rather than
        # showing a nonsensical inverted chart.
        scale = 1.0
        baseline_probability = raw_model_probability - float(mean_contributions.sum())

    scaled_contributions = mean_contributions * scale

    patient_row = input_df.iloc[0]
    features = [
        {
            "label": _readable_label(name, patient_row[name]),
            "value": float(scaled_contributions[i]),
        }
        for i, name in enumerate(feature_names)
    ]
    features.sort(key=lambda f: abs(f["value"]), reverse=True)

    return {
        "baseline_probability": baseline_probability,
        "calibrated_probability": calibrated_probability,
        "num_folds_averaged": len(model.calibrated_classifiers_),
        "features": features,
    }


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
    import matplotlib.pyplot as plt

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
