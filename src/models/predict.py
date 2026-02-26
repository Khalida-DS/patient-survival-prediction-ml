"""
src/models/predict.py

Inference module with runtime input validation.
Ranges derived from actual Survival dataset EDA.
"""

import pickle
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Tuple, List

logger = logging.getLogger(__name__)

# Ranges from real dataset (after capping age outliers at 100)
VALID_RANGES = {
    "Patient_Age": (0, 100),
    "Patient_Body_Mass_Index": (1.0, 30.0),
    "Number_of_prev_cond": (1, 5),
}


def load_model_artifacts(config: Dict[str, Any]) -> Tuple[Any, list]:
    artifacts_dir = Path(config["model"]["artifacts_dir"])
    model_path = artifacts_dir / "calibrated_gb_model.pkl"
    features_path = artifacts_dir / "feature_names.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Run train_model.py first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path, "rb") as f:
        feature_names = pickle.load(f)

    return model, feature_names


def validate_input(input_df: pd.DataFrame) -> List[str]:
    """
    Validates a prediction input.
    Returns list of warnings (empty if all clear).
    """
    warnings = []
    row = input_df.iloc[0]

    for col, (min_val, max_val) in VALID_RANGES.items():
        if col not in input_df.columns:
            warnings.append(f"Missing feature: '{col}'")
            continue
        val = row[col]
        if pd.isna(val):
            warnings.append(f"'{col}' is missing — will be imputed.")
        elif val < min_val:
            warnings.append(
                f"'{col}' value {val} is below minimum {min_val}."
            )
        elif val > max_val:
            warnings.append(
                f"'{col}' value {val} is above maximum {max_val}."
            )

    return warnings


def predict(
    model: Any,
    input_df: pd.DataFrame,
) -> Tuple[int, float, List[str]]:
    """
    Runs validated inference on a single patient record.

    Returns:
        prediction (int): 0 = did not survive, 1 = survived
        survival_probability (float): calibrated probability
        warnings (list): input validation warnings
    """
    warnings = validate_input(input_df)
    survival_prob = model.predict_proba(input_df)[0][1]
    prediction = int(survival_prob >= 0.5)
    return prediction, float(survival_prob), warnings