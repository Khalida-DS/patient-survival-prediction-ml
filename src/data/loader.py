"""
src/data/loader.py

Loads and cleans the real Survival dataset.

Data quality issues discovered in EDA:
- Treated_with_drugs: has trailing whitespace (e.g. "DX1 ") — stripped
- Patient_Smoker: "Cannot say" rows have 100% survival rate (data artifact) — dropped
- Patient_mental_condition: only one value ("Stable") — dropped (useless feature)
- Patient_Age: 9 rows with age > 120 (likely data entry errors) — capped at 100
- ID columns: ID_Patient_Care_Situation, Patient_ID — dropped (not features)
- Diagnosed_Condition: near-zero correlation with survival — dropped
"""

import logging
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Features actually used for modeling
NUMERIC_FEATURES = [
    "Patient_Age",
    "Patient_Body_Mass_Index",
    "Number_of_prev_cond",
    "A", "B", "C", "D", "E", "F", "Z",
]

CATEGORICAL_FEATURES = [
    "Treated_with_drugs",
    "Patient_Smoker",
    "Patient_Rural_Urban",
]

TARGET_COL = "Survived_1_year"

# Columns to drop (IDs, constant, near-zero correlation)
DROP_COLS = [
    "ID_Patient_Care_Situation",
    "Patient_ID",
    "Diagnosed_Condition",
    "Patient_mental_condition",  # constant — only value is "Stable"
]


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(data_path: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and clean the Survival dataset.
    Returns a clean DataFrame ready for modeling.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")

    # ── Strip whitespace from drug column ────────────────────────────────
    # Raw data has values like "DX1 " (trailing space)
    df["Treated_with_drugs"] = df["Treated_with_drugs"].str.strip()

    # ── Drop useless columns ──────────────────────────────────────────────
    df = df.drop(
        columns=[c for c in DROP_COLS if c in df.columns],
        errors="ignore",
    )

    # ── Drop "Cannot say" smokers ─────────────────────────────────────────
    # These rows have 100% survival rate — a data artifact, not a real signal
    before = len(df)
    df = df[df["Patient_Smoker"] != "Cannot say"]
    logger.info(f"Dropped {before - len(df)} 'Cannot say' smoker rows")

    # ── Cap age outliers ──────────────────────────────────────────────────
    # 9 rows with age > 120 (max 149) — almost certainly data entry errors
    outliers = (df["Patient_Age"] > 100).sum()
    if outliers > 0:
        df["Patient_Age"] = df["Patient_Age"].clip(upper=100)
        logger.info(f"Capped {outliers} age outliers to 100")

    # ── Validate required columns ─────────────────────────────────────────
    all_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(
        f"Clean data: {len(df)} rows | "
        f"Positive rate: {df[TARGET_COL].mean():.2%} | "
        f"Features: {len(NUMERIC_FEATURES + CATEGORICAL_FEATURES)}"
    )
    return df


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[target_column]
    return X, y
