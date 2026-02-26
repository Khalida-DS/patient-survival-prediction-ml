"""
src/data/preprocessor.py

Builds a scikit-learn Pipeline for preprocessing using real dataset columns.

KEY UPGRADE over original preprocess.py (pd.get_dummies):
- OrdinalEncoder handles the 32 drug combination categories robustly
- All imputation fit INSIDE CV folds — no data leakage
- Unknown categories at inference time handled gracefully (unknown_value=-1)
- Missing values in A-F, Z, Number_of_prev_cond (1235 rows) properly imputed
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from typing import List, Tuple

# Real dataset column definitions (after dropping useless columns)
NUMERIC_FEATURES = [
    "Patient_Age",
    "Patient_Body_Mass_Index",
    "Number_of_prev_cond",
    "A", "B", "C", "D", "E", "F", "Z",
]

CATEGORICAL_FEATURES = [
    "Treated_with_drugs",   # 32 combinations of DX1-DX6
    "Patient_Smoker",       # YES / NO
    "Patient_Rural_Urban",  # URBAN / RURAL
]


def build_preprocessor() -> ColumnTransformer:
    """
    Builds a ColumnTransformer that:
    - Imputes missing values (1235 rows have NaN in clinical indicators)
    - Scales numeric features
    - Ordinal-encodes categorical features (handles 32 drug combos)

    Safe to use inside cross-validation — no leakage.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names() -> List[str]:
    """Returns ordered feature names matching ColumnTransformer output."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_feature_columns() -> Tuple[List[str], List[str]]:
    return NUMERIC_FEATURES, CATEGORICAL_FEATURES
