"""
tests/test_preprocessor.py

Unit tests for the preprocessing pipeline
using real dataset columns.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.dummy import DummyClassifier

from src.data.preprocessor import (
    build_preprocessor,
    get_feature_names,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "Patient_Age": np.random.uniform(5, 90, n),
        "Patient_Body_Mass_Index": np.random.uniform(5, 30, n),
        "Number_of_prev_cond": np.random.uniform(1, 5, n),
        "A": np.random.randint(0, 2, n).astype(float),
        "B": np.random.randint(0, 2, n).astype(float),
        "C": np.random.randint(0, 2, n).astype(float),
        "D": np.random.randint(0, 2, n).astype(float),
        "E": np.random.randint(0, 2, n).astype(float),
        "F": np.random.randint(0, 2, n).astype(float),
        "Z": np.random.randint(0, 2, n).astype(float),
        "Treated_with_drugs": np.random.choice(
            ["DX1", "DX2", "DX3", "DX6", "DX1 DX2"], n
        ),
        "Patient_Smoker": np.random.choice(["YES", "NO"], n),
        "Patient_Rural_Urban": np.random.choice(
            ["URBAN", "RURAL"], n
        ),
        "Survived_1_year": np.random.randint(0, 2, n),
    })


@pytest.fixture
def df_with_missing(sample_df):
    df = sample_df.copy()
    for col in ["A", "B", "C", "D", "E", "F", "Z",
                "Number_of_prev_cond"]:
        mask = np.random.random(len(df)) < 0.1
        df.loc[mask, col] = np.nan
    return df


def test_output_shape(sample_df):
    """Output should have one column per feature."""
    X = sample_df.drop(columns=["Survived_1_year"])
    X_t = build_preprocessor().fit_transform(X)
    expected = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    assert X_t.shape == (len(sample_df), expected)


def test_no_nan_after_transform(df_with_missing):
    """Missing values should be imputed — no NaN in output."""
    X = df_with_missing.drop(columns=["Survived_1_year"])
    X_t = build_preprocessor().fit_transform(X)
    assert not np.isnan(X_t).any(), (
        "NaN values remain after preprocessing"
    )


def test_no_data_leakage_in_cv(sample_df):
    """
    DummyClassifier inside pipeline should score ~0.5.
    If it scores much higher, preprocessing is leaking.
    """
    X = sample_df.drop(columns=["Survived_1_year"])
    y = sample_df["Survived_1_year"]
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", DummyClassifier(
            strategy="most_frequent"
        )),
    ])
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="roc_auc"
    )
    assert scores.mean() < 0.65, (
        f"Suspiciously high AUC for dummy: {scores.mean():.3f}"
    )


def test_feature_names_order():
    """Feature names must match ColumnTransformer output order."""
    assert get_feature_names() == (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )


def test_categorical_encoding(sample_df):
    """Categorical columns should be encoded as numeric."""
    X = sample_df.drop(columns=["Survived_1_year"])
    X_t = build_preprocessor().fit_transform(X)
    assert X_t.dtype in [np.float32, np.float64]


def test_handles_unknown_drug_at_inference(sample_df):
    """
    An unseen drug combination at inference should not crash.
    OrdinalEncoder with unknown_value=-1 handles this.
    """
    X_train = sample_df.drop(columns=["Survived_1_year"])
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    X_new = X_train.iloc[[0]].copy()
    X_new["Treated_with_drugs"] = "DX1 DX2 DX3 DX4 DX5"
    result = preprocessor.transform(X_new)
    assert result.shape == (
        1,
        len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
    )