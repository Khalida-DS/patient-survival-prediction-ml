"""
tests/test_app_smoke.py

End-to-end smoke test for the deployed app's exact runtime path:
load the committed model artifacts, run one prediction, and build
the SHAP explanation. This is the check that would have caught the
original Streamlit Cloud outage — it installs and runs against
requirements.txt only (see .github/workflows/ci.yml), the same lean
set Streamlit Community Cloud installs for the live demo.
"""

import pandas as pd
import pytest

from src.data.loader import load_config, load_data, split_features_target
from src.models.predict import load_model_artifacts, predict
from src.explainability.shap_explainer import explain_patient


@pytest.fixture(scope="module")
def resources():
    config = load_config("config/config.yaml")
    model, feature_names = load_model_artifacts(config)
    return config, model, feature_names


@pytest.fixture(scope="module")
def background_sample(resources):
    config, _, _ = resources
    df = load_data(config["data"]["path"], config)
    X, _ = split_features_target(df, config["data"]["target_column"])
    return X.sample(n=30, random_state=config["data"]["random_state"])


@pytest.fixture
def sample_patient():
    return pd.DataFrame([{
        "Patient_Age": 62.0,
        "Patient_Body_Mass_Index": 28.0,
        "Number_of_prev_cond": 3.0,
        "A": 1.0, "B": 0.0, "C": 0.0,
        "D": 0.0, "E": 1.0, "F": 0.0, "Z": 0.0,
        "Treated_with_drugs": "DX6",
        "Patient_Smoker": "YES",
        "Patient_Rural_Urban": "URBAN",
    }])


def test_model_loads(resources):
    """The committed model artifacts must unpickle without error."""
    config, model, feature_names = resources
    assert model is not None
    assert len(feature_names) == 13


def test_model_predicts(resources, sample_patient):
    """The committed model must produce a valid calibrated prediction."""
    _, model, _ = resources
    prediction, survival_prob, warnings = predict(model, sample_patient)
    assert prediction in (0, 1)
    assert 0.0 <= survival_prob <= 1.0
    assert warnings == []


def test_shap_explanation_builds(resources, background_sample, sample_patient):
    """
    The full "Why this prediction?" pipeline must run end to end:
    averaging SHAP across every calibrated fold, and the resulting
    contributions must sum back to the calibrated probability.
    """
    config, model, feature_names = resources
    _, survival_prob, _ = predict(model, sample_patient)

    explanation = explain_patient(
        model=model,
        input_df=sample_patient,
        feature_names=feature_names,
        background_df=background_sample,
        calibrated_probability=survival_prob,
    )

    assert explanation["num_folds_averaged"] == len(model.calibrated_classifiers_)
    assert len(explanation["features"]) == len(feature_names)

    total = explanation["baseline_probability"] + sum(
        f["value"] for f in explanation["features"]
    )
    assert total == pytest.approx(survival_prob, abs=1e-6)
