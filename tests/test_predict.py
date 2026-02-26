"""
tests/test_predict.py

Inference contract tests using real dataset column names.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.predict import validate_input


@pytest.fixture
def valid_input():
    return pd.DataFrame([{
        "Patient_Age": 45.0,
        "Patient_Body_Mass_Index": 22.5,
        "Number_of_prev_cond": 2.0,
        "A": 1.0, "B": 0.0, "C": 0.0,
        "D": 0.0, "E": 1.0, "F": 0.0, "Z": 0.0,
        "Treated_with_drugs": "DX1",
        "Patient_Smoker": "NO",
        "Patient_Rural_Urban": "URBAN",
    }])


def test_valid_input_no_warnings(valid_input):
    """Valid input should produce zero warnings."""
    assert validate_input(valid_input) == []


def test_age_out_of_range_flagged():
    """Age of 150 should trigger a validation warning."""
    df = pd.DataFrame([{
        "Patient_Age": 150.0,
        "Patient_Body_Mass_Index": 22.0,
        "Number_of_prev_cond": 1.0,
    }])
    warnings = validate_input(df)
    assert any("Patient_Age" in w for w in warnings)


def test_bmi_out_of_range_flagged():
    """BMI of 0.5 should trigger a validation warning."""
    df = pd.DataFrame([{
        "Patient_Age": 40.0,
        "Patient_Body_Mass_Index": 0.5,
        "Number_of_prev_cond": 1.0,
    }])
    warnings = validate_input(df)
    assert any("Patient_Body_Mass_Index" in w for w in warnings)


def test_missing_feature_flagged():
    """A missing required feature should produce a warning."""
    df = pd.DataFrame([{"Patient_Age": 40.0}])
    assert len(validate_input(df)) > 0