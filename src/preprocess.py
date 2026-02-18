# src/preprocess.py
# --------------------------------------------------
# Patient Survival Prediction - Data Preprocessing
# --------------------------------------------------
# WHY THIS FILE EXISTS:
# Instead of mixing data cleaning with model training,
# we isolate all data logic here. This means:
# - If the data changes, you only edit this file
# - train.py stays clean and focused on models only
# - You can reuse this preprocessing for any new model
# --------------------------------------------------

import os
import pandas as pd


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
# We define column names here instead of hardcoding
# them inside functions. If a column name changes in
# the CSV, you only update it in one place.

FEATURE_COLS = [
    'Treated_with_drugs',
    'Patient_Age',
    'Patient_Body_Mass_Index',
    'Patient_Smoker',
    'Patient_Rural_Urban',
    'Patient_mental_condition',
    'A', 'B', 'C', 'D', 'E', 'F', 'Z',
    'Number_of_prev_cond',
]

TARGET_COL = 'Survived_1_year'

CATEGORICAL_COLS = [
    'Treated_with_drugs',
    'Patient_Smoker',
    'Patient_Rural_Urban',
    'Patient_mental_condition',
]


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------
# This is the only function train.py needs to call.
# It takes a file path, returns X (features) and y (target).
#
# WHY RETURN X AND y SEPARATELY?
# Because train.py only needs the clean features and
# the target — it doesn't care how we got there.

def load_and_preprocess(data_path: str):
    """
    Load and preprocess the survival dataset.

    Args:
        data_path (str): Full path to Survival.csv

    Returns:
        X (pd.DataFrame): Encoded feature matrix
        y (pd.Series): Target variable (Survived_1_year)
        feature_names (list): List of feature column names
    """

    # ── 1. Load ────────────────────────────────────
    # Read the raw CSV into a DataFrame
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded — shape: {df.shape}")

    # ── 2. Select Columns ──────────────────────────
    # We only keep the columns we actually use.
    # This avoids accidentally training on irrelevant
    # columns or leaking future information.
    df = df[FEATURE_COLS + [TARGET_COL]]

    # ── 3. Remove Ambiguous Smoker Rows ───────────
    # 'Cannot say' is not a meaningful category for
    # a model — it adds noise. We drop these rows.
    df = df[df['Patient_Smoker'] != 'Cannot say']

    # ── 4. Fill Missing Values ────────────────────
    # For numerical columns like Number_of_prev_cond,
    # we use the MEDIAN (not mean) because it's more
    # robust to outliers in medical data.
    df['Number_of_prev_cond'] = df['Number_of_prev_cond'].fillna(
        df['Number_of_prev_cond'].median()
    )

    # For all remaining columns, fill with MODE
    # (most frequent value) — safe for both categorical
    # and numerical columns.
    for col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("✅ Data cleaning completed")

    # ── 5. One-Hot Encoding ───────────────────────
    # Machine learning models work with numbers, not
    # text. One-hot encoding converts categories like
    # 'Yes'/'No'/'Never' into separate 0/1 columns.
    #
    # Example:
    # Patient_Smoker = 'Yes'  →  Patient_Smoker_Yes = 1
    #                             Patient_Smoker_No  = 0
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS)

    # ── 6. Split Features and Target ─────────────
    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL]
    feature_names = X.columns.tolist()

    print(f"✅ Preprocessing complete — {len(feature_names)} features ready")

    return X, y, feature_names
