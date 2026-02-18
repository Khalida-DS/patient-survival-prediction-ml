# src/train.py
# --------------------------------------------------
# Patient Survival Prediction - Training Pipeline
# --------------------------------------------------
# WHY THIS FILE EXISTS:
# This is the main entry point. It ties everything
# together — it calls preprocess.py to get clean data,
# applies SMOTE to fix class imbalance, trains 3
# different models, and calls evaluate.py to score
# and log each one to MLflow.
#
# HOW TO RUN:
#   python -m src.train
# --------------------------------------------------

import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from src.preprocess import load_and_preprocess
from src.evaluate import evaluate_and_log


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Survival.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
MLFLOW_DIR = os.path.join(BASE_DIR, "mlflow_tracking")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

# --------------------------------------------------
# MLFLOW SETUP
# --------------------------------------------------

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR}/mlflow.db")
mlflow.set_experiment("patient-survival-prediction-v2")

# --------------------------------------------------
# MODELS TO TRAIN
# --------------------------------------------------
# We define all 3 models in a list of tuples:
# (name, model_object)
#
# WHY A LIST?
# So we can loop over them and train/evaluate each
# one with the exact same code. No copy-pasting.
#
# Each model is initialized with its hyperparameters:
#
# GradientBoosting — your original baseline model
# XGBoost          — faster, uses regularization to
#                    prevent overfitting (scale_pos_weight
#                    handles imbalance natively)
# LightGBM         — fastest of the three, great on
#                    larger datasets, num_leaves controls
#                    tree complexity

MODELS = [
    (
        "GradientBoosting",
        GradientBoostingClassifier(
            max_depth=4,
            n_estimators=150,
            learning_rate=0.2,
            random_state=45,
        )
    ),
    (
        "XGBoost",
        XGBClassifier(
            max_depth=4,
            n_estimators=150,
            learning_rate=0.2,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=45,
            verbosity=0,
        )
    ),
    (
        "LightGBM",
        LGBMClassifier(
            max_depth=4,
            n_estimators=150,
            learning_rate=0.2,
            num_leaves=31,
            random_state=45,
            verbose=-1,
        )
    ),
]


# --------------------------------------------------
# MAIN TRAINING PIPELINE
# --------------------------------------------------

def main():

    # ── 1. Load & Preprocess ───────────────────────
    # One function call — all cleaning happens in
    # preprocess.py, we just get back X, y, features
    X, y, feature_names = load_and_preprocess(DATA_PATH)

    # ── 2. Check Class Balance ────────────────────
    # Before applying SMOTE we check how imbalanced
    # the dataset actually is.
    #
    # Example output:
    #   Survived_1_year
    #   1    0.78   ← 78% survived
    #   0    0.22   ← 22% did not survive
    #
    # If the split is close to 50/50, SMOTE may not
    # be necessary. If it's 80/20 or worse, SMOTE
    # will meaningfully help recall for the minority.
    print("\n📊 Class Distribution (before SMOTE):")
    print(y.value_counts(normalize=True).round(3))

    # ── 3. Train/Test Split ───────────────────────
    # IMPORTANT: We split BEFORE applying SMOTE.
    # Why? Because SMOTE creates synthetic samples.
    # If we apply it before splitting, synthetic
    # samples could leak into the test set — the
    # model would be evaluated on data it has
    # effectively already seen. That inflates scores.
    # Always: split first → SMOTE only on train set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=45
    )

    # ── 4. Apply SMOTE to Training Set Only ───────
    # SMOTE = Synthetic Minority Over-sampling Technique
    #
    # It looks at existing minority class samples,
    # finds their nearest neighbors, and creates new
    # synthetic samples between them.
    #
    # Example:
    #   Before SMOTE: 6000 survived, 1800 not survived
    #   After SMOTE:  6000 survived, 6000 not survived
    #
    # random_state=45 ensures reproducibility — same
    # synthetic samples every time you run the script.
    print("\n⚖️  Applying SMOTE to training set...")
    smote = SMOTE(random_state=45)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print("Class Distribution (after SMOTE):")
    import pandas as pd
    print(pd.Series(y_train_balanced).value_counts(normalize=True).round(3))

    # ── 5. Train & Evaluate All Models ───────────
    # We loop over every model in our MODELS list.
    # For each one we:
    #   a) Start a new MLflow run (separate entry in UI)
    #   b) Log the model name and hyperparameters
    #   c) Train on the SMOTE-balanced training data
    #   d) Call evaluate_and_log() from evaluate.py
    #   e) Save the model locally as a .pkl file
    #   f) Register it in the MLflow Model Registry

    results = {}

    for model_name, model in MODELS:
        print(f"\n🚀 Training {model_name}...")

        with mlflow.start_run(run_name=model_name):

            # Log which model and its settings
            mlflow.log_param("model", model_name)
            mlflow.log_param("smote_applied", True)
            mlflow.log_params(model.get_params())

            # Train the model
            model.fit(X_train_balanced, y_train_balanced)
            print(f"✅ {model_name} training completed")

            # Evaluate + log metrics, ROC curve,
            # and feature importance to MLflow
            metrics = evaluate_and_log(
                model_name=model_name,
                model=model,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                output_dir=REPORT_DIR,
            )

            results[model_name] = metrics

            # Log and register the model in MLflow
            # Model Registry lets you version models
            # and later mark one as "Production"
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=f"PatientSurvival_{model_name}",
                input_example=X_train.iloc[:5],
            )

            # Save locally as .pkl for the Streamlit app
            model_path = os.path.join(MODEL_DIR, f"{model_name.lower()}.pkl")
            joblib.dump(model, model_path)
            print(f"💾 Saved locally → {model_path}")

    # ── 6. Save Feature Names ─────────────────────
    # The Streamlit app needs to know the exact column
    # names and order the model was trained on.
    feature_path = os.path.join(MODEL_DIR, "feature_names.pkl")
    joblib.dump(feature_names, feature_path)
    print(f"\n💾 Feature names saved → {feature_path}")

    # ── 7. Print Final Comparison ─────────────────
    # After all 3 models are done, print a summary
    # table so you can see the winner at a glance
    # right in the terminal.
    print(f"\n{'='*60}")
    print("  FINAL MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Accuracy':>10} {'ROC AUC':>10} {'Recall':>10}")
    print(f"  {'-'*50}")
    for name, m in results.items():
        print(f"  {name:<20} {m['accuracy']:>10.4f} {m['roc_auc']:>10.4f} {m['recall']:>10.4f}")
    print(f"{'='*60}")
    print("\n✅ All models trained and logged to MLflow")
    print("👉 Run: mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db --port 5001")


if __name__ == "__main__":
    main()
