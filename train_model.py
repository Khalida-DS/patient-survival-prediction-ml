"""
train_model.py

Main entrypoint for the full training pipeline. Run this once to:
  1. Load and clean data
  2. Benchmark 5 models with stratified CV
  3. Tune best model with RandomizedSearchCV
  4. Calibrate probabilities (Platt scaling)
  5. Evaluate: AUC, F1, Sensitivity, Specificity, Brier Score
  6. Generate calibration curve and SHAP plots
  7. Log everything to MLflow

Usage:
    python train_model.py
"""

import logging
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import load_config, load_data, split_features_target
from src.data.preprocessor import build_preprocessor, get_feature_names
from src.models.benchmark import run_benchmark
from src.models.train import train_and_log
from src.evaluation.metrics import compute_metrics, print_evaluation_report
from src.evaluation.calibration import plot_calibration_curve
from src.explainability.shap_explainer import (
    compute_shap_values,
    plot_global_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # ── 1. Load config and data ──────────────────────────────────────
    config = load_config("config/config.yaml")
    df = load_data(config["data"]["path"], config)

    # ── 2. Split features and target ────────────────────────────────
    X, y = split_features_target(df, config["data"]["target_column"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )
    logger.info(
        f"Train: {len(X_train)} | Test: {len(X_test)} | "
        f"Positive rate (train): {y_train.mean():.2%}"
    )

    # ── 3. Build preprocessor ────────────────────────────────────────
    preprocessor = build_preprocessor()
    feature_names = get_feature_names()
    logger.info(f"Features ({len(feature_names)}): {feature_names}")

    # ── 4. Multi-model benchmark ─────────────────────────────────────
    logger.info("\nRunning 5-model CV benchmark...")
    benchmark_results, best_model_name = run_benchmark(
        X_train, y_train, preprocessor, config
    )
    print(
        f"\nBenchmark Results:\n"
        f"{benchmark_results[['model','mean_auc','std_auc']].to_string(index=False)}\n"
    )

    # ── 5. Train, calibrate, and log (single call) ───────────────────
    final_model = train_and_log(
        X_train, y_train, X_test, y_test,
        preprocessor, feature_names,
        benchmark_results, config,
    )

    # ── 6. Evaluation report ─────────────────────────────────────────
    y_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    print_evaluation_report(y_test.values, y_pred, y_prob)

    # ── 7. Calibration curve ─────────────────────────────────────────
    plot_calibration_curve(
        y_test.values, y_prob,
        output_dir=config["evaluation"]["output_dir"],
        n_bins=config["evaluation"]["calibration_bins"],
        log_to_mlflow=False,
    )
    logger.info("Calibration curve saved.")

    # ── 8. SHAP global explainability ────────────────────────────────
    logger.info("Computing SHAP values...")
    try:
        inner_model = final_model.calibrated_classifiers_[0].estimator
        inner_preprocessor = inner_model.named_steps["preprocessor"]
        inner_clf = inner_model.named_steps["classifier"]

        X_test_shap = inner_preprocessor.transform(X_test)
        shap_values = compute_shap_values(
            inner_clf,
            X_test_shap,
            background_samples=config["explainability"]["background_samples"],
        )
        plot_global_summary(
            shap_values,
            feature_names=feature_names,
            output_dir=config["explainability"]["output_dir"],
            max_display=config["explainability"]["max_display"],
            log_to_mlflow=False,
        )
        logger.info("SHAP summary plot saved.")
    except Exception as e:
        logger.warning(f"SHAP computation skipped: {e}")

    logger.info("\n✅ Training pipeline complete.")
    logger.info("   MLflow UI:  mlflow ui  →  http://localhost:5000")
    logger.info("   App:        streamlit run app/app.py")


if __name__ == "__main__":
    main()
