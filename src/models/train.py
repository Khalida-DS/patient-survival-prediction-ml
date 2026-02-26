"""
src/models/train.py

Full training pipeline:
  1. RandomizedSearchCV for hyperparameter tuning (no leakage)
  2. Probability calibration via Platt scaling
  3. MLflow experiment tracking
"""

import logging
import pickle
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import Any, Dict, Tuple

from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


def _get_param_distributions(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = config["model"]["tuning"]["param_distributions"]
    return {
        "classifier__n_estimators": raw["n_estimators"],
        "classifier__max_depth": raw["max_depth"],
        "classifier__learning_rate": raw["learning_rate"],
        "classifier__subsample": raw["subsample"],
        "classifier__min_samples_leaf": raw["min_samples_leaf"],
    }


def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    config: Dict[str, Any],
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Run RandomizedSearchCV over GradientBoosting hyperparameters.
    Preprocessing inside pipeline — no leakage.
    """
    random_state = config["model"]["random_state"]
    n_iter = config["model"]["tuning"]["n_iter"]
    cv_folds = config["model"]["cv_folds"]
    scoring = config["model"]["scoring"]

    base_clf = GradientBoostingClassifier(random_state=random_state)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", base_clf),
    ])

    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=_get_param_distributions(config),
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
        refit=True,
    )

    logger.info(
        f"Starting RandomizedSearchCV: "
        f"{n_iter} iterations, {cv_folds}-fold CV..."
    )
    search.fit(X_train, y_train)

    logger.info(f"Best CV AUC: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.best_score_


def calibrate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
) -> CalibratedClassifierCV:
    """
    Wrap tuned pipeline in Platt scaling calibration.
    Ensures predicted probabilities are reliable.
    """
    calibrated = CalibratedClassifierCV(
    pipeline,
    method=config["model"]["calibration_method"],
)
    calibrated.fit(X_train, y_train)
    logger.info("Probability calibration (Platt scaling) applied.")
    return calibrated


def save_artifacts(
    model: Any,
    feature_names: list,
    config: Dict[str, Any],
) -> None:
    artifacts_dir = Path(config["model"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with open(artifacts_dir / "calibrated_gb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(artifacts_dir / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    logger.info(f"Artifacts saved to {artifacts_dir}")


def train_and_log(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    feature_names: list,
    benchmark_results: pd.DataFrame,
    config: Dict[str, Any],
) -> CalibratedClassifierCV:
    """
    Single entry point: tunes, calibrates, evaluates,
    and logs everything to MLflow. Saves artifacts to disk.
    """
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="GradientBoosting_calibrated"):

        # Log benchmark results
        for _, row in benchmark_results.iterrows():
            mlflow.log_metric(
                f"benchmark_{row['model']}_mean_auc", row["mean_auc"]
            )
            mlflow.log_metric(
                f"benchmark_{row['model']}_std_auc", row["std_auc"]
            )

        # Tune
        best_pipeline, best_params, best_cv_auc = tune_model(
            X_train, y_train, preprocessor, config
        )

        for param, val in best_params.items():
            mlflow.log_param(param.replace("classifier__", ""), val)
        mlflow.log_metric("best_cv_auc", best_cv_auc)

        # Calibrate
        calibrated_model = calibrate_model(
            best_pipeline, X_train, y_train, config
        )

        # Evaluate
        y_prob = calibrated_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        test_metrics = compute_metrics(y_test.values, y_pred, y_prob)

        for metric_name, value in test_metrics.items():
            mlflow.log_metric(metric_name, value)

        mlflow.log_artifact("config/config.yaml")
        mlflow.sklearn.log_model(
            calibrated_model,
            artifact_path="model",
            registered_model_name="PatientSurvivalPredictor",
        )

        logger.info(
            f"MLflow run complete. "
            f"Test AUC: {test_metrics.get('test_auc', 0):.4f} | "
            f"Brier: {test_metrics.get('test_brier_score', 0):.4f}"
        )

    save_artifacts(calibrated_model, feature_names, config)
    return calibrated_model