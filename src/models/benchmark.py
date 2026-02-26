"""
src/models/benchmark.py

Runs a rigorous multi-model benchmark using stratified 5-fold CV.
Reports mean AUC ± std for each candidate.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def get_candidate_models(random_state: int) -> Dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=1,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1,
        ),
    }


def run_benchmark(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, str]:
    """
    Benchmarks all candidate models with stratified k-fold CV.
    Returns results DataFrame and name of best model.
    """
    random_state = config["model"]["random_state"]
    cv_folds = config["model"]["cv_folds"]
    scoring = config["model"]["scoring"]

    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    candidates = get_candidate_models(random_state)

    results = []
    for name, clf in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1
        )
        results.append({
            "model": name,
            "mean_auc": scores.mean(),
            "std_auc": scores.std(),
            "min_auc": scores.min(),
        })
        logger.info(
            f"{name:25s}  AUC: {scores.mean():.4f} ± "
            f"{scores.std():.4f}  (min: {scores.min():.4f})"
        )

    results_df = pd.DataFrame(results).sort_values(
        "mean_auc", ascending=False
    )
    best_model_name = results_df.iloc[0]["model"]
    logger.info(
        f"\nSelected: {best_model_name} "
        f"(AUC: {results_df.iloc[0]['mean_auc']:.4f})"
    )

    return results_df, best_model_name