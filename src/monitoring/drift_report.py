"""
src/monitoring/drift_report.py

Detects data drift between training distribution
and new patient batches using Evidently.

WHY THIS EXISTS:
Models degrade silently when input distributions shift.
A hospital that updates treatment protocols or data
collection methods will cause your model to fail without
warning. This module catches that before it becomes a
clinical problem.

Usage:
    python -m src.monitoring.drift_report \
        --reference data/Survival.csv \
        --current data/new_batch.csv \
        --output reports/drift/drift_report.html
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

logger = logging.getLogger(__name__)

TARGET_COL = "Survived_1_year"


def generate_drift_report(
    reference_path: str,
    current_path: str,
    output_path: str,
    psi_threshold: float = 0.2,
) -> dict:
    """
    Compares reference and current datasets.
    Saves an HTML drift report and returns a summary dict.
    """
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    # Strip whitespace from drug column
    for df in [reference_df, current_df]:
        if "Treated_with_drugs" in df.columns:
            df["Treated_with_drugs"] = (
                df["Treated_with_drugs"].str.strip()
            )

    # Drop target — compare features only
    reference_df = reference_df.drop(
        columns=[TARGET_COL], errors="ignore"
    )
    current_df = current_df.drop(
        columns=[TARGET_COL], errors="ignore"
    )

    # Align columns
    shared_cols = [
        c for c in reference_df.columns
        if c in current_df.columns
    ]
    reference_df = reference_df[shared_cols]
    current_df = current_df[shared_cols]

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    logger.info(f"Drift report saved to: {output_path}")

    result = report.as_dict()
    drifted_features = []

    for metric in result.get("metrics", []):
        if metric.get("metric") == "DataDriftTable":
            for col_result in (
                metric.get("result", {})
                .get("drift_by_columns", {})
                .values()
            ):
                if col_result.get("drift_detected", False):
                    drifted_features.append(
                        col_result.get("column_name")
                    )

    if drifted_features:
        logger.warning(
            f"DRIFT DETECTED in {len(drifted_features)} "
            f"feature(s): {drifted_features}. "
            f"Consider retraining."
        )
    else:
        logger.info("No significant data drift detected.")

    return {
        "drift_detected": len(drifted_features) > 0,
        "drifted_features": drifted_features,
        "report_path": output_path,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument(
        "--output",
        default="reports/drift/drift_report.html",
    )
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    args = parser.parse_args()

    result = generate_drift_report(
        args.reference,
        args.current,
        args.output,
        args.psi_threshold,
    )
    print(f"\nDrift detected: {result['drift_detected']}")
    if result["drifted_features"]:
        print(f"Drifted features: {result['drifted_features']}")
    print(f"Report: {result['report_path']}")