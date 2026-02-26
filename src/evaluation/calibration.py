"""
src/evaluation/calibration.py

Generates a reliability diagram (calibration curve).

WHY THIS MATTERS:
A model that says "70% survival probability" should be right
about 70% of the time. Without calibration, raw GBM outputs
are often overconfident. Platt scaling fixes this, and this
plot proves it worked.
"""

import numpy as np
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
from sklearn.calibration import calibration_curve


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str,
    n_bins: int = 10,
    log_to_mlflow: bool = True,
) -> str:
    """
    Plots and saves a reliability diagram.
    Returns the path to the saved figure.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / "calibration_curve.png")

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", color="gray",
        label="Perfectly calibrated",
    )
    ax.plot(
        prob_pred, prob_true,
        marker="o", color="#2563eb",
        label="Model (Platt scaling)",
    )

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(
        "Reliability Diagram (Calibration Curve)", fontsize=14
    )
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    if log_to_mlflow:
        mlflow.log_artifact(output_path)

    return output_path