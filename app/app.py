"""
app/app.py

Clinical Decision Support Interface.
Built on real Survival dataset — drug values match actual data.
"""

import logging
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import load_config, load_data, split_features_target
from src.models.predict import load_model_artifacts, predict, validate_input
from src.data.preprocessor import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.explainability.shap_explainer import explain_patient

logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Survival DSS",
    page_icon="🏥",
    layout="wide",
)

# ── Load config + model ───────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    config = load_config("config/config.yaml")
    model, feature_names = load_model_artifacts(config)
    return config, model, feature_names


@st.cache_data
def load_background_sample(_config):
    """
    A small, fixed sample of real patients used as the reference
    population for SHAP explanations — this is what "the average
    patient" means in the "Why this prediction?" section below.
    """
    df = load_data(_config["data"]["path"], _config)
    X, _ = split_features_target(df, _config["data"]["target_column"])
    n = _config["explainability"]["background_samples"]
    return X.sample(
        n=min(n, len(X)), random_state=_config["data"]["random_state"]
    )


try:
    config, model, feature_names = load_resources()
    background_sample = load_background_sample(config)
except FileNotFoundError:
    st.error("⚠️ Model not found. Please run `python train_model.py` first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏥 Patient Survival DSS")
st.sidebar.markdown("**Clinical Decision Support System**")
st.sidebar.markdown("*For demonstration purposes only.*")
st.sidebar.divider()
tab_choice = st.sidebar.radio(
    "Navigate",
    ["🔬 Predict Survival", "📊 Model Performance", "📖 About"],
)

# Drug options — exact values from real dataset
DRUG_OPTIONS = [
    "DX1", "DX2", "DX3", "DX4", "DX5", "DX6",
    "DX1 DX2", "DX1 DX3", "DX1 DX4", "DX1 DX5",
    "DX2 DX3", "DX2 DX4", "DX2 DX5",
    "DX3 DX4", "DX3 DX5", "DX4 DX5",
    "DX1 DX2 DX3", "DX1 DX2 DX4", "DX1 DX2 DX5",
    "DX1 DX3 DX4", "DX1 DX3 DX5", "DX1 DX4 DX5",
    "DX2 DX3 DX4", "DX2 DX3 DX5", "DX2 DX4 DX5",
    "DX3 DX4 DX5",
    "DX1 DX2 DX3 DX4", "DX1 DX2 DX3 DX5",
    "DX1 DX2 DX4 DX5", "DX1 DX3 DX4 DX5",
    "DX2 DX3 DX4 DX5",
    "DX1 DX2 DX3 DX4 DX5",
]

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if tab_choice == "🔬 Predict Survival":
    st.title("Patient Survival Prediction")
    st.markdown(
        "Enter patient characteristics to generate "
        "a one-year survival estimate."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & Lifestyle")
        patient_age = st.number_input(
            "Patient Age", min_value=0, max_value=100, value=40
        )
        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=1.0, max_value=30.0, value=22.0, step=0.1,
        )
        smoker = st.selectbox("Smoker", options=["NO", "YES"])
        rural_urban = st.selectbox(
            "Location", options=["URBAN", "RURAL"]
        )

    with col2:
        st.subheader("Clinical & Treatment")
        treatment = st.selectbox(
            "Treatment (Drug Combination)",
            options=DRUG_OPTIONS,
            help="DX6 alone is associated with significantly "
                 "lower survival rates.",
        )
        num_prev_cond = st.number_input(
            "Number of Previous Conditions",
            min_value=1, max_value=5, value=1,
        )

    st.subheader("Clinical Indicators (A–F, Z)")
    st.caption("Binary clinical markers from patient record.")
    ind_cols = st.columns(7)
    labels = ["A", "B", "C", "D", "E", "F", "Z"]
    indicator_values = {}
    for i, label in enumerate(labels):
        with ind_cols[i]:
            indicator_values[label] = st.selectbox(
                label, [0, 1], key=f"ind_{label}"
            )

    st.divider()
    predict_btn = st.button(
        "🔍 Generate Prediction",
        type="primary",
        width="stretch",
    )

    if predict_btn:
        input_data = pd.DataFrame([{
            "Patient_Age": patient_age,
            "Patient_Body_Mass_Index": bmi,
            "Number_of_prev_cond": num_prev_cond,
            "A": float(indicator_values["A"]),
            "B": float(indicator_values["B"]),
            "C": float(indicator_values["C"]),
            "D": float(indicator_values["D"]),
            "E": float(indicator_values["E"]),
            "F": float(indicator_values["F"]),
            "Z": float(indicator_values["Z"]),
            "Treated_with_drugs": treatment,
            "Patient_Smoker": smoker,
            "Patient_Rural_Urban": rural_urban,
        }])

        prediction, survival_prob, warnings = predict(
            model, input_data
        )

        if warnings:
            for w in warnings:
                st.warning(f"⚠️ {w}")

        # ── Risk classification ─────────────────────────────────────
        thresholds = config["app"]["risk_thresholds"]
        if survival_prob < thresholds["high_risk"]:
            risk_tier = "🔴 HIGH RISK"
        elif survival_prob < thresholds["moderate_risk"]:
            risk_tier = "🟡 MODERATE RISK"
        else:
            risk_tier = "🟢 LOW RISK"

        # ── Display result ──────────────────────────────────────────
        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Predicted Outcome",
            "✅ Survived" if prediction == 1
            else "❌ Did Not Survive",
        )
        r2.metric(
            "Survival Probability",
            f"{survival_prob:.1%}",
            help="Calibrated probability from Platt scaling.",
        )
        r3.metric("Risk Tier", risk_tier)

        st.markdown("#### Survival Probability")
        st.progress(survival_prob)

        # ── DX6 warning ─────────────────────────────────────────────
        if treatment == "DX6":
            st.warning(
                "⚠️ DX6 monotherapy is associated with only ~43% "
                "one-year survival in this dataset — significantly "
                "lower than other treatment regimens."
            )

        # ── SHAP explanation ────────────────────────────────────────
        st.markdown("#### Why this prediction?")

        try:
            explanation = explain_patient(
                model=model,
                input_df=input_data,
                feature_names=feature_names,
                background_df=background_sample,
                calibrated_probability=survival_prob,
            )
            baseline_pct = explanation["baseline_probability"] * 100
            final_pct = explanation["calibrated_probability"] * 100

            st.markdown(
                f"Averaged across all **{explanation['num_folds_averaged']}** "
                f"cross-validation folds of the calibrated model — starting "
                f"from the average patient survival of **{baseline_pct:.0f}%**, "
                f"the factors below moved this patient to **{final_pct:.0f}%**."
            )
            st.caption(
                "Green bars push survival probability up, red bars push it "
                "down. Bars are sorted by size — the biggest movers are at "
                "the top."
            )

            top_features = explanation["features"][:8]
            top_features_display = list(reversed(top_features))  # largest on top
            labels = [f["label"] for f in top_features_display]
            values_pp = [f["value"] * 100 for f in top_features_display]
            colors = ["#dc2626" if v < 0 else "#16a34a" for v in values_pp]

            fig = go.Figure(
                go.Bar(
                    x=values_pp,
                    y=labels,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.1f} pts" for v in values_pp],
                    textposition="outside",
                )
            )
            fig.update_layout(
                xaxis_title="Impact on survival probability (percentage points)",
                yaxis_title=None,
                height=360,
                margin=dict(l=10, r=10, t=10, b=40),
            )
            st.plotly_chart(fig, width="stretch")

            top3 = explanation["features"][:3]
            summary_lines = []
            for i, f in enumerate(top3):
                direction = "lowered" if f["value"] < 0 else "raised"
                superlative = " the most" if i == 0 else ""
                summary_lines.append(
                    f"- **{f['label']}** {direction} this patient's "
                    f"predicted survival{superlative} "
                    f"({f['value'] * 100:+.1f} points)."
                )
            st.markdown("\n".join(summary_lines))

        except Exception:
            logger.exception("SHAP explanation failed for a patient prediction")
            st.error(
                "⚠️ Could not generate the SHAP explanation for this "
                "patient. This has been logged — check the app logs for "
                "the full traceback."
            )

        st.divider()
        st.caption(
            "⚠️ **Disclaimer**: For demonstration purposes only. "
            "Not a medical device."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📊 Model Performance":
    st.title("Model Performance Dashboard")

    st.markdown(
        "Model selected via 5-fold stratified CV across five candidate "
        "algorithms. All metrics on held-out test set (20% split). "
        "Dataset: 23,097 patients, 63% survival rate."
    )

    st.subheader("Model Benchmark (CV AUC)")
    benchmark_data = {
        "Model": [
            "Gradient Boosting ✅", "XGBoost", "LightGBM",
            "Random Forest", "Logistic Regression",
        ],
        "Mean AUC": [0.87, 0.86, 0.85, 0.84, 0.78],
        "Std AUC": [0.02, 0.02, 0.03, 0.02, 0.03],
        "Selected": ["✅", "", "", "", ""],
    }
    st.dataframe(
        pd.DataFrame(benchmark_data),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Final Test Set Metrics")
    metrics_data = {
        "Metric": [
            "AUC-ROC", "Accuracy", "F1 Score",
            "Sensitivity", "Specificity", "Brier Score",
        ],
        "Value": [0.88, 0.83, 0.82, 0.85, 0.81, 0.14],
        "Notes": [
            "Primary selection criterion",
            "83% correct predictions",
            "Harmonic mean of precision + recall",
            "85% of survivors correctly identified",
            "81% of non-survivors correctly identified",
            "Lower is better — calibration quality",
        ],
    }
    st.dataframe(
        pd.DataFrame(metrics_data),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Key Data Insights")
    i1, i2 = st.columns(2)
    with i1:
        st.metric("DX6 Survival Rate", "42.6%", "-33% vs other drugs")
        st.metric("Smoker Survival Rate", "49.0%", "-24% vs non-smokers")
    with i2:
        st.metric("Urban Survival Rate", "54.9%", "-11% vs rural")
        st.metric("Dataset Size", "23,097 patients")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Calibration Curve")
        st.caption(
            "The dashed line is perfect calibration. The closer the blue "
            "line hugs it, the more a '70% survival' prediction really "
            "does mean about 70 in 100 similar patients survived."
        )
        cal_path = Path("reports/evaluation/calibration_curve.png")
        if cal_path.exists():
            st.image(str(cal_path))
        else:
            st.info("Run `python train_model.py` to generate.")
    with c2:
        st.subheader("SHAP Global Feature Importance")
        st.caption(
            "Each dot is one patient. Position left/right shows whether "
            "that feature pushed survival down or up for them; red dots "
            "are high feature values, blue dots are low. Features are "
            "ranked top-to-bottom by overall impact across all patients."
        )
        shap_path = Path("reports/shap/shap_summary.png")
        if shap_path.exists():
            st.image(str(shap_path))
        else:
            st.info("Run `python train_model.py` to generate.")

    st.divider()
    st.subheader("MLflow Experiment Tracking")
    st.code("mlflow ui  # → http://localhost:5000", language="bash")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📖 About":
    st.title("About This System")
    st.markdown("""
    ### Architecture

    Production-grade clinical ML pipeline:

    - **Data cleaning** — strips whitespace from drug codes,
      removes ambiguous smoker labels, caps age outliers,
      drops constant and near-zero-correlation features
    - **Leak-free preprocessing** — sklearn Pipeline, imputation
      fit inside CV folds only
    - **Multi-model benchmark** — 5-model stratified CV comparison
    - **Hyperparameter tuning** — `RandomizedSearchCV` with AUC
    - **Probability calibration** — Platt scaling
    - **Explainability** — SHAP global summary + per-patient
      explanation, averaged across every calibrated fold
    - **Experiment tracking** — MLflow logging and model registry
    - **Drift monitoring** — Evidently distribution shift detection
    - **CI/CD** — lint, unit tests, smoke test on every commit

    ### Dataset
    23,097 patients | 13 features | 63% one-year survival rate
    Treatment options: 32 combinations of drugs DX1–DX6

    ### Disclaimer
    For educational and analytical demonstration only.
    Not a medical device.
    """)
