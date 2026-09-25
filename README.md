# Patient Survival Prediction — Clinical Decision Support System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-tracking-orange)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> A production-grade machine learning system for one-year patient survival
> prediction. Built with rigorous statistical validation, full model
> explainability (SHAP), experiment tracking (MLflow), data drift monitoring
> (Evidently), and a clinical Streamlit interface.

🚀 **[Live Demo](https://patient-survival-prediction-ml-mzrrwnlfgjsjnvxygywcoa.streamlit.app/)**

📖 **[Full guide](docs/GUIDE.md)** — how the pipeline works end to end, how
the SHAP explanation works for a non-technical reader, exact commands to
run/retrain/redeploy, and a troubleshooting table.

---

## Problem Statement

Healthcare providers often lack actionable, data-driven insight into which
factors most strongly influence one-year patient survival. This results in
delayed risk identification, inconsistent treatment comparisons, and limited
support for evidence-based decisions.

This system addresses that gap by combining machine learning with clinical
interpretability — producing not just predictions, but *explainable,
calibrated* predictions that clinicians can scrutinise and trust.

---

## Key Data Insights (from EDA)

Before any modeling, EDA revealed several important findings:

| Finding | Action Taken |
|---|---|
| `Treated_with_drugs` had trailing whitespace ("DX1 ") | Stripped on load |
| `Patient_mental_condition` constant ("Stable" for all rows) | Dropped — zero information |
| `"Cannot say"` smokers have 100% survival rate | Dropped — data artifact |
| 9 patients with age > 120 (max: 149) | Capped at 100 — data entry errors |
| `Diagnosed_Condition` correlation with survival = 0.004 | Dropped — near-zero signal |
| **DX6 alone = only 42.6% survival** vs 75–88% for others | Highlighted in app |

---

## Model Performance

| Metric | Score | Notes |
|---|---|---|
| AUC-ROC | 0.88 | Primary selection criterion |
| Accuracy | 0.83 | |
| F1 Score | 0.82 | |
| Sensitivity | 0.85 | 85% of survivors correctly identified |
| Specificity | 0.81 | |
| Brier Score | 0.14 | Calibration quality — lower is better |

### 5-Model Benchmark (CV AUC)

| Model | Mean AUC | Std | Selected |
|---|---|---|---|
| **Gradient Boosting** | **0.87** | 0.02 | ✅ |
| XGBoost | 0.86 | 0.02 | |
| LightGBM | 0.85 | 0.03 | |
| Random Forest | 0.84 | 0.02 | |
| Logistic Regression | 0.78 | 0.03 | |

Selected based on AUC stability across folds — not just peak score.

---

## How to read the explanation

Every prediction comes with a "Why this prediction?" chart on the Predict
tab: a horizontal bar chart where **green bars push survival probability
up** and **red bars push it down**, sorted by size, using the patient's
real values ("Smoker = Yes", "BMI = 28.0") rather than internal scaled
numbers. It's built by averaging SHAP attributions across every fold of
the calibrated model and starts from a stated baseline — "the average
patient survival of X%" — so you can see exactly how each factor moved
one specific patient away from that average, in probability points that
add up to their final predicted probability. A 3-line plain-English
summary below the chart calls out the biggest factors. Full walkthrough
with a worked example: [docs/GUIDE.md, Section 3](docs/GUIDE.md#3-how-the-shap-explanation-works).

---

## Architecture
```
┌─────────────────────────────────────────────────────┐
│                    DATA LAYER                       │
│  Raw CSV → EDA → Validation → sklearn Pipeline     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  TRAINING LAYER                     │
│  5-Model CV Benchmark → RandomizedSearchCV         │
│  Platt Scaling Calibration → MLflow Tracking       │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               EXPLAINABILITY LAYER                  │
│  SHAP Global Summary → Per-Patient Waterfall       │
│  Reliability Diagram (Calibration Curve)           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                 INFERENCE LAYER                     │
│  Input Validation → Streamlit App → Risk Tier      │
│  SHAP Explanation → Evidently Drift Detection      │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure
```
patient-survival-prediction-ml/
│
├── .github/workflows/ci.yml        # Lint → tests → smoke test
├── src/
│   ├── data/
│   │   ├── loader.py               # Loading, cleaning, validation
│   │   └── preprocessor.py        # Leak-free sklearn pipeline
│   ├── models/
│   │   ├── benchmark.py            # 5-model CV benchmark
│   │   ├── train.py                # Tuning + calibration + MLflow
│   │   └── predict.py              # Inference with validation
│   ├── evaluation/
│   │   ├── metrics.py              # AUC, F1, Sensitivity, Brier
│   │   └── calibration.py         # Reliability diagram
│   ├── explainability/
│   │   └── shap_explainer.py       # SHAP global + per-patient
│   └── monitoring/
│       └── drift_report.py         # Evidently drift detection
│
├── app/app.py                       # Streamlit clinical interface
├── tests/                           # 18 pytest unit + smoke tests
├── config/config.yaml               # Centralised config
├── train_model.py                   # Pipeline entrypoint
├── requirements.txt                 # Lean, pinned — what the deployed demo installs
├── requirements-dev.txt             # + training/testing/lint tools (MLflow, XGBoost, pytest, ...)
├── runtime.txt                      # Python version for Streamlit Community Cloud
├── docs/GUIDE.md                    # Full owner/reviewer guide
└── Dockerfile
```

---

## Quickstart

The deployed demo only ever installs `requirements.txt` (lean, pinned —
see [docs/GUIDE.md](docs/GUIDE.md) for why that separation exists and
what broke when it wasn't there). For local development, use
`requirements-dev.txt`, which includes everything in `requirements.txt`
plus training, testing, and linting tools.

```bash
# 1. Clone and install (dev = app + training + tests + lint)
git clone https://github.com/Khalida-DS/patient-survival-prediction-ml.git
cd patient-survival-prediction-ml
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

# 2. Run tests
pytest tests/ -v

# 3. Launch the app (the committed model in models/ is ready to use —
#    no need to retrain first)
streamlit run app/app.py

# 4. (Optional) Retrain the model
python train_model.py

# 5. (Optional) View MLflow experiments
mlflow ui  # → http://localhost:5000

# 6. (Optional) Drift monitoring
python -m src.monitoring.drift_report \
  --reference data/Survival.csv \
  --current data/new_batch.csv \
  --output reports/drift/drift_report.html
```

Full step-by-step instructions (including Streamlit Community Cloud
deploy/redeploy): [docs/GUIDE.md](docs/GUIDE.md).

---

## Tech Stack

Python · scikit-learn · XGBoost · LightGBM · SHAP ·
MLflow · Evidently · Streamlit · Docker · GitHub Actions

---

## Disclaimer

For **educational and analytical demonstration purposes only**.
Not a medical device. Must not be used for real clinical
decision-making without regulatory review and prospective validation.