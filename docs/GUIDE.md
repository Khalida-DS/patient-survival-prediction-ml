# Patient Survival Prediction — Owner & Reviewer Guide

This is the single reference for how this project works, what broke on the
live demo, how it was fixed, and the exact commands to run, retrain, and
redeploy it. It's written for two readers: **you** (the project owner) and a
**technical client** reviewing the project before trusting it.

> Not a medical device. Everything here supports a demonstration ML system,
> not a clinical tool.

---

## Contents

1. [How the project works end to end](#1-how-the-project-works-end-to-end)
2. [What was broken, and why](#2-what-was-broken-and-why)
3. [How the SHAP explanation works](#3-how-the-shap-explanation-works)
4. [Run it locally on a Mac (Apple Silicon)](#4-run-it-locally-on-a-mac-apple-silicon)
5. [Retrain the model](#5-retrain-the-model)
6. [Deploy or redeploy on Streamlit Community Cloud](#6-deploy-or-redeploy-on-streamlit-community-cloud)
7. [The GitHub workflow: review, merge, confirm](#7-the-github-workflow-review-merge-confirm)
8. [Troubleshooting table](#8-troubleshooting-table)
9. [Checklist before sharing the demo link](#9-checklist-before-sharing-the-demo-link)

---

## 1. How the project works end to end

```
 data/Survival.csv
        │
        ▼
 ┌─────────────────────────┐
 │  LOAD + CLEAN            │  src/data/loader.py
 │  strip whitespace, drop   │  Strips "DX1 " → "DX1", drops constant/
 │  junk columns, cap ages   │  near-zero-signal columns, caps age at 100.
 └────────────┬─────────────┘
              ▼
 ┌─────────────────────────┐
 │  PREPROCESS              │  src/data/preprocessor.py
 │  impute → scale (numeric) │  A leak-free sklearn ColumnTransformer:
 │  impute → ordinal-encode  │  numeric features are median-imputed +
 │  (categorical)             │  scaled; the 3 categorical features
 │                            │  (drug combo, smoker, rural/urban) are
 │                            │  ordinal-encoded, not one-hot.
 └────────────┬─────────────┘
              ▼
 ┌─────────────────────────┐
 │  BENCHMARK + TUNE         │  src/models/benchmark.py, src/models/train.py
 │  5-model stratified CV,   │  Compares GradientBoosting, XGBoost,
 │  RandomizedSearchCV        │  LightGBM, RandomForest, LogisticRegression;
 │  on the winner              │  tunes the winner (GradientBoosting) with
 │                            │  RandomizedSearchCV, scoring on AUC.
 └────────────┬─────────────┘
              ▼
 ┌─────────────────────────┐
 │  CALIBRATE                │  src/models/train.py (calibrate_model)
 │  Platt scaling via 5-fold │  Wraps the tuned pipeline in
 │  CalibratedClassifierCV    │  CalibratedClassifierCV — so "70%
 │                            │  survival" really means ~70/100 patients
 │                            │  like this one survived. Produces 5
 │                            │  internal fold classifiers, not 1.
 └────────────┬─────────────┘
              ▼
 ┌─────────────────────────┐
 │  SAVE ARTIFACTS            │  src/models/train.py (save_artifacts)
 │  models/calibrated_gb_     │  Pickles the calibrated model + the
 │  model.pkl,                │  ordered feature name list. These two
 │  models/feature_names.pkl  │  files are what the Streamlit app loads.
 └────────────┬─────────────┘
              ▼
 ┌─────────────────────────┐
 │  STREAMLIT APP             │  app/app.py
 │  loads the pickled model,  │  3 tabs: Predict Survival, Model
 │  runs predict(), builds    │  Performance, About. Predict tab takes
 │  the SHAP explanation       │  form inputs, calls src/models/predict.py,
 │                            │  shows the calibrated probability + risk tier.
 └────────────┬─────────────┘
              ▼
 ┌─────────────────────────┐
 │  SHAP EXPLANATION           │  src/explainability/shap_explainer.py
 │  "Why this prediction?"     │  explain_patient() averages SHAP
 │                            │  attributions across all 5 calibrated
 │                            │  folds, in probability terms, against
 │                            │  real (not scaled) patient values —
 │                            │  rendered as a green/red Plotly bar chart.
 └───────────────────────────┘
```

**Two separate installs.** `requirements.txt` is the *lean* set the
Streamlit Cloud demo installs (Streamlit, pandas, numpy, scikit-learn, shap,
plotly, PyYAML — nothing else). `requirements-dev.txt` adds everything the
app itself never imports: MLflow, Evidently, XGBoost, LightGBM, pytest,
flake8, matplotlib, seaborn. You need `requirements-dev.txt` to train,
test, or lint; the deployed demo only ever needs `requirements.txt`.

---

## 2. What was broken, and why

**The symptom:** the public Streamlit Cloud demo didn't load.

**The root cause, in plain words:** `requirements.txt` said
`scikit-learn>=1.3.0` — no upper bound. The committed model file,
`models/calibrated_gb_model.pkl`, was saved with scikit-learn **1.8.0**.
Every time Streamlit Cloud rebuilds the app, it installs whatever
`>=1.3.0` resolves to *that day* — which, by the time this was diagnosed,
was scikit-learn 1.9.1. Scikit-learn reorganized an internal module
(`sklearn._loss`) between those versions, so the newer scikit-learn could
not read a model pickled by the older one. The app crashed on the very
first line that loads the model, before rendering anything, with:

```
ModuleNotFoundError: No module named '_loss'
```

This was reproduced locally: a clean virtualenv built from the original
`requirements.txt` installed scikit-learn 1.9.1 and hit the exact same
error; pinning `scikit-learn==1.8.0` in that same venv loaded the model
with no errors.

**Everything that was changed, and why:**

| File(s) | What changed | Why it matters |
|---|---|---|
| `requirements.txt` | Trimmed to only what `app/app.py` imports; every version pinned exactly (`scikit-learn==1.8.0` in particular) | This was the actual bug. An unpinned scikit-learn is free to drift to a version that can't read the committed model. Pinning ties the deployed environment to the environment the model was actually saved in. |
| `requirements-dev.txt` (new) | Training/testing/lint-only packages (MLflow, Evidently, XGBoost, LightGBM, pytest, flake8, matplotlib, seaborn) moved here | These are never imported by the deployed app. Installing them on Streamlit Cloud only adds install time and failure surface for no runtime benefit. |
| `runtime.txt` (new) | Pins `python-3.12` | Streamlit Cloud reads this to pick the Python version. Without it, Cloud picks a default that may not match what was tested. |
| `.gitignore` | `models/*.pkl` and `reports/` are still ignored by default, but the two model files and the two report images are explicitly un-ignored (`!models/calibrated_gb_model.pkl`, etc.) | The model files were already force-committed, so they weren't actually missing on deploy — but the rule was a footgun: regenerating the model and running a plain `git add` would have silently dropped it again. `reports/` *was* genuinely missing (not committed, and the directory didn't even exist locally), which is why the Model Performance tab only ever showed "Run `python train_model.py`" placeholders. |
| `reports/evaluation/calibration_curve.png`, `reports/shap/shap_summary.png` (new, committed) | Regenerated from the **existing, already-trained model** (no retraining — same frozen `models/calibrated_gb_model.pkl`, same deterministic train/test split) | The Model Performance tab has real charts to show instead of "run training first". Model metrics (AUC ≈0.88) are unchanged because nothing was retrained. |
| `src/models/train.py`, `src/evaluation/calibration.py`, `src/explainability/shap_explainer.py` | `import mlflow` / `import mlflow.sklearn` moved from module level into the functions that use them | Hardening: makes it impossible for a future refactor to accidentally make `app.py`'s import chain depend on MLflow. (Investigation confirmed `app.py`'s actual import chain — `src/data/loader.py`, `src/models/predict.py`, `src/data/preprocessor.py` — never imported MLflow or Evidently in the first place; this wasn't the cause of the outage, but it's now structurally guaranteed to stay that way.) |
| `Dockerfile`, `.github/workflows/ci.yml` | Python 3.11 → 3.12; CI installs `requirements-dev.txt`; a new CI job installs *only* `requirements.txt` and runs the smoke test | CI now has a job that installs exactly what Streamlit Cloud installs and proves the model loads and predicts under it — this is the check that would have caught the original bug before it ever reached production. |
| `tests/test_app_smoke.py` (new) | Loads the real committed model, runs one prediction, builds the full SHAP explanation | Directly covers the failure mode that took the demo down. |
| `app/app.py`, `src/explainability/shap_explainer.py` | SHAP explanation reworked — see [Section 3](#3-how-the-shap-explanation-works) | Was showing scaled/encoded numbers and log-odds from one arbitrary calibration fold; now shows real values and real probability points, averaged across the whole calibrated model. |

---

## 3. How the SHAP explanation works

This section is written for a non-technical reader.

### The baseline

Before looking at any single patient, the model has an "average" opinion:
if you handed it 100 random patients from the training data, what
fraction would it predict survive? Call this the **baseline** — it's the
starting point every individual prediction is measured against.

### How one patient moves away from the baseline

Every patient has some features that make them look *better* than
average (never smoked, no prior conditions) and some that make them look
*worse* (older, treated with a drug combination that historically does
poorly). SHAP ("SHapley Additive exPlanations") is a well-established
method for splitting the *gap* between the baseline and this patient's
final prediction into a contribution from each individual feature — how
many percentage points did *this* feature push survival up or down for
*this* patient.

**Why the pushes add up exactly.** This isn't a rough approximation —
it's an arithmetic identity:

```
baseline probability + (sum of every feature's push) = this patient's final probability
```

The app enforces this exactly (see below), so the numbers you see in the
chart always reconcile with the headline probability shown above it.

### Which model is being explained

The calibrated model isn't one classifier — internally it's **5** slightly
different classifiers (one per cross-validation fold used during Platt
scaling calibration), and their opinions are combined into the final
probability. The old version of this app only explained fold `[0]` — an
arbitrary one of the five. The current version (`explain_patient()` in
`src/explainability/shap_explainer.py`) computes SHAP values against
**all 5 folds and averages them**, and the app says so explicitly
("Averaged across all 5 cross-validation folds..."). That average is
computed in probability terms directly (via SHAP's
`model_output="probability"` mode), then rescaled by a single constant so
it lands exactly on the real calibrated probability — this preserves
each feature's relative importance and direction while guaranteeing the
math matches the number shown elsewhere in the app.

### Reading the chart

- **Green bar** = this factor pushed survival probability *up*.
- **Red bar** = this factor pushed survival probability *down*.
- Bars are sorted by size, biggest at the top, showing the top 8 factors.
- The number on each bar is in **percentage points** of survival
  probability, not log-odds or a raw model score.
- Feature labels use real values ("BMI = 28.0", "Smoker = Yes",
  "Treatment = DX6"), not scaled numbers or category codes.

Underneath the chart, a 3-line plain-English summary calls out the top
factors, e.g. *"BMI = 28.0 lowered this patient's predicted survival the
most (−20.6 points)."*

### Worked example

Patient: Age 62, BMI 28.0, 3 previous conditions, indicator E positive
(all others negative), treated with **DX6** only, a **smoker**, in an
**urban** location.

```
Starting from the average patient survival of 67%, these factors moved
this patient to 2%.

  BMI = 28.0               -20.6 pts
  Smoker = Yes              -18.5 pts
  Age = 62                  -12.4 pts
  Treatment = DX6            -11.8 pts
  Location = Urban            -8.6 pts
  Indicator A = No             +6.4 pts
  Previous conditions = 3      -0.8 pts
  ...
```

`67% − 20.6 − 18.5 − 12.4 − 11.8 − 8.6 + 6.4 − 0.8 + ... ≈ 2%` — matching
the calibrated prediction the app shows for this patient exactly.

### The global chart (Model Performance tab)

The beeswarm chart there is a *different* view: instead of one patient,
it plots SHAP values for every patient in the test set at once. Each dot
is one patient; its horizontal position shows whether that feature
pushed their survival up or down, and its color shows whether that
patient's own value for that feature was high (red) or low (blue).
Reading top to bottom ranks features by their overall impact across the
whole population — this is "what matters in general", where the
per-patient chart is "what mattered for this one person."

### If SHAP fails

Previously, any error was hidden behind a soft `st.info(...)` box. Now a
failure shows a clear `st.error(...)` in the app **and** logs the full
exception (`logger.exception(...)`), so a real bug is visible instead of
silently swallowed.

---

## 4. Run it locally on a Mac (Apple Silicon)

**1. Install Python 3.12** (matches `runtime.txt` and what's been tested):

```bash
brew install python@3.12
```

**2. Clone the repo and create a virtual environment:**

```bash
git clone https://github.com/Khalida-DS/patient-survival-prediction-ml.git
cd patient-survival-prediction-ml
python3.12 -m venv venv
source venv/bin/activate
```

**3. Install dependencies** (use the dev file locally — it includes
everything, including the lean app requirements):

```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```

**4. Run the tests:**

```bash
pytest tests/ -v
```

You should see 18 tests pass, including `tests/test_app_smoke.py`.

**5. Start the app:**

```bash
streamlit run app/app.py
```

**6. What you should see in the browser** (opens automatically at
`http://localhost:8501`): a page titled "Patient Survival Prediction DSS"
with a sidebar showing three tabs — 🔬 Predict Survival, 📊 Model
Performance, 📖 About. On the Predict tab, fill in the form and click
"🔍 Generate Prediction" — you'll see the predicted outcome, calibrated
survival probability, risk tier, and the "Why this prediction?" chart
described in Section 3.

---

## 5. Retrain the model

Only do this if you're intentionally changing the data or model — the
committed model already reflects AUC ≈0.88 and doesn't need retraining
just to run the app.

**1. Make sure dev dependencies are installed** (see Section 4, step 3).

**2. Run the training pipeline:**

```bash
python train_model.py
```

This runs the full pipeline from Section 1: load/clean data, 5-model
benchmark, `RandomizedSearchCV` tuning, Platt scaling calibration,
evaluation, calibration curve, and SHAP global summary. It also logs
everything to MLflow locally (`mlruns/`) — view it with `mlflow ui`.

**3. Where the outputs go:**

- `models/calibrated_gb_model.pkl`, `models/feature_names.pkl` — the
  artifacts the app loads.
- `reports/evaluation/calibration_curve.png`
- `reports/shap/shap_summary.png`
- `mlruns/` — MLflow experiment tracking data (not committed).

**4. Which files must then be committed:**

```bash
git add models/calibrated_gb_model.pkl models/feature_names.pkl \
        reports/evaluation/calibration_curve.png \
        reports/shap/shap_summary.png
git commit -m "Retrain model"
```

If you bump the scikit-learn version as part of this, update the pin in
`requirements.txt` to match — that mismatch is exactly what broke the
demo the first time (Section 2).

---

## 6. Deploy or redeploy on Streamlit Community Cloud

**Repo / branch / main file:**
- Repository: `Khalida-DS/patient-survival-prediction-ml`
- Branch: `main`
- Main file path: `app/app.py`

**Python version:** Streamlit Cloud reads `runtime.txt` (now
`python-3.12`) automatically when the app is first created. **Streamlit
Cloud does not let you change an existing app's Python version** — if it
was already deployed against an older `runtime.txt` (or none), you must
delete the app in Cloud settings and create it again pointing at the same
repo/branch/file so it picks up the new `runtime.txt`.

**To reboot the app** (after a config change or to clear a stuck state):
open the app on [share.streamlit.io](https://share.streamlit.io), open
its menu (⋮ in the bottom-right of the app card, or top-right inside the
app itself), and choose **Reboot app**. A push to `main` triggers an
automatic redeploy on its own; a manual reboot is only needed if it gets
stuck or you want to force a clean reinstall.

**Where to read the logs:** inside the running app, click **Manage app**
in the bottom-right corner — this opens a log panel showing the build
(pip install) log and the running app's stdout/stderr, including any
`logger.exception(...)` output from the SHAP error handling in Section 3.

**If the Python version has to change again:** delete the app from
Cloud's app list and redeploy from the same repo/branch/file — there is
no in-place upgrade path.

---

## 7. The GitHub workflow: review, merge, confirm

**1. Review the pull request.** Open the PR on GitHub — it lists every
file changed and why (see Section 2's table for the full rationale).
Check that CI (`CI Pipeline`) is green on the PR: both the
`lint-and-test` job and the `app-deploy-smoke-test` job (which installs
only `requirements.txt`, the same lean set Cloud installs, and runs the
model smoke test).

**2. Merge into `main`.** Use a normal merge (or squash, per your usual
preference) once CI is green and you're satisfied with the review.

**3. Confirm the live demo updated.** Streamlit Cloud watches `main` and
redeploys automatically on merge. Open the live demo URL, wait for the
"Your app is in the oven" rebuild banner (if shown) to finish, then check
the Predict tab loads without the old "Model not found" error and the
new chart-based SHAP explanation appears after clicking "Generate
Prediction". If it doesn't update within a few minutes, reboot it
manually (Section 6).

---

## 8. Troubleshooting table

| Symptom | Likely cause | Exact fix |
|---|---|---|
| App crashes on load with `ModuleNotFoundError: No module named '_loss'` (or any unpickle error) | scikit-learn version installed on Cloud doesn't match the version the model was pickled with | Pin the exact scikit-learn version in `requirements.txt` to match `models/calibrated_gb_model.pkl` (check with a quick local load-test if unsure), then reboot the app. |
| Cloud install hangs or times out | Heavy, unused packages in the deploy requirements (MLflow, Evidently, XGBoost, LightGBM, etc.) | Make sure the app installs `requirements.txt` (lean), never `requirements-dev.txt`. Training-only packages belong in `requirements-dev.txt` only. |
| "⚠️ Model not found. Please run `python train_model.py` first." | `models/calibrated_gb_model.pkl` or `models/feature_names.pkl` isn't committed / wasn't pulled | `git ls-files models/` should list both `.pkl` files. If missing, check `.gitignore` still has the `!models/calibrated_gb_model.pkl` / `!models/feature_names.pkl` exceptions, then `git add -f` them and commit. |
| Model Performance tab shows "Run `python train_model.py` to generate" instead of charts | `reports/evaluation/calibration_curve.png` or `reports/shap/shap_summary.png` isn't committed | Regenerate (Section 5) and commit both PNGs; confirm `.gitignore`'s `!reports/...` exceptions are intact. |
| "⚠️ Could not generate the SHAP explanation for this patient" | An actual runtime error in `explain_patient()` | Check **Manage app → logs** (Section 6) or local terminal output for the full traceback logged by `logger.exception(...)`; it names the real cause instead of hiding it. |
| Tests pass locally but CI fails on the `app-deploy-smoke-test` job specifically | Something the app needs isn't in the lean `requirements.txt`, only in `requirements-dev.txt` | Add the missing package to `requirements.txt` (with an exact pin) rather than relying on it being present via the dev file. |
| Streamlit Cloud app stuck on an old Python version after updating `runtime.txt` | Cloud doesn't support in-place Python version changes | Delete the app in Cloud settings and redeploy fresh from the same repo/branch/file (Section 6). |
| `flake8` fails in CI | A genuine style violation in `src/` or `train_model.py` | Run `flake8 src/ train_model.py --max-line-length=100 --ignore=E501,W503 --exclude=__pycache__` locally and fix what it reports. |

---

## 9. Checklist before sharing the demo link

- [ ] `pytest tests/ -v` passes locally (18/18).
- [ ] `flake8 src/ train_model.py --max-line-length=100 --ignore=E501,W503 --exclude=__pycache__` is clean.
- [ ] CI is green on `main` (both `lint-and-test` and `app-deploy-smoke-test`).
- [ ] Live demo URL loads the Predict tab without a "Model not found" error.
- [ ] A test prediction on the Predict tab shows a probability, a risk
      tier, and the green/red "Why this prediction?" chart with no error
      box.
- [ ] Model Performance tab shows the calibration curve and SHAP summary
      images (not "run training first" placeholders).
- [ ] `runtime.txt` (`python-3.12`) matches what the deployed Cloud app is
      actually running (check under **Manage app → Settings**).
- [ ] You've skimmed the "not a medical device" disclaimer is still
      visible on the Predict tab and the About tab.
