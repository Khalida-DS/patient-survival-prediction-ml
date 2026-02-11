
# Patient Survival Decision Support System

A machine learning–powered clinical decision support application that predicts one-year patient survival probability using demographic, clinical, lifestyle, and treatment-related features.
The system demonstrates a production-ready ML workflow, from model training and validation to deployment via an interactive web application.

##  Problem Statement

Healthcare providers often lack clear, data-driven insights into which factors most strongly influence patient survival following treatment. This makes it difficult to:

*  Identify high-risk patients early

* Compare treatment effectiveness across cohorts

* Support evidence-based clinical decision-making

This project addresses that challenge by leveraging historical clinical data and machine learning to estimate survival likelihood and surface key risk drivers.

##  Solution Overview

💡 Solution Overview

The solution consists of three core components:

* **Machine Learning Model**
A Gradient Boosting classifier trained to predict one-year survival probability.

* **Interactive Web Application**
A Streamlit-based UI that allows users to input patient characteristics and receive real-time survival predictions with risk scores and visual explanations.

* **Reproducible ML Pipeline**
A modular training and inference workflow with saved model artifacts to ensure consistent, production-style deployment.



## 🧠 Machine Learning Approach
### Model

* **Algorithm:** Gradient Boosting Classifier

* **Task:** Binary classification

* **Target Variable:** Survived_1_year

### AutoML & Validation

* **PyCaret AutoML** was used to benchmark multiple classification models.

* Gradient Boosting was selected based on performance, stability, and interpretability.

* Manual hyperparameter tuning was applied to improve generalization.

### Features

* **Demographics:** Age, BMI

* **Lifestyle Factors:** Smoking status

* **Clinical History:** Previous conditions, mental condition

* **Treatment Information:** Treatment type
  
## Performance

**Accuracy:** ~83% after tuning

* Additional evaluation metrics include:

* ROC Curve & AUC

* Confusion Matrix

* Feature Importance rankings

⚠️ Note: Metrics are reported for demonstration purposes and are not intended for clinical use.

## 🛠 Tech Stack

* **Programming Language:** Python

* **Data Processing:** Pandas, NumPy

* **Machine Learning:** Scikit-learn, PyCaret

* **Web Application:** Streamlit

* **Visualization:** Plotly

* **Model Persistence:** Joblib

##  Project Structure

```
patient-survival-prediction-app/
│
├── app/
│ └── app.py # Streamlit application
│
├── data/
│ └── Survival.csv # Sample /  data
│
├── models/
│ ├── gradient_boosting.pkl # Trained Gradient Boosting model
│ └── feature_names.pkl # Saved feature schema for inference
│
├── train_model.py # Model training & evaluation pipeline
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```




##  ▶️ How to Run Locally
#### 1️⃣ Install Dependencies

```
pip install -r requirements.txt
```
#### 2️⃣ Train the Model
```
python train_model.py
```
### 3️⃣ Launch the Application

```
streamlit run app/app.py
```

## 📊 Application Features

* Interactive patient data input form

* Real-time survival probability prediction

* Probability-based risk scoring

* Exploratory data visualizations

* Model performance evaluation:

     * ROC curve

     * Confusion matrix

* Feature importance analysis for interpretability

## 📊 Dashboard Screenshots
### 🔹 Application Interface
Interactive form for entering patient demographic, clinical, and treatment data.

![Application Interface](screenshots/app_input.png)

---

### 🔹 Prediction Output
Predicted one-year survival outcome with probability-based risk scoring.

![Prediction Result](screenshots/prediction_result.png)

---

### 🔹 Feature Importance
Top features influencing the survival prediction using Gradient Boosting.

![Feature Importance](screenshots/feature_importance.png)


## 📌 Notes & Limitations

* This project is intended for educational and analytical demonstration purposes only.

* It is not a medical device and should not be used for real-world clinical decision-making.

* Future improvements could include:

     * Survival analysis models (Cox, Kaplan–Meier)

     * Time-to-event prediction

     * Model explainability with SHAP

     * Role-based dashboards for clinicians and executives

The model and feature schema are persisted to ensure training–inference consistency

## ⭐ Why This Project Matters

This system demonstrates:

   * End-to-end ML pipeline design

   * Model selection and validation

   * Deployment-ready inference workflows

   * Practical application of ML in healthcare analytics

It bridges the gap between data science experimentation and real-world decision support systems.



