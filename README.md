
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

##  Machine Learning Approach

Model: Gradient Boosting Classifier

Target: One-year survival (Survived_1_year)

PyCaret AutoML was used to benchmark multiple classification models and validate the manually tuned Gradient Boosting model.

#### Features:

Demographics (Age, BMI)

Lifestyle factors (Smoking status)

Clinical history (Previous conditions, mental condition)

Treatment type

#### Performance: ~83% accuracy after hyperparameter tuning

##  Tech Stack

Python

Pandas / NumPy

Scikit-learn

Streamlit

Plotly

Joblib

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




##  How to Run Locally
#### 1. Install dependencies
pip install -r requirements.txt

#### 2. Train the model
python train_model.py

#### 3. Run the app
streamlit run app/app.py

##  Application Features

Interactive patient data input

Real-time survival prediction

Probability-based risk scoring

Exploratory visualizations

Model performance evaluation (ROC, confusion matrix)

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


## 📌 Notes

The model and feature schema are persisted to ensure training–inference consistency



## 👤 Author
Khalida Khaldi

M.S. Data Science

Focus: Machine Learning, Analytics, Deployment

