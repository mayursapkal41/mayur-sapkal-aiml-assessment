# AI-Powered Lead Conversion Prediction System

## Overview

This project was developed as part of the Vynqe AI/ML Engineer Assessment.

The objective is to predict whether a lead is likely to convert into a customer using company information and behavioral interaction data.

The solution includes:

* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning Model Training
* FastAPI-based Prediction API
* Gemini-powered Explanation Endpoint
* Model Evaluation and Explainability

---

# Project Structure

```text
mayur-sapkal-aiml-assessment/
│
├── data/
│   ├── leads.csv
│   └── interactions.csv
│
├── models/
│   ├── model.pkl
│   └── feature_columns.pkl
│
├── outputs/
│   ├── model_metrics.json
│   ├── feature_importance.png
│   ├── funnel_stage_distribution.png
│   ├── device_distribution.png
│   ├── monthly_leads.png
│   └── session_duration_distribution.png
│
├── analysis.md
├── README.md
├── app.py
├── train.py
├── requirements.txt
├── .gitignore
└── .env
```

---

# Problem Statement

Organizations generate thousands of leads through marketing campaigns, but not every lead converts into a paying customer.

The goal of this project is to:

* Predict lead conversion probability
* Identify high-intent leads
* Improve sales prioritization
* Increase marketing efficiency

---

# Dataset Information

## Leads Dataset

Contains:

* Lead Source
* Campaign Information
* Industry
* Company Size
* Revenue Band
* Location
* Account Type

Shape:

* 2,045 rows
* 21 columns

---

## Interactions Dataset

Contains:

* Session Information
* Page Visits
* Scroll Depth
* Click Activity
* Mouse Activity
* Funnel Stages
* Device Information
* Visitor Behavior

Shape:

* 40,000 rows
* 36 columns

---

# Exploratory Data Analysis

The following analyses were performed:

* Missing Value Analysis
* Duplicate Detection
* Funnel Stage Analysis
* Form Completion Analysis
* Device Usage Analysis
* Visitor Behavior Analysis
* Conversion Distribution Analysis
* Temporal Trend Analysis

Generated Visualizations:

* Funnel Stage Distribution
* Device Distribution
* Monthly Lead Trend
* Session Duration Distribution
* Feature Importance Analysis

Detailed findings are available in:

```text
analysis.md
```

---

# Feature Engineering

Lead-level behavioral features were engineered from interaction data.

Features created:

* Average Session Duration
* Average Scroll Depth
* Total Clicks
* Average Mouse Activity
* Total Sessions
* Total Page Visits
* Return Visitor Indicator

These features capture engagement and purchase intent.

---

# Machine Learning Approach

Three machine learning models were trained and evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* AUC-ROC

---

# Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.8663   | 0.8596    | 0.8500 | 0.8547   | 0.9317  |
| Random Forest       | 0.8869   | 0.8864    | 0.8667 | 0.8764   | 0.9389  |
| XGBoost             | 0.8792   | 0.8556    | 0.8889 | 0.8719   | 0.9392  |

Selected Model:

**XGBoost**

Reason:

* Highest AUC-ROC Score
* Strong performance on unseen data
* Handles non-linear relationships effectively

---

# Feature Importance

Top predictive features identified:

1. Total Sessions
2. Return Visitor
3. Total Clicks
4. Average Session Duration
5. Account Type

Key observation:

Behavioral engagement metrics were more predictive than demographic attributes.

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd mayur-sapkal-aiml-assessment
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

# Training the Model

Run:

```bash
python train.py
```

Generated Outputs:

```text
models/model.pkl
models/feature_columns.pkl
outputs/model_metrics.json
outputs/feature_importance.png
```

---

# Running the API

Start FastAPI server:

```bash
python -m uvicorn app:app --reload
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## GET /

Health Check Endpoint

Response:

```json
{
  "message": "Lead Conversion Prediction API is running!"
}
```

---

## POST /predict

Predict lead conversion probability.

Request:

```json
{
  "avg_session_duration": 900,
  "avg_scroll_depth": 70,
  "total_clicks": 25,
  "avg_mouse_activity": 60,
  "total_sessions": 8,
  "total_page_visits": 30,
  "return_visitor": 1
}
```

Response:

```json
{
  "prediction": 1,
  "conversion_probability": 0.912,
  "confidence": "High",
  "risk_level": "Low"
}
```

---

## POST /explain

Generate AI-powered lead analysis using Gemini.

Request:

```json
{
  "conversion_probability": 0.912,
  "total_sessions": 8,
  "total_page_visits": 30,
  "return_visitor": 1
}
```

Response:

```json
{
  "ai_explanation": "This lead demonstrates strong engagement and high purchase intent..."
}
```

---

# Deployment

The API can be deployed using:

* Render
* Railway
* AWS EC2
* Docker

### Live API

https://lead-conversion-prediction-api.onrender.com/

### Swagger Documentation

https://lead-conversion-prediction-api.onrender.com/docs

---

# Limitations

* Conversion labels were inferred from behavioral signals.
* Dataset size is limited.
* Additional CRM and sales outcome data could improve predictive performance.

---

# Future Improvements

* Hyperparameter Optimization
* SHAP Explainability
* Cross Validation
* Real-Time Prediction Pipeline
* Model Monitoring
* Automated Retraining
* Cloud-Native Deployment

---

# Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* FastAPI
* Gemini API
* Matplotlib

---

# Author

**Mayur Sapkal**

AI/ML Engineer Assessment Submission
