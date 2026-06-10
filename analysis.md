# Lead Conversion Prediction - Exploratory Data Analysis

## 1. Project Objective

The objective of this project is to analyze lead behavior and predict whether a lead is likely to convert into a customer using machine learning techniques.

The analysis combines company information from leads.csv with behavioral interaction data from interactions.csv to identify key conversion drivers.

---

# 2. Dataset Overview

## Leads Dataset

* Rows: 2,045
* Columns: 21

Contains:

* Lead source
* Campaign information
* Geographic information
* Industry
* Company size
* Revenue band
* Job role
* Account type

## Interactions Dataset

* Rows: 40,000
* Columns: 36

Contains:

* Session activity
* Page visits
* Scroll depth
* Click activity
* Mouse activity
* Funnel stage progression
* Device information
* Visitor behavior

---

# 3. Data Quality Assessment

## Missing Values

### Leads Dataset

Missing values were identified in:

* City (40)
* Browser (101)
* Company Size (101)
* Annual Revenue Band (41)

### Interactions Dataset

Major missing values were observed in:

* Page Name (829)
* Button Name (3,869)
* CTA Type (7,944)
* Form Details (32,220)
* Browser (1,906)
* UTM Parameters
* Previous Session Gap Days (6,271)

### Handling Strategy

* Missing categorical values were replaced with "Unknown".
* Missing numerical values were replaced using median imputation.

---

## Duplicate Records

| Dataset      | Duplicate Records |
| ------------ | ----------------- |
| Leads        | 20                |
| Interactions | 0                 |

Duplicate lead records were removed during preprocessing.

---

# 4. Funnel Analysis

## Funnel Stage Distribution

| Funnel Stage  | Interactions |
| ------------- | ------------ |
| Awareness     | 13,537       |
| Consideration | 11,488       |
| Evaluation    | 10,304       |
| Decision      | 4,671        |

### Insight

Most users enter the Awareness stage, but only a smaller proportion reach the Decision stage.

### Business Recommendation

Improve lead nurturing and retargeting campaigns to increase progression through the funnel.

---

# 5. Form Completion Analysis

| Status | Count  |
| ------ | ------ |
| False  | 38,719 |
| True   | 1,281  |

### Insight

The majority of visitors do not complete forms.

### Business Recommendation

Reduce form complexity and improve CTA placement to increase lead capture rates.

---

# 6. Device Analysis

| Device  | Count  |
| ------- | ------ |
| Mobile  | 25,767 |
| Desktop | 12,241 |
| Tablet  | 1,992  |

### Insight

Mobile users represent the majority of platform traffic.

### Business Recommendation

Continue prioritizing mobile-first design and optimization.

---

# 7. Visitor Behavior Analysis

| Visitor Type       | Count  |
| ------------------ | ------ |
| Returning Visitor  | 33,729 |
| First-Time Visitor | 6,271  |

### Insight

Returning visitors account for the majority of interactions.

### Business Recommendation

Invest in remarketing and re-engagement strategies to encourage repeat visits.

---

# 8. Conversion Analysis

Target variable was created using:

* Form Completion
* Decision Funnel Stage

Target Distribution:

| Converted | Count  |
| --------- | ------ |
| No        | 34,637 |
| Yes       | 5,363  |

Lead-Level Conversion Distribution:

| Converted | Count |
| --------- | ----- |
| No        | 1,015 |
| Yes       | 930   |

Overall Lead Conversion Rate:

47.81%

---

# 9. Feature Engineering

Several lead-level behavioral features were engineered from interaction data.

Created Features:

* Average Session Duration
* Average Scroll Depth
* Total Clicks
* Average Mouse Activity
* Total Sessions
* Total Page Visits
* Return Visitor Indicator

These features capture user engagement, interaction frequency, and purchase intent.

---

# 10. Model Performance

Three machine learning models were evaluated.

| Model               | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.8663   | 0.8596    | 0.8500 | 0.8547   | 0.9317  |
| Random Forest       | 0.8869   | 0.8864    | 0.8667 | 0.8764   | 0.9389  |
| XGBoost             | 0.8792   | 0.8556    | 0.8889 | 0.8719   | 0.9392  |

Selected Model:

**XGBoost**

Reason:

* Highest AUC-ROC Score
* Strong predictive capability
* Better handling of complex non-linear relationships

---

# 11. Feature Importance Analysis

Top Predictive Features:

| Feature                  | Importance |
| ------------------------ | ---------- |
| Total Sessions           | 0.558      |
| Return Visitor           | 0.050      |
| Total Clicks             | 0.047      |
| Average Session Duration | 0.026      |
| Account Type             | 0.022      |
| Industry                 | 0.020      |
| Employee Growth Band     | 0.019      |
| Source                   | 0.018      |
| State                    | 0.018      |
| Total Page Visits        | 0.018      |

### Key Finding

Behavioral engagement metrics are significantly more predictive than company demographic information.

---

# 12. Key Business Findings

### Finding 1

Session frequency is the strongest predictor of conversion.

### Finding 2

Returning visitors show stronger purchase intent than first-time visitors.

### Finding 3

Mobile traffic dominates user engagement.

### Finding 4

Users drop significantly between Awareness and Decision stages.

### Finding 5

Behavioral signals outperform demographic information for conversion prediction.

---

# 13. Business Recommendations

1. Increase retargeting efforts for returning visitors.
2. Improve funnel progression between Awareness and Decision stages.
3. Continue mobile-first optimization.
4. Simplify lead capture forms.
5. Prioritize outreach to highly engaged leads identified by the model.
6. Use session frequency as a key lead scoring signal.

---

# 14. Limitations

* Conversion labels were inferred from behavioral signals.
* Limited historical time range.
* Additional CRM and sales outcome data could improve prediction quality.

---

# 15. Future Improvements

* Hyperparameter tuning using GridSearchCV
* Cross-validation
* SHAP-based explainability
* Real-time prediction pipeline
* Model monitoring and drift detection

```
```
