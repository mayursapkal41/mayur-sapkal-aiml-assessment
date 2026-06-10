import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# =========================
# LOAD DATASETS
# =========================

print("\nLoading datasets...")

leads_df = pd.read_csv("data/leads.csv")
interactions_df = pd.read_csv("data/interactions.csv")

print("Datasets loaded successfully!")

# =========================
# BASIC CLEANING
# =========================

# Remove duplicate rows
leads_df = leads_df.drop_duplicates()

# Convert datetime columns
leads_df["created_at"] = pd.to_datetime(
    leads_df["created_at"]
)

interactions_df["timestamp"] = pd.to_datetime(
    interactions_df["timestamp"]
)

print("\nBasic cleaning completed!")

# =========================
# MISSING VALUES SUMMARY
# =========================

print("\n========== LEADS MISSING VALUES ==========")
print(leads_df.isnull().sum())

print("\n========== INTERACTIONS MISSING VALUES ==========")
print(interactions_df.isnull().sum())

# =========================
# BUSINESS INSIGHTS
# =========================

print("\n========== FUNNEL STAGES ==========")
print(interactions_df["funnel_stage"].value_counts())

print("\n========== FORM COMPLETION ==========")
print(interactions_df["form_completed"].value_counts())

print("\n========== DEVICE TYPES ==========")
print(interactions_df["device_type"].value_counts())

print("\n========== RETURN VISITORS ==========")
print(
    interactions_df["is_return_visitor"]
    .value_counts()
)

# =========================
# CREATE TARGET VARIABLE
# =========================

interactions_df["converted"] = np.where(
    (interactions_df["form_completed"] == True) |
    (interactions_df["funnel_stage"] == "Decision"),
    1,
    0
)

print("\n========== TARGET DISTRIBUTION ==========")
print(interactions_df["converted"].value_counts())

# =========================
# FEATURE ENGINEERING
# =========================

print("\nCreating lead-level features...")

lead_features = interactions_df.groupby("lead_id").agg({

    # Engagement metrics
    "session_duration_seconds": "mean",
    "scroll_depth_percent": "mean",
    "click_count": "sum",
    "mouse_activity_score": "mean",

    # Behavioral metrics
    "session_id": "nunique",
    "page_name": "count",

    # Visitor behavior
    "is_return_visitor": "max",

    # Target
    "converted": "max"

}).reset_index()

# Rename columns
lead_features.columns = [
    "lead_id",
    "avg_session_duration",
    "avg_scroll_depth",
    "total_clicks",
    "avg_mouse_activity",
    "total_sessions",
    "total_page_visits",
    "return_visitor",
    "converted"
]

print("\n========== FEATURE DATASET ==========")
print(lead_features.head())

# =========================
# MERGE DATASETS
# =========================

final_df = pd.merge(
    leads_df,
    lead_features,
    on="lead_id",
    how="inner"
)

print("\n========== FINAL DATASET ==========")
print(final_df.head())

print("\nFinal Dataset Shape:", final_df.shape)

# =========================
# TARGET DISTRIBUTION
# =========================

print("\n========== FINAL TARGET DISTRIBUTION ==========")
print(final_df["converted"].value_counts())

conversion_rate = final_df["converted"].mean() * 100

print(f"\nConversion Rate: {conversion_rate:.2f}%")

# =========================
# PREPARE DATA FOR MODELING
# =========================

print("\nPreparing data for modeling...")

# Drop unnecessary columns
drop_cols = [
    "lead_id",
    "business_email",
    "created_at"
]

final_df = final_df.drop(columns=drop_cols)

# =========================
# ENCODE CATEGORICAL COLUMNS
# =========================

categorical_cols = final_df.select_dtypes(
    include=["object", "bool", "string"]
).columns

label_encoders = {}

# Fill missing categorical values
for col in categorical_cols:
    final_df[col] = final_df[col].fillna("Unknown")

# Encode categorical columns
for col in categorical_cols:

    le = LabelEncoder()

    final_df[col] = le.fit_transform(
        final_df[col].astype(str)
    )

    label_encoders[col] = le

# =========================
# HANDLE MISSING VALUES
# =========================

numerical_cols = final_df.select_dtypes(
    include=["int64", "float64"]
).columns

for col in numerical_cols:

    final_df[col] = final_df[col].fillna(
        final_df[col].median()
    )

# =========================
# FEATURES AND TARGET
# =========================

X = final_df.drop("converted", axis=1)
y = final_df["converted"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train-test split completed!")

# =========================
# MODEL TRAINING
# =========================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

best_model = None
best_model_name = None
best_auc = 0

results = {}

print("\nTraining models...\n")

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4)
    }

    print(f"\n{name} Results:")
    print(results[name])

    # Select best model
    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

# =========================
# SAVE BEST MODEL
# =========================

joblib.dump(best_model, "models/model.pkl")

print(f"\nBest Model: {best_model_name}")

# =========================
# SAVE MODEL METRICS
# =========================

with open("outputs/model_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nModel metrics saved successfully!")

print("\nMODEL TRAINING COMPLETED!")

# =========================
# FEATURE IMPORTANCE
# =========================

if best_model_name in ["Random Forest", "XGBoost"]:

    importance = best_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n========== TOP FEATURES ==========")
    print(feature_importance_df.head(10))

    # Plot top 10 features
    plt.figure(figsize=(10, 6))

    plt.barh(
        feature_importance_df["Feature"].head(10),
        feature_importance_df["Importance"].head(10)
    )

    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Top 10 Feature Importances")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig("outputs/feature_importance.png")

    print("\nFeature importance graph saved successfully!")