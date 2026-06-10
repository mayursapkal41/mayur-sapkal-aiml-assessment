import pandas as pd
import numpy as np

# =========================
# LOAD DATASETS
# =========================

print("\nLoading datasets...")

leads_df = pd.read_csv("data/leads.csv")
interactions_df = pd.read_csv("data/interactions.csv")

print("Datasets loaded successfully!")

# =========================
# DATA OVERVIEW
# =========================

print("\n========== DATA SHAPES ==========")
print(f"Leads Dataset Shape: {leads_df.shape}")
print(f"Interactions Dataset Shape: {interactions_df.shape}")

# =========================
# MISSING VALUES
# =========================

print("\n========== LEADS MISSING VALUES ==========")
print(leads_df.isnull().sum())

print("\n========== INTERACTIONS MISSING VALUES ==========")
print(interactions_df.isnull().sum())

# =========================
# DUPLICATES
# =========================

print("\n========== DUPLICATE ROWS ==========")
print(f"Leads Duplicates: {leads_df.duplicated().sum()}")
print(f"Interactions Duplicates: {interactions_df.duplicated().sum()}")

# =========================
# BASIC CLEANING
# =========================

# Remove duplicate rows
leads_df = leads_df.drop_duplicates()

# Convert datetime columns
leads_df["created_at"] = pd.to_datetime(leads_df["created_at"])
interactions_df["timestamp"] = pd.to_datetime(
    interactions_df["timestamp"]
)

print("\nBasic cleaning completed!")

# =========================
# IMPORTANT BUSINESS INSIGHTS
# =========================

print("\n========== FUNNEL STAGES ==========")
print(interactions_df["funnel_stage"].value_counts())

print("\n========== FORM COMPLETION ==========")
print(interactions_df["form_completed"].value_counts())

print("\n========== TOP PAGE CATEGORIES ==========")
print(
    interactions_df["page_category"]
    .value_counts()
    .head(10)
)

print("\n========== DEVICE TYPES ==========")
print(interactions_df["device_type"].value_counts())

print("\n========== RETURN VISITORS ==========")
print(
    interactions_df["is_return_visitor"]
    .value_counts()
)

# =========================
# SESSION ANALYSIS
# =========================

print("\n========== SESSION ANALYSIS ==========")

avg_session_duration = interactions_df[
    "session_duration_seconds"
].mean()

avg_scroll_depth = interactions_df[
    "scroll_depth_percent"
].mean()

avg_clicks = interactions_df[
    "click_count"
].mean()

print(f"Average Session Duration: {avg_session_duration:.2f} seconds")
print(f"Average Scroll Depth: {avg_scroll_depth:.2f}%")
print(f"Average Click Count: {avg_clicks:.2f}")

# =========================
# TEMPORAL ANALYSIS
# =========================

leads_df["month"] = leads_df["created_at"].dt.month_name()

print("\n========== LEADS BY MONTH ==========")
print(leads_df["month"].value_counts())

print("\nEDA completed successfully!")