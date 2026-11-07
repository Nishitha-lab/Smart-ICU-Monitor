import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "hourly_patient_state_alerts.csv"
OUTPUT_FILE = PROCESSED_DIR / "anomaly_features_v2.csv"


# ============================================================
# FEATURE SET
# ============================================================

core_features = [
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp"
]

short_term_features = [
    "heart_rate_change_1h",
    "spo2_change_1h",
    "mean_bp_change_1h"
]

recent_features = [
    "heart_rate_rolling_mean_3h",
    "spo2_rolling_mean_3h",
    "respiration_rolling_mean_3h",
    "mean_bp_rolling_mean_3h"
]

fluid_features = [
    "net_fluid_balance_ml",
    "urine_output_ml"
]

treatment_features = [
    "active_infusion_count",
    "vasoactive_support"
]

feature_columns = (
    core_features
    + short_term_features
    + recent_features
    + fluid_features
    + treatment_features
)


# ============================================================
# LOAD
# ============================================================

print("Loading alert dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Input shape:", df.shape)
print("ICU stays:", df["patientunitstayid"].nunique())


# ============================================================
# CHECK COLUMNS
# ============================================================

missing_columns = [
    col
    for col in feature_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# SELECT FEATURES
# ============================================================

features = df[
    [
        "patientunitstayid",
        "hour"
    ] + feature_columns
].copy()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for col in feature_columns:
    features[col] = pd.to_numeric(
        features[col],
        errors="coerce"
    )


# ============================================================
# REMOVE IMPOSSIBLE VALUES
# ============================================================

print("\nChecking physiological ranges...")

features.loc[
    (features["heart_rate"] <= 0) |
    (features["heart_rate"] > 300),
    "heart_rate"
] = np.nan

features.loc[
    (features["spo2"] <= 0) |
    (features["spo2"] > 100),
    "spo2"
] = np.nan

features.loc[
    (features["respiration"] <= 0) |
    (features["respiration"] > 100),
    "respiration"
] = np.nan

features.loc[
    (features["systolic_bp"] <= 0) |
    (features["systolic_bp"] > 300),
    "systolic_bp"
] = np.nan

features.loc[
    (features["diastolic_bp"] <= 0) |
    (features["diastolic_bp"] > 200),
    "diastolic_bp"
] = np.nan

features.loc[
    (features["mean_bp"] <= 0) |
    (features["mean_bp"] > 250),
    "mean_bp"
] = np.nan


# ============================================================
# VALIDATION
# ============================================================

print("\n============================================================")
print("FEATURE AVAILABILITY")
print("============================================================")

availability = (
    features[feature_columns]
    .notna()
    .mean()
    .mul(100)
    .round(2)
    .sort_values()
)

print(
    availability.to_string()
)


print("\n============================================================")
print("VALIDATION")
print("============================================================")

print(
    "Rows:",
    len(features)
)

print(
    "ICU stays:",
    features["patientunitstayid"].nunique()
)

print(
    "Duplicate patient-hours:",
    features.duplicated(
        subset=["patientunitstayid", "hour"]
    ).sum()
)

print(
    "Complete feature rows:",
    features[feature_columns]
    .notna()
    .all(axis=1)
    .sum()
)


# ============================================================
# SAVE
# ============================================================

features.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n============================================================")
print("SUCCESS")
print("============================================================")

print(
    "ML features:",
    len(feature_columns)
)

print(
    "Output:",
    OUTPUT_FILE
)