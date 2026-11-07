import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "hourly_patient_state_alerts.csv"
OUTPUT_FILE = PROCESSED_DIR / "anomaly_features.csv"


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
# FEATURE GROUPS
# ============================================================

core_features = [
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp"
]

trend_features = [
    "heart_rate_change_1h",
    "spo2_change_1h",
    "mean_bp_change_1h",

    "heart_rate_rolling_mean_3h",
    "heart_rate_rolling_std_3h",

    "spo2_rolling_mean_3h",
    "spo2_rolling_std_3h",

    "respiration_rolling_mean_3h",
    "respiration_rolling_std_3h",

    "systolic_bp_rolling_mean_3h",
    "systolic_bp_rolling_std_3h",

    "diastolic_bp_rolling_mean_3h",
    "diastolic_bp_rolling_std_3h",

    "mean_bp_rolling_mean_3h",
    "mean_bp_rolling_std_3h",

    "heart_rate_change_3h",
    "spo2_change_3h",
    "mean_bp_change_3h",

    "heart_rate_change_6h",
    "spo2_change_6h",
    "mean_bp_change_6h"
]

fluid_features = [
    "net_fluid_balance_ml",
    "urine_output_ml",
    "cumulative_net_fluid_balance_ml",
    "cumulative_urine_output_ml"
]

treatment_features = [
    "active_infusion_count",
    "vasoactive_support"
]


feature_columns = (
    core_features
    + trend_features
    + fluid_features
    + treatment_features
)


# ============================================================
# CHECK FEATURES
# ============================================================

missing_columns = [
    col for col in feature_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# SELECT DATA
# ============================================================

features = df[
    [
        "patientunitstayid",
        "hour"
    ] + feature_columns
].copy()


# ============================================================
# REMOVE IMPOSSIBLE VALUES
#
# We already cleaned the main vitals, but this protects the
# ML pipeline from unexpected values.
# ============================================================

print("\nChecking feature ranges...")

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
# MISSINGNESS INDICATORS
#
# Only for physiological measurements and trends.
#
# Fluid/treatment fields are already complete.
# ============================================================

print("Creating missingness indicators...")

missingness_features = []

for col in core_features + trend_features:

    missing_col = col + "_missing"

    features[missing_col] = (
        features[col].isna().astype(int)
    )

    missingness_features.append(
        missing_col
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================================")
print("FEATURE MISSINGNESS")
print("============================================================")

missing_summary = (
    features[feature_columns]
    .isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(
    missing_summary.round(2).to_string()
)


# ============================================================
# SAVE
# ============================================================

features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n============================================================")
print("VALIDATION")
print("============================================================")

print(
    "Rows:",
    len(features)
)

print(
    "Columns:",
    len(features.columns)
)

print(
    "ICU stays:",
    features["patientunitstayid"].nunique()
)

duplicates = features.duplicated(
    subset=["patientunitstayid", "hour"]
).sum()

print(
    "Duplicate patient-hours:",
    duplicates
)

print(
    "Original rows:",
    len(df)
)

if len(features) != len(df):
    raise ValueError(
        "Row count changed during feature preparation."
    )

if duplicates > 0:
    raise ValueError(
        "Duplicate patient-hour rows detected."
    )


print("\n============================================================")
print("FEATURE GROUPS")
print("============================================================")

print(
    "Core physiological features:",
    len(core_features)
)

print(
    "Trend features:",
    len(trend_features)
)

print(
    "Fluid features:",
    len(fluid_features)
)

print(
    "Treatment features:",
    len(treatment_features)
)

print(
    "Missingness indicators:",
    len(missingness_features)
)

print(
    "Total ML features:",
    len(
        feature_columns +
        missingness_features
    )
)


print("\n============================================================")
print("SUCCESS")
print("============================================================")

print(
    "Output:",
    OUTPUT_FILE
)