import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

STATE_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_alerts.csv"
)

LABEL_FILE = (
    PROCESSED_DIR /
    "deterioration_labels.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "deterioration_features.csv"
)

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

feature_columns = [

    # Current physiology
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",

    # Short-term trends
    "heart_rate_change_1h",
    "spo2_change_1h",
    "respiration_change_1h",
    "systolic_bp_change_1h",
    "diastolic_bp_change_1h",
    "mean_bp_change_1h",

    # Recent trends
    "heart_rate_rolling_mean_3h",
    "spo2_rolling_mean_3h",
    "respiration_rolling_mean_3h",
    "mean_bp_rolling_mean_3h",

    # Fluid
    "net_fluid_balance_ml",
    "urine_output_ml",

    # Treatment
    "active_infusion_count",
    "vasoactive_support",

    # Interpretable clinical signal counts
    "clinical_warning_count",
    "clinical_severe_count"
]

print("Loading patient state...")
state = pd.read_csv(
    STATE_FILE,
    low_memory=False
)

print("Loading deterioration labels...")
labels = pd.read_csv(
    LABEL_FILE,
    low_memory=False
)

print("\nState shape:", state.shape)
print("Label shape:", labels.shape)

# ------------------------------------------------------------
# VALIDATE
# ------------------------------------------------------------

required_state = [
    "patientunitstayid",
    "hour"
]

required_label = [
    "patientunitstayid",
    "hour",
    "deterioration_label_24h"
]

for col in required_state:
    if col not in state.columns:
        raise ValueError(
            f"Missing state column: {col}"
        )

for col in required_label:
    if col not in labels.columns:
        raise ValueError(
            f"Missing label column: {col}"
        )

missing_features = [
    col
    for col in feature_columns
    if col not in state.columns
]

if missing_features:
    raise ValueError(
        f"Missing feature columns: {missing_features}"
    )

# ------------------------------------------------------------
# SELECT FEATURES
# ------------------------------------------------------------

features = state[
    [
        "patientunitstayid",
        "hour"
    ] +
    feature_columns
].copy()

# ------------------------------------------------------------
# RENAME RULE COUNTS
# ------------------------------------------------------------

features["clinical_warning_count"] = (
    pd.to_numeric(
        features["clinical_warning_count"],
        errors="coerce"
    )
)

features["clinical_severe_count"] = (
    pd.to_numeric(
        features["clinical_severe_count"],
        errors="coerce"
    )
)

# ------------------------------------------------------------
# NUMERIC CONVERSION
# ------------------------------------------------------------

for col in feature_columns:

    features[col] = pd.to_numeric(
        features[col],
        errors="coerce"
    )

# ------------------------------------------------------------
# CONSERVATIVE PHYSIOLOGICAL CLEANING
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# MERGE LABEL
# ------------------------------------------------------------

labels = labels[
    [
        "patientunitstayid",
        "hour",
        "deterioration_label_24h"
    ]
].copy()

merged = features.merge(
    labels,
    on=[
        "patientunitstayid",
        "hour"
    ],
    how="inner",
    validate="one_to_one"
)

# ------------------------------------------------------------
# REMOVE INVALID LABEL ROWS
# ------------------------------------------------------------

merged = merged[
    merged["deterioration_label_24h"]
    .notna()
].copy()

merged["deterioration_label_24h"] = (
    pd.to_numeric(
        merged["deterioration_label_24h"],
        errors="coerce"
    )
    .astype(int)
)

# ------------------------------------------------------------
# FEATURE AVAILABILITY
# ------------------------------------------------------------

print("\n============================================================")
print("FEATURE AVAILABILITY")
print("============================================================")

availability = (
    merged[feature_columns]
    .notna()
    .mean()
    .mul(100)
    .round(2)
    .sort_values()
)

print(
    availability.to_string()
)

# ------------------------------------------------------------
# LABEL SUMMARY
# ------------------------------------------------------------

print("\n============================================================")
print("LABEL SUMMARY")
print("============================================================")

print(
    merged[
        "deterioration_label_24h"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

positive_rate = (
    merged[
        "deterioration_label_24h"
    ].mean() * 100
)

print(
    "Positive rate:",
    round(
        positive_rate,
        3
    ),
    "%"
)

print(
    "Positive patients:",
    merged.loc[
        merged["deterioration_label_24h"] == 1,
        "patientunitstayid"
    ].nunique()
)

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n============================================================")
print("VALIDATION")
print("============================================================")

print(
    "Rows:",
    len(merged)
)

print(
    "ICU stays:",
    merged["patientunitstayid"].nunique()
)

print(
    "Duplicate patient-hours:",
    merged.duplicated(
        subset=[
            "patientunitstayid",
            "hour"
        ]
    ).sum()
)

print(
    "Missing labels:",
    merged[
        "deterioration_label_24h"
    ].isna().sum()
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

merged.to_csv(
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