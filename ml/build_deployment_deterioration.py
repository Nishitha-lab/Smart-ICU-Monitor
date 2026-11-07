import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

TRAIN_FILE = (
    PROCESSED_DIR /
    "deterioration_train.csv"
)

STATE_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_alerts.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "deployment_deterioration_risk.csv"
)

feature_columns = [
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",

    "heart_rate_change_1h",
    "spo2_change_1h",
    "respiration_change_1h",
    "systolic_bp_change_1h",
    "diastolic_bp_change_1h",
    "mean_bp_change_1h",

    "heart_rate_rolling_mean_3h",
    "spo2_rolling_mean_3h",
    "respiration_rolling_mean_3h",
    "mean_bp_rolling_mean_3h",

    "net_fluid_balance_ml",
    "urine_output_ml",

    "active_infusion_count",
    "vasoactive_support",

    "clinical_warning_count",
    "clinical_severe_count"
]

TARGET = "deterioration_label_24h"

print("Loading deterioration training data...")
train = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

print("Loading complete patient state...")
state = pd.read_csv(
    STATE_FILE,
    low_memory=False
)

print("Train shape:", train.shape)
print("State shape:", state.shape)

# ------------------------------------------------------------
# PREPARE
# ------------------------------------------------------------

X_train = train[
    feature_columns
].copy()

y_train = train[
    TARGET
].astype(int)

X_all = state[
    feature_columns
].copy()

for col in feature_columns:

    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )

    X_all[col] = pd.to_numeric(
        X_all[col],
        errors="coerce"
    )

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
#
# This is the probability-oriented model.
#
# We intentionally do NOT use class_weight="balanced"
# because we want the score to remain on the natural
# observed prevalence scale.
#

print("\n============================================================")
print("TRAINING DEPLOYMENT RISK MODEL")
print("============================================================")

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                class_weight=None,
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

model.fit(
    X_train,
    y_train
)

print("Training complete.")

# ------------------------------------------------------------
# SCORE ALL ICU HOURS
# ------------------------------------------------------------

print("Scoring all ICU hours...")

risk = model.predict_proba(
    X_all
)[:, 1]

# ------------------------------------------------------------
# CREATE OUTPUT
# ------------------------------------------------------------

output = state[
    [
        "patientunitstayid",
        "hour"
    ]
].copy()

output["risk_probability"] = risk

# ------------------------------------------------------------
# RISK BAND
# ------------------------------------------------------------

def risk_band(value):

    if value < 0.01:
        return "very_low"

    if value < 0.02:
        return "low"

    if value < 0.05:
        return "elevated"

    if value < 0.10:
        return "high"

    if value < 0.25:
        return "very_high"

    return "extreme"


output["risk_band"] = (
    output["risk_probability"]
    .apply(risk_band)
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n============================================================")
print("DEPLOYMENT RISK SUMMARY")
print("============================================================")

print(
    "Rows:",
    len(output)
)

print(
    "ICU stays:",
    output["patientunitstayid"].nunique()
)

print("\nRisk probability statistics:")

print(
    output["risk_probability"]
    .describe()
    .round(6)
    .to_string()
)

print("\nRisk band distribution:")

print(
    output["risk_band"]
    .value_counts()
    .to_string()
)

print(
    "\nRisk >= 10%:",
    int(
        (
            output["risk_probability"] >= 0.10
        ).sum()
    )
)

print(
    "Risk >= 25%:",
    int(
        (
            output["risk_probability"] >= 0.25
        ).sum()
    )
)

print(
    "Risk >= 50%:",
    int(
        (
            output["risk_probability"] >= 0.50
        ).sum()
    )
)

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n============================================================")
print("VALIDATION")
print("============================================================")

print(
    "Duplicate patient-hours:",
    output.duplicated(
        subset=[
            "patientunitstayid",
            "hour"
        ]
    ).sum()
)

print(
    "Missing risk scores:",
    output["risk_probability"].isna().sum()
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n============================================================")
print("SUCCESS")
print("============================================================")

print(
    "Output:",
    OUTPUT_FILE
)