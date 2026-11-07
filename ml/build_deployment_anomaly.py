import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

TRAIN_FILE = (
    PROCESSED_DIR /
    "anomaly_train_v2.csv"
)

STATE_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_alerts.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "deployment_anomaly_scores.csv"
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
    "mean_bp_change_1h",

    "heart_rate_rolling_mean_3h",
    "spo2_rolling_mean_3h",
    "respiration_rolling_mean_3h",
    "mean_bp_rolling_mean_3h",

    "net_fluid_balance_ml",
    "urine_output_ml",

    "active_infusion_count",
    "vasoactive_support"
]

print("Loading anomaly training data...")
train = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

print("Loading complete patient state...")
state = pd.read_csv(
    STATE_FILE,
    low_memory=False
)

X_train = train[feature_columns].copy()
X_all = state[feature_columns].copy()

for col in feature_columns:
    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )
    X_all[col] = pd.to_numeric(
        X_all[col],
        errors="coerce"
    )

print("\nTraining deployment anomaly model...")

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "model",
            IsolationForest(
                n_estimators=300,
                contamination=0.02,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

model.fit(X_train)

print("Scoring all ICU hours...")

scores = -model.decision_function(
    X_all
)

flags = (
    model.predict(X_all) == -1
).astype(int)

output = state[
    [
        "patientunitstayid",
        "hour"
    ]
].copy()

output["anomaly_score"] = scores
output["anomaly_flag"] = flags

print("\n============================================================")
print("DEPLOYMENT ANOMALY SUMMARY")
print("============================================================")

print(
    "Rows:",
    len(output)
)

print(
    "Anomalies:",
    int(output["anomaly_flag"].sum())
)

print(
    "Anomaly rate:",
    round(
        output["anomaly_flag"].mean() * 100,
        2
    ),
    "%"
)

print(
    "ICU stays:",
    output["patientunitstayid"].nunique()
)

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n============================================================")
print("SUCCESS")
print("============================================================")

print("Output:", OUTPUT_FILE)