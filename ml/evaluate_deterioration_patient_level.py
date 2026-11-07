import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

PREDICTION_FILE = (
    PROCESSED_DIR /
    "deterioration_predictions.csv"
)

TEST_FILE = (
    PROCESSED_DIR /
    "deterioration_test.csv"
)

STATE_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_alerts.csv"
)

print("Loading predictions...")
pred = pd.read_csv(
    PREDICTION_FILE,
    low_memory=False
)

print("Loading deterioration test data...")
test = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print("Loading patient state data...")
state = pd.read_csv(
    STATE_FILE,
    low_memory=False
)

# ------------------------------------------------------------
# BUILD TIME-TO-DISCHARGE FROM ORIGINAL STATE
# ------------------------------------------------------------

time_info = state[
    [
        "patientunitstayid",
        "hour",
        "unitdischargeoffset"
    ]
].copy()

time_info = time_info.drop_duplicates(
    subset=[
        "patientunitstayid",
        "hour"
    ]
)

time_info["unitdischargeoffset"] = pd.to_numeric(
    time_info["unitdischargeoffset"],
    errors="coerce"
)

time_info["hour"] = pd.to_numeric(
    time_info["hour"],
    errors="coerce"
)

time_info["hours_until_discharge"] = (
    time_info["unitdischargeoffset"] / 60.0
    - time_info["hour"]
)

# ------------------------------------------------------------
# MERGE
# ------------------------------------------------------------

df = pred.merge(
    test[
        [
            "patientunitstayid",
            "hour",
            "deterioration_label_24h"
        ]
    ],
    on=[
        "patientunitstayid",
        "hour"
    ],
    how="inner",
    validate="one_to_one"
)

df = df.merge(
    time_info[
        [
            "patientunitstayid",
            "hour",
            "hours_until_discharge"
        ]
    ],
    on=[
        "patientunitstayid",
        "hour"
    ],
    how="left",
    validate="one_to_one"
)

# ------------------------------------------------------------
# TYPES
# ------------------------------------------------------------

df["risk_score"] = pd.to_numeric(
    df["risk_probability"],
    errors="coerce"
)

df["actual"] = pd.to_numeric(
    df["deterioration_label_24h"],
    errors="coerce"
).astype(int)

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n============================================================")
print("MERGE VALIDATION")
print("============================================================")

print(
    "Evaluation rows:",
    len(df)
)

print(
    "Missing hours_until_discharge:",
    df["hours_until_discharge"].isna().sum()
)

print(
    "Duplicate patient-hours:",
    df.duplicated(
        subset=[
            "patientunitstayid",
            "hour"
        ]
    ).sum()
)

if df["hours_until_discharge"].isna().any():
    raise ValueError(
        "Missing hours_until_discharge after merge."
    )

# ------------------------------------------------------------
# PATIENT-LEVEL OUTCOME
# ------------------------------------------------------------

patient_actual = (
    df
    .groupby("patientunitstayid")
    .agg(
        actual_event=(
            "actual",
            "max"
        ),
        max_risk=(
            "risk_score",
            "max"
        ),
        mean_risk=(
            "risk_score",
            "mean"
        )
    )
    .reset_index()
)

print("\n============================================================")
print("PATIENT-LEVEL DATASET")
print("============================================================")

print(
    "Patients:",
    len(patient_actual)
)

print(
    "Positive patients:",
    int(
        patient_actual["actual_event"].sum()
    )
)

print(
    "Negative patients:",
    int(
        (patient_actual["actual_event"] == 0).sum()
    )
)

# ------------------------------------------------------------
# PATIENT-LEVEL ROC / PR
# ------------------------------------------------------------

patient_roc = roc_auc_score(
    patient_actual["actual_event"],
    patient_actual["max_risk"]
)

patient_pr = average_precision_score(
    patient_actual["actual_event"],
    patient_actual["max_risk"]
)

patient_baseline = (
    patient_actual["actual_event"].mean()
)

print("\n============================================================")
print("PATIENT-LEVEL RANKING")
print("============================================================")

print(
    "ROC-AUC:",
    round(patient_roc, 4)
)

print(
    "PR-AUC:",
    round(patient_pr, 4)
)

print(
    "Baseline PR-AUC:",
    round(patient_baseline, 4)
)

# ------------------------------------------------------------
# THRESHOLD ANALYSIS
# ------------------------------------------------------------

print("\n============================================================")
print("PATIENT-LEVEL THRESHOLD ANALYSIS")
print("============================================================")

thresholds = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

for threshold in thresholds:

    predicted = (
        patient_actual["max_risk"]
        >= threshold
    ).astype(int)

    precision = precision_score(
        patient_actual["actual_event"],
        predicted,
        zero_division=0
    )

    recall = recall_score(
        patient_actual["actual_event"],
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        patient_actual["actual_event"],
        predicted,
        zero_division=0
    )

    tp = int(
        (
            (predicted == 1) &
            (patient_actual["actual_event"] == 1)
        ).sum()
    )

    fp = int(
        (
            (predicted == 1) &
            (patient_actual["actual_event"] == 0)
        ).sum()
    )

    print(
        f"\nThreshold {threshold:.2f}"
    )

    print(
        "Predicted positive patients:",
        int(predicted.sum())
    )

    print(
        "True positives:",
        tp
    )

    print(
        "False positives:",
        fp
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall:",
        round(recall, 4)
    )

    print(
        "F1:",
        round(f1, 4)
    )

# ------------------------------------------------------------
# EARLY WARNING ANALYSIS
# ------------------------------------------------------------

print("\n============================================================")
print("EARLY WARNING ANALYSIS")
print("============================================================")

positive_patients = patient_actual.loc[
    patient_actual["actual_event"] == 1,
    "patientunitstayid"
].tolist()

for threshold in [
    0.50,
    0.70,
    0.90
]:

    candidates = df[
        (df["patientunitstayid"].isin(
            positive_patients
        ))
        &
        (df["risk_score"] >= threshold)
        &
        (df["hours_until_discharge"] >= 0)
    ].copy()

    if len(candidates) == 0:

        print(
            f"\nThreshold {threshold:.2f}: no predictions"
        )

        continue

    earliest_detection = (
        candidates
        .groupby(
            "patientunitstayid"
        )["hours_until_discharge"]
        .max()
    )

    print(
        f"\nThreshold {threshold:.2f}"
    )

    print(
        "Positive patients detected:",
        len(earliest_detection)
    )

    print(
        "Median lead time:",
        round(
            earliest_detection.median(),
            2
        ),
        "hours"
    )

    print(
        "25th percentile lead time:",
        round(
            earliest_detection.quantile(0.25),
            2
        ),
        "hours"
    )

    print(
        "75th percentile lead time:",
        round(
            earliest_detection.quantile(0.75),
            2
        ),
        "hours"
    )

# ------------------------------------------------------------
# TOP PATIENTS
# ------------------------------------------------------------

print("\n============================================================")
print("TOP 20 PATIENTS BY MAX RISK")
print("============================================================")

print(
    patient_actual
    .sort_values(
        "max_risk",
        ascending=False
    )
    .head(20)
    .round(4)
    .to_string(index=False)
)

print("\nEvaluation complete.")