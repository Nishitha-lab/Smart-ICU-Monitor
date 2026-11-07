import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

TRAIN_FILE = (
    PROCESSED_DIR /
    "deterioration_train.csv"
)

TEST_FILE = (
    PROCESSED_DIR /
    "deterioration_test.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "deterioration_predictions_calibrated.csv"
)

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print("Loading training data...")
train = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

print("Loading test data...")
test = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

# ------------------------------------------------------------
# PATIENT-WISE TRAIN / CALIBRATION SPLIT
# ------------------------------------------------------------

patients = train[
    "patientunitstayid"
].unique()

model_patients, calibration_patients = train_test_split(
    patients,
    test_size=0.20,
    random_state=123
)

model_train = train[
    train["patientunitstayid"].isin(
        model_patients
    )
].copy()

calibration = train[
    train["patientunitstayid"].isin(
        calibration_patients
    )
].copy()

print("\n============================================================")
print("PATIENT-WISE CALIBRATION SPLIT")
print("============================================================")

print(
    "Model-training patients:",
    len(model_patients)
)

print(
    "Calibration patients:",
    len(calibration_patients)
)

print(
    "Model-training rows:",
    len(model_train)
)

print(
    "Calibration rows:",
    len(calibration)
)

overlap = (
    set(model_patients) &
    set(calibration_patients)
)

print(
    "Patient overlap:",
    len(overlap)
)

if overlap:
    raise ValueError(
        "Patient leakage detected."
    )

# ------------------------------------------------------------
# PREPARE
# ------------------------------------------------------------

X_model = model_train[
    feature_columns
].copy()

y_model = model_train[
    TARGET
].astype(int)

X_cal = calibration[
    feature_columns
].copy()

y_cal = calibration[
    TARGET
].astype(int)

X_test = test[
    feature_columns
].copy()

y_test = test[
    TARGET
].astype(int)

for col in feature_columns:

    X_model[col] = pd.to_numeric(
        X_model[col],
        errors="coerce"
    )

    X_cal[col] = pd.to_numeric(
        X_cal[col],
        errors="coerce"
    )

    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce"
    )

# ------------------------------------------------------------
# BASE MODEL
# ------------------------------------------------------------

print("\n============================================================")
print("TRAINING BASE MODEL")
print("============================================================")

base_model = Pipeline(
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
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            )
        )
    ]
)

base_model.fit(
    X_model,
    y_model
)

print("Base model trained.")

# ------------------------------------------------------------
# RAW CALIBRATION SCORES
# ------------------------------------------------------------

calibration_raw = base_model.predict_proba(
    X_cal
)[:, 1]

test_raw = base_model.predict_proba(
    X_test
)[:, 1]

# ------------------------------------------------------------
# ISOTONIC CALIBRATION
# ------------------------------------------------------------

print("\n============================================================")
print("FITTING ISOTONIC CALIBRATION")
print("============================================================")

calibrator = IsotonicRegression(
    y_min=0.0,
    y_max=1.0,
    out_of_bounds="clip"
)

calibrator.fit(
    calibration_raw,
    y_cal
)

calibrated_test = calibrator.predict(
    test_raw
)

print("Calibration complete.")

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

raw_roc = roc_auc_score(
    y_test,
    test_raw
)

calibrated_roc = roc_auc_score(
    y_test,
    calibrated_test
)

raw_pr = average_precision_score(
    y_test,
    test_raw
)

calibrated_pr = average_precision_score(
    y_test,
    calibrated_test
)

raw_brier = brier_score_loss(
    y_test,
    test_raw
)

calibrated_brier = brier_score_loss(
    y_test,
    calibrated_test
)

print("\n============================================================")
print("CALIBRATION RESULTS")
print("============================================================")

print(
    "Raw ROC-AUC:",
    round(raw_roc, 4)
)

print(
    "Calibrated ROC-AUC:",
    round(calibrated_roc, 4)
)

print(
    "Raw PR-AUC:",
    round(raw_pr, 4)
)

print(
    "Calibrated PR-AUC:",
    round(calibrated_pr, 4)
)

print(
    "Raw Brier score:",
    round(raw_brier, 6)
)

print(
    "Calibrated Brier score:",
    round(calibrated_brier, 6)
)

# ------------------------------------------------------------
# RISK BANDS
# ------------------------------------------------------------

results = test[
    [
        "patientunitstayid",
        "hour"
    ]
].copy()

results["actual"] = y_test.values
results["raw_risk_score"] = test_raw
results["calibrated_risk"] = calibrated_test

print("\n============================================================")
print("CALIBRATED RISK BANDS")
print("============================================================")

bands = [
    ("<0.10", 0.0, 0.10),
    ("0.10-0.25", 0.10, 0.25),
    ("0.25-0.50", 0.25, 0.50),
    ("0.50-0.75", 0.50, 0.75),
    ("0.75-0.90", 0.75, 0.90),
    (">=0.90", 0.90, 1.01)
]

for name, low, high in bands:

    mask = (
        (results["calibrated_risk"] >= low)
        &
        (results["calibrated_risk"] < high)
    )

    rows = int(mask.sum())

    positives = int(
        results.loc[
            mask,
            "actual"
        ].sum()
    )

    observed_rate = (
        positives / rows * 100
        if rows > 0
        else 0
    )

    mean_predicted = (
        results.loc[
            mask,
            "calibrated_risk"
        ].mean()
        if rows > 0
        else 0
    )

    print(
        f"\n{name}"
    )

    print(
        "Rows:",
        rows
    )

    print(
        "Positive events:",
        positives
    )

    print(
        "Observed positive rate:",
        round(
            observed_rate,
            2
        ),
        "%"
    )

    print(
        "Mean calibrated score:",
        round(
            mean_predicted,
            4
        )
    )

# ------------------------------------------------------------
# PATIENT-LEVEL CALIBRATED RISK
# ------------------------------------------------------------

patient = (
    results
    .groupby("patientunitstayid")
    .agg(
        actual_event=(
            "actual",
            "max"
        ),
        max_raw_risk=(
            "raw_risk_score",
            "max"
        ),
        max_calibrated_risk=(
            "calibrated_risk",
            "max"
        ),
        mean_calibrated_risk=(
            "calibrated_risk",
            "mean"
        )
    )
    .reset_index()
)

print("\n============================================================")
print("PATIENT-LEVEL CALIBRATED RISK")
print("============================================================")

print(
    "Patients:",
    len(patient)
)

print(
    "Positive patients:",
    int(
        patient["actual_event"].sum()
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            patient["actual_event"],
            patient["max_calibrated_risk"]
        ),
        4
    )
)

print(
    "PR-AUC:",
    round(
        average_precision_score(
            patient["actual_event"],
            patient["max_calibrated_risk"]
        ),
        4
    )
)

print("\nTop 20 patients:")

print(
    patient
    .sort_values(
        "max_calibrated_risk",
        ascending=False
    )
    .head(20)
    .round(4)
    .to_string(index=False)
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

results.to_csv(
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