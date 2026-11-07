import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
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
    "deterioration_predictions.csv"
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

    # Short-term changes
    "heart_rate_change_1h",
    "spo2_change_1h",
    "respiration_change_1h",
    "systolic_bp_change_1h",
    "diastolic_bp_change_1h",
    "mean_bp_change_1h",

    # Recent state
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

    # Rule-derived counts
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

test = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print("Train shape:", train.shape)
print("Test shape :", test.shape)

# ------------------------------------------------------------
# VALIDATE
# ------------------------------------------------------------

missing_features = [
    col
    for col in feature_columns
    if col not in train.columns
]

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )

if TARGET not in train.columns:
    raise ValueError(
        f"Missing target: {TARGET}"
    )

# ------------------------------------------------------------
# PREPARE
# ------------------------------------------------------------

X_train = train[
    feature_columns
].copy()

X_test = test[
    feature_columns
].copy()

y_train = train[
    TARGET
].astype(int)

y_test = test[
    TARGET
].astype(int)

for col in feature_columns:

    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )

    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce"
    )

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

print("\n============================================================")
print("BUILDING LOGISTIC REGRESSION")
print("============================================================")

pipeline = Pipeline(
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

print("Training model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")

# ------------------------------------------------------------
# PREDICT PROBABILITIES
# ------------------------------------------------------------

print("\nScoring test data...")

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]

# Default threshold
threshold = 0.50

predictions = (
    probabilities >= threshold
).astype(int)

# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

pr_auc = average_precision_score(
    y_test,
    probabilities
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    predictions
)

# ------------------------------------------------------------
# PRINT METRICS
# ------------------------------------------------------------

print("\n============================================================")
print("DETERIORATION MODEL RESULTS")
print("============================================================")

print(
    "Test rows:",
    len(test)
)

print(
    "Positive test rows:",
    int(y_test.sum())
)

print(
    "ROC-AUC:",
    round(roc_auc, 4)
)

print(
    "PR-AUC:",
    round(pr_auc, 4)
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

print("\nConfusion matrix:")

print(
    cm
)

# ------------------------------------------------------------
# BASELINE PR-AUC
# ------------------------------------------------------------

baseline_pr = y_test.mean()

print(
    "\nBaseline PR-AUC (positive prevalence):",
    round(
        baseline_pr,
        4
    )
)

# ------------------------------------------------------------
# PROBABILITY DISTRIBUTION
# ------------------------------------------------------------

print("\n============================================================")
print("RISK PROBABILITY DISTRIBUTION")
print("============================================================")

print(
    pd.Series(probabilities)
    .describe()
    .round(4)
    .to_string()
)

# ------------------------------------------------------------
# PATIENT-WISE RESULTS
# ------------------------------------------------------------

results = test[
    [
        "patientunitstayid",
        "hour"
    ]
].copy()

results["actual"] = y_test.values
results["risk_probability"] = probabilities
results["predicted"] = predictions

# ------------------------------------------------------------
# TOP RISK CASES
# ------------------------------------------------------------

print("\n============================================================")
print("TOP 20 PREDICTED-RISK HOURS")
print("============================================================")

top = (
    results
    .sort_values(
        "risk_probability",
        ascending=False
    )
    .head(20)
)

print(
    top.to_string(
        index=False
    )
)

# ------------------------------------------------------------
# POSITIVE CAPTURE
# ------------------------------------------------------------

print("\n============================================================")
print("POSITIVE EVENT CAPTURE")
print("============================================================")

actual_positive = (
    results["actual"] == 1
)

top_1pct_n = max(
    1,
    int(
        len(results) *
        0.01
    )
)

top_5pct_n = max(
    1,
    int(
        len(results) *
        0.05
    )
)

for name, n in [
    ("Top 1%", top_1pct_n),
    ("Top 5%", top_5pct_n)
]:

    top_rows = (
        results
        .nlargest(
            n,
            "risk_probability"
        )
    )

    captured = (
        top_rows["actual"]
        .sum()
    )

    total_positive = (
        actual_positive.sum()
    )

    capture_rate = (
        captured /
        total_positive *
        100
        if total_positive > 0
        else 0
    )

    print(
        f"{name}:",
        int(captured),
        "/",
        int(total_positive),
        "positive events captured =",
        round(
            capture_rate,
            2
        ),
        "%"
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