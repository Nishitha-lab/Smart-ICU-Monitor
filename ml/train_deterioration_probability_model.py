import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

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
    "deterioration_probability_predictions.csv"
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

test = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print(
    "Train shape:",
    train.shape
)

print(
    "Test shape:",
    test.shape
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
# PROBABILITY-ORIENTED MODEL
# ------------------------------------------------------------
#
# No class_weight="balanced".
#
# This preserves the observed training prevalence
# rather than artificially reweighting positive cases.
#

print("\n============================================================")
print("TRAINING PROBABILITY MODEL")
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

MODEL_DIR = ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump({"pipeline": model, "feature_columns": feature_columns, "target": TARGET, "model_type": "LogisticRegression"}, MODEL_DIR / "deterioration_logistic_v1.joblib")
print("Saved model:", MODEL_DIR / "deterioration_logistic_v1.joblib")

print("Training complete.")

# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

probabilities = model.predict_proba(
    X_test
)[:, 1]

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

brier = brier_score_loss(
    y_test,
    probabilities
)

baseline_pr = y_test.mean()

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n============================================================")
print("PROBABILITY MODEL RESULTS")
print("============================================================")

print(
    "ROC-AUC:",
    round(
        roc_auc,
        4
    )
)

print(
    "PR-AUC:",
    round(
        pr_auc,
        4
    )
)

print(
    "Baseline PR-AUC:",
    round(
        baseline_pr,
        4
    )
)

print(
    "Brier score:",
    round(
        brier,
        6
    )
)

print("\nProbability distribution:")

print(
    pd.Series(probabilities)
    .describe()
    .round(6)
    .to_string()
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
results["risk_probability"] = probabilities

print("\n============================================================")
print("PROBABILITY BANDS")
print("============================================================")

bands = [
    ("<0.01", 0.0, 0.01),
    ("0.01-0.02", 0.01, 0.02),
    ("0.02-0.05", 0.02, 0.05),
    ("0.05-0.10", 0.05, 0.10),
    ("0.10-0.25", 0.10, 0.25),
    ("0.25-0.50", 0.25, 0.50),
    (">=0.50", 0.50, 1.01)
]

for name, low, high in bands:

    mask = (
        (results["risk_probability"] >= low)
        &
        (results["risk_probability"] < high)
    )

    rows = int(mask.sum())

    positives = int(
        results.loc[
            mask,
            "actual"
        ].sum()
    )

    observed_rate = (
        positives /
        rows *
        100
        if rows > 0
        else 0
    )

    mean_probability = (
        results.loc[
            mask,
            "risk_probability"
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
        "Mean predicted probability:",
        round(
            mean_probability,
            4
        )
    )

# ------------------------------------------------------------
# HIGH-RISK THRESHOLDS
# ------------------------------------------------------------

print("\n============================================================")
print("PROBABILITY THRESHOLDS")
print("============================================================")

for threshold in [
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.50
]:

    flagged = (
        probabilities >= threshold
    )

    flagged_count = int(
        flagged.sum()
    )

    true_positive = int(
        (
            flagged &
            (y_test.values == 1)
        ).sum()
    )

    false_positive = int(
        (
            flagged &
            (y_test.values == 0)
        ).sum()
    )

    precision = (
        true_positive /
        flagged_count
        if flagged_count > 0
        else 0
    )

    recall = (
        true_positive /
        y_test.sum()
        if y_test.sum() > 0
        else 0
    )

    print(
        f"\nThreshold {threshold:.2f}"
    )

    print(
        "Flagged rows:",
        flagged_count
    )

    print(
        "True positives:",
        true_positive
    )

    print(
        "False positives:",
        false_positive
    )

    print(
        "Precision:",
        round(
            precision,
            4
        )
    )

    print(
        "Recall:",
        round(
            recall,
            4
        )
    )

# ------------------------------------------------------------
# TOP RISK ROWS
# ------------------------------------------------------------

print("\n============================================================")
print("TOP 20 RISK HOURS")
print("============================================================")

print(
    results
    .sort_values(
        "risk_probability",
        ascending=False
    )
    .head(20)
    .round(6)
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