import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
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

pred = pd.read_csv(
    PREDICTION_FILE,
    low_memory=False
)

test = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

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

y = df["deterioration_label_24h"].astype(int)

risk = pd.to_numeric(
    df["risk_probability"],
    errors="coerce"
)

print("\n============================================================")
print("DETERIORATION RISK BANDS")
print("============================================================")

bands = [
    ("<0.50", 0.0, 0.50),
    ("0.50-0.70", 0.50, 0.70),
    ("0.70-0.90", 0.70, 0.90),
    (">=0.90", 0.90, 1.01)
]

for name, low, high in bands:

    mask = (
        (risk >= low) &
        (risk < high)
    )

    count = int(mask.sum())
    positives = int(y[mask].sum())

    rate = (
        positives / count * 100
        if count > 0
        else 0
    )

    print(
        f"\n{name}"
    )

    print(
        "Rows:",
        count
    )

    print(
        "Positive events:",
        positives
    )

    print(
        "Positive rate:",
        round(rate, 2),
        "%"
    )

# ------------------------------------------------------------
# THRESHOLD METRICS
# ------------------------------------------------------------

print("\n============================================================")
print("HOURLY THRESHOLD PERFORMANCE")
print("============================================================")

thresholds = [
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
        risk >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        y,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predicted,
        zero_division=0
    )

    print(
        f"\nThreshold {threshold:.2f}"
    )

    print(
        "Flagged rows:",
        int(predicted.sum())
    )

    print(
        "True positives:",
        int(
            (
                (predicted == 1) &
                (y == 1)
            ).sum()
        )
    )

    print(
        "False positives:",
        int(
            (
                (predicted == 1) &
                (y == 0)
            ).sum()
        )
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

print("\nEvaluation complete.")