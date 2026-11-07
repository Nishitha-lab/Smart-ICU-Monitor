import pandas as pd
import numpy as np
from pathlib import Path

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

print("Loading prediction data...")

pred = pd.read_csv(
    PREDICTION_FILE,
    low_memory=False
)

print("Loading test labels...")

test = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print("Loading patient state...")

state = pd.read_csv(
    STATE_FILE,
    low_memory=False
)

# ------------------------------------------------------------
# GET TIME UNTIL DISCHARGE
# ------------------------------------------------------------

time_info = state[
    [
        "patientunitstayid",
        "hour",
        "unitdischargeoffset"
    ]
].drop_duplicates(
    subset=[
        "patientunitstayid",
        "hour"
    ]
).copy()

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

df["risk_score"] = pd.to_numeric(
    df["risk_probability"],
    errors="coerce"
)

df["actual"] = pd.to_numeric(
    df["deterioration_label_24h"],
    errors="coerce"
).astype(int)

# ------------------------------------------------------------
# POSITIVE PATIENTS
# ------------------------------------------------------------

positive_patients = (
    df.loc[
        df["actual"] == 1,
        "patientunitstayid"
    ]
    .unique()
)

print("\n============================================================")
print("EARLY WARNING EVALUATION")
print("============================================================")

print(
    "Positive patients:",
    len(positive_patients)
)

# ------------------------------------------------------------
# EVALUATE FIXED HORIZONS
# ------------------------------------------------------------

horizons = [
    24,
    12,
    6
]

thresholds = [
    0.50,
    0.70,
    0.90
]

for horizon in horizons:

    print("\n============================================================")
    print(
        f"DETECTION AT LEAST {horizon} HOURS BEFORE DISCHARGE"
    )
    print("============================================================")

    positive = df[
        (
            df["patientunitstayid"]
            .isin(positive_patients)
        )
        &
        (
            df["hours_until_discharge"]
            >= horizon
        )
    ].copy()

    for threshold in thresholds:

        detected_patients = (
            positive.loc[
                positive["risk_score"]
                >= threshold,
                "patientunitstayid"
            ]
            .nunique()
        )

        total_positive = len(
            positive_patients
        )

        recall = (
            detected_patients /
            total_positive *
            100
            if total_positive > 0
            else 0
        )

        print(
            f"Threshold {threshold:.2f}: "
            f"{detected_patients}/{total_positive} "
            f"patients detected "
            f"({recall:.2f}%)"
        )

# ------------------------------------------------------------
# FALSE POSITIVE BURDEN AT FIXED HORIZONS
# ------------------------------------------------------------

print("\n============================================================")
print("RISK SCORE DISTRIBUTION")
print("============================================================")

for horizon in horizons:

    eligible = df[
        df["hours_until_discharge"]
        >= horizon
    ].copy()

    if len(eligible) == 0:
        continue

    print(
        f"\nAt least {horizon}h before discharge"
    )

    print(
        eligible["risk_score"]
        .describe()
        .round(4)
        .to_string()
    )

# ------------------------------------------------------------
# POSITIVE PATIENT DETAIL
# ------------------------------------------------------------

print("\n============================================================")
print("POSITIVE PATIENT EARLIEST HIGH-RISK DETECTION")
print("============================================================")

rows = []

for patient in positive_patients:

    patient_df = df[
        df["patientunitstayid"]
        == patient
    ].copy()

    record = {
        "patientunitstayid": patient
    }

    for threshold in thresholds:

        candidates = patient_df[
            patient_df["risk_score"]
            >= threshold
        ]

        if len(candidates) == 0:

            record[
                f"lead_time_{threshold}"
            ] = np.nan

        else:

            record[
                f"lead_time_{threshold}"
            ] = candidates[
                "hours_until_discharge"
            ].max()

    rows.append(record)

lead_time_df = pd.DataFrame(rows)

print(
    lead_time_df
    .sort_values(
        "lead_time_0.9",
        ascending=False
    )
    .round(2)
    .to_string(index=False)
)

print("\nEvaluation complete.")