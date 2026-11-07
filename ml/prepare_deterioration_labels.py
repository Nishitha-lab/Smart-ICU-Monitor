import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_alerts.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "deterioration_labels.csv"
)

HORIZON = 24

print("Loading patient state data...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Rows:", len(df))
print(
    "ICU stays:",
    df["patientunitstayid"].nunique()
)

# ------------------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "patientunitstayid",
    "hour",
    "unitdischargeoffset",
    "unitdischargestatus"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# ------------------------------------------------------------
# NUMERIC CONVERSION
# ------------------------------------------------------------

df["hour"] = pd.to_numeric(
    df["hour"],
    errors="coerce"
)

df["unitdischargeoffset"] = pd.to_numeric(
    df["unitdischargeoffset"],
    errors="coerce"
)

# ------------------------------------------------------------
# ICU OUTCOME
# ------------------------------------------------------------

df["death_outcome"] = (
    df["unitdischargestatus"]
    .astype(str)
    .str.lower()
    .eq("expired")
)

stay_outcomes = (
    df[
        [
            "patientunitstayid",
            "death_outcome"
        ]
    ]
    .drop_duplicates(
        subset=["patientunitstayid"]
    )
)

print("\n============================================================")
print("ICU OUTCOME")
print("============================================================")

print(
    "Expired ICU stays:",
    int(
        stay_outcomes["death_outcome"].sum()
    )
)

print(
    "Non-expired ICU stays:",
    int(
        (~stay_outcomes["death_outcome"]).sum()
    )
)

# ------------------------------------------------------------
# TIME UNTIL ICU DISCHARGE
# ------------------------------------------------------------

df["hours_until_discharge"] = (
    df["unitdischargeoffset"] / 60.0
    - df["hour"]
)

# ------------------------------------------------------------
# 24-HOUR FUTURE DEATH LABEL
# ------------------------------------------------------------
#
# Positive:
#   Patient eventually expires AND the expiration/discharge
#   occurs within the next 24 ICU hours.
#
# Negative:
#   All other prediction hours.
#
# This produces a prospective prediction target:
#
#       information available at hour t
#                    ↓
#       predict death within next 24h
#
# No future clinical features are used here.
#

df["deterioration_label_24h"] = (
    df["death_outcome"]
    &
    (df["hours_until_discharge"] >= 0)
    &
    (df["hours_until_discharge"] <= HORIZON)
).astype(int)

# ------------------------------------------------------------
# LABEL SUMMARY
# ------------------------------------------------------------

print("\n============================================================")
print("24-HOUR DETERIORATION LABEL")
print("============================================================")

positive = int(
    (
        df["deterioration_label_24h"] == 1
    ).sum()
)

negative = int(
    (
        df["deterioration_label_24h"] == 0
    ).sum()
)

print(
    "Positive rows:",
    positive
)

print(
    "Negative rows:",
    negative
)

total = positive + negative

if total > 0:
    print(
        "Positive rate:",
        round(
            positive / total * 100,
            3
        ),
        "%"
    )

# ------------------------------------------------------------
# POSITIVE PATIENTS
# ------------------------------------------------------------

positive_patients = (
    df.loc[
        df["deterioration_label_24h"] == 1,
        "patientunitstayid"
    ]
    .nunique()
)

print(
    "Patients with positive labels:",
    positive_patients
)

# ------------------------------------------------------------
# POSITIVE ROWS PER EXPIRED PATIENT
# ------------------------------------------------------------

positive_per_patient = (
    df.loc[
        df["deterioration_label_24h"] == 1
    ]
    .groupby(
        "patientunitstayid"
    )
    .size()
)

print("\n============================================================")
print("POSITIVE LABELS PER EXPIRED PATIENT")
print("============================================================")

if len(positive_per_patient) > 0:

    print(
        positive_per_patient
        .describe()
        .round(2)
        .to_string()
    )

else:

    print("No positive patients found.")

# ------------------------------------------------------------
# LABEL DISTRIBUTION BY TIME TO DISCHARGE
# ------------------------------------------------------------

print("\n============================================================")
print("LABEL DISTRIBUTION BY TIME TO DISCHARGE")
print("============================================================")

bins = [
    -0.001,
    6,
    12,
    18,
    24
]

labels = [
    "0-6h",
    "6-12h",
    "12-18h",
    "18-24h"
]

expired_24h = df[
    (df["death_outcome"]) &
    (df["hours_until_discharge"] >= 0) &
    (df["hours_until_discharge"] <= 24)
].copy()

expired_24h["time_to_discharge_group"] = pd.cut(
    expired_24h["hours_until_discharge"],
    bins=bins,
    labels=labels
)

if len(expired_24h) > 0:

    time_summary = (
        expired_24h
        .groupby(
            "time_to_discharge_group",
            observed=False
        )
        .agg(
            rows=(
                "deterioration_label_24h",
                "size"
            ),
            positives=(
                "deterioration_label_24h",
                "sum"
            )
        )
    )

    time_summary["positive_rate"] = (
        time_summary["positives"] /
        time_summary["rows"] *
        100
    )

    print(
        time_summary
        .round(2)
        .to_string()
    )

else:

    print(
        "No expired rows in the 24-hour prediction window."
    )

# ------------------------------------------------------------
# CHECK NEGATIVE EXAMPLES
# ------------------------------------------------------------

print("\n============================================================")
print("NEGATIVE-LABEL CHECK")
print("============================================================")

negative_expired_far_from_discharge = (
    df[
        df["death_outcome"] &
        (df["hours_until_discharge"] > 24)
    ]
)

print(
    "Expired patients >24h from discharge:",
    len(negative_expired_far_from_discharge)
)

if len(negative_expired_far_from_discharge) > 0:

    print(
        "All correctly labelled as 0:",
        bool(
            (
                negative_expired_far_from_discharge[
                    "deterioration_label_24h"
                ] == 0
            ).all()
        )
    )

# ------------------------------------------------------------
# CHECK FOR IMPOSSIBLE VALUES
# ------------------------------------------------------------

print("\n============================================================")
print("VALIDATION")
print("============================================================")

print(
    "Missing hours:",
    df["hour"].isna().sum()
)

print(
    "Missing discharge offsets:",
    df["unitdischargeoffset"].isna().sum()
)

print(
    "Negative time-to-discharge:",
    (
        df["hours_until_discharge"] < 0
    ).sum()
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

label_output = df[
    [
        "patientunitstayid",
        "hour",
        "deterioration_label_24h",
        "hours_until_discharge"
    ]
].copy()

label_output.to_csv(
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