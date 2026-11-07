import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DIR /
    "deterioration_features.csv"
)

TRAIN_FILE = (
    PROCESSED_DIR /
    "deterioration_train.csv"
)

TEST_FILE = (
    PROCESSED_DIR /
    "deterioration_test.csv"
)

print("Loading deterioration features...")

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
# PATIENT-WISE SPLIT
# ------------------------------------------------------------

patients = (
    df["patientunitstayid"]
    .unique()
)

train_patients, test_patients = train_test_split(
    patients,
    test_size=0.20,
    random_state=42
)

train_df = df[
    df["patientunitstayid"]
    .isin(train_patients)
].copy()

test_df = df[
    df["patientunitstayid"]
    .isin(test_patients)
].copy()

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n============================================================")
print("PATIENT-WISE SPLIT")
print("============================================================")

print(
    "Total ICU stays:",
    len(patients)
)

print(
    "Train ICU stays:",
    len(train_patients)
)

print(
    "Test ICU stays:",
    len(test_patients)
)

print(
    "\nTrain rows:",
    len(train_df)
)

print(
    "Test rows:",
    len(test_df)
)

overlap = (
    set(train_patients) &
    set(test_patients)
)

print(
    "\nPatient overlap:",
    len(overlap)
)

if len(overlap) != 0:
    raise ValueError(
        "ERROR: Patient leakage detected!"
    )

# ------------------------------------------------------------
# LABEL COUNTS
# ------------------------------------------------------------

print("\n============================================================")
print("TRAIN LABEL DISTRIBUTION")
print("============================================================")

print(
    train_df[
        "deterioration_label_24h"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n============================================================")
print("TEST LABEL DISTRIBUTION")
print("============================================================")

print(
    test_df[
        "deterioration_label_24h"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nPositive train patients:")

print(
    train_df.loc[
        train_df["deterioration_label_24h"] == 1,
        "patientunitstayid"
    ].nunique()
)

print(
    "\nPositive test patients:"
)

print(
    test_df.loc[
        test_df["deterioration_label_24h"] == 1,
        "patientunitstayid"
    ].nunique()
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)

print("\n============================================================")
print("SUCCESS")
print("============================================================")

print(
    "Train:",
    TRAIN_FILE
)

print(
    "Test :",
    TEST_FILE
)