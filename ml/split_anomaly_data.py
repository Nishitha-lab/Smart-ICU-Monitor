import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "anomaly_features.csv"

TRAIN_FILE = PROCESSED_DIR / "anomaly_train.csv"
TEST_FILE = PROCESSED_DIR / "anomaly_test.csv"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42
TEST_FRACTION = 0.20


# ============================================================
# LOAD
# ============================================================

print("Loading anomaly features...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Rows:", len(df))
print("ICU stays:", df["patientunitstayid"].nunique())


# ============================================================
# GET UNIQUE ICU STAYS
# ============================================================

patient_ids = (
    df["patientunitstayid"]
    .dropna()
    .unique()
)

print(
    "Unique ICU stays available:",
    len(patient_ids)
)


# ============================================================
# SHUFFLE PATIENTS
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

patient_ids = rng.permutation(patient_ids)


# ============================================================
# SPLIT PATIENTS
# ============================================================

n_test = int(
    len(patient_ids) * TEST_FRACTION
)

test_patients = set(
    patient_ids[:n_test]
)

train_patients = set(
    patient_ids[n_test:]
)


# ============================================================
# CREATE DATASETS
# ============================================================

train_df = df[
    df["patientunitstayid"].isin(train_patients)
].copy()

test_df = df[
    df["patientunitstayid"].isin(test_patients)
].copy()


# ============================================================
# VALIDATION
# ============================================================

print("\n============================================================")
print("SPLIT VALIDATION")
print("============================================================")

print(
    "Training ICU stays:",
    train_df["patientunitstayid"].nunique()
)

print(
    "Testing ICU stays:",
    test_df["patientunitstayid"].nunique()
)

print(
    "Training rows:",
    len(train_df)
)

print(
    "Testing rows:",
    len(test_df)
)

print(
    "Total rows:",
    len(train_df) + len(test_df)
)


# ============================================================
# CHECK PATIENT LEAKAGE
# ============================================================

overlap = (
    set(train_df["patientunitstayid"])
    &
    set(test_df["patientunitstayid"])
)

print(
    "Patient overlap:",
    len(overlap)
)


if len(overlap) > 0:
    raise ValueError(
        "Patient leakage detected!"
    )


if len(train_df) + len(test_df) != len(df):
    raise ValueError(
        "Train + test rows do not equal original rows."
    )


# ============================================================
# SORT
# ============================================================

train_df = train_df.sort_values(
    ["patientunitstayid", "hour"]
)

test_df = test_df.sort_values(
    ["patientunitstayid", "hour"]
)


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n============================================================")
print("SUCCESS")
print("============================================================")

print(
    "Training file:",
    TRAIN_FILE
)

print(
    "Testing file:",
    TEST_FILE
)