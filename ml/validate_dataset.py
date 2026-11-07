import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "hourly_vitals_labs.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 1. BASIC INFORMATION
# ============================================================

print("\n============================================")
print("1. BASIC DATASET INFORMATION")
print("============================================")

print(f"Unique ICU stays: {df['patientunitstayid'].nunique():,}")

print(
    f"Unique hours: "
    f"{df[['patientunitstayid', 'hour']].drop_duplicates().shape[0]:,}"
)


# ============================================================
# 2. DUPLICATE PATIENT-HOUR CHECK
# ============================================================

print("\n============================================")
print("2. DUPLICATE PATIENT-HOUR CHECK")
print("============================================")

duplicates = df.duplicated(
    subset=["patientunitstayid", "hour"]
).sum()

print(f"Duplicate patient-hour rows: {duplicates:,}")


# ============================================================
# 3. HOUR RANGE CHECK
# ============================================================

print("\n============================================")
print("3. HOUR RANGE CHECK")
print("============================================")

print(f"Minimum hour: {df['hour'].min()}")
print(f"Maximum hour: {df['hour'].max()}")

negative_hours = (df["hour"] < 0).sum()

print(f"Negative hours: {negative_hours:,}")


# ============================================================
# 4. HOURS PER ICU STAY
# ============================================================

print("\n============================================")
print("4. HOURS PER ICU STAY")
print("============================================")

hours_per_patient = (
    df.groupby("patientunitstayid")["hour"]
    .nunique()
)

print(hours_per_patient.describe())


# ============================================================
# 5. VITAL SIGN RANGES
# ============================================================

print("\n============================================")
print("5. VITAL SIGN RANGES")
print("============================================")

vitals = [
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
]

print(
    df[vitals].describe().T[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
)


# ============================================================
# 6. LAB RANGES
# ============================================================

print("\n============================================")
print("6. LABORATORY RANGES")
print("============================================")

labs = [
    "potassium",
    "sodium",
    "glucose",
    "creatinine",
    "BUN",
    "WBC",
    "hemoglobin",
    "platelets",
    "bicarbonate",
    "calcium",
    "chloride",
    "anion_gap",
]

print(
    df[labs].describe().T[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
)


# ============================================================
# 7. PATIENT COVERAGE OF LABS
# ============================================================

print("\n============================================")
print("7. PATIENT COVERAGE OF LABS")
print("============================================")

for lab in labs:

    patients_with_lab = df.loc[
        df[lab].notna(),
        "patientunitstayid"
    ].nunique()

    print(
        f"{lab:15s}: "
        f"{patients_with_lab:,} patients"
    )


# ============================================================
# 8. VITAL COVERAGE
# ============================================================

print("\n============================================")
print("8. VITAL SIGN COVERAGE")
print("============================================")

for vital in vitals:

    patients_with_vital = df.loc[
        df[vital].notna(),
        "patientunitstayid"
    ].nunique()

    print(
        f"{vital:15s}: "
        f"{patients_with_vital:,} patients"
    )


# ============================================================
# 9. COMPLETELY EMPTY ROWS
# ============================================================

print("\n============================================")
print("9. EMPTY MEASUREMENT ROWS")
print("============================================")

measurement_columns = vitals + labs

empty_rows = df[
    df[measurement_columns]
    .notna()
    .sum(axis=1)
    == 0
]

print(
    f"Rows with no vital/lab measurements: "
    f"{len(empty_rows):,}"
)


# ============================================================
# 10. PATIENTS WITH NO MEASUREMENTS
# ============================================================

patients_with_measurements = (
    df.loc[
        df[measurement_columns]
        .notna()
        .any(axis=1),
        "patientunitstayid"
    ]
    .nunique()
)

total_patients = df["patientunitstayid"].nunique()

print(
    f"Patients with at least one measurement: "
    f"{patients_with_measurements:,}"
)

print(
    f"Patients with no measurements: "
    f"{total_patients - patients_with_measurements:,}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n============================================")
print("VALIDATION COMPLETE")
print("============================================")