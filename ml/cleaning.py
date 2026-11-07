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

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "hourly_vitals_labs_clean.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# DEFINE MEASUREMENT COLUMNS
# ============================================================

vitals = [
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
]

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


# ============================================================
# 1. REMOVE PHYSIOLOGICALLY IMPOSSIBLE VITAL VALUES
# ============================================================

print("\n[1/5] Cleaning impossible vital-sign values...")


# Heart rate
invalid_hr = (
    (df["heart_rate"] <= 0)
    | (df["heart_rate"] > 300)
)

# SpO2
invalid_spo2 = (
    (df["spo2"] <= 0)
    | (df["spo2"] > 100)
)

# Respiration
invalid_resp = (
    (df["respiration"] <= 0)
    | (df["respiration"] > 100)
)

# Blood pressure
invalid_sbp = (
    (df["systolic_bp"] <= 0)
    | (df["systolic_bp"] > 300)
)

invalid_dbp = (
    (df["diastolic_bp"] <= 0)
    | (df["diastolic_bp"] > 200)
)

invalid_map = (
    (df["mean_bp"] <= 0)
    | (df["mean_bp"] > 250)
)


invalid_counts = {
    "heart_rate": invalid_hr.sum(),
    "spo2": invalid_spo2.sum(),
    "respiration": invalid_resp.sum(),
    "systolic_bp": invalid_sbp.sum(),
    "diastolic_bp": invalid_dbp.sum(),
    "mean_bp": invalid_map.sum(),
}


for column, count in invalid_counts.items():

    print(
        f"{column:15s}: "
        f"{count:,} invalid values"
    )


# Convert impossible values to missing

df.loc[invalid_hr, "heart_rate"] = pd.NA
df.loc[invalid_spo2, "spo2"] = pd.NA
df.loc[invalid_resp, "respiration"] = pd.NA
df.loc[invalid_sbp, "systolic_bp"] = pd.NA
df.loc[invalid_dbp, "diastolic_bp"] = pd.NA
df.loc[invalid_map, "mean_bp"] = pd.NA


# ============================================================
# 2. CLEAN IMPOSSIBLE LAB VALUES
# ============================================================

print("\n[2/5] Cleaning impossible laboratory values...")


# These are conservative physical validity checks.
# Extreme but medically possible values are retained.

lab_limits = {

    "potassium": (0, 15),

    "sodium": (80, 220),

    "glucose": (0, 1500),

    "creatinine": (0, 30),

    "BUN": (0, 300),

    "WBC": (0, 1000),

    "hemoglobin": (0, 30),

    "platelets": (0, 2000),

    "bicarbonate": (0, 100),

    "calcium": (0, 20),

    "chloride": (0, 200),

    "anion_gap": (-20, 100),
}


for column, (lower, upper) in lab_limits.items():

    invalid = (
        (df[column] < lower)
        | (df[column] > upper)
    )

    count = invalid.sum()

    if count > 0:

        print(
            f"{column:15s}: "
            f"{count:,} invalid values"
        )

        df.loc[invalid, column] = pd.NA


# ============================================================
# 3. REMOVE ROWS WITH NO MEASUREMENTS
# ============================================================

print("\n[3/5] Removing completely empty measurement rows...")


measurement_columns = vitals + labs

before_rows = len(df)

df = df[
    df[measurement_columns]
    .notna()
    .any(axis=1)
].copy()

removed_rows = before_rows - len(df)

print(
    f"Removed rows: {removed_rows:,}"
)


# ============================================================
# 4. REMOVE PATIENTS WITH NO MEASUREMENTS
# ============================================================

print("\n[4/5] Checking patient measurement coverage...")


patients_before = df["patientunitstayid"].nunique()


patients_with_measurements = (
    df.groupby("patientunitstayid")[measurement_columns]
    .apply(lambda x: x.notna().any().any())
)

valid_patients = patients_with_measurements[
    patients_with_measurements
].index


df = df[
    df["patientunitstayid"].isin(valid_patients)
].copy()


patients_after = df["patientunitstayid"].nunique()


print(
    f"Patients before: {patients_before:,}"
)

print(
    f"Patients after:  {patients_after:,}"
)


# ============================================================
# 5. SAVE CLEAN DATASET
# ============================================================

print("\n[5/5] Saving cleaned dataset...")


df = df.sort_values(
    ["patientunitstayid", "hour"]
).reset_index(drop=True)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print("CLEANING COMPLETE")
print("============================================")

print(
    f"Output: {OUTPUT_FILE}"
)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    f"ICU stays: "
    f"{df['patientunitstayid'].nunique():,}"
)

print("\nRemaining missingness:")

print(
    df[measurement_columns]
    .isna()
    .mean()
    .mul(100)
    .round(2)
    .sort_values()
)

print("\nFirst 5 rows:")

print(df.head())