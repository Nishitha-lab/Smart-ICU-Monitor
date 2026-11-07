import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"

VITALS_FILE = PROCESSED_DIR / "hourly_vitals_labs_clean.csv"
INFUSION_FILE = PROCESSED_DIR / "hourly_infusions.csv"

OUTPUT_FILE = PROCESSED_DIR / "hourly_vitals_labs_infusions.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("\n[1/5] Loading cleaned vitals + labs...")

vitals = pd.read_csv(VITALS_FILE)

print(f"Vitals/labs rows: {len(vitals):,}")
print(f"Vitals/labs columns: {len(vitals.columns)}")


print("\n[2/5] Loading hourly infusion data...")

infusions = pd.read_csv(INFUSION_FILE)

print(f"Infusion rows: {len(infusions):,}")
print(f"Infusion columns: {len(infusions.columns)}")


# ============================================================
# CHECK KEYS
# ============================================================

print("\n[3/5] Checking patient-hour keys...")

key_columns = [
    "patientunitstayid",
    "hour",
]


duplicate_infusions = infusions.duplicated(
    subset=key_columns
).sum()

duplicate_vitals = vitals.duplicated(
    subset=key_columns
).sum()

print(
    f"Duplicate vitals/labs patient-hours: "
    f"{duplicate_vitals}"
)

print(
    f"Duplicate infusion patient-hours: "
    f"{duplicate_infusions}"
)


# ============================================================
# MERGE
# ============================================================

print("\n[4/5] Merging datasets...")

merged = vitals.merge(
    infusions,
    on=key_columns,
    how="left",
    validate="one_to_one",
)


# ============================================================
# FILL ABSENT INFUSIONS WITH ZERO
# ============================================================

infusion_columns = [
    "norepinephrine",
    "vasopressin",
    "phenylephrine",
    "dobutamine",
    "insulin",
    "propofol",
    "fentanyl",
    "midazolam",
]


for column in infusion_columns:

    merged[column] = (
        merged[column]
        .fillna(0)
        .astype(int)
    )


# ============================================================
# SORT
# ============================================================

merged = merged.sort_values(
    [
        "patientunitstayid",
        "hour",
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

print("\n[5/5] Saving unified dataset...")

merged.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION SUMMARY
# ============================================================

print("\n============================================")
print("INFUSION MERGE COMPLETE")
print("============================================")

print(f"Output: {OUTPUT_FILE}")
print(f"Rows: {len(merged):,}")
print(f"Columns: {len(merged.columns):,}")

print(
    f"Unique ICU stays: "
    f"{merged['patientunitstayid'].nunique():,}"
)

print(
    f"Unique patient-hours: "
    f"{merged[['patientunitstayid', 'hour']].drop_duplicates().shape[0]:,}"
)


print("\nInfusion exposure summary:")

for column in infusion_columns:

    exposure_hours = merged[column].sum()

    exposure_patients = merged.loc[
        merged[column] == 1,
        "patientunitstayid"
    ].nunique()

    print(
        f"{column:18s}: "
        f"{exposure_hours:,} hours | "
        f"{exposure_patients:,} patients"
    )


print("\nChecking for missing infusion values:")

print(
    merged[infusion_columns]
    .isna()
    .sum()
)


print("\nFirst 5 rows:")

print(
    merged[
        [
            "patientunitstayid",
            "hour",
            "heart_rate",
            "spo2",
            "mean_bp",
            "glucose",
            "norepinephrine",
            "vasopressin",
            "propofol",
            "fentanyl",
        ]
    ].head()
)