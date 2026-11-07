import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR
    / "data"
    / "real"
    / "eicu_demo"
    / "eicu-collaborative-research-database-demo-2.0.1"
)

INFUSION_FILE = DATA_DIR / "infusiondrug.csv.gz"
PATIENT_FILE = DATA_DIR / "patient.csv.gz"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "hourly_infusions.csv"


# ============================================================
# DRUG NORMALIZATION
# ============================================================

DRUG_MAP = {
    "Norepinephrine": "norepinephrine",
    "Vasopressin": "vasopressin",
    "Phenylephrine": "phenylephrine",
    "Dobutamine": "dobutamine",
    "Insulin": "insulin",
    "Propofol": "propofol",
    "Fentanyl": "fentanyl",
    "Midazolam": "midazolam",
}


# ============================================================
# LOAD PATIENT ICU WINDOWS
# ============================================================

print("\n[1/7] Loading patient ICU windows...")

patients = pd.read_csv(
    PATIENT_FILE,
    usecols=[
        "patientunitstayid",
        "unitdischargeoffset",
    ],
)

print(f"Patients loaded: {len(patients):,}")


# ============================================================
# LOAD INFUSION DATA
# ============================================================

print("\n[2/7] Loading infusion data...")

columns = [
    "patientunitstayid",
    "infusionoffset",
    "drugname",
    "drugrate",
    "infusionrate",
    "drugamount",
    "volumeoffluid",
]

infusions = pd.read_csv(
    INFUSION_FILE,
    usecols=columns,
)

print(f"Infusion records loaded: {len(infusions):,}")


# ============================================================
# NORMALIZE DRUG NAMES
# ============================================================

print("\n[3/7] Normalizing drug names...")


def normalize_drug_name(name):

    if pd.isna(name):
        return None

    name = str(name).strip()

    base_name = name.split("(")[0].strip()

    for original, normalized in DRUG_MAP.items():

        if base_name.lower() == original.lower():
            return normalized

    return None


infusions["drug"] = infusions["drugname"].apply(
    normalize_drug_name
)

infusions = infusions[
    infusions["drug"].notna()
].copy()

print(
    f"Selected infusion records: "
    f"{len(infusions):,}"
)

print("\nSelected drugs:")
print(infusions["drug"].value_counts())


# ============================================================
# CONVERT NUMERIC FIELDS
# ============================================================

print("\n[4/7] Converting numeric fields...")

numeric_columns = [
    "infusionoffset",
    "drugrate",
    "infusionrate",
    "drugamount",
    "volumeoffluid",
]

for column in numeric_columns:

    infusions[column] = pd.to_numeric(
        infusions[column],
        errors="coerce"
    )


# ============================================================
# ICU WINDOW FILTER
# ============================================================

print("\n[5/7] Restricting infusions to ICU stay...")

infusions = infusions.merge(
    patients,
    on="patientunitstayid",
    how="left",
)

infusions = infusions[
    infusions["infusionoffset"].notna()
].copy()

infusions = infusions[
    infusions["unitdischargeoffset"].notna()
].copy()

# Keep only events occurring during the ICU stay
infusions = infusions[
    (infusions["infusionoffset"] >= 0)
    &
    (infusions["infusionoffset"] <= infusions["unitdischargeoffset"])
].copy()

print(
    f"Infusion records inside ICU window: "
    f"{len(infusions):,}"
)

# Convert minute offset → ICU hour
infusions["hour"] = (
    infusions["infusionoffset"] // 60
).astype(int)


# ============================================================
# CREATE HOURLY EXPOSURE FEATURES
# ============================================================

print("\n[6/7] Creating hourly infusion features...")

exposure = (
    infusions
    .groupby(
        [
            "patientunitstayid",
            "hour",
            "drug",
        ]
    )
    .size()
    .reset_index(name="records")
)

exposure = (
    exposure
    .pivot(
        index=[
            "patientunitstayid",
            "hour",
        ],
        columns="drug",
        values="records",
    )
    .reset_index()
)

exposure.columns.name = None


# ============================================================
# ENSURE ALL EXPECTED DRUG FEATURES EXIST
# ============================================================

expected_drugs = [
    "norepinephrine",
    "vasopressin",
    "phenylephrine",
    "dobutamine",
    "insulin",
    "propofol",
    "fentanyl",
    "midazolam",
]

for drug in expected_drugs:

    if drug not in exposure.columns:
        exposure[drug] = 0


# Convert counts → binary exposure

for drug in expected_drugs:

    exposure[drug] = (
        exposure[drug]
        .fillna(0)
        .gt(0)
        .astype(int)
    )


# ============================================================
# SAVE
# ============================================================

print("\n[7/7] Saving hourly infusion dataset...")

exposure = exposure.sort_values(
    [
        "patientunitstayid",
        "hour",
    ]
).reset_index(drop=True)

exposure.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n============================================")
print("INFUSION PROCESSING COMPLETE")
print("============================================")

print(f"Output: {OUTPUT_FILE}")
print(f"Rows: {len(exposure):,}")
print(f"Columns: {len(exposure.columns)}")

print("\nDrug exposure counts:")

for drug in expected_drugs:

    count = exposure[drug].sum()

    patients_count = exposure.loc[
        exposure[drug] == 1,
        "patientunitstayid"
    ].nunique()

    print(
        f"{drug:18s}: "
        f"{count:,} hourly exposures | "
        f"{patients_count:,} patients"
    )

print("\nFirst 10 rows:")
print(exposure.head(10))