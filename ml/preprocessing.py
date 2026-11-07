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

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PATIENT_FILE = DATA_DIR / "patient.csv.gz"
PERIODIC_FILE = DATA_DIR / "vitalPeriodic.csv.gz"
APERIODIC_FILE = DATA_DIR / "vitalAperiodic.csv.gz"
LAB_FILE = DATA_DIR / "lab.csv.gz"


# ============================================================
# STEP 1 — LOAD PATIENT DATA
# ============================================================

print("\n[1/7] Loading patient data...")

patient_cols = [
    "patientunitstayid",
    "gender",
    "age",
    "ethnicity",
    "unittype",
    "unitdischargeoffset",
    "unitdischargestatus",
]

patients = pd.read_csv(
    PATIENT_FILE,
    usecols=patient_cols
)

print(f"Patients loaded: {len(patients):,}")


# ============================================================
# STEP 2 — LOAD PERIODIC VITALS
# ============================================================

print("\n[2/7] Loading periodic vital signs...")

periodic_cols = [
    "patientunitstayid",
    "observationoffset",
    "heartrate",
    "sao2",
    "respiration",
]

periodic = pd.read_csv(
    PERIODIC_FILE,
    usecols=periodic_cols
)

print(f"Periodic observations: {len(periodic):,}")


# Add ICU discharge offset to every observation
periodic = periodic.merge(
    patients[
        [
            "patientunitstayid",
            "unitdischargeoffset"
        ]
    ],
    on="patientunitstayid",
    how="inner"
)

# Keep only observations occurring during ICU stay
periodic = periodic[
    (periodic["observationoffset"] >= 0)
    & (
        periodic["observationoffset"]
        <= periodic["unitdischargeoffset"]
    )
].copy()

print(
    f"Periodic observations inside ICU stay: "
    f"{len(periodic):,}"
)


# ============================================================
# STEP 3 — LOAD APERIODIC BLOOD PRESSURE
# ============================================================

print("\n[3/7] Loading aperiodic blood pressure...")

aperiodic_cols = [
    "patientunitstayid",
    "observationoffset",
    "noninvasivesystolic",
    "noninvasivediastolic",
    "noninvasivemean",
]

aperiodic = pd.read_csv(
    APERIODIC_FILE,
    usecols=aperiodic_cols
)

print(f"Aperiodic observations: {len(aperiodic):,}")


aperiodic = aperiodic.merge(
    patients[
        [
            "patientunitstayid",
            "unitdischargeoffset"
        ]
    ],
    on="patientunitstayid",
    how="inner"
)

# Keep only observations occurring during ICU stay
aperiodic = aperiodic[
    (aperiodic["observationoffset"] >= 0)
    & (
        aperiodic["observationoffset"]
        <= aperiodic["unitdischargeoffset"]
    )
].copy()

print(
    f"Aperiodic observations inside ICU stay: "
    f"{len(aperiodic):,}"
)


# ============================================================
# STEP 4 — CREATE HOURLY VITAL FEATURES
# ============================================================

print("\n[4/7] Creating hourly vital-sign features...")


periodic["hour"] = (
    periodic["observationoffset"] // 60
).astype(int)


periodic_hourly = (
    periodic
    .groupby(
        ["patientunitstayid", "hour"],
        as_index=False
    )
    .agg(
        heart_rate=("heartrate", "mean"),
        spo2=("sao2", "mean"),
        respiration=("respiration", "mean"),
    )
)


aperiodic["hour"] = (
    aperiodic["observationoffset"] // 60
).astype(int)


aperiodic_hourly = (
    aperiodic
    .groupby(
        ["patientunitstayid", "hour"],
        as_index=False
    )
    .agg(
        systolic_bp=("noninvasivesystolic", "mean"),
        diastolic_bp=("noninvasivediastolic", "mean"),
        mean_bp=("noninvasivemean", "mean"),
    )
)


hourly_vitals = pd.merge(
    periodic_hourly,
    aperiodic_hourly,
    on=["patientunitstayid", "hour"],
    how="outer"
)


print(
    f"Hourly vital rows: "
    f"{len(hourly_vitals):,}"
)


# ============================================================
# STEP 5 — LOAD AND PROCESS LABORATORY DATA
# ============================================================

print("\n[5/7] Loading laboratory data...")

lab_cols = [
    "patientunitstayid",
    "labresultoffset",
    "labname",
    "labresult",
]

labs = pd.read_csv(
    LAB_FILE,
    usecols=lab_cols
)

print(f"Laboratory records: {len(labs):,}")


# ------------------------------------------------------------
# Initial clinically useful laboratory measurements
# ------------------------------------------------------------

LAB_MAP = {
    "potassium": "potassium",
    "sodium": "sodium",
    "glucose": "glucose",
    "creatinine": "creatinine",
    "BUN": "BUN",
    "WBC x 1000": "WBC",
    "Hgb": "hemoglobin",
    "platelets x 1000": "platelets",
    "bicarbonate": "bicarbonate",
    "calcium": "calcium",
    "chloride": "chloride",
    "anion gap": "anion_gap",
}


labs = labs[
    labs["labname"].isin(LAB_MAP.keys())
].copy()


print(
    f"Selected laboratory records: "
    f"{len(labs):,}"
)


# Map original names to clean feature names
labs["lab_feature"] = labs["labname"].map(LAB_MAP)


# Convert results to numeric
labs["labresult"] = pd.to_numeric(
    labs["labresult"],
    errors="coerce"
)


# Remove invalid laboratory results
labs = labs.dropna(
    subset=["labresult"]
)


# ------------------------------------------------------------
# Restrict labs to actual ICU stay
# ------------------------------------------------------------

labs = labs.merge(
    patients[
        [
            "patientunitstayid",
            "unitdischargeoffset"
        ]
    ],
    on="patientunitstayid",
    how="inner"
)


labs = labs[
    (labs["labresultoffset"] >= 0)
    & (
        labs["labresultoffset"]
        <= labs["unitdischargeoffset"]
    )
].copy()


# Convert minutes → ICU hour
labs["hour"] = (
    labs["labresultoffset"] // 60
).astype(int)


# ------------------------------------------------------------
# If multiple results occur in one hour,
# take the mean for that laboratory.
# ------------------------------------------------------------

labs_hourly = (
    labs
    .groupby(
        [
            "patientunitstayid",
            "hour",
            "lab_feature"
        ],
        as_index=False
    )["labresult"]
    .mean()
)


# Convert laboratory names into columns
labs_hourly = (
    labs_hourly
    .pivot(
        index=["patientunitstayid", "hour"],
        columns="lab_feature",
        values="labresult"
    )
    .reset_index()
)


labs_hourly.columns.name = None


print(
    f"Hourly laboratory rows: "
    f"{len(labs_hourly):,}"
)


# ============================================================
# STEP 6 — MERGE VITALS + LABS + PATIENT INFORMATION
# ============================================================

print("\n[6/7] Building unified hourly patient timeline...")


hourly = pd.merge(
    hourly_vitals,
    labs_hourly,
    on=["patientunitstayid", "hour"],
    how="outer"
)


hourly = pd.merge(
    patients,
    hourly,
    on="patientunitstayid",
    how="left"
)


hourly = hourly.sort_values(
    [
        "patientunitstayid",
        "hour"
    ]
).reset_index(drop=True)


# ============================================================
# STEP 7 — SAVE OUTPUT
# ============================================================

print("\n[7/7] Saving processed dataset...")


output_file = (
    OUTPUT_DIR
    / "hourly_vitals_labs.csv"
)


hourly.to_csv(
    output_file,
    index=False
)


# ============================================================
# VALIDATION OUTPUT
# ============================================================

print("\n============================================")
print("PREPROCESSING COMPLETE")
print("============================================")

print(f"Output: {output_file}")

print(
    f"Rows: {len(hourly):,}"
)

print(
    f"Columns: {len(hourly.columns)}"
)


print("\nColumns:")

print(
    hourly.columns.tolist()
)


print("\nLaboratory missingness:")

lab_features = [
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
    hourly[lab_features]
    .isna()
    .mean()
    .mul(100)
    .round(2)
    .sort_values()
)


print("\nFirst 10 rows:")

print(
    hourly.head(10)
)