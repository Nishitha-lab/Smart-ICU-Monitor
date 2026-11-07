import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT
    / "data"
    / "real"
    / "eicu_demo"
    / "eicu-collaborative-research-database-demo-2.0.1"
)

PROCESSED_DIR = ROOT / "data" / "processed"

PATIENT_FILE = DATA_DIR / "patient.csv.gz"
FLUID_FILE = DATA_DIR / "intakeOutput.csv.gz"

OUTPUT_FILE = PROCESSED_DIR / "hourly_fluid_balance.csv"


# ============================================================
# Load ICU stay windows
# ============================================================

print("Loading patient ICU windows...")

patients = pd.read_csv(
    PATIENT_FILE,
    usecols=[
        "patientunitstayid",
        "unitdischargeoffset"
    ]
)

patients["unitdischargeoffset"] = pd.to_numeric(
    patients["unitdischargeoffset"],
    errors="coerce"
)

patients = patients.dropna(
    subset=[
        "patientunitstayid",
        "unitdischargeoffset"
    ]
)

print("ICU stays available:", len(patients))


# ============================================================
# Load intake/output data
# ============================================================

print("\nLoading intakeOutput.csv.gz...")

fluid = pd.read_csv(
    FLUID_FILE,
    usecols=[
        "patientunitstayid",
        "intakeoutputoffset",
        "intaketotal",
        "outputtotal",
        "dialysistotal",
        "nettotal",
        "celllabel",
        "cellvaluenumeric"
    ]
)

print("Raw records loaded:", len(fluid))


# ============================================================
# Numeric conversion
# ============================================================

numeric_cols = [
    "intakeoutputoffset",
    "intaketotal",
    "outputtotal",
    "dialysistotal",
    "nettotal",
    "cellvaluenumeric"
]

for col in numeric_cols:
    fluid[col] = pd.to_numeric(
        fluid[col],
        errors="coerce"
    )


# ============================================================
# Restrict to ICU window
# ============================================================

fluid = fluid.merge(
    patients,
    on="patientunitstayid",
    how="inner",
    validate="many_to_one"
)

fluid = fluid[
    (fluid["intakeoutputoffset"] >= 0)
    & (
        fluid["intakeoutputoffset"]
        <= fluid["unitdischargeoffset"]
    )
].copy()

print("Records inside ICU window:", len(fluid))


# ============================================================
# PART 1
# Interval-level aggregate totals
#
# Multiple celllabel rows can share the same patient + offset.
# The aggregate totals are repeated on those rows.
#
# Therefore, retain ONE copy per patient + offset.
# ============================================================

interval_totals = (
    fluid[
        [
            "patientunitstayid",
            "intakeoutputoffset",
            "intaketotal",
            "outputtotal",
            "dialysistotal",
            "nettotal"
        ]
    ]
    .drop_duplicates(
        subset=[
            "patientunitstayid",
            "intakeoutputoffset"
        ]
    )
    .copy()
)

print(
    "\nUnique patient-offset intervals:",
    len(interval_totals)
)


# ============================================================
# Validate net-fluid calculation
#
# eICU uses signed dialysis values.
#
# net = intake - output + dialysis
# ============================================================

interval_totals["calculated_net_ml"] = (
    interval_totals["intaketotal"].fillna(0)
    - interval_totals["outputtotal"].fillna(0)
    + interval_totals["dialysistotal"].fillna(0)
)

interval_totals["net_difference_ml"] = (
    interval_totals["nettotal"]
    - interval_totals["calculated_net_ml"]
)

valid_net = interval_totals[
    "net_difference_ml"
].dropna()

print("\nNet calculation validation:")
print(
    "Intervals checked:",
    len(valid_net)
)

print(
    "Max absolute difference:",
    valid_net.abs().max()
)

print(
    "Intervals differing > 1 mL:",
    (valid_net.abs() > 1).sum()
)


# ============================================================
# PART 2
# Convert interval records to hourly records
# ============================================================

interval_totals["hour"] = (
    interval_totals["intakeoutputoffset"] // 60
).astype(int)

hourly = (
    interval_totals
    .groupby(
        [
            "patientunitstayid",
            "hour"
        ],
        as_index=False
    )
    .agg(
        total_intake_ml=(
            "intaketotal",
            "sum"
        ),
        total_output_ml=(
            "outputtotal",
            "sum"
        ),
        dialysis_ml=(
            "dialysistotal",
            "sum"
        ),
        reported_net_ml=(
            "nettotal",
            "sum"
        )
    )
)


# ============================================================
# Calculate hourly net fluid balance
# ============================================================

hourly["net_fluid_balance_ml"] = (
    hourly["total_intake_ml"]
    - hourly["total_output_ml"]
    + hourly["dialysis_ml"]
)


# ============================================================
# PART 3
# Conservative urine-volume extraction
# ============================================================

fluid["celllabel_clean"] = (
    fluid["celllabel"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)


# ------------------------------------------------------------
# Explicit urine-volume labels
#
# IMPORTANT:
# Do NOT match generic "urine" blindly because that would
# include:
#   - Urine Count
#   - Urine Occurrence
#
# Also exclude mixed urine/stool from pure urine volume.
# ------------------------------------------------------------

urine_patterns = [
    r"^urine$",
    r"^urine catheter$",
    r"^urine output[- ]foley$",
    r"^urine output \(ml\)-urethral catheter$",
    r"^foley cath$",
    r"^foley cath$",
    r"^foley$",
    r"^sn urine output\(ml\)$",
    r"^output \(ml\)-external urinary catheter$",
    r"^urinary catheter output:"
]

urine_pattern = "|".join(
    f"({pattern})"
    for pattern in urine_patterns
)

urine = fluid[
    fluid["celllabel_clean"].str.contains(
        urine_pattern,
        regex=True,
        na=False
    )
].copy()

print(
    "\nExplicit urine-volume records:",
    len(urine)
)


# ============================================================
# Convert urine records to hourly buckets
# ============================================================

urine["hour"] = (
    urine["intakeoutputoffset"] // 60
).astype(int)


# ============================================================
# Multiple urine labels at same patient + offset may represent
# duplicate/alternative charting.
#
# Use MAX rather than SUM to avoid artificial double counting.
# ============================================================

urine_interval = (
    urine
    .groupby(
        [
            "patientunitstayid",
            "intakeoutputoffset"
        ],
        as_index=False
    )
    .agg(
        urine_output_interval_ml=(
            "cellvaluenumeric",
            "max"
        )
    )
)


# Convert interval urine values to hourly values
urine_interval["hour"] = (
    urine_interval["intakeoutputoffset"] // 60
).astype(int)


urine_hourly = (
    urine_interval
    .groupby(
        [
            "patientunitstayid",
            "hour"
        ],
        as_index=False
    )
    .agg(
        urine_output_ml=(
            "urine_output_interval_ml",
            "sum"
        )
    )
)


# ============================================================
# Merge urine with hourly fluid balance
# ============================================================

hourly = hourly.merge(
    urine_hourly,
    on=[
        "patientunitstayid",
        "hour"
    ],
    how="left",
    validate="one_to_one"
)

hourly["urine_output_ml"] = (
    hourly["urine_output_ml"]
    .fillna(0)
)


# ============================================================
# Sort and reset index
# ============================================================

hourly = hourly.sort_values(
    [
        "patientunitstayid",
        "hour"
    ]
).reset_index(drop=True)


# ============================================================
# Save
# ============================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

hourly.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n============================================================")
print("Fluid balance processing complete")
print("============================================================")

print(
    "Output:",
    OUTPUT_FILE
)

print(
    "Rows:",
    len(hourly)
)

print(
    "Columns:",
    len(hourly.columns)
)

print(
    "ICU stays:",
    hourly["patientunitstayid"].nunique()
)

print("\nColumns:")
print(hourly.columns.tolist())


# ============================================================
# Summary statistics
# ============================================================

print("\nFluid summary:")

print(
    hourly[
        [
            "total_intake_ml",
            "total_output_ml",
            "dialysis_ml",
            "net_fluid_balance_ml",
            "urine_output_ml"
        ]
    ].describe().to_string()
)


# ============================================================
# Largest hourly intake
# ============================================================

print("\nLargest hourly intake:")

print(
    hourly.nlargest(
        10,
        "total_intake_ml"
    )[
        [
            "patientunitstayid",
            "hour",
            "total_intake_ml",
            "total_output_ml",
            "dialysis_ml",
            "net_fluid_balance_ml"
        ]
    ].to_string(index=False)
)


# ============================================================
# Largest hourly output
# ============================================================

print("\nLargest hourly output:")

print(
    hourly.nlargest(
        10,
        "total_output_ml"
    )[
        [
            "patientunitstayid",
            "hour",
            "total_intake_ml",
            "total_output_ml",
            "dialysis_ml",
            "net_fluid_balance_ml"
        ]
    ].to_string(index=False)
)