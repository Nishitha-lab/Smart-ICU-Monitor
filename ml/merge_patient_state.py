import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

VITALS_FILE = PROCESSED_DIR / "hourly_vitals_labs_clean.csv"
INFUSIONS_FILE = PROCESSED_DIR / "hourly_infusions.csv"
FLUID_FILE = PROCESSED_DIR / "hourly_fluid_balance.csv"

OUTPUT_FILE = PROCESSED_DIR / "hourly_patient_state.csv"

KEYS = ["patientunitstayid", "hour"]


# ============================================================
# Load datasets
# ============================================================

print("Loading cleaned vitals/labs...")
vitals = pd.read_csv(VITALS_FILE)

print("Loading infusion features...")
infusions = pd.read_csv(INFUSIONS_FILE)

print("Loading fluid balance...")
fluid = pd.read_csv(FLUID_FILE)


# ============================================================
# Basic information
# ============================================================

print("\n============================================================")
print("INPUT DATASETS")
print("============================================================")

print(
    "Vitals/Labs:",
    vitals.shape,
    "| ICU stays:",
    vitals["patientunitstayid"].nunique()
)

print(
    "Infusions:",
    infusions.shape,
    "| ICU stays:",
    infusions["patientunitstayid"].nunique()
)

print(
    "Fluid:",
    fluid.shape,
    "| ICU stays:",
    fluid["patientunitstayid"].nunique()
)


# ============================================================
# Validate key columns
# ============================================================

for name, df in [
    ("vitals/labs", vitals),
    ("infusions", infusions),
    ("fluid", fluid)
]:
    missing_keys = [
        col for col in KEYS
        if col not in df.columns
    ]

    if missing_keys:
        raise ValueError(
            f"{name} is missing key columns: {missing_keys}"
        )


# ============================================================
# Check duplicate patient-hour keys
# ============================================================

print("\n============================================================")
print("DUPLICATE KEY CHECK")
print("============================================================")

for name, df in [
    ("vitals/labs", vitals),
    ("infusions", infusions),
    ("fluid", fluid)
]:
    duplicates = df.duplicated(
        subset=KEYS
    ).sum()

    print(
        f"{name}: duplicate patient-hours = {duplicates}"
    )

    if duplicates > 0:
        raise ValueError(
            f"{name} contains duplicate patient-hour keys."
        )


# ============================================================
# Preserve original main timeline
#
# Vitals/labs is our master hourly timeline.
# ============================================================

original_rows = len(vitals)
original_stays = vitals["patientunitstayid"].nunique()


# ============================================================
# Merge infusions
# ============================================================

print("\nMerging infusion features...")

state = vitals.merge(
    infusions,
    on=KEYS,
    how="left",
    validate="one_to_one"
)

print(
    "Rows after infusion merge:",
    len(state)
)


# ============================================================
# Fill absent infusion exposure with 0
#
# Missing means there was no recorded exposure in that hour.
# ============================================================

infusion_cols = [
    "dobutamine",
    "fentanyl",
    "insulin",
    "midazolam",
    "norepinephrine",
    "phenylephrine",
    "propofol",
    "vasopressin"
]

for col in infusion_cols:
    state[col] = state[col].fillna(0)


# ============================================================
# Merge fluid balance
# ============================================================

print("Merging fluid features...")

state = state.merge(
    fluid,
    on=KEYS,
    how="left",
    validate="one_to_one"
)

print(
    "Rows after fluid merge:",
    len(state)
)


# ============================================================
# Fill fluid features
#
# If there is no intake/output record for a patient-hour,
# represent it as zero recorded fluid activity.
# ============================================================

fluid_cols = [
    "total_intake_ml",
    "total_output_ml",
    "dialysis_ml",
    "reported_net_ml",
    "net_fluid_balance_ml",
    "urine_output_ml"
]

for col in fluid_cols:
    state[col] = state[col].fillna(0)


# ============================================================
# Sort
# ============================================================

state = state.sort_values(
    KEYS
).reset_index(drop=True)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n============================================================")
print("FINAL VALIDATION")
print("============================================================")

print(
    "Original rows:",
    original_rows
)

print(
    "Final rows:",
    len(state)
)

print(
    "Original ICU stays:",
    original_stays
)

print(
    "Final ICU stays:",
    state["patientunitstayid"].nunique()
)

print(
    "Duplicate patient-hour rows:",
    state.duplicated(subset=KEYS).sum()
)

print(
    "Minimum hour:",
    state["hour"].min()
)

print(
    "Maximum hour:",
    state["hour"].max()
)


# ============================================================
# Check that the master timeline was preserved
# ============================================================

if len(state) != original_rows:
    raise ValueError(
        "Row count changed after merging. "
        "The master timeline was not preserved."
    )

if state["patientunitstayid"].nunique() != original_stays:
    raise ValueError(
        "ICU stay count changed after merging."
    )

if state.duplicated(subset=KEYS).sum() != 0:
    raise ValueError(
        "Final dataset contains duplicate patient-hour rows."
    )


# ============================================================
# Missingness report
# ============================================================

print("\n============================================================")
print("MISSINGNESS")
print("============================================================")

missing = (
    state.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(
    missing.to_string()
)


# ============================================================
# Infusion exposure summary
# ============================================================

print("\n============================================================")
print("INFUSION EXPOSURE")
print("============================================================")

for col in infusion_cols:
    print(
        f"{col:16s}",
        "hours =",
        int(state[col].sum()),
        "| patients =",
        int(
            state.loc[
                state[col] > 0,
                "patientunitstayid"
            ].nunique()
        )
    )


# ============================================================
# Fluid summary
# ============================================================

print("\n============================================================")
print("FLUID FEATURES")
print("============================================================")

print(
    state[
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
# Final columns
# ============================================================

print("\n============================================================")
print("FINAL DATASET")
print("============================================================")

print(
    "Rows:",
    len(state)
)

print(
    "Columns:",
    len(state.columns)
)

print("\nColumns:")
print(state.columns.tolist())


# ============================================================
# Save
# ============================================================

state.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nStep 5 unified patient state created successfully.")