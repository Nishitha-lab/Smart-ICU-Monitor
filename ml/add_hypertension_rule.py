import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "hourly_patient_state_rules.csv"
OUTPUT_FILE = PROCESSED_DIR / "hourly_patient_state_rules_v2.csv"

print("Loading clinical rules...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Rows:", len(df))
print("ICU stays:", df["patientunitstayid"].nunique())

# ------------------------------------------------------------
# ADD SEVERE HYPERTENSION RULE
# ------------------------------------------------------------
#
# Research/demo threshold:
# SBP > 180 mmHg OR DBP > 120 mmHg
#
# This indicates severe hypertension.
# It does NOT by itself establish hypertensive emergency,
# because acute target-organ injury is required.

df["rule_severe_hypertension"] = (
    (df["systolic_bp"] > 180) |
    (df["diastolic_bp"] > 120)
).fillna(False).astype(int)

# ------------------------------------------------------------
# EXPLANATION
# ------------------------------------------------------------

def hypertension_explanation(row):
    if row["rule_severe_hypertension"] == 1:
        sbp = row["systolic_bp"]
        dbp = row["diastolic_bp"]

        if pd.notna(sbp) and pd.notna(dbp):
            return (
                f"severe hypertension "
                f"(SBP {sbp:.1f}; DBP {dbp:.1f})"
            )

        if pd.notna(sbp):
            return f"severe hypertension (SBP {sbp:.1f})"

        if pd.notna(dbp):
            return f"severe hypertension (DBP {dbp:.1f})"

    return ""


df["hypertension_explanation"] = df.apply(
    hypertension_explanation,
    axis=1
)

# ------------------------------------------------------------
# CREATE V2 WARNING / SEVERE COUNTS
# ------------------------------------------------------------

df["clinical_warning_count_v2"] = (
    pd.to_numeric(
        df["clinical_warning_count"],
        errors="coerce"
    )
    .fillna(0)
    + df["rule_severe_hypertension"]
)

df["clinical_severe_count_v2"] = (
    pd.to_numeric(
        df["clinical_severe_count"],
        errors="coerce"
    )
    .fillna(0)
    + df["rule_severe_hypertension"]
)

# ------------------------------------------------------------
# UPDATE SEVERITY
# ------------------------------------------------------------

severity_order = {
    "normal": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4
}

reverse_severity = {
    0: "normal",
    1: "low",
    2: "moderate",
    3: "high",
    4: "critical"
}

def update_severity(row):
    current = str(row["clinical_rule_severity"]).lower()

    current_level = severity_order.get(
        current,
        0
    )

    # Severe hypertension upgrades a normal/low/moderate
    # rule state to HIGH, but never downgrades an existing
    # high/critical state.
    if row["rule_severe_hypertension"] == 1:
        current_level = max(
            current_level,
            3
        )

    return reverse_severity[current_level]


df["clinical_rule_severity_v2"] = df.apply(
    update_severity,
    axis=1
)

# ------------------------------------------------------------
# UPDATE EXPLANATION
# ------------------------------------------------------------

def update_explanation(row):

    existing = str(
        row["clinical_rule_explanation"]
    )

    hypertension = str(
        row["hypertension_explanation"]
    )

    if not hypertension:
        return existing

    if (
        existing == ""
        or existing == "No rule-based abnormality"
        or existing == "nan"
    ):
        return hypertension

    return existing + "; " + hypertension


df["clinical_rule_explanation_v2"] = df.apply(
    update_explanation,
    axis=1
)

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n============================================================")
print("HYPERTENSION RULE")
print("============================================================")

hypertension_count = int(
    df["rule_severe_hypertension"].sum()
)

hypertension_patients = (
    df.loc[
        df["rule_severe_hypertension"] == 1,
        "patientunitstayid"
    ]
    .nunique()
)

print(
    "Severe-hypertension hours:",
    hypertension_count
)

print(
    "Patients affected:",
    hypertension_patients
)

print("\nUpdated severity distribution:")

print(
    df["clinical_rule_severity_v2"]
    .value_counts()
    .reindex(
        ["normal", "low", "moderate", "high", "critical"],
        fill_value=0
    )
    .to_string()
)

# ------------------------------------------------------------
# CHECK NEW RULE EXAMPLES
# ------------------------------------------------------------

print("\n============================================================")
print("SAMPLE SEVERE-HYPERTENSION CASES")
print("============================================================")

examples = (
    df[
        df["rule_severe_hypertension"] == 1
    ]
    [
        [
            "patientunitstayid",
            "hour",
            "systolic_bp",
            "diastolic_bp",
            "mean_bp",
            "clinical_rule_severity",
            "clinical_rule_severity_v2",
            "clinical_rule_explanation_v2"
        ]
    ]
    .sort_values(
        ["systolic_bp", "diastolic_bp"],
        ascending=False
    )
    .head(20)
)

if len(examples) > 0:
    print(
        examples
        .round(2)
        .to_string(index=False)
    )
else:
    print("No severe-hypertension cases found.")

# ------------------------------------------------------------
# VALIDATE UNIQUE PATIENT-HOURS
# ------------------------------------------------------------

duplicates = df.duplicated(
    subset=[
        "patientunitstayid",
        "hour"
    ]
).sum()

print("\nDuplicate patient-hours:", duplicates)

if duplicates != 0:
    raise ValueError(
        "Duplicate patient-hours detected."
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n============================================================")
print("SUCCESS")
print("============================================================")

print("Output:", OUTPUT_FILE)