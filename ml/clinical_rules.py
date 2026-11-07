import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "hourly_patient_state_trends.csv"
OUTPUT_FILE = PROCESSED_DIR / "hourly_patient_state_rules.csv"


# ============================================================
# Load
# ============================================================

print("Loading trend dataset...")

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)
print("ICU stays:", df["patientunitstayid"].nunique())


# ============================================================
# Helper function
# ============================================================

def available(series):
    return series.notna()


# ============================================================
# 1. HEART RATE RULES
# ============================================================

print("\nCreating heart-rate rules...")

df["rule_tachycardia"] = (
    df["heart_rate"] > 120
).astype(int)

df["rule_bradycardia"] = (
    df["heart_rate"] < 50
).astype(int)

df["rule_severe_tachycardia"] = (
    df["heart_rate"] > 140
).astype(int)

df["rule_severe_bradycardia"] = (
    df["heart_rate"] < 40
).astype(int)


# ============================================================
# 2. OXYGENATION RULES
# ============================================================

print("Creating oxygenation rules...")

df["rule_hypoxemia"] = (
    df["spo2"] < 92
).astype(int)

df["rule_severe_hypoxemia"] = (
    df["spo2"] < 88
).astype(int)


# ============================================================
# 3. RESPIRATORY RATE RULES
# ============================================================

print("Creating respiratory rules...")

df["rule_abnormal_respiration"] = (
    (df["respiration"] < 10) |
    (df["respiration"] > 30)
).astype(int)

df["rule_severe_respiration"] = (
    (df["respiration"] < 8) |
    (df["respiration"] > 35)
).astype(int)


# ============================================================
# 4. BLOOD PRESSURE RULES
# ============================================================

print("Creating blood-pressure rules...")

df["rule_hypotension_sbp"] = (
    df["systolic_bp"] < 90
).astype(int)

df["rule_severe_hypotension_sbp"] = (
    df["systolic_bp"] < 80
).astype(int)

df["rule_low_map"] = (
    df["mean_bp"] < 65
).astype(int)

df["rule_severe_low_map"] = (
    df["mean_bp"] < 55
).astype(int)


# ============================================================
# 5. RAPID HEART-RATE CHANGE
# ============================================================

print("Creating heart-rate trend rules...")

df["rule_rapid_hr_rise"] = (
    df["heart_rate_change_1h"] > 20
).astype(int)

df["rule_rapid_hr_fall"] = (
    df["heart_rate_change_1h"] < -20
).astype(int)

df["rule_severe_hr_rise"] = (
    df["heart_rate_change_1h"] > 30
).astype(int)

df["rule_severe_hr_fall"] = (
    df["heart_rate_change_1h"] < -30
).astype(int)


# ============================================================
# 6. RAPID SpO2 CHANGE
# ============================================================

print("Creating oxygen trend rules...")

df["rule_falling_spo2"] = (
    df["spo2_change_1h"] < -3
).astype(int)

df["rule_rapid_falling_spo2"] = (
    df["spo2_change_1h"] < -5
).astype(int)


# ============================================================
# 7. RAPID MAP CHANGE
# ============================================================

print("Creating blood-pressure trend rules...")

df["rule_falling_map"] = (
    df["mean_bp_change_1h"] < -15
).astype(int)

df["rule_rapid_falling_map"] = (
    df["mean_bp_change_1h"] < -25
).astype(int)


# ============================================================
# 8. VASOACTIVE SUPPORT
# ============================================================

print("Creating treatment-context rules...")

df["rule_vasoactive_support"] = (
    df["vasoactive_support"] > 0
).astype(int)


# ============================================================
# 9. MULTIPLE ACTIVE INFUSIONS
# ============================================================

df["rule_multiple_infusions"] = (
    df["active_infusion_count"] >= 2
).astype(int)


# ============================================================
# 10. FLUID BALANCE RULES
#
# These are deliberately broad because hourly fluid totals
# can contain charting/measurement variability.
# ============================================================

print("Creating fluid-balance rules...")

df["rule_large_positive_fluid_balance"] = (
    df["net_fluid_balance_ml"] >= 500
).astype(int)

df["rule_large_negative_fluid_balance"] = (
    df["net_fluid_balance_ml"] <= -500
).astype(int)


# ============================================================
# 11. VERY LOW RECORDED URINE OUTPUT
#
# This is NOT labeled as oliguria.
#
# It simply identifies hours with zero recorded urine output.
# Later we can build a weight-normalized multi-hour urine
# feature if the necessary data quality supports it.
# ============================================================

df["rule_zero_recorded_urine"] = (
    df["urine_output_ml"] == 0
).astype(int)


# ============================================================
# 12. COMPOSITE WARNING COUNT
#
# Count independent warning signals.
# Severe signals are NOT double-counted here.
# ============================================================

warning_columns = [
    "rule_tachycardia",
    "rule_bradycardia",
    "rule_hypoxemia",
    "rule_abnormal_respiration",
    "rule_hypotension_sbp",
    "rule_low_map",
    "rule_rapid_hr_rise",
    "rule_rapid_hr_fall",
    "rule_falling_spo2",
    "rule_falling_map",
    "rule_vasoactive_support",
    "rule_multiple_infusions",
    "rule_large_positive_fluid_balance",
    "rule_large_negative_fluid_balance"
]

df["clinical_warning_count"] = (
    df[warning_columns]
    .sum(axis=1)
)


# ============================================================
# 13. COMPOSITE SEVERE COUNT
# ============================================================

severe_columns = [
    "rule_severe_tachycardia",
    "rule_severe_bradycardia",
    "rule_severe_hypoxemia",
    "rule_severe_respiration",
    "rule_severe_hypotension_sbp",
    "rule_severe_low_map",
    "rule_severe_hr_rise",
    "rule_severe_hr_fall",
    "rule_rapid_falling_spo2",
    "rule_rapid_falling_map"
]

df["clinical_severe_count"] = (
    df[severe_columns]
    .sum(axis=1)
)


# ============================================================
# 14. RULE SEVERITY
#
# This is a transparent rule-based classification.
# It is NOT a machine-learning prediction.
# ============================================================

df["clinical_rule_severity"] = np.select(
    [
        df["clinical_severe_count"] >= 2,
        df["clinical_severe_count"] == 1,
        df["clinical_warning_count"] >= 2,
        df["clinical_warning_count"] == 1
    ],
    [
        "critical",
        "high",
        "moderate",
        "low"
    ],
    default="normal"
)


# ============================================================
# 15. EXPLANATION STRING
#
# Useful later for the ICU dashboard.
# ============================================================

rule_names = {
    "rule_tachycardia": "tachycardia",
    "rule_bradycardia": "bradycardia",
    "rule_hypoxemia": "low SpO2",
    "rule_abnormal_respiration": "abnormal respiration",
    "rule_hypotension_sbp": "low systolic BP",
    "rule_low_map": "low MAP",
    "rule_rapid_hr_rise": "rapid HR rise",
    "rule_rapid_hr_fall": "rapid HR fall",
    "rule_falling_spo2": "falling SpO2",
    "rule_falling_map": "falling MAP",
    "rule_vasoactive_support": "vasoactive support",
    "rule_multiple_infusions": "multiple active infusions",
    "rule_large_positive_fluid_balance": "large positive fluid balance",
    "rule_large_negative_fluid_balance": "large negative fluid balance"
}

def build_explanation(row):
    reasons = []

    for column, label in rule_names.items():
        if row[column] == 1:
            reasons.append(label)

    if not reasons:
        return "No rule-based abnormality"

    return "; ".join(reasons)


print("\nCreating rule explanations...")

df["clinical_rule_explanation"] = df.apply(
    build_explanation,
    axis=1
)


# ============================================================
# VALIDATION
# ============================================================

print("\n============================================================")
print("RULE VALIDATION")
print("============================================================")

duplicates = df.duplicated(
    subset=["patientunitstayid", "hour"]
).sum()

print(
    "Duplicate patient-hours:",
    duplicates
)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)

print(
    "ICU stays:",
    df["patientunitstayid"].nunique()
)


if duplicates > 0:
    raise ValueError(
        "Duplicate patient-hour rows detected."
    )

if len(df) != 139432:
    raise ValueError(
        f"Expected 139432 rows, got {len(df)}"
    )

if df["patientunitstayid"].nunique() != 2419:
    raise ValueError(
        "ICU stay count changed."
    )


# ============================================================
# Rule frequency
# ============================================================

print("\n============================================================")
print("RULE FREQUENCIES")
print("============================================================")

rule_columns = list(rule_names.keys())

for col in rule_columns:
    count = int(df[col].sum())
    patients = int(
        df.loc[
            df[col] == 1,
            "patientunitstayid"
        ].nunique()
    )

    print(
        f"{col:40s}"
        f" hours = {count:6d}"
        f" | patients = {patients:4d}"
    )


# ============================================================
# Severity distribution
# ============================================================

print("\n============================================================")
print("CLINICAL RULE SEVERITY")
print("============================================================")

print(
    df["clinical_rule_severity"]
    .value_counts()
    .to_string()
)


# ============================================================
# Composite counts
# ============================================================

print("\n============================================================")
print("COMPOSITE COUNTS")
print("============================================================")

print(
    "Rows with >=1 warning:",
    int(
        (df["clinical_warning_count"] >= 1).sum()
    )
)

print(
    "Rows with >=2 warnings:",
    int(
        (df["clinical_warning_count"] >= 2).sum()
    )
)

print(
    "Rows with >=1 severe signal:",
    int(
        (df["clinical_severe_count"] >= 1).sum()
    )
)

print(
    "Rows with >=2 severe signals:",
    int(
        (df["clinical_severe_count"] >= 2).sum()
    )
)


# ============================================================
# Example explanations
# ============================================================

print("\n============================================================")
print("EXAMPLE ALERT EXPLANATIONS")
print("============================================================")

examples = df.loc[
    df["clinical_warning_count"] > 0,
    [
        "patientunitstayid",
        "hour",
        "heart_rate",
        "spo2",
        "mean_bp",
        "clinical_rule_severity",
        "clinical_rule_explanation"
    ]
].head(10)

print(
    examples.to_string(index=False)
)


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n============================================================")
print("SUCCESS")
print("============================================================")

print(
    "Output:",
    OUTPUT_FILE
)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)