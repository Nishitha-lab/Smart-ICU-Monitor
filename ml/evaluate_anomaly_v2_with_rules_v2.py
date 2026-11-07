import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

ANOMALY_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_anomalies_v2.csv"
)

RULE_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_rules_v2.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "anomaly_v2_evaluation_rules_v2.csv"
)

print("Loading anomaly results...")
anomalies = pd.read_csv(
    ANOMALY_FILE,
    low_memory=False
)

print("Loading clinical rules v2...")
rules = pd.read_csv(
    RULE_FILE,
    low_memory=False
)

print("Anomaly rows:", len(anomalies))
print("Rule rows:", len(rules))

# ------------------------------------------------------------
# SELECT RULE FIELDS
# ------------------------------------------------------------

rule_columns = [
    "patientunitstayid",
    "hour",
    "clinical_warning_count_v2",
    "clinical_severe_count_v2",
    "clinical_rule_severity_v2",
    "clinical_rule_explanation_v2"
]

missing_columns = [
    col
    for col in rule_columns
    if col not in rules.columns
]

if missing_columns:
    raise ValueError(
        f"Missing rule columns: {missing_columns}"
    )

rule_subset = rules[
    rule_columns
].copy()

# ------------------------------------------------------------
# MERGE
# ------------------------------------------------------------

merged = anomalies.merge(
    rule_subset,
    on=[
        "patientunitstayid",
        "hour"
    ],
    how="left",
    validate="one_to_one"
)

# ------------------------------------------------------------
# NUMERIC CONVERSION
# ------------------------------------------------------------

merged["clinical_warning_count_v2"] = (
    pd.to_numeric(
        merged["clinical_warning_count_v2"],
        errors="coerce"
    )
    .fillna(0)
)

merged["clinical_severe_count_v2"] = (
    pd.to_numeric(
        merged["clinical_severe_count_v2"],
        errors="coerce"
    )
    .fillna(0)
)

# ------------------------------------------------------------
# CLASSIFICATION
# ------------------------------------------------------------

merged["ml_anomaly"] = (
    merged["anomaly_flag"] == 1
)

merged["rule_abnormal"] = (
    merged["clinical_warning_count_v2"] >= 1
)

merged["rule_severe"] = (
    merged["clinical_severe_count_v2"] >= 1
)

merged["ml_and_rule"] = (
    merged["ml_anomaly"] &
    merged["rule_abnormal"]
)

merged["ml_only"] = (
    merged["ml_anomaly"] &
    ~merged["rule_abnormal"]
)

merged["rule_only"] = (
    ~merged["ml_anomaly"] &
    merged["rule_abnormal"]
)

merged["both_normal"] = (
    ~merged["ml_anomaly"] &
    ~merged["rule_abnormal"]
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n============================================================")
print("ML + CLINICAL RULE V2 AGREEMENT")
print("============================================================")

print(
    "Total rows:",
    len(merged)
)

print(
    "ML anomalies:",
    int(merged["ml_anomaly"].sum())
)

print(
    "Rule abnormal:",
    int(merged["rule_abnormal"].sum())
)

print(
    "Rule severe:",
    int(merged["rule_severe"].sum())
)

print(
    "ML + rule:",
    int(merged["ml_and_rule"].sum())
)

print(
    "ML only:",
    int(merged["ml_only"].sum())
)

print(
    "Rule only:",
    int(merged["rule_only"].sum())
)

print(
    "Both normal:",
    int(merged["both_normal"].sum())
)

# ------------------------------------------------------------
# PERCENTAGES
# ------------------------------------------------------------

print("\n============================================================")
print("PERCENTAGES")
print("============================================================")

ml_count = int(
    merged["ml_anomaly"].sum()
)

if ml_count > 0:

    supported = (
        merged.loc[
            merged["ml_anomaly"],
            "rule_abnormal"
        ]
        .mean()
        * 100
    )

else:

    supported = 0

print(
    "ML anomalies supported by rules:",
    round(supported, 2),
    "%"
)

print(
    "ML anomalies not supported by rules:",
    round(100 - supported, 2),
    "%"
)

# ------------------------------------------------------------
# ML ANOMALIES BY RULE SEVERITY
# ------------------------------------------------------------

print("\n============================================================")
print("ML ANOMALIES BY RULE V2 SEVERITY")
print("============================================================")

severity_summary = (
    merged[
        merged["ml_anomaly"]
    ]
    .groupby(
        "clinical_rule_severity_v2",
        dropna=False
    )
    .size()
    .sort_values(
        ascending=False
    )
)

print(
    severity_summary.to_string()
)

# ------------------------------------------------------------
# SEVERE HYPERTENSION OVERLAP
# ------------------------------------------------------------

print("\n============================================================")
print("SEVERE HYPERTENSION + ML OVERLAP")
print("============================================================")

hypertension_ml = (
    merged["ml_anomaly"] &
    merged["clinical_rule_explanation_v2"]
    .fillna("")
    .str.contains(
        "severe hypertension",
        case=False,
        regex=False
    )
)

print(
    "ML anomalies with severe hypertension:",
    int(hypertension_ml.sum())
)

print(
    "All severe-hypertension rows:",
    int(
        rules["rule_severe_hypertension"]
        .sum()
    )
)

# ------------------------------------------------------------
# TOP ML-ONLY
# ------------------------------------------------------------

print("\n============================================================")
print("TOP ML-ONLY ANOMALIES")
print("============================================================")

ml_only = (
    merged[
        merged["ml_only"]
    ]
    .sort_values(
        "anomaly_score",
        ascending=False
    )
    .head(20)
)

display_columns = [
    "patientunitstayid",
    "hour",
    "anomaly_score",
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "heart_rate_change_1h",
    "spo2_change_1h",
    "mean_bp_change_1h",
    "net_fluid_balance_ml",
    "urine_output_ml",
    "active_infusion_count",
    "vasoactive_support",
    "clinical_warning_count_v2",
    "clinical_severe_count_v2",
    "clinical_rule_severity_v2",
    "clinical_rule_explanation_v2"
]

available_columns = [
    col
    for col in display_columns
    if col in ml_only.columns
]

if len(ml_only) > 0:

    print(
        ml_only[
            available_columns
        ]
        .round(2)
        .to_string(index=False)
    )

else:

    print("No ML-only anomalies found.")

# ------------------------------------------------------------
# TOP COMBINED ANOMALIES
# ------------------------------------------------------------

print("\n============================================================")
print("TOP ML + RULE ANOMALIES")
print("============================================================")

combined = (
    merged[
        merged["ml_and_rule"]
    ]
    .sort_values(
        [
            "clinical_severe_count_v2",
            "anomaly_score"
        ],
        ascending=False
    )
    .head(20)
)

if len(combined) > 0:

    print(
        combined[
            available_columns
        ]
        .round(2)
        .to_string(index=False)
    )

else:

    print(
        "No ML + clinical-rule anomalies found."
    )

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

merged.to_csv(
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