import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

ANOMALY_FILE = PROCESSED_DIR / "hourly_patient_state_anomalies_v2.csv"
RULE_FILE = PROCESSED_DIR / "hourly_patient_state_rules.csv"
OUTPUT_FILE = PROCESSED_DIR / "anomaly_v2_evaluation.csv"

print("Loading anomaly results...")
anomalies = pd.read_csv(ANOMALY_FILE, low_memory=False)

print("Loading clinical rules...")
rules = pd.read_csv(RULE_FILE, low_memory=False)

print("Anomaly rows:", len(anomalies))
print("Rule rows:", len(rules))

# ------------------------------------------------------------
# LOAD RELEVANT CLINICAL RULE FIELDS
# ------------------------------------------------------------

rule_columns = [
    "patientunitstayid",
    "hour",
    "clinical_warning_count",
    "clinical_severe_count",
    "clinical_rule_severity",
    "clinical_rule_explanation"
]

missing_columns = [
    col for col in rule_columns
    if col not in rules.columns
]

if missing_columns:
    raise ValueError(
        f"Missing clinical rule columns: {missing_columns}"
    )

rule_subset = rules[rule_columns].copy()

# ------------------------------------------------------------
# MERGE
# ------------------------------------------------------------

merged = anomalies.merge(
    rule_subset,
    on=["patientunitstayid", "hour"],
    how="left",
    validate="one_to_one"
)

# ------------------------------------------------------------
# NUMERIC CONVERSION
# ------------------------------------------------------------

merged["clinical_warning_count"] = (
    pd.to_numeric(
        merged["clinical_warning_count"],
        errors="coerce"
    )
    .fillna(0)
)

merged["clinical_severe_count"] = (
    pd.to_numeric(
        merged["clinical_severe_count"],
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
    merged["clinical_warning_count"] >= 1
)

merged["rule_severe"] = (
    merged["clinical_severe_count"] >= 1
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
print("ML + CLINICAL RULE AGREEMENT")
print("============================================================")

print("Total rows:", len(merged))

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

ml_count = merged["ml_anomaly"].sum()

if ml_count > 0:

    supported_pct = (
        merged.loc[
            merged["ml_anomaly"],
            "rule_abnormal"
        ].mean() * 100
    )

    unsupported_pct = 100 - supported_pct

else:
    supported_pct = 0
    unsupported_pct = 0

print(
    "ML anomalies supported by clinical rules:",
    round(supported_pct, 2),
    "%"
)

print(
    "ML anomalies not supported by clinical rules:",
    round(unsupported_pct, 2),
    "%"
)

# ------------------------------------------------------------
# ML ANOMALIES BY CLINICAL SEVERITY
# ------------------------------------------------------------

print("\n============================================================")
print("ML ANOMALIES BY CLINICAL RULE SEVERITY")
print("============================================================")

severity_summary = (
    merged[
        merged["ml_anomaly"]
    ]
    .groupby(
        "clinical_rule_severity",
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
# ML ANOMALIES BY RULE CATEGORY
# ------------------------------------------------------------

print("\n============================================================")
print("ML ANOMALIES WITH CLINICAL WARNINGS")
print("============================================================")

supported = merged[
    merged["ml_and_rule"]
].copy()

if len(supported) > 0:

    print(
        supported[
            "clinical_rule_explanation"
        ]
        .value_counts()
        .head(20)
        .to_string()
    )

else:

    print("No ML anomalies overlap with clinical warnings.")

# ------------------------------------------------------------
# TOP ML-ONLY ANOMALIES
# ------------------------------------------------------------

print("\n============================================================")
print("TOP 20 ML-ONLY ANOMALIES")
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
    "clinical_warning_count",
    "clinical_severe_count",
    "clinical_rule_severity",
    "clinical_rule_explanation"
]

available_display_columns = [
    col
    for col in display_columns
    if col in ml_only.columns
]

if len(ml_only) > 0:

    print(
        ml_only[
            available_display_columns
        ]
        .round(2)
        .to_string(index=False)
    )

else:

    print("No ML-only anomalies found.")

# ------------------------------------------------------------
# TOP ML + RULE ANOMALIES
# ------------------------------------------------------------

print("\n============================================================")
print("TOP 20 ML + CLINICAL-RULE ANOMALIES")
print("============================================================")

both = (
    merged[
        merged["ml_and_rule"]
    ]
    .sort_values(
        [
            "clinical_severe_count",
            "anomaly_score"
        ],
        ascending=False
    )
    .head(20)
)

if len(both) > 0:

    print(
        both[
            available_display_columns
        ]
        .round(2)
        .to_string(index=False)
    )

else:

    print(
        "No rows where ML anomaly and clinical warning overlap."
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

print("Output:", OUTPUT_FILE)