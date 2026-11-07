import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

STATE_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_alerts.csv"
)

RULE_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_rules_v2.csv"
)

ANOMALY_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_anomalies_v2.csv"
)

RISK_FILE = (
    PROCESSED_DIR /
    "deterioration_probability_predictions.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "smart_icu_master_alert_dataset.csv"
)

print("Loading patient state...")
state = pd.read_csv(
    STATE_FILE,
    low_memory=False
)

print("Loading clinical rules v2...")
rules = pd.read_csv(
    RULE_FILE,
    low_memory=False
)

print("Loading anomaly results...")
anomaly = pd.read_csv(
    ANOMALY_FILE,
    low_memory=False
)

print("Loading deterioration risk...")
risk = pd.read_csv(
    RISK_FILE,
    low_memory=False
)

# ------------------------------------------------------------
# SELECT RULE DATA
# ------------------------------------------------------------

rule_cols = [
    "patientunitstayid",
    "hour",
    "clinical_warning_count_v2",
    "clinical_severe_count_v2",
    "clinical_rule_severity_v2",
    "clinical_rule_explanation_v2"
]

rules = rules[rule_cols].copy()

# ------------------------------------------------------------
# SELECT ANOMALY DATA
# ------------------------------------------------------------

anomaly_cols = [
    "patientunitstayid",
    "hour",
    "anomaly_score",
    "anomaly_flag"
]

anomaly = anomaly[anomaly_cols].copy()

# ------------------------------------------------------------
# SELECT RISK DATA
# ------------------------------------------------------------

risk_cols = [
    "patientunitstayid",
    "hour",
    "risk_probability"
]

risk = risk[risk_cols].copy()

# ------------------------------------------------------------
# MASTER DATASET
# ------------------------------------------------------------

print("\nMerging datasets...")

master = state.merge(
    rules,
    on=[
        "patientunitstayid",
        "hour"
    ],
    how="left",
    validate="one_to_one"
)

master = master.merge(
    anomaly,
    on=[
        "patientunitstayid",
        "hour"
    ],
    how="left",
    validate="one_to_one"
)

master = master.merge(
    risk,
    on=[
        "patientunitstayid",
        "hour"
    ],
    how="left",
    validate="one_to_one"
)

# ------------------------------------------------------------
# TYPES
# ------------------------------------------------------------

master["anomaly_flag"] = (
    pd.to_numeric(
        master["anomaly_flag"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

master["anomaly_score"] = pd.to_numeric(
    master["anomaly_score"],
    errors="coerce"
)

master["risk_probability"] = pd.to_numeric(
    master["risk_probability"],
    errors="coerce"
)

master["clinical_warning_count_v2"] = (
    pd.to_numeric(
        master["clinical_warning_count_v2"],
        errors="coerce"
    )
    .fillna(0)
)

master["clinical_severe_count_v2"] = (
    pd.to_numeric(
        master["clinical_severe_count_v2"],
        errors="coerce"
    )
    .fillna(0)
)

# ------------------------------------------------------------
# RISK BAND
# ------------------------------------------------------------

def risk_band(value):

    if pd.isna(value):
        return "unknown"

    if value < 0.01:
        return "very_low"

    if value < 0.02:
        return "low"

    if value < 0.05:
        return "elevated"

    if value < 0.10:
        return "high"

    if value < 0.25:
        return "very_high"

    return "extreme"


master["deterioration_risk_band"] = (
    master["risk_probability"]
    .apply(risk_band)
)

# ------------------------------------------------------------
# BASIC SIGNAL FLAGS
# ------------------------------------------------------------

master["ml_anomaly"] = (
    master["anomaly_flag"] == 1
)

master["clinical_warning"] = (
    master["clinical_warning_count_v2"] >= 1
)

master["clinical_severe"] = (
    master["clinical_severe_count_v2"] >= 1
)

master["high_risk_signal"] = (
    master["risk_probability"] >= 0.10
)

master["very_high_risk_signal"] = (
    master["risk_probability"] >= 0.25
)

master["extreme_risk_signal"] = (
    master["risk_probability"] >= 0.50
)

# ------------------------------------------------------------
# CONSISTENCY CHECK
# ------------------------------------------------------------

print("\n============================================================")
print("MASTER DATASET VALIDATION")
print("============================================================")

print(
    "Rows:",
    len(master)
)

print(
    "ICU stays:",
    master["patientunitstayid"].nunique()
)

print(
    "Duplicate patient-hours:",
    master.duplicated(
        subset=[
            "patientunitstayid",
            "hour"
        ]
    ).sum()
)

print(
    "Missing anomaly scores:",
    master["anomaly_score"].isna().sum()
)

print(
    "Missing risk probabilities:",
    master["risk_probability"].isna().sum()
)

print(
    "Clinical warning rows:",
    int(
        master["clinical_warning"].sum()
    )
)

print(
    "Clinical severe rows:",
    int(
        master["clinical_severe"].sum()
    )
)

print(
    "ML anomaly rows:",
    int(
        master["ml_anomaly"].sum()
    )
)

print(
    "Risk >=10% rows:",
    int(
        master["high_risk_signal"].sum()
    )
)

print(
    "Risk >=25% rows:",
    int(
        master["very_high_risk_signal"].sum()
    )
)

print(
    "Risk >=50% rows:",
    int(
        master["extreme_risk_signal"].sum()
    )
)

# ------------------------------------------------------------
# RISK BAND DISTRIBUTION
# ------------------------------------------------------------

print("\n============================================================")
print("RISK BAND DISTRIBUTION")
print("============================================================")

print(
    master[
        "deterioration_risk_band"
    ]
    .value_counts()
    .to_string()
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

master.to_csv(
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