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
    "deployment_anomaly_scores.csv"
)

RISK_FILE = (
    PROCESSED_DIR /
    "deployment_deterioration_risk.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "smart_icu_deployment_master.csv"
)

print("Loading patient state...")
state = pd.read_csv(
    STATE_FILE,
    low_memory=False
)

print("Loading clinical rules...")
rules = pd.read_csv(
    RULE_FILE,
    low_memory=False
)

print("Loading deployment anomaly scores...")
anomaly = pd.read_csv(
    ANOMALY_FILE,
    low_memory=False
)

print("Loading deployment deterioration risk...")
risk = pd.read_csv(
    RISK_FILE,
    low_memory=False
)

# ------------------------------------------------------------
# SELECT COLUMNS
# ------------------------------------------------------------

rules = rules[
    [
        "patientunitstayid",
        "hour",
        "clinical_warning_count_v2",
        "clinical_severe_count_v2",
        "clinical_rule_severity_v2",
        "clinical_rule_explanation_v2"
    ]
].copy()

anomaly = anomaly[
    [
        "patientunitstayid",
        "hour",
        "anomaly_score",
        "anomaly_flag"
    ]
].copy()

risk = risk[
    [
        "patientunitstayid",
        "hour",
        "risk_probability",
        "risk_band"
    ]
].copy()

# ------------------------------------------------------------
# MERGE
# ------------------------------------------------------------

print("\nMerging deployment layers...")

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

master["anomaly_score"] = pd.to_numeric(
    master["anomaly_score"],
    errors="coerce"
)

master["anomaly_flag"] = (
    pd.to_numeric(
        master["anomaly_flag"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
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

master["risk_high"] = (
    master["risk_probability"] >= 0.10
)

master["risk_very_high"] = (
    master["risk_probability"] >= 0.25
)

master["risk_extreme"] = (
    master["risk_probability"] >= 0.50
)

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n============================================================")
print("DEPLOYMENT MASTER VALIDATION")
print("============================================================")

print(
    "Rows:",
    len(master)
)

print(
    "ICU stays:",
    master[
        "patientunitstayid"
    ].nunique()
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
    master[
        "anomaly_score"
    ].isna().sum()
)

print(
    "Missing risk probabilities:",
    master[
        "risk_probability"
    ].isna().sum()
)

print(
    "Missing clinical severity:",
    master[
        "clinical_rule_severity_v2"
    ].isna().sum()
)

# ------------------------------------------------------------
# SIGNAL COUNTS
# ------------------------------------------------------------

print("\n============================================================")
print("SIGNAL COUNTS")
print("============================================================")

print(
    "ML anomalies:",
    int(
        master["ml_anomaly"].sum()
    )
)

print(
    "Clinical warnings:",
    int(
        master["clinical_warning"].sum()
    )
)

print(
    "Clinical severe:",
    int(
        master["clinical_severe"].sum()
    )
)

print(
    "Risk >=10%:",
    int(
        master["risk_high"].sum()
    )
)

print(
    "Risk >=25%:",
    int(
        master["risk_very_high"].sum()
    )
)

print(
    "Risk >=50%:",
    int(
        master["risk_extreme"].sum()
    )
)

# ------------------------------------------------------------
# CROSS-SIGNAL COUNTS
# ------------------------------------------------------------

print("\n============================================================")
print("CROSS-SIGNAL COUNTS")
print("============================================================")

print(
    "ML + clinical warning:",
    int(
        (
            master["ml_anomaly"] &
            master["clinical_warning"]
        ).sum()
    )
)

print(
    "ML + severe clinical:",
    int(
        (
            master["ml_anomaly"] &
            master["clinical_severe"]
        ).sum()
    )
)

print(
    "Clinical severe + risk >=25%:",
    int(
        (
            master["clinical_severe"] &
            master["risk_very_high"]
        ).sum()
    )
)

print(
    "ML + severe clinical + risk >=25%:",
    int(
        (
            master["ml_anomaly"] &
            master["clinical_severe"] &
            master["risk_very_high"]
        ).sum()
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
        "risk_band"
    ]
    .value_counts()
    .to_string()
)

# ------------------------------------------------------------
# CLINICAL SEVERITY DISTRIBUTION
# ------------------------------------------------------------

print("\n============================================================")
print("CLINICAL SEVERITY DISTRIBUTION")
print("============================================================")

print(
    master[
        "clinical_rule_severity_v2"
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