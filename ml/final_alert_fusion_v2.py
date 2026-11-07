import pandas as pd
import numpy as np
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
    "hourly_patient_state_final_alerts.csv"
)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("Loading ML anomaly results...")
ml = pd.read_csv(
    ANOMALY_FILE,
    low_memory=False
)

print("Loading clinical rules v2...")
rules = pd.read_csv(
    RULE_FILE,
    low_memory=False
)

print("ML rows:", len(ml))
print("Rule rows:", len(rules))

# ------------------------------------------------------------
# SELECT RULE DATA
# ------------------------------------------------------------

rule_columns = [
    "patientunitstayid",
    "hour",
    "clinical_warning_count_v2",
    "clinical_severe_count_v2",
    "clinical_rule_severity_v2",
    "clinical_rule_explanation_v2"
]

rules_subset = rules[rule_columns].copy()

# ------------------------------------------------------------
# MERGE
# ------------------------------------------------------------

df = ml.merge(
    rules_subset,
    on=[
        "patientunitstayid",
        "hour"
    ],
    how="left",
    validate="one_to_one"
)

# ------------------------------------------------------------
# CLEAN TYPES
# ------------------------------------------------------------

df["clinical_warning_count_v2"] = (
    pd.to_numeric(
        df["clinical_warning_count_v2"],
        errors="coerce"
    )
    .fillna(0)
)

df["clinical_severe_count_v2"] = (
    pd.to_numeric(
        df["clinical_severe_count_v2"],
        errors="coerce"
    )
    .fillna(0)
)

df["anomaly_flag"] = (
    pd.to_numeric(
        df["anomaly_flag"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

df["anomaly_score"] = pd.to_numeric(
    df["anomaly_score"],
    errors="coerce"
)

# ------------------------------------------------------------
# BASIC SIGNALS
# ------------------------------------------------------------

df["ml_anomaly"] = (
    df["anomaly_flag"] == 1
)

df["rule_warning"] = (
    df["clinical_warning_count_v2"] >= 1
)

df["rule_severe"] = (
    df["clinical_severe_count_v2"] >= 1
)

# ------------------------------------------------------------
# CURRENT PHYSIOLOGICAL ABNORMALITIES
# ------------------------------------------------------------

current_signal_columns = [
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp"
]

# Count explicitly abnormal current values.
# These are aligned with the rule layer.

df["current_abnormality_count"] = 0

df["current_abnormality_count"] += (
    (df["heart_rate"] < 50) |
    (df["heart_rate"] > 120)
).fillna(False).astype(int)

df["current_abnormality_count"] += (
    (df["spo2"] < 92)
).fillna(False).astype(int)

df["current_abnormality_count"] += (
    (df["respiration"] < 10) |
    (df["respiration"] > 30)
).fillna(False).astype(int)

df["current_abnormality_count"] += (
    (df["systolic_bp"] < 90) |
    (df["systolic_bp"] > 180)
).fillna(False).astype(int)

df["current_abnormality_count"] += (
    (df["mean_bp"] < 65)
).fillna(False).astype(int)

# ------------------------------------------------------------
# TREND SIGNALS
# ------------------------------------------------------------

df["trend_signal_count"] = 0

df["trend_signal_count"] += (
    (
        df["heart_rate_change_1h"] > 20
    ) |
    (
        df["heart_rate_change_1h"] < -20
    )
).fillna(False).astype(int)

df["trend_signal_count"] += (
    df["spo2_change_1h"] < -3
).fillna(False).astype(int)

df["trend_signal_count"] += (
    df["mean_bp_change_1h"] < -15
).fillna(False).astype(int)

# ------------------------------------------------------------
# TREATMENT CONTEXT
# ------------------------------------------------------------

df["treatment_support"] = (
    (
        df["active_infusion_count"] >= 1
    ) |
    (
        df["vasoactive_support"] == 1
    )
).fillna(False)

# ------------------------------------------------------------
# FLUID CONTEXT
# ------------------------------------------------------------

df["fluid_concern"] = (
    (
        df["net_fluid_balance_ml"].abs() >= 500
    )
).fillna(False)

# ------------------------------------------------------------
# MULTI-SIGNAL SCORE
# ------------------------------------------------------------

df["multi_signal_count"] = (
    (
        df["current_abnormality_count"] > 0
    ).astype(int)
    +
    (
        df["trend_signal_count"] > 0
    ).astype(int)
    +
    df["treatment_support"].astype(int)
    +
    df["fluid_concern"].astype(int)
)

# ------------------------------------------------------------
# FINAL PRIORITY
# ------------------------------------------------------------

def determine_priority(row):

    severe = row["rule_severe"]
    warning = row["rule_warning"]
    ml_anomaly = row["ml_anomaly"]

    current_count = row[
        "current_abnormality_count"
    ]

    trend_count = row[
        "trend_signal_count"
    ]

    treatment = row[
        "treatment_support"
    ]

    multi_signal = row[
        "multi_signal_count"
    ]

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    if (
        row["clinical_rule_severity_v2"]
        == "critical"
    ):
        return "critical"

    if (
        severe
        and ml_anomaly
    ):
        return "critical"

    if (
        severe
        and current_count >= 2
    ):
        return "critical"

    if (
        current_count >= 3
        and trend_count >= 1
    ):
        return "critical"

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if severe:
        return "high"

    if (
        warning
        and ml_anomaly
    ):
        return "high"

    if (
        current_count >= 2
        and trend_count >= 1
    ):
        return "high"

    if (
        current_count >= 2
        and treatment
    ):
        return "high"

    if (
        multi_signal >= 3
    ):
        return "high"

    # --------------------------------------------------------
    # MODERATE
    # --------------------------------------------------------

    if (
        warning
        and trend_count >= 1
    ):
        return "moderate"

    if (
        warning
        and treatment
    ):
        return "moderate"

    if (
        ml_anomaly
        and warning
    ):
        return "moderate"

    if (
        current_count >= 1
        and trend_count >= 1
    ):
        return "moderate"

    if (
        multi_signal >= 2
    ):
        return "moderate"

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    if warning:
        return "low"

    if ml_anomaly:
        return "low"

    if current_count >= 1:
        return "low"

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    return "normal"


df["final_alert_priority"] = df.apply(
    determine_priority,
    axis=1
)

# ------------------------------------------------------------
# EXPLANATION
# ------------------------------------------------------------

def build_explanation(row):

    parts = []

    rule_explanation = str(
        row["clinical_rule_explanation_v2"]
    )

    if (
        rule_explanation
        and rule_explanation != "No rule-based abnormality"
        and rule_explanation != "nan"
    ):
        parts.append(rule_explanation)

    if row["ml_anomaly"]:
        parts.append(
            f"ML anomaly detected "
            f"(score {row['anomaly_score']:.3f})"
        )

    if (
        row["current_abnormality_count"] > 0
        and not rule_explanation
    ):
        parts.append(
            "abnormal physiological state"
        )

    if (
        row["trend_signal_count"] > 0
    ):
        parts.append(
            "recent physiological deterioration"
        )

    if row["treatment_support"]:
        parts.append(
            "active treatment support"
        )

    if row["fluid_concern"]:
        parts.append(
            "significant fluid-balance change"
        )

    if not parts:
        return "No significant abnormality detected"

    return " | ".join(parts)


df["final_alert_explanation"] = df.apply(
    build_explanation,
    axis=1
)

# ------------------------------------------------------------
# ALERT TYPE
# ------------------------------------------------------------

def classify_alert_type(row):

    if row["ml_anomaly"] and row["rule_warning"]:
        return "ML + clinical"

    if row["ml_anomaly"]:
        return "ML anomaly"

    if row["rule_warning"]:
        return "Clinical rule"

    if row["current_abnormality_count"] > 0:
        return "Physiological"

    return "None"


df["alert_type"] = df.apply(
    classify_alert_type,
    axis=1
)

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n============================================================")
print("FINAL ALERT FUSION")
print("============================================================")

print(
    "Rows:",
    len(df)
)

print(
    "ICU stays:",
    df["patientunitstayid"].nunique()
)

print(
    "Duplicate patient-hours:",
    df.duplicated(
        subset=[
            "patientunitstayid",
            "hour"
        ]
    ).sum()
)

print("\nFinal alert priorities:")

print(
    df["final_alert_priority"]
    .value_counts()
    .reindex(
        [
            "normal",
            "low",
            "moderate",
            "high",
            "critical"
        ],
        fill_value=0
    )
    .to_string()
)

print("\nAlert types:")

print(
    df["alert_type"]
    .value_counts()
    .to_string()
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

print(
    "Output:",
    OUTPUT_FILE
)