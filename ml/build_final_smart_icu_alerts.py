import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DIR /
    "smart_icu_deployment_master.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "smart_icu_final_alerts.csv"
)

print("Loading Smart-ICU deployment master...")
df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

# ------------------------------------------------------------
# PRIORITY RANK
# ------------------------------------------------------------

priority_rank = {
    "normal": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4
}

rank_priority = {
    0: "normal",
    1: "low",
    2: "moderate",
    3: "high",
    4: "critical"
}

df["clinical_severity_rank"] = (
    df["clinical_rule_severity_v2"]
    .map(priority_rank)
    .fillna(0)
)

# ------------------------------------------------------------
# SIGNAL FLAGS
# ------------------------------------------------------------

df["risk_10"] = (
    df["risk_probability"] >= 0.10
)

df["risk_25"] = (
    df["risk_probability"] >= 0.25
)

df["risk_50"] = (
    df["risk_probability"] >= 0.50
)

df["ml_anomaly"] = (
    df["anomaly_flag"] == 1
)

df["clinical_warning"] = (
    df["clinical_warning_count_v2"] >= 1
)

df["clinical_severe"] = (
    df["clinical_severe_count_v2"] >= 1
)

# ------------------------------------------------------------
# MULTI-SIGNAL COUNT
# ------------------------------------------------------------

df["intelligence_signal_count"] = (
    df["ml_anomaly"].astype(int)
    +
    df["clinical_warning"].astype(int)
    +
    df["risk_10"].astype(int)
)

# ------------------------------------------------------------
# FINAL ALERT PRIORITY
# ------------------------------------------------------------

def calculate_priority(row):

    clinical_rank = int(
        row["clinical_severity_rank"]
    )

    ml = bool(row["ml_anomaly"])
    warning = bool(row["clinical_warning"])

    risk10 = bool(row["risk_10"])
    risk25 = bool(row["risk_25"])
    risk50 = bool(row["risk_50"])

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    # Existing critical clinical state always wins.
    if clinical_rank >= 4:
        return "critical"

    # Severe clinical state + very high risk.
    if (
        clinical_rank >= 3
        and risk25
    ):
        return "critical"

    # Severe clinical state + anomaly.
    if (
        clinical_rank >= 3
        and ml
    ):
        return "critical"

    # Multiple independent signals + very high risk.
    if (
        risk25
        and ml
        and warning
    ):
        return "critical"

    # Extreme risk combined with any clinical concern.
    if (
        risk50
        and warning
    ):
        return "critical"

    # --------------------------------------------------------
    # HIGH
    # --------------------------------------------------------

    if clinical_rank >= 3:
        return "high"

    if (
        clinical_rank >= 2
        and risk25
    ):
        return "high"

    if (
        clinical_rank >= 2
        and ml
    ):
        return "high"

    if (
        clinical_rank >= 1
        and risk25
    ):
        return "high"

    if (
        risk25
        and ml
    ):
        return "high"

    if (
        risk50
    ):
        return "high"

    # --------------------------------------------------------
    # MODERATE
    # --------------------------------------------------------

    if (
        clinical_rank >= 2
    ):
        return "moderate"

    if (
        clinical_rank >= 1
        and risk10
    ):
        return "moderate"

    if (
        clinical_rank >= 1
        and ml
    ):
        return "moderate"

    if (
        risk10
        and ml
    ):
        return "moderate"

    if (
        risk25
    ):
        return "moderate"

    # --------------------------------------------------------
    # LOW
    # --------------------------------------------------------

    if clinical_rank >= 1:
        return "low"

    if ml:
        return "low"

    if risk10:
        return "low"

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    return "normal"


df["final_alert_priority"] = df.apply(
    calculate_priority,
    axis=1
)

# ------------------------------------------------------------
# EXPLANATION
# ------------------------------------------------------------

def build_explanation(row):

    pieces = []

    clinical = str(
        row["clinical_rule_explanation_v2"]
    )

    if (
        clinical
        and clinical != "No rule-based abnormality"
        and clinical != "nan"
    ):
        pieces.append(clinical)

    if row["ml_anomaly"]:

        score = row["anomaly_score"]

        pieces.append(
            f"ML anomaly detected "
            f"(score {score:.3f})"
        )

    if row["risk_probability"] >= 0.10:

        risk = (
            row["risk_probability"] * 100
        )

        pieces.append(
            f"24h deterioration risk "
            f"{risk:.1f}%"
        )

    if row["active_infusion_count"] > 0:

        pieces.append(
            f"{int(row['active_infusion_count'])} "
            f"active infusion(s)"
        )

    if row["vasoactive_support"] == 1:

        pieces.append(
            "vasoactive support"
        )

    if abs(
        row["net_fluid_balance_ml"]
    ) >= 500:

        pieces.append(
            f"fluid balance "
            f"{row['net_fluid_balance_ml']:.0f} mL"
        )

    if not pieces:

        return (
            "No significant abnormality detected"
        )

    return " | ".join(pieces)


df["final_alert_explanation"] = df.apply(
    build_explanation,
    axis=1
)

# ------------------------------------------------------------
# ALERT CATEGORY
# ------------------------------------------------------------

def classify_category(row):

    if (
        row["clinical_severe"]
        and row["ml_anomaly"]
        and row["risk_25"]
    ):
        return "multi-signal"

    if (
        row["clinical_severe"]
        and row["risk_25"]
    ):
        return "clinical + risk"

    if (
        row["clinical_severe"]
        and row["ml_anomaly"]
    ):
        return "clinical + anomaly"

    if (
        row["ml_anomaly"]
        and row["risk_25"]
    ):
        return "anomaly + risk"

    if row["clinical_warning"]:
        return "clinical"

    if row["ml_anomaly"]:
        return "anomaly"

    if row["risk_10"]:
        return "risk"

    return "none"


df["alert_category"] = df.apply(
    classify_category,
    axis=1
)

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

print("\n============================================================")
print("FINAL SMART-ICU ALERT ENGINE")
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

print("\nFinal priority distribution:")

priority_counts = (
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
)

priority_percent = (
    priority_counts /
    len(df) *
    100
)

summary = pd.DataFrame({
    "hours": priority_counts,
    "percentage": priority_percent.round(2)
})

print(
    summary.to_string()
)

# ------------------------------------------------------------
# CATEGORY DISTRIBUTION
# ------------------------------------------------------------

print("\n============================================================")
print("ALERT CATEGORY DISTRIBUTION")
print("============================================================")

print(
    df[
        "alert_category"
    ]
    .value_counts()
    .to_string()
)

# ------------------------------------------------------------
# MULTI-SIGNAL ALERTS
# ------------------------------------------------------------

print("\n============================================================")
print("MULTI-SIGNAL ALERTS")
print("============================================================")

multi = df[
    (
        df["alert_category"]
        == "multi-signal"
    )
]

print(
    "Multi-signal hours:",
    len(multi)
)

print(
    "Multi-signal patients:",
    multi[
        "patientunitstayid"
    ].nunique()
)

# ------------------------------------------------------------
# TOP CRITICAL ALERTS
# ------------------------------------------------------------

print("\n============================================================")
print("TOP CRITICAL ALERTS")
print("============================================================")

critical = df[
    df["final_alert_priority"]
    == "critical"
].copy()

critical = critical.sort_values(
    [
        "risk_probability",
        "anomaly_score"
    ],
    ascending=False
)

display_columns = [
    "patientunitstayid",
    "hour",
    "final_alert_priority",
    "alert_category",
    "clinical_rule_severity_v2",
    "clinical_rule_explanation_v2",
    "anomaly_score",
    "risk_probability",
    "active_infusion_count",
    "vasoactive_support",
    "net_fluid_balance_ml",
    "final_alert_explanation"
]

print(
    critical[
        display_columns
    ]
    .head(20)
    .round(3)
    .to_string(index=False)
)

# ------------------------------------------------------------
# SAFETY VALIDATION
# ------------------------------------------------------------

print("\n============================================================")
print("SAFETY VALIDATION")
print("============================================================")

# A clinical CRITICAL state must never become lower than
# CRITICAL in the final engine.

critical_clinical = df[
    df["clinical_rule_severity_v2"]
    == "critical"
]

downgraded = critical_clinical[
    critical_clinical[
        "final_alert_priority"
    ] != "critical"
]

print(
    "Clinical critical rows:",
    len(critical_clinical)
)

print(
    "Clinical critical rows downgraded:",
    len(downgraded)
)

if len(downgraded) > 0:
    raise ValueError(
        "CRITICAL clinical alerts were downgraded."
    )

print(
    "Critical safety check: PASSED"
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