import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_final_alerts.csv"
)

print("Loading final alerts...")
df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Rows:", len(df))
print("ICU stays:", df["patientunitstayid"].nunique())

# ------------------------------------------------------------
# 1. PRIORITY DISTRIBUTION
# ------------------------------------------------------------

print("\n============================================================")
print("FINAL PRIORITY DISTRIBUTION")
print("============================================================")

priority_order = [
    "normal",
    "low",
    "moderate",
    "high",
    "critical"
]

priority_counts = (
    df["final_alert_priority"]
    .value_counts()
    .reindex(
        priority_order,
        fill_value=0
    )
)

priority_pct = (
    priority_counts /
    len(df) *
    100
)

summary = pd.DataFrame({
    "hours": priority_counts,
    "percentage": priority_pct.round(2)
})

print(summary.to_string())

# ------------------------------------------------------------
# 2. HIGH / CRITICAL DRIVER ANALYSIS
# ------------------------------------------------------------

print("\n============================================================")
print("HIGH / CRITICAL DRIVER ANALYSIS")
print("============================================================")

severe_alerts = df[
    df["final_alert_priority"].isin(
        ["high", "critical"]
    )
].copy()

print(
    "High + Critical hours:",
    len(severe_alerts)
)

print(
    "High + Critical patients:",
    severe_alerts[
        "patientunitstayid"
    ].nunique()
)

print("\nClinical rule severity:")

print(
    severe_alerts[
        "clinical_rule_severity_v2"
    ]
    .value_counts()
    .to_string()
)

print("\nAlert type:")

print(
    severe_alerts[
        "alert_type"
    ]
    .value_counts()
    .to_string()
)

# ------------------------------------------------------------
# 3. MOST COMMON EXPLANATIONS
# ------------------------------------------------------------

print("\n============================================================")
print("MOST COMMON HIGH/CRITICAL EXPLANATIONS")
print("============================================================")

explanation_counts = (
    severe_alerts[
        "final_alert_explanation"
    ]
    .value_counts()
    .head(30)
)

print(
    explanation_counts.to_string()
)

# ------------------------------------------------------------
# 4. CRITICAL DRIVER BREAKDOWN
# ------------------------------------------------------------

print("\n============================================================")
print("CRITICAL ALERT DRIVER BREAKDOWN")
print("============================================================")

critical = df[
    df["final_alert_priority"] == "critical"
].copy()

print(
    "Critical hours:",
    len(critical)
)

print(
    critical[
        "clinical_rule_explanation_v2"
    ]
    .value_counts()
    .head(30)
    .to_string()
)

# ------------------------------------------------------------
# 5. HIGH ALERT DRIVER BREAKDOWN
# ------------------------------------------------------------

print("\n============================================================")
print("HIGH ALERT DRIVER BREAKDOWN")
print("============================================================")

high = df[
    df["final_alert_priority"] == "high"
].copy()

print(
    "High hours:",
    len(high)
)

print(
    high[
        "clinical_rule_explanation_v2"
    ]
    .value_counts()
    .head(30)
    .to_string()
)

# ------------------------------------------------------------
# 6. ALERTS PER PATIENT
# ------------------------------------------------------------

print("\n============================================================")
print("ALERT BURDEN PER PATIENT")
print("============================================================")

patient_alerts = (
    df[
        df["final_alert_priority"].isin(
            ["moderate", "high", "critical"]
        )
    ]
    .groupby("patientunitstayid")
    .agg(
        alert_hours=(
            "final_alert_priority",
            "size"
        ),
        high_hours=(
            "final_alert_priority",
            lambda x: (x == "high").sum()
        ),
        critical_hours=(
            "final_alert_priority",
            lambda x: (x == "critical").sum()
        )
    )
    .sort_values(
        [
            "critical_hours",
            "high_hours",
            "alert_hours"
        ],
        ascending=False
    )
)

print(
    patient_alerts.head(20).to_string()
)

# ------------------------------------------------------------
# 7. CONSECUTIVE HIGH/CRITICAL RUNS
# ------------------------------------------------------------

print("\n============================================================")
print("CONSECUTIVE HIGH/CRITICAL RUNS")
print("============================================================")

df_sorted = df.sort_values(
    [
        "patientunitstayid",
        "hour"
    ]
).copy()

df_sorted["severe_flag"] = (
    df_sorted["final_alert_priority"]
    .isin(["high", "critical"])
)

# Group consecutive hourly severe alerts.
df_sorted["previous_severe"] = (
    df_sorted
    .groupby("patientunitstayid")["severe_flag"]
    .shift(1)
    .fillna(False)
)

df_sorted["new_run"] = (
    df_sorted["severe_flag"] &
    ~df_sorted["previous_severe"]
)

df_sorted["run_id"] = (
    df_sorted
    .groupby("patientunitstayid")["new_run"]
    .cumsum()
)

runs = (
    df_sorted[
        df_sorted["severe_flag"]
    ]
    .groupby(
        [
            "patientunitstayid",
            "run_id"
        ]
    )
    .agg(
        start_hour=("hour", "min"),
        end_hour=("hour", "max"),
        hours=("hour", "size"),
        max_priority=(
            "final_alert_priority",
            lambda x:
                "critical"
                if (x == "critical").any()
                else "high"
        )
    )
    .reset_index()
)

print(
    "Number of severe-alert runs:",
    len(runs)
)

print("\nRun-length distribution:")

print(
    runs["hours"]
    .describe()
    .round(2)
    .to_string()
)

print("\nLongest 20 runs:")

print(
    runs.sort_values(
        "hours",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 8. CHECK HIGH/CRITICAL WITHOUT CLINICAL WARNING
# ------------------------------------------------------------

print("\n============================================================")
print("HIGH/CRITICAL WITHOUT CLINICAL WARNING")
print("============================================================")

no_rule_severe = severe_alerts[
    severe_alerts["clinical_warning_count_v2"] == 0
]

print(
    "High/Critical with zero clinical warnings:",
    len(no_rule_severe)
)

if len(no_rule_severe) > 0:

    print(
        no_rule_severe[
            [
                "patientunitstayid",
                "hour",
                "final_alert_priority",
                "anomaly_flag",
                "anomaly_score",
                "current_abnormality_count",
                "trend_signal_count",
                "active_infusion_count",
                "vasoactive_support",
                "net_fluid_balance_ml",
                "final_alert_explanation"
            ]
        ]
        .sort_values(
            "anomaly_score",
            ascending=False
        )
        .head(30)
        .round(2)
        .to_string(index=False)
    )

# ------------------------------------------------------------
# 9. VALIDATION
# ------------------------------------------------------------

print("\n============================================================")
print("VALIDATION")
print("============================================================")

print(
    "Duplicate patient-hours:",
    df.duplicated(
        subset=[
            "patientunitstayid",
            "hour"
        ]
    ).sum()
)

print(
    "Missing final priority:",
    df["final_alert_priority"].isna().sum()
)

print(
    "Missing final explanation:",
    df["final_alert_explanation"].isna().sum()
)

print("\nAudit complete.")