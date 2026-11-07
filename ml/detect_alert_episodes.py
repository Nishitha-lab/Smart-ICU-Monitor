import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_final_alerts.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_alert_episodes.csv"
)

print("Loading final alerts...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Rows:", len(df))
print(
    "ICU stays:",
    df["patientunitstayid"].nunique()
)

# ------------------------------------------------------------
# PREPARE
# ------------------------------------------------------------

df = df.sort_values(
    [
        "patientunitstayid",
        "hour"
    ]
).copy()

# Severe alert = HIGH or CRITICAL
df["severe_alert"] = (
    df["final_alert_priority"]
    .isin(["high", "critical"])
)

# ------------------------------------------------------------
# DETECT EPISODES
# ------------------------------------------------------------
#
# Allow gaps of up to 2 ICU hours.
#
# Example:
# 20, 21, 23, 24 -> SAME episode
# 20, 21, 24       -> DIFFERENT episodes
#
# This handles intermittent ICU measurements.

df["previous_hour"] = (
    df.groupby("patientunitstayid")["hour"]
    .shift(1)
)

df["hour_gap"] = (
    df["hour"] -
    df["previous_hour"]
)

df["previous_severe"] = (
    df.groupby("patientunitstayid")["severe_alert"]
    .shift(1)
    .fillna(False)
)

df["new_episode"] = (
    df["severe_alert"] &
    (
        ~df["previous_severe"] |
        (df["hour_gap"] > 2)
    )
)

df["episode_id"] = (
    df.groupby("patientunitstayid")["new_episode"]
    .cumsum()
)

# ------------------------------------------------------------
# EXTRACT SEVERE ALERT ROWS
# ------------------------------------------------------------

severe = df[
    df["severe_alert"]
].copy()

print("\nSevere alert rows:", len(severe))

# ------------------------------------------------------------
# BUILD EPISODES
# ------------------------------------------------------------

episodes = (
    severe
    .groupby(
        [
            "patientunitstayid",
            "episode_id"
        ]
    )
    .agg(
        start_hour=(
            "hour",
            "min"
        ),
        end_hour=(
            "hour",
            "max"
        ),
        alert_hours=(
            "hour",
            "size"
        ),

        peak_priority=(
            "final_alert_priority",
            lambda x:
                "critical"
                if (x == "critical").any()
                else "high"
        ),

        max_anomaly_score=(
            "anomaly_score",
            "max"
        ),

        max_warning_count=(
            "clinical_warning_count_v2",
            "max"
        ),

        max_severe_count=(
            "clinical_severe_count_v2",
            "max"
        ),

        max_current_abnormalities=(
            "current_abnormality_count",
            "max"
        ),

        max_trend_signals=(
            "trend_signal_count",
            "max"
        ),

        max_active_infusions=(
            "active_infusion_count",
            "max"
        ),

        vasoactive_present=(
            "vasoactive_support",
            "max"
        )
    )
    .reset_index()
)

# ------------------------------------------------------------
# EPISODE DURATION
# ------------------------------------------------------------

episodes["episode_span_hours"] = (
    episodes["end_hour"] -
    episodes["start_hour"] +
    1
)

episodes["intermittent_gap_hours"] = (
    episodes["episode_span_hours"] -
    episodes["alert_hours"]
)

# ------------------------------------------------------------
# EPISODE SEVERITY
# ------------------------------------------------------------

severity_rank = {
    "high": 1,
    "critical": 2
}

episodes["severity_rank"] = (
    episodes["peak_priority"]
    .map(severity_rank)
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n============================================================")
print("ALERT EPISODE SUMMARY")
print("============================================================")

print(
    "Severe alert rows:",
    len(severe)
)

print(
    "Alert episodes:",
    len(episodes)
)

print(
    "Patients with severe episodes:",
    episodes[
        "patientunitstayid"
    ].nunique()
)

print("\nEpisode peak priority:")

print(
    episodes[
        "peak_priority"
    ]
    .value_counts()
    .to_string()
)

# ------------------------------------------------------------
# EPISODE LENGTH
# ------------------------------------------------------------

print("\n============================================================")
print("EPISODE LENGTH")
print("============================================================")

print(
    episodes[
        "episode_span_hours"
    ]
    .describe()
    .round(2)
    .to_string()
)

print("\nLongest episodes:")

print(
    episodes
    .sort_values(
        "episode_span_hours",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

# ------------------------------------------------------------
# INTERMITTENT EPISODES
# ------------------------------------------------------------

print("\n============================================================")
print("INTERMITTENT EPISODES")
print("============================================================")

intermittent = episodes[
    episodes["intermittent_gap_hours"] > 0
]

print(
    "Episodes containing gaps:",
    len(intermittent)
)

print(
    "Percentage:",
    round(
        len(intermittent) /
        len(episodes) *
        100,
        2
    ),
    "%"
)

# ------------------------------------------------------------
# REDUCTION POTENTIAL
# ------------------------------------------------------------

print("\n============================================================")
print("ALERT REDUCTION POTENTIAL")
print("============================================================")

original_severe_alerts = len(severe)
episode_count = len(episodes)

if original_severe_alerts > 0:

    reduction = (
        1 -
        episode_count /
        original_severe_alerts
    ) * 100

else:

    reduction = 0

print(
    "Original severe alert rows:",
    original_severe_alerts
)

print(
    "Potential unique episodes:",
    episode_count
)

print(
    "Potential reduction:",
    round(reduction, 2),
    "%"
)

# ------------------------------------------------------------
# TOP PATIENTS BY EPISODE BURDEN
# ------------------------------------------------------------

print("\n============================================================")
print("PATIENTS WITH MOST SEVERE EPISODES")
print("============================================================")

patient_summary = (
    episodes
    .groupby("patientunitstayid")
    .agg(
        episodes=(
            "episode_id",
            "size"
        ),
        total_alert_hours=(
            "alert_hours",
            "sum"
        ),
        critical_episodes=(
            "peak_priority",
            lambda x:
                (x == "critical").sum()
        ),
        longest_episode=(
            "episode_span_hours",
            "max"
        )
    )
    .sort_values(
        [
            "critical_episodes",
            "total_alert_hours"
        ],
        ascending=False
    )
)

print(
    patient_summary
    .head(20)
    .to_string()
)

# ------------------------------------------------------------
# SAVE EPISODES
# ------------------------------------------------------------

episodes.to_csv(
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