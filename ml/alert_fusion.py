import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "hourly_patient_state_rules.csv"
OUTPUT_FILE = PROCESSED_DIR / "hourly_patient_state_alerts.csv"


# ============================================================
# LOAD
# ============================================================

print("Loading clinical rule dataset...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Input shape:", df.shape)
print("ICU stays:", df["patientunitstayid"].nunique())


# ============================================================
# CURRENT PHYSIOLOGICAL ABNORMALITY
# ============================================================

print("\nCreating current-abnormality indicators...")


# Heart rate
df["current_hr_abnormal"] = (
    (df["heart_rate"] < 50) |
    (df["heart_rate"] > 120)
).astype(int)


# Oxygenation
df["current_spo2_abnormal"] = (
    df["spo2"] < 92
).astype(int)


# Respiratory rate
df["current_resp_abnormal"] = (
    (df["respiration"] < 10) |
    (df["respiration"] > 30)
).astype(int)


# Blood pressure
df["current_bp_abnormal"] = (
    (df["systolic_bp"] < 90) |
    (df["mean_bp"] < 65)
).astype(int)


# Severe current abnormalities
df["severe_current_abnormality"] = (
    (df["heart_rate"] < 40) |
    (df["heart_rate"] > 140) |
    (df["spo2"] < 88) |
    (df["respiration"] < 8) |
    (df["respiration"] > 35) |
    (df["systolic_bp"] < 80) |
    (df["mean_bp"] < 55)
).astype(int)


# Count current abnormalities
current_columns = [
    "current_hr_abnormal",
    "current_spo2_abnormal",
    "current_resp_abnormal",
    "current_bp_abnormal"
]

df["current_abnormality_count"] = (
    df[current_columns].sum(axis=1)
)


# ============================================================
# TREND DETERIORATION
# ============================================================

print("Creating trend-deterioration indicators...")


df["trend_hr_deterioration"] = (
    (df["heart_rate_change_1h"] > 20) |
    (df["heart_rate_change_1h"] < -20)
).astype(int)


df["trend_spo2_deterioration"] = (
    df["spo2_change_1h"] < -3
).astype(int)


df["trend_map_deterioration"] = (
    df["mean_bp_change_1h"] < -15
).astype(int)


trend_columns = [
    "trend_hr_deterioration",
    "trend_spo2_deterioration",
    "trend_map_deterioration"
]

df["trend_deterioration_count"] = (
    df[trend_columns].sum(axis=1)
)


# ============================================================
# TREATMENT CONTEXT
# ============================================================

print("Creating treatment-context indicators...")


df["treatment_context"] = (
    (df["vasoactive_support"] > 0) |
    (df["active_infusion_count"] >= 2)
).astype(int)


# ============================================================
# FLUID CONTEXT
# ============================================================

print("Creating fluid-context indicators...")


df["fluid_context"] = (
    (df["net_fluid_balance_ml"] >= 500) |
    (df["net_fluid_balance_ml"] <= -500)
).astype(int)


# ============================================================
# COMBINED DETERIORATION
# ============================================================

print("Creating combined deterioration logic...")


# Current abnormality + concerning trend
df["combined_current_and_trend"] = (
    (df["current_abnormality_count"] >= 1) &
    (df["trend_deterioration_count"] >= 1)
).astype(int)


# Multiple current physiological abnormalities
df["multiple_current_abnormalities"] = (
    df["current_abnormality_count"] >= 2
).astype(int)


# Current abnormality + treatment support
df["abnormality_with_treatment_support"] = (
    (df["current_abnormality_count"] >= 1) &
    (df["treatment_context"] == 1)
).astype(int)


# Strong multi-signal agreement
df["multi_signal_deterioration"] = (
    (
        (df["current_abnormality_count"] >= 1) &
        (df["trend_deterioration_count"] >= 1) &
        (df["treatment_context"] == 1)
    )
).astype(int)


# ============================================================
# ALERT PRIORITY
# ============================================================

print("Assigning alert priorities...")


df["alert_priority"] = np.select(
    [
        # Critical:
        # Severe current abnormality + another concerning signal
        (
            (df["severe_current_abnormality"] == 1) &
            (
                (df["trend_deterioration_count"] >= 1) |
                (df["treatment_context"] == 1)
            )
        ),

        # High:
        # Strong agreement between multiple signal types
        (
            (df["multi_signal_deterioration"] == 1) |
            (
                (df["severe_current_abnormality"] == 1) &
                (df["current_abnormality_count"] >= 1)
            )
        ),

        # Moderate:
        # Current abnormality + trend
        (df["combined_current_and_trend"] == 1),

        # Moderate:
        # Multiple simultaneous current abnormalities
        (df["multiple_current_abnormalities"] == 1),

        # Low:
        # Single current abnormality
        (df["current_abnormality_count"] == 1),

        # Low:
        # Trend without current abnormality
        (df["trend_deterioration_count"] >= 1),

        # Low:
        # Treatment context by itself
        (df["treatment_context"] == 1),

        # Low:
        # Fluid context by itself
        (df["fluid_context"] == 1)
    ],
    [
        "critical",
        "high",
        "moderate",
        "moderate",
        "low",
        "low",
        "low",
        "low"
    ],
    default="normal"
)


# ============================================================
# ALERT BOOLEAN
# ============================================================

df["alert_active"] = (
    df["alert_priority"] != "normal"
).astype(int)


# ============================================================
# ALERT EXPLANATION
# ============================================================

print("Creating human-readable explanations...")


def build_alert_explanation(row):

    reasons = []

    # Current physiology
    if row["current_hr_abnormal"] == 1:
        if row["heart_rate"] < 50:
            reasons.append("low heart rate")
        elif row["heart_rate"] > 120:
            reasons.append("high heart rate")

    if row["current_spo2_abnormal"] == 1:
        reasons.append("low SpO2")

    if row["current_resp_abnormal"] == 1:
        if pd.notna(row["respiration"]):
            if row["respiration"] < 10:
                reasons.append("low respiratory rate")
            elif row["respiration"] > 30:
                reasons.append("high respiratory rate")

    if row["current_bp_abnormal"] == 1:
        if pd.notna(row["mean_bp"]) and row["mean_bp"] < 65:
            reasons.append("low MAP")
        elif pd.notna(row["systolic_bp"]) and row["systolic_bp"] < 90:
            reasons.append("low systolic BP")

    # Trends
    if row["trend_hr_deterioration"] == 1:
        if pd.notna(row["heart_rate_change_1h"]):
            if row["heart_rate_change_1h"] > 20:
                reasons.append("rapid HR rise")
            elif row["heart_rate_change_1h"] < -20:
                reasons.append("rapid HR fall")

    if row["trend_spo2_deterioration"] == 1:
        reasons.append("falling SpO2")

    if row["trend_map_deterioration"] == 1:
        reasons.append("falling MAP")

    # Treatment
    if row["vasoactive_support"] > 0:
        reasons.append("vasoactive support")

    if row["active_infusion_count"] >= 2:
        reasons.append("multiple active infusions")

    # Fluid
    if row["net_fluid_balance_ml"] >= 500:
        reasons.append("positive fluid balance")

    if row["net_fluid_balance_ml"] <= -500:
        reasons.append("negative fluid balance")

    if not reasons:
        return "No significant rule-based concern"

    return "; ".join(reasons)


df["alert_explanation"] = df.apply(
    build_alert_explanation,
    axis=1
)


# ============================================================
# VALIDATION
# ============================================================

print("\n============================================================")
print("ALERT FUSION VALIDATION")
print("============================================================")

duplicates = df.duplicated(
    subset=["patientunitstayid", "hour"]
).sum()

print("Duplicate patient-hours:", duplicates)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("ICU stays:", df["patientunitstayid"].nunique())


if duplicates > 0:
    raise ValueError("Duplicate patient-hour rows detected.")


# ============================================================
# PRIORITY DISTRIBUTION
# ============================================================

print("\n============================================================")
print("ALERT PRIORITY DISTRIBUTION")
print("============================================================")

print(
    df["alert_priority"]
    .value_counts()
    .to_string()
)


# ============================================================
# CURRENT ABNORMALITY DISTRIBUTION
# ============================================================

print("\n============================================================")
print("CURRENT ABNORMALITY")
print("============================================================")

print(
    "Rows with >=1 current abnormality:",
    int(
        (df["current_abnormality_count"] >= 1).sum()
    )
)

print(
    "Rows with >=2 current abnormalities:",
    int(
        (df["current_abnormality_count"] >= 2).sum()
    )
)

print(
    "Rows with severe current abnormality:",
    int(
        (df["severe_current_abnormality"] == 1).sum()
    )
)


# ============================================================
# TREND DISTRIBUTION
# ============================================================

print("\n============================================================")
print("TREND DETERIORATION")
print("============================================================")

print(
    "Rows with >=1 concerning trend:",
    int(
        (df["trend_deterioration_count"] >= 1).sum()
    )
)

print(
    "Rows with >=2 concerning trends:",
    int(
        (df["trend_deterioration_count"] >= 2).sum()
    )
)


# ============================================================
# COMBINED SIGNALS
# ============================================================

print("\n============================================================")
print("MULTI-SIGNAL FUSION")
print("============================================================")

print(
    "Current + trend:",
    int(
        df["combined_current_and_trend"].sum()
    )
)

print(
    "Multiple current abnormalities:",
    int(
        df["multiple_current_abnormalities"].sum()
    )
)

print(
    "Abnormality + treatment support:",
    int(
        df["abnormality_with_treatment_support"].sum()
    )
)

print(
    "Strong multi-signal deterioration:",
    int(
        df["multi_signal_deterioration"].sum()
    )
)


# ============================================================
# EXAMPLES
# ============================================================

print("\n============================================================")
print("EXAMPLE FUSED ALERTS")
print("============================================================")

example_columns = [
    "patientunitstayid",
    "hour",
    "heart_rate",
    "spo2",
    "respiration",
    "mean_bp",
    "heart_rate_change_1h",
    "spo2_change_1h",
    "mean_bp_change_1h",
    "vasoactive_support",
    "alert_priority",
    "alert_explanation"
]

examples = df.loc[
    df["alert_active"] == 1,
    example_columns
].head(15)

print(
    examples.to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n============================================================")
print("SUCCESS")
print("============================================================")

print("Output:", OUTPUT_FILE)
print("Rows:", len(df))
print("Columns:", len(df.columns))