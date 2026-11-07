import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_final_alerts.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_notifications.csv"
)

COOLDOWN_HOURS = 6

print("Loading final alerts...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

df = df.sort_values(
    [
        "patientunitstayid",
        "hour"
    ]
).copy()

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

df["priority_rank"] = (
    df["final_alert_priority"]
    .map(priority_rank)
    .fillna(0)
)

df["severe_alert"] = (
    df["final_alert_priority"]
    .isin(
        ["high", "critical"]
    )
)

# ------------------------------------------------------------
# PREVIOUS STATE
# ------------------------------------------------------------

grouped = df.groupby("patientunitstayid", sort=False)

df["previous_hour"] = grouped["hour"].shift(1)

df["hours_since_previous"] = (
    df["hour"] - df["previous_hour"]
)

df["previous_priority_rank"] = (
    grouped["priority_rank"].shift(1)
)

df["previous_severe"] = (
    grouped["severe_alert"]
    .shift(1)
    .fillna(False)
    .astype(bool)
)

df["previous_explanation"] = (
    grouped["clinical_rule_explanation_v2"].shift(1)
)

# ------------------------------------------------------------
# REASON CHANGE
# ------------------------------------------------------------

current_reason = (
    df["clinical_rule_explanation_v2"]
    .fillna("")
    .astype(str)
)

previous_reason = (
    df["previous_explanation"]
    .fillna("")
    .astype(str)
)

df["reason_changed"] = (
    current_reason != previous_reason
)

# ------------------------------------------------------------
# SEVERITY ESCALATION
# ------------------------------------------------------------

df["severity_escalation"] = (
    df["priority_rank"] >
    df["previous_priority_rank"]
)

# ------------------------------------------------------------
# NEW SEVERE EVENT
# ------------------------------------------------------------

df["new_severe_event"] = (
    df["severe_alert"] &
    ~df["previous_severe"]
)

# Independent consistency check
expected_new_events = (
    df["severe_alert"] &
    ~df["previous_severe"]
)

mismatches = (
    df["new_severe_event"] != expected_new_events
).sum()

print("\n===== EVENT VALIDATION =====")
print("New severe events:", df["new_severe_event"].sum())
print("Mismatches:", mismatches)

if mismatches != 0:
    raise ValueError("New severe event validation failed.")

print(
    "Previous severe:",
    df.loc[df["severe_alert"], "previous_severe"].sum()
)

# ------------------------------------------------------------
# COOLDOWN EXPIRY
# ------------------------------------------------------------

df["cooldown_expired"] = (
    df["hours_since_previous"] >=
    COOLDOWN_HOURS
)

# ------------------------------------------------------------
# CRITICAL ESCALATION
# ------------------------------------------------------------

df["critical_escalation"] = (
    (
        df["final_alert_priority"]
        == "critical"
    )
    &
    (
        df["previous_priority_rank"]
        < priority_rank["critical"]
    )
)

# ------------------------------------------------------------
# DECISION LOGIC
# ------------------------------------------------------------
#
# ALWAYS emit:
#   - first severe alert
#   - new severe event after a non-severe state
#   - severity escalation
#   - critical escalation
#   - materially changed reason
#   - cooldown expired
#
# Otherwise suppress repeated severe alerts.

def notification_decision(row):

    if not row["severe_alert"]:
        return "NO_ALERT"

    if pd.isna(
        row["previous_priority_rank"]
    ):
        return "EMIT"

    if row["critical_escalation"]:
        return "ESCALATION"

    if row["severity_escalation"]:
        return "ESCALATION"

    if row["new_severe_event"]:
        return "NEW_EVENT"

    if row["reason_changed"]:
        return "NEW_REASON"

    if row["cooldown_expired"]:
        return "COOLDOWN_EXPIRED"

    return "SUPPRESS"


df["notification_action"] = df.apply(
    notification_decision,
    axis=1
)

# ------------------------------------------------------------
# IMPORTANT SAFETY CHECK
# ------------------------------------------------------------

# A critical row must NEVER be suppressed when it represents
# an escalation from a lower priority state.

critical_suppressed = df[
    (
        df["final_alert_priority"]
        == "critical"
    )
    &
    (
        df["notification_action"]
        == "SUPPRESS"
    )
    &
    (
        df["severity_escalation"]
    )
]

if len(critical_suppressed) > 0:

    raise ValueError(
        "Critical escalation was suppressed."
    )

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n============================================================")
print("NOTIFICATION LAYER")
print("============================================================")

print(
    "Total rows:",
    len(df)
)

print("\nNotification actions:")

print(
    df["notification_action"]
    .value_counts()
    .to_string()
)

# ------------------------------------------------------------
# SEVERE ALERT REDUCTION
# ------------------------------------------------------------

original_severe = int(
    df["severe_alert"].sum()
)

emitted_severe = int(
    (
        (
            df["notification_action"] != "SUPPRESS"
        )
        &
        df["severe_alert"]
    ).sum()
)

suppressed_severe = (
    original_severe -
    emitted_severe
)

print("\n============================================================")
print("SEVERE ALERT REDUCTION")
print("============================================================")

print(
    "Original severe alerts:",
    original_severe
)

print(
    "Emitted severe notifications:",
    emitted_severe
)

print(
    "Suppressed severe alerts:",
    suppressed_severe
)

if original_severe > 0:

    reduction = (
        suppressed_severe /
        original_severe *
        100
    )

else:

    reduction = 0

print(
    "Reduction:",
    round(reduction, 2),
    "%"
)

# ------------------------------------------------------------
# CRITICAL CHECK
# ------------------------------------------------------------

original_critical = int(
    (
        df["final_alert_priority"]
        == "critical"
    ).sum()
)

suppressed_critical = int(
    (
        (df["final_alert_priority"] == "critical")
        &
        (df["notification_action"] == "SUPPRESS")
    ).sum()
)

emitted_critical = (
    original_critical -
    suppressed_critical
)

print("\n============================================================")
print("CRITICAL ALERT CHECK")
print("============================================================")

print(
    "Original critical alerts:",
    original_critical
)

print(
    "Critical notifications emitted:",
    emitted_critical
)

print(
    "Critical alerts suppressed:",
    suppressed_critical
)

# ------------------------------------------------------------
# ACTION DISTRIBUTION BY PRIORITY
# ------------------------------------------------------------

print("\n============================================================")
print("ACTION BY PRIORITY")
print("============================================================")

action_priority = pd.crosstab(
    df["final_alert_priority"],
    df["notification_action"]
)

print(
    action_priority.to_string()
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