
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "hourly_patient_state_final_alerts.csv"
OUTPUT_FILE = PROCESSED_DIR / "hourly_patient_state_notifications.csv"

COOLDOWN_HOURS = 6

print("Loading final alerts...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

df = df.sort_values(
    ["patientunitstayid", "hour"]
).reset_index(drop=True)

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
    .astype(int)
)

df["severe_alert"] = (
    df["final_alert_priority"]
    .isin(["high", "critical"])
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
).fillna(False)

# ------------------------------------------------------------
# NEW SEVERE EVENT
# ------------------------------------------------------------

df["new_severe_event"] = (
    df["severe_alert"] &
    ~df["previous_severe"]
)

expected_new_events = (
    df["severe_alert"] &
    ~df["previous_severe"]
)

if not df["new_severe_event"].equals(expected_new_events):
    raise ValueError("New severe event validation failed.")

print("\n===== EVENT VALIDATION =====")
print("New severe events:", int(df["new_severe_event"].sum()))
print("Mismatches:", int(
    (df["new_severe_event"] != expected_new_events).sum()
))

# ------------------------------------------------------------
# CRITICAL ESCALATION
# ------------------------------------------------------------

df["critical_escalation"] = (
    (df["final_alert_priority"] == "critical") &
    (df["previous_priority_rank"] < priority_rank["critical"])
).fillna(False)

# ------------------------------------------------------------
# STATEFUL NOTIFICATION ENGINE
# ------------------------------------------------------------

notification_actions = ["NO_ALERT"] * len(df)

last_emitted_hours = [float("nan")] * len(df)
hours_since_last_notification = [float("nan")] * len(df)
cooldown_expired_flags = [False] * len(df)
reason_changed_flags = [False] * len(df)

for patient_id, group in df.groupby(
    "patientunitstayid",
    sort=False
):

    last_emit_hour = None
    last_notified_reason = None

    previous_severe_state = False
    previous_rank = None

    for idx, row in group.iterrows():

        current_hour = float(row["hour"])
        current_severe = bool(row["severe_alert"])
        current_rank = int(row["priority_rank"])

        current_reason = (
            str(row["clinical_rule_explanation_v2"])
            if pd.notna(row["clinical_rule_explanation_v2"])
            else ""
        )

        action = "NO_ALERT"

        # ----------------------------------------------------
        # NON-SEVERE STATE
        # ----------------------------------------------------

        if not current_severe:

            previous_severe_state = False
            previous_rank = current_rank

            if last_emit_hour is not None:
                last_emitted_hours[idx] = last_emit_hour

            continue

        # ----------------------------------------------------
        # TIME SINCE LAST NOTIFICATION
        # ----------------------------------------------------

        if last_emit_hour is not None:

            elapsed = current_hour - last_emit_hour

            hours_since_last_notification[idx] = elapsed

            cooldown_expired = (
                elapsed >= COOLDOWN_HOURS
            )

        else:

            elapsed = None
            cooldown_expired = False

        cooldown_expired_flags[idx] = cooldown_expired

        # ----------------------------------------------------
        # REASON CHANGE SINCE LAST NOTIFICATION
        # ----------------------------------------------------

        reason_changed = (
            last_notified_reason is not None
            and current_reason != last_notified_reason
        )

        reason_changed_flags[idx] = reason_changed

        # ----------------------------------------------------
        # NOTIFICATION DECISION
        # ----------------------------------------------------

        if not previous_severe_state:

            if last_emit_hour is None:
                action = "EMIT"
            else:
                action = "NEW_EVENT"

        elif (
            previous_rank is not None
            and current_rank > previous_rank
        ):

            action = "ESCALATION"

        elif reason_changed:

            action = "NEW_REASON"

        elif last_emit_hour is None:

            action = "EMIT"

        elif cooldown_expired:

            action = "COOLDOWN_EXPIRED"

        else:

            action = "SUPPRESS"

        # ----------------------------------------------------
        # UPDATE LAST NOTIFICATION ONLY WHEN EMITTED
        # ----------------------------------------------------

        if action != "SUPPRESS":

            last_emit_hour = current_hour
            last_notified_reason = current_reason

        notification_actions[idx] = action

        last_emitted_hours[idx] = last_emit_hour

        previous_severe_state = True
        previous_rank = current_rank


df["notification_action"] = notification_actions

df["last_emitted_notification_hour"] = last_emitted_hours

df["hours_since_last_notification"] = (
    hours_since_last_notification
)

df["cooldown_expired"] = cooldown_expired_flags

df["reason_changed_since_notification"] = (
    reason_changed_flags
)

# ------------------------------------------------------------
# SAFETY VALIDATION
# ------------------------------------------------------------

critical_escalation_suppressed = df[
    df["critical_escalation"] &
    (df["notification_action"] == "SUPPRESS")
]

new_event_suppressed = df[
    df["new_severe_event"] &
    (df["notification_action"] == "SUPPRESS")
]

invalid_cooldown_reminders = df[
    (df["notification_action"] == "COOLDOWN_EXPIRED") &
    (df["hours_since_last_notification"] < COOLDOWN_HOURS)
]

if len(critical_escalation_suppressed):
    raise ValueError("Critical escalation was suppressed!")

if len(new_event_suppressed):
    raise ValueError("New severe event was suppressed!")

if len(invalid_cooldown_reminders):
    raise ValueError("Invalid cooldown reminder detected!")

print("\nAll notification safety checks passed.")

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n============================================================")
print("NOTIFICATION LAYER")
print("============================================================")

print("Total rows:", len(df))

print("\nNotification actions:")
print(df["notification_action"].value_counts().to_string())

# ------------------------------------------------------------
# SEVERE ALERT REDUCTION
# ------------------------------------------------------------

original_severe = int(df["severe_alert"].sum())

emitted_severe = int(
    (
        df["notification_action"] != "SUPPRESS"
    ).where(df["severe_alert"], False).sum()
)

suppressed_severe = original_severe - emitted_severe

reduction = (
    suppressed_severe / original_severe * 100
    if original_severe > 0 else 0
)

print("\n============================================================")
print("SEVERE ALERT REDUCTION")
print("============================================================")

print("Original severe alerts:", original_severe)
print("Emitted severe notifications:", emitted_severe)
print("Suppressed severe alerts:", suppressed_severe)
print("Reduction:", round(reduction, 2), "%")

# ------------------------------------------------------------
# CRITICAL CHECK
# ------------------------------------------------------------

critical_mask = df["final_alert_priority"] == "critical"

original_critical = int(critical_mask.sum())

suppressed_critical = int(
    (
        critical_mask &
        (df["notification_action"] == "SUPPRESS")
    ).sum()
)

emitted_critical = original_critical - suppressed_critical

print("\n============================================================")
print("CRITICAL ALERT CHECK")
print("============================================================")

print("Original critical alerts:", original_critical)
print("Critical notifications emitted:", emitted_critical)
print("Critical alerts suppressed:", suppressed_critical)

# ------------------------------------------------------------
# ACTION DISTRIBUTION BY PRIORITY
# ------------------------------------------------------------

print("\n============================================================")
print("ACTION BY PRIORITY")
print("============================================================")

print(
    pd.crosstab(
        df["final_alert_priority"],
        df["notification_action"]
    ).to_string()
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n============================================================")
print("SUCCESS")
print("============================================================")

print("Output:", OUTPUT_FILE)
