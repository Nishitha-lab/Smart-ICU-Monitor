import pandas as pd
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

# ------------------------------------------------------------
# PREPARE
# ------------------------------------------------------------

df = df.sort_values(
    [
        "patientunitstayid",
        "hour"
    ]
).copy()

df["is_severe"] = (
    df["final_alert_priority"]
    .isin(["high", "critical"])
)

df["priority_rank"] = (
    df["final_alert_priority"]
    .map({
        "normal": 0,
        "low": 1,
        "moderate": 2,
        "high": 3,
        "critical": 4
    })
)

# A simplified clinical signature.
# Same explanation = same underlying reason.
df["alert_signature"] = (
    df["clinical_rule_explanation_v2"]
    .fillna("")
    .astype(str)
)

# ------------------------------------------------------------
# SIMULATION FUNCTION
# ------------------------------------------------------------

def simulate_cooldown(data, cooldown_hours):

    emitted = []
    
    last_alert_hour = {}
    last_alert_rank = {}
    last_signature = {}

    for row in data.itertuples(index=False):

        if not row.is_severe:
            emitted.append(False)
            continue

        patient = row.patientunitstayid
        hour = row.hour
        rank = row.priority_rank
        signature = row.alert_signature

        if patient not in last_alert_hour:
            emitted.append(True)

            last_alert_hour[patient] = hour
            last_alert_rank[patient] = rank
            last_signature[patient] = signature

            continue

        elapsed = (
            hour -
            last_alert_hour[patient]
        )

        previous_rank = last_alert_rank[patient]
        previous_signature = last_signature[patient]

        # Always emit if severity increases.
        escalation = (
            rank > previous_rank
        )

        # Emit if the clinical reason changes.
        reason_changed = (
            signature != previous_signature
        )

        # Emit if cooldown has expired.
        cooldown_expired = (
            elapsed >= cooldown_hours
        )

        should_emit = (
            escalation
            or reason_changed
            or cooldown_expired
        )

        emitted.append(should_emit)

        if should_emit:
            last_alert_hour[patient] = hour
            last_alert_rank[patient] = rank
            last_signature[patient] = signature

    return pd.Series(
        emitted,
        index=data.index
    )


# ------------------------------------------------------------
# ORIGINAL
# ------------------------------------------------------------

severe_count = int(
    df["is_severe"].sum()
)

print("\n============================================================")
print("ORIGINAL ALERT BURDEN")
print("============================================================")

print(
    "High/Critical alert hours:",
    severe_count
)

print(
    "Patients with High/Critical alerts:",
    df.loc[
        df["is_severe"],
        "patientunitstayid"
    ].nunique()
)

# ------------------------------------------------------------
# TEST POLICIES
# ------------------------------------------------------------

print("\n============================================================")
print("COOLDOWN SIMULATION")
print("============================================================")

for cooldown in [3, 6, 12]:

    emitted = simulate_cooldown(
        df,
        cooldown
    )

    severe_emitted = (
        emitted &
        df["is_severe"]
    )

    emitted_count = int(
        severe_emitted.sum()
    )

    reduction = (
        1 -
        emitted_count /
        severe_count
    ) * 100

    print(
        f"\nCooldown: {cooldown} hours"
    )

    print(
        "Emitted severe alerts:",
        emitted_count
    )

    print(
        "Suppressed severe alerts:",
        severe_count - emitted_count
    )

    print(
        "Reduction:",
        round(reduction, 2),
        "%"
    )

    # Patient-level burden
    patient_counts = (
        df.loc[
            severe_emitted
        ]
        .groupby(
            "patientunitstayid"
        )
        .size()
    )

    print(
        "Patients still generating alerts:",
        len(patient_counts)
    )

    print(
        "Median alerts/patient:",
        round(
            patient_counts.median(),
            2
        )
        if len(patient_counts)
        else 0
    )

    print(
        "Maximum alerts/patient:",
        int(
            patient_counts.max()
        )
        if len(patient_counts)
        else 0
    )

# ------------------------------------------------------------
# DETAILED 6-HOUR POLICY
# ------------------------------------------------------------

print("\n============================================================")
print("6-HOUR POLICY DETAILED ANALYSIS")
print("============================================================")

df["emit_6h"] = simulate_cooldown(
    df,
    6
)

six_hour = df[
    df["is_severe"] &
    df["emit_6h"]
].copy()

print(
    "Original severe alerts:",
    severe_count
)

print(
    "6-hour emitted alerts:",
    len(six_hour)
)

print(
    "6-hour suppression:",
    severe_count - len(six_hour)
)

print(
    "Reduction:",
    round(
        (
            1 -
            len(six_hour) /
            severe_count
        ) * 100,
        2
    ),
    "%"
)

# ------------------------------------------------------------
# ESCALATIONS PRESERVED
# ------------------------------------------------------------

print("\n============================================================")
print("ESCALATION PRESERVATION")
print("============================================================")

df["previous_priority_rank"] = (
    df.groupby(
        "patientunitstayid"
    )["priority_rank"]
    .shift(1)
)

df["escalation"] = (
    df["priority_rank"] >
    df["previous_priority_rank"]
)

# Recreate the 6-hour emitted subset AFTER
# adding the escalation column.
six_hour = df[
    df["is_severe"] &
    df["emit_6h"]
].copy()

severe_escalations = df[
    df["is_severe"] &
    df["escalation"]
]

emitted_escalations = six_hour[
    six_hour["escalation"]
]
print(
    "Severe escalation events:",
    len(severe_escalations)
)

print(
    "Escalations preserved by 6-hour policy:",
    len(emitted_escalations)
)

# ------------------------------------------------------------
# CRITICAL PRESERVATION
# ------------------------------------------------------------

print("\n============================================================")
print("CRITICAL ALERT PRESERVATION")
print("============================================================")

original_critical = (
    df["final_alert_priority"]
    == "critical"
).sum()

emitted_critical = (
    six_hour["final_alert_priority"]
    == "critical"
).sum()

print(
    "Original critical alerts:",
    int(original_critical)
)

print(
    "6-hour emitted critical alerts:",
    int(emitted_critical)
)

# ------------------------------------------------------------
# SAVE SIMULATION
# ------------------------------------------------------------

OUTPUT_FILE = (
    PROCESSED_DIR /
    "alert_suppression_simulation.csv"
)

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