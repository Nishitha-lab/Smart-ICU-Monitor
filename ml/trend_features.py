import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "hourly_patient_state.csv"
OUTPUT_FILE = PROCESSED_DIR / "hourly_patient_state_trends.csv"


# ============================================================
# Load
# ============================================================

print("Loading unified hourly patient state...")

df = pd.read_csv(INPUT_FILE)

print("Input shape:", df.shape)
print("ICU stays:", df["patientunitstayid"].nunique())


# ============================================================
# Sort patient timelines
# ============================================================

df = df.sort_values(
    ["patientunitstayid", "hour"]
).reset_index(drop=True)


# ============================================================
# Variables for temporal features
# ============================================================

trend_variables = [
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "total_intake_ml",
    "total_output_ml",
    "net_fluid_balance_ml",
    "urine_output_ml"
]


# ============================================================
# Validate variables
# ============================================================

missing_variables = [
    col for col in trend_variables
    if col not in df.columns
]

if missing_variables:
    raise ValueError(
        f"Missing trend variables: {missing_variables}"
    )


# ============================================================
# Detect timeline gaps
# ============================================================

df["hour_gap"] = (
    df.groupby("patientunitstayid")["hour"]
    .diff()
)

print("\n============================================================")
print("TIMELINE GAP AUDIT")
print("============================================================")

gap_count = (df["hour_gap"] > 1).sum()

print(
    "Rows with gap > 1 hour:",
    gap_count
)

print(
    "Percentage:",
    round(gap_count / len(df) * 100, 2),
    "%"
)

print(
    "Maximum gap:",
    df["hour_gap"].max()
)


# ============================================================
# Time-aware change features
#
# Instead of blindly using diff(3), we only calculate a
# 3-hour change when the current observation is actually
# 3 hours after the previous relevant observation.
#
# For 1h / 3h / 6h:
# - exact elapsed-hour matching is required
# - otherwise feature remains NaN
# ============================================================

print("\n============================================================")
print("CREATING TIME-AWARE TEMPORAL FEATURES")
print("============================================================")


for col in trend_variables:

    print(f"Processing: {col}")

    grouped = df.groupby(
        "patientunitstayid",
        sort=False
    )

    for hours in [1, 3, 6]:

        value_previous = grouped[col].shift(
            1
        )

        time_previous = grouped["hour"].shift(
            1
        )

        elapsed = (
            df["hour"] - time_previous
        )

        change = (
            df[col] - value_previous
        )

        df[f"{col}_change_{hours}h"] = np.where(
            elapsed == hours,
            change,
            np.nan
        )

        # ----------------------------------------------------
        # For 3h and 6h changes, search for an exact
        # observation at t-hours using a self-merge.
        # ----------------------------------------------------

        if hours in [3, 6]:

            lookup = df[
                [
                    "patientunitstayid",
                    "hour",
                    col
                ]
            ].copy()

            lookup["target_hour"] = (
                lookup["hour"] + hours
            )

            lookup = lookup.rename(
                columns={
                    col: f"{col}_previous_{hours}h"
                }
            )

            lookup = lookup[
                [
                    "patientunitstayid",
                    "target_hour",
                    f"{col}_previous_{hours}h"
                ]
            ]

            df = df.merge(
                lookup,
                left_on=[
                    "patientunitstayid",
                    "hour"
                ],
                right_on=[
                    "patientunitstayid",
                    "target_hour"
                ],
                how="left"
            )

            # The merge above finds t-hours by using
            # target_hour = previous_hour + hours.
            #
            # Correct it by looking backward from current hour.

            df[f"{col}_previous_{hours}h"] = (
                df[f"{col}_previous_{hours}h"]
            )

            # Recalculate using a direct indexed lookup
            # to ensure exact patient/hour matching.

            previous_lookup = (
                df[
                    [
                        "patientunitstayid",
                        "hour",
                        col
                    ]
                ]
                .drop_duplicates(
                    ["patientunitstayid", "hour"]
                )
                .set_index(
                    ["patientunitstayid", "hour"]
                )[col]
            )

            keys = list(
                zip(
                    df["patientunitstayid"],
                    df["hour"] - hours
                )
            )

            previous_values = [
                previous_lookup.get(key, np.nan)
                for key in keys
            ]

            df[f"{col}_change_{hours}h"] = (
                df[col] -
                pd.Series(
                    previous_values,
                    index=df.index
                )
            )

            df.loc[
                pd.isna(
                    pd.Series(
                        previous_values,
                        index=df.index
                    )
                ),
                f"{col}_change_{hours}h"
            ] = np.nan

            df = df.drop(
                columns=[
                    "target_hour",
                    f"{col}_previous_{hours}h"
                ],
                errors="ignore"
            )


# ============================================================
# Rolling features
#
# A rolling feature is only allowed to use observations from
# a continuous timeline segment. If there is a gap > 1 hour,
# we restart the rolling window.
# ============================================================

print("\n============================================================")
print("CREATING GAP-AWARE ROLLING FEATURES")
print("============================================================")


rolling_variables = [
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp"
]


for col in rolling_variables:

    print(f"Rolling features: {col}")

    # Segment identifier:
    # every gap > 1 hour starts a new segment.

    gap_break = (
        df["hour_gap"].fillna(1) > 1
    )

    segment = (
        gap_break
        .groupby(df["patientunitstayid"])
        .cumsum()
    )

    segment_id = (
        df["patientunitstayid"].astype(str)
        + "_"
        + segment.astype(str)
    )

    grouped = df.groupby(
        segment_id,
        sort=False
    )[col]

    # 3-hour rolling mean
    df[f"{col}_rolling_mean_3h"] = (
        grouped
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=2
            ).mean()
        )
    )

    # 3-hour rolling standard deviation
    df[f"{col}_rolling_std_3h"] = (
        grouped
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=2
            ).std()
        )
    )

    # 6-hour rolling mean
    df[f"{col}_rolling_mean_6h"] = (
        grouped
        .transform(
            lambda x: x.rolling(
                window=6,
                min_periods=2
            ).mean()
        )
    )

    # 6-hour rolling standard deviation
    df[f"{col}_rolling_std_6h"] = (
        grouped
        .transform(
            lambda x: x.rolling(
                window=6,
                min_periods=2
            ).std()
        )
    )


# ============================================================
# Remove helper column
# ============================================================

df = df.drop(
    columns=["hour_gap"]
)


# ============================================================
# Cumulative fluid balance
# ============================================================

print("\nCreating cumulative fluid balance...")

df["cumulative_net_fluid_balance_ml"] = (
    df.groupby("patientunitstayid")[
        "net_fluid_balance_ml"
    ].cumsum()
)


# ============================================================
# Cumulative urine output
# ============================================================

df["cumulative_urine_output_ml"] = (
    df.groupby("patientunitstayid")[
        "urine_output_ml"
    ].cumsum()
)


# ============================================================
# Active infusion count
# ============================================================

infusion_columns = [
    "dobutamine",
    "fentanyl",
    "insulin",
    "midazolam",
    "norepinephrine",
    "phenylephrine",
    "propofol",
    "vasopressin"
]

df["active_infusion_count"] = (
    df[infusion_columns]
    .sum(axis=1)
)


# ============================================================
# Vasoactive support
# ============================================================

vasoactive_columns = [
    "dobutamine",
    "norepinephrine",
    "phenylephrine",
    "vasopressin"
]

df["vasoactive_support"] = (
    df[vasoactive_columns]
    .max(axis=1)
)


# ============================================================
# Validation
# ============================================================

print("\n============================================================")
print("VALIDATION")
print("============================================================")

duplicates = df.duplicated(
    subset=["patientunitstayid", "hour"]
).sum()

print(
    "Duplicate patient-hours:",
    duplicates
)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)

print(
    "ICU stays:",
    df["patientunitstayid"].nunique()
)

print(
    "Minimum hour:",
    df["hour"].min()
)

print(
    "Maximum hour:",
    df["hour"].max()
)


if duplicates > 0:
    raise ValueError(
        "Duplicate patient-hour rows detected."
    )


if len(df) != 139432:
    raise ValueError(
        f"Row count changed. Expected 139432, "
        f"got {len(df)}"
    )


if df["patientunitstayid"].nunique() != 2419:
    raise ValueError(
        "ICU stay count changed."
    )


# ============================================================
# New feature list
# ============================================================

original_columns = pd.read_csv(
    INPUT_FILE,
    nrows=1
).columns.tolist()

new_columns = [
    col
    for col in df.columns
    if col not in original_columns
]

print("\n============================================================")
print("NEW FEATURES")
print("============================================================")

print(
    "Number of new features:",
    len(new_columns)
)

for col in new_columns:
    print(col)


# ============================================================
# Missingness
# ============================================================

print("\n============================================================")
print("TREND FEATURE MISSINGNESS")
print("============================================================")

missingness = (
    df[new_columns]
    .isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(
    missingness.to_string()
)


# ============================================================
# Save
# ============================================================

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

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)