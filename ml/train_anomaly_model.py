import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

TRAIN_FILE = PROCESSED_DIR / "anomaly_train.csv"
TEST_FILE = PROCESSED_DIR / "anomaly_test.csv"

OUTPUT_FILE = (
    PROCESSED_DIR /
    "hourly_patient_state_anomalies.csv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

# Initial operating point.
# This is an ML parameter, NOT a clinical prevalence estimate.
CONTAMINATION = 0.02


# ============================================================
# LOAD DATA
# ============================================================

print("Loading training data...")

train_df = pd.read_csv(
    TRAIN_FILE,
    low_memory=False
)

print(
    "Training shape:",
    train_df.shape
)

print(
    "Training ICU stays:",
    train_df["patientunitstayid"].nunique()
)


print("\nLoading testing data...")

test_df = pd.read_csv(
    TEST_FILE,
    low_memory=False
)

print(
    "Testing shape:",
    test_df.shape
)

print(
    "Testing ICU stays:",
    test_df["patientunitstayid"].nunique()
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

identifier_columns = [
    "patientunitstayid",
    "hour"
]

feature_columns = [
    col
    for col in train_df.columns
    if col not in identifier_columns
]


print("\nNumber of ML features:", len(feature_columns))


# ============================================================
# CHECK TRAIN / TEST FEATURE CONSISTENCY
# ============================================================

test_feature_columns = [
    col
    for col in test_df.columns
    if col not in identifier_columns
]

if feature_columns != test_feature_columns:
    raise ValueError(
        "Training and testing feature columns do not match."
    )


# ============================================================
# CONVERT TO NUMERIC
# ============================================================

X_train = train_df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = test_df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# REMOVE NON-FINITE VALUES
# ============================================================

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# CHECK MISSINGNESS
# ============================================================

print("\n============================================================")
print("INPUT MISSINGNESS")
print("============================================================")

train_missing = (
    X_train.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(
    train_missing.head(15).round(2).to_string()
)


# ============================================================
# PIPELINE
#
# Imputer and scaler are fitted ONLY on training data.
# ============================================================

print("\nBuilding ML pipeline...")

pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "isolation_forest",
            IsolationForest(
                n_estimators=300,
                contamination=CONTAMINATION,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\n============================================================")
print("TRAINING ISOLATION FOREST")
print("============================================================")

print(
    "Estimators:",
    300
)

print(
    "Contamination:",
    CONTAMINATION
)

print(
    "Training rows:",
    len(X_train)
)

pipeline.fit(X_train)


print("Training complete.")


# ============================================================
# PREDICT TEST DATA
# ============================================================

print("\n============================================================")
print("SCORING TEST DATA")
print("============================================================")

# Isolation Forest:
# decision_function:
#   higher = more normal
#   lower  = more anomalous
#
# We invert it so:
#   higher anomaly_score = more anomalous

raw_score = pipeline.decision_function(
    X_test
)

anomaly_score = -raw_score

prediction = pipeline.predict(
    X_test
)

# Isolation Forest returns:
#   1  = normal
#  -1  = anomaly

is_anomaly = (
    prediction == -1
)


# ============================================================
# CREATE OUTPUT
# ============================================================

output = test_df[
    identifier_columns
].copy()

output["anomaly_score"] = anomaly_score

output["is_anomaly"] = (
    is_anomaly.astype(int)
)


# ============================================================
# ANOMALY SEVERITY
#
# These are ML severity bands for dashboard prioritization,
# not clinical diagnoses.
# ============================================================

q90 = np.quantile(
    anomaly_score,
    0.90
)

q97 = np.quantile(
    anomaly_score,
    0.97
)

q995 = np.quantile(
    anomaly_score,
    0.995
)


def classify_severity(score):

    if score >= q995:
        return "critical"

    if score >= q97:
        return "high"

    if score >= q90:
        return "moderate"

    return "low"


output["anomaly_severity"] = (
    output["anomaly_score"]
    .apply(classify_severity)
)


# ============================================================
# VALIDATION
# ============================================================

print("\n============================================================")
print("ANOMALY VALIDATION")
print("============================================================")

print(
    "Output rows:",
    len(output)
)

print(
    "Output ICU stays:",
    output["patientunitstayid"].nunique()
)

print(
    "Duplicate patient-hours:",
    output.duplicated(
        subset=[
            "patientunitstayid",
            "hour"
        ]
    ).sum()
)

print(
    "Anomalies detected:",
    int(output["is_anomaly"].sum())
)

print(
    "Anomaly percentage:",
    round(
        output["is_anomaly"].mean() * 100,
        2
    ),
    "%"
)


print("\nAnomaly score distribution:")

print(
    output["anomaly_score"]
    .describe(
        percentiles=[
            0.50,
            0.90,
            0.95,
            0.97,
            0.99,
            0.995
        ]
    )
    .to_string()
)


print("\nSeverity distribution:")

print(
    output["anomaly_severity"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# CHECK PATIENT LEAKAGE
# ============================================================

train_patients = set(
    train_df["patientunitstayid"]
)

test_patients = set(
    test_df["patientunitstayid"]
)

overlap = (
    train_patients &
    test_patients
)

print(
    "\nPatient overlap:",
    len(overlap)
)

if len(overlap) > 0:
    raise ValueError(
        "Patient leakage detected."
    )


# ============================================================
# SAVE
# ============================================================

output.to_csv(
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