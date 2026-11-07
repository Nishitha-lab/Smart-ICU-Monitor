import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
import joblib

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

TRAIN_FILE = PROCESSED_DIR / "anomaly_train_v2.csv"
TEST_FILE = PROCESSED_DIR / "anomaly_test_v2.csv"
OUTPUT_FILE = PROCESSED_DIR / "hourly_patient_state_anomalies_v2.csv"

# ------------------------------------------------------------
# FEATURES
# ------------------------------------------------------------

feature_columns = [
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",

    "heart_rate_change_1h",
    "spo2_change_1h",
    "mean_bp_change_1h",

    "heart_rate_rolling_mean_3h",
    "spo2_rolling_mean_3h",
    "respiration_rolling_mean_3h",
    "mean_bp_rolling_mean_3h",

    "net_fluid_balance_ml",
    "urine_output_ml",

    "active_infusion_count",
    "vasoactive_support"
]

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("Loading training data...")

train_df = pd.read_csv(TRAIN_FILE, low_memory=False)
test_df = pd.read_csv(TEST_FILE, low_memory=False)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

print("Train ICU stays:", train_df["patientunitstayid"].nunique())
print("Test ICU stays :", test_df["patientunitstayid"].nunique())

# ------------------------------------------------------------
# PREPARE FEATURES
# ------------------------------------------------------------

X_train = train_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()

for col in feature_columns:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

# ------------------------------------------------------------
# IMPUTATION + ISOLATION FOREST
# ------------------------------------------------------------

print("\nBuilding Isolation Forest pipeline...")

pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            IsolationForest(
                n_estimators=300,
                contamination=0.02,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("Training model...")

pipeline.fit(X_train)

print("Training complete.")
MODEL_DIR = ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump({"pipeline": pipeline, "feature_columns": feature_columns, "model_type": "IsolationForest", "version": "v2"}, MODEL_DIR / "anomaly_isolation_forest_v2.joblib")
print("Saved model:", MODEL_DIR / "anomaly_isolation_forest_v2.joblib")

# ------------------------------------------------------------
# PREDICT TEST SET
# ------------------------------------------------------------

print("\nScoring test data...")

# sklearn:
# decision_function -> higher = more normal
# We negate it so higher = more anomalous.

decision_scores = pipeline.decision_function(X_test)

anomaly_scores = -decision_scores

predictions = pipeline.predict(X_test)

anomaly_flag = (predictions == -1).astype(int)

# ------------------------------------------------------------
# CREATE OUTPUT
# ------------------------------------------------------------

results = test_df[
    ["patientunitstayid", "hour"]
].copy()

results["anomaly_score"] = anomaly_scores
results["anomaly_flag"] = anomaly_flag

# Add original features for analysis
for col in feature_columns:
    results[col] = test_df[col].values

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n============================================================")
print("ANOMALY MODEL V2 SUMMARY")
print("============================================================")

print("Test rows:", len(results))

print(
    "Anomalies:",
    results["anomaly_flag"].sum()
)

print(
    "Anomaly rate:",
    round(results["anomaly_flag"].mean() * 100, 2),
    "%"
)

print(
    "Anomalous ICU stays:",
    results.loc[
        results["anomaly_flag"] == 1,
        "patientunitstayid"
    ].nunique()
)

print("\nAnomaly score statistics:")

print(
    results["anomaly_score"]
    .describe()
    .to_string()
)

# ------------------------------------------------------------
# ADMISSION-HOUR BIAS CHECK
# ------------------------------------------------------------

print("\n============================================================")
print("ANOMALY RATE BY ICU TIME")
print("============================================================")

bins = [
    -1,
    1,
    3,
    6,
    12,
    24,
    48,
    72,
    168,
    np.inf
]

labels = [
    "0-1",
    "2-3",
    "4-6",
    "7-12",
    "13-24",
    "25-48",
    "49-72",
    "73-168",
    "169+"
]

results["icu_hour_group"] = pd.cut(
    results["hour"],
    bins=bins,
    labels=labels
)

hour_summary = (
    results
    .groupby("icu_hour_group", observed=False)
    .agg(
        rows=("anomaly_flag", "size"),
        anomalies=("anomaly_flag", "sum"),
        anomaly_rate=("anomaly_flag", "mean")
    )
)

hour_summary["anomaly_rate"] *= 100

print(hour_summary.round(2).to_string())

# ------------------------------------------------------------
# MISSINGNESS BIAS CHECK
# ------------------------------------------------------------

print("\n============================================================")
print("MISSINGNESS BIAS CHECK")
print("============================================================")

physiology_features = [
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp"
]

results["missing_physiology_count"] = (
    results[physiology_features]
    .isna()
    .sum(axis=1)
)

missing_summary = (
    results
    .groupby("anomaly_flag")
    ["missing_physiology_count"]
    .agg(["count", "mean", "median", "max"])
)

print(missing_summary.round(2).to_string())

# ------------------------------------------------------------
# TOP ANOMALIES
# ------------------------------------------------------------

print("\n============================================================")
print("TOP 20 ANOMALIES")
print("============================================================")

top = (
    results
    .sort_values("anomaly_score", ascending=False)
    .head(20)
)

display_columns = [
    "patientunitstayid",
    "hour",
    "anomaly_score",
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "heart_rate_change_1h",
    "spo2_change_1h",
    "mean_bp_change_1h",
    "net_fluid_balance_ml",
    "urine_output_ml",
    "active_infusion_count",
    "vasoactive_support"
]

print(
    top[display_columns]
    .round(2)
    .to_string(index=False)
)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n============================================================")
print("SUCCESS")
print("============================================================")

print("Output:", OUTPUT_FILE)