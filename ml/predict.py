"""Load saved Smart ICU models for offline/research inference."""
from pathlib import Path
import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"

def load_model(filename: str):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}. Run the corresponding training script first.")
    return joblib.load(path)

def predict_deterioration(records: pd.DataFrame):
    artifact = load_model("deterioration_logistic_v1.joblib")
    features = artifact["feature_columns"]
    missing = [c for c in features if c not in records.columns]
    if missing: raise ValueError(f"Missing required model features: {missing}")
    probabilities = artifact["pipeline"].predict_proba(records[features])[:, 1]
    return probabilities

def score_anomalies(records: pd.DataFrame):
    artifact = load_model("anomaly_isolation_forest_v2.joblib")
    features = artifact["feature_columns"]
    missing = [c for c in features if c not in records.columns]
    if missing: raise ValueError(f"Missing required model features: {missing}")
    pipeline = artifact["pipeline"]
    return {"anomaly_score": -pipeline.decision_function(records[features]), "anomaly_flag": (pipeline.predict(records[features]) == -1).astype(int)}
