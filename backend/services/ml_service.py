"""
ML Service for Smart ICU Patient Safety System.

Manages inspection, status reporting, and safe inference interfaces for:
1. Historical deterioration model (Target: 'ICU discharge marked expired within 24 hours').
2. Historical unsupervised anomaly detection (IsolationForest v2).

Clinical Safety Guarantees:
- If model artifacts are missing, returns explicit UNAVAILABLE states.
- Never fabricates probabilities, scores, or predictions.
- Never feeds live laboratory or urine output into models not trained on them.
- Clear separation between historical dataset precomputed research metrics and live inferences.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ml" / "models"
DETERIORATION_ARTIFACT = MODEL_DIR / "deterioration_logistic_v1.joblib"
ANOMALY_ARTIFACT = MODEL_DIR / "anomaly_isolation_forest_v2.joblib"


class MLService:
    def __init__(self) -> None:
        self.deterioration_target = "ICU discharge marked expired within 24 hours"
        self._cached_deterioration_model = None
        self._cached_anomaly_model = None

    def get_status(self) -> dict[str, Any]:
        """
        Inspect the local filesystem for trained model artifacts and report exact availability.
        """
        det_exists = False
        try:
            det_exists = DETERIORATION_ARTIFACT.exists()
        except Exception:
            det_exists = False

        anom_exists = False
        try:
            anom_exists = ANOMALY_ARTIFACT.exists()
        except Exception:
            anom_exists = False

        return {
            "deterioration_model": {
                "model_name": "deterioration_logistic_v1",
                "model_type": "LogisticRegression (calibrated)",
                "artifact_path": str(DETERIORATION_ARTIFACT),
                "available": det_exists,
                "status": "AVAILABLE" if det_exists else "UNAVAILABLE",
                "target_definition": self.deterioration_target,
                "notes": (
                    "Trained on historical eICU dataset for predicting ICU discharge marked expired within 24 hours."
                    if det_exists
                    else f"Model artifact not found at {DETERIORATION_ARTIFACT}. In compliance with clinical safety rules, no predictions are fabricated."
                ),
            },
            "anomaly_model": {
                "model_name": "anomaly_isolation_forest_v2",
                "model_type": "IsolationForest",
                "artifact_path": str(ANOMALY_ARTIFACT),
                "available": anom_exists,
                "status": "AVAILABLE" if anom_exists else "UNAVAILABLE",
                "statistical_interpretation": "Statistical multi-variable outlier detection; not a medical diagnosis.",
                "notes": (
                    "Available for offline feature inference."
                    if anom_exists
                    else f"Model artifact not found at {ANOMALY_ARTIFACT}. Unsupervised anomaly scores are unavailable."
                ),
            },
            "clinical_rules_status": {
                "status": "ACTIVE",
                "engine": "ClinicalAlertEngine v1.0.0",
                "coverage": ["KDIGO Urine Output", "Electrolyte & Renal Panels", "Complete Blood Count", "Vitals"],
                "notes": "Deterministic rules operate independently of ML model artifact presence.",
            },
        }

    def predict_deterioration(self, records: Any) -> dict[str, Any]:
        """
        Safely score deterioration probability. Returns UNAVAILABLE if artifact is missing.
        """
        if not DETERIORATION_ARTIFACT.exists():
            return {
                "available": False,
                "status": "UNAVAILABLE",
                "error": f"Model artifact '{DETERIORATION_ARTIFACT.name}' is not available on this system.",
                "target": self.deterioration_target,
                "probabilities": None,
            }
        try:
            import joblib
            if self._cached_deterioration_model is None:
                self._cached_deterioration_model = joblib.load(DETERIORATION_ARTIFACT)
            bundle = self._cached_deterioration_model
            features = bundle["feature_columns"]
            missing = [c for c in features if c not in records.columns]
            if missing:
                return {
                    "available": False,
                    "status": "INCOMPATIBLE_FEATURES",
                    "error": f"Missing required features: {missing}",
                    "probabilities": None,
                }
            probs = bundle["pipeline"].predict_proba(records[features])[:, 1].tolist()
            return {
                "available": True,
                "status": "SUCCESS",
                "target": self.deterioration_target,
                "probabilities": probs,
            }
        except Exception as exc:
            return {
                "available": False,
                "status": "ERROR",
                "error": str(exc),
                "probabilities": None,
            }

    def score_anomalies(self, records: Any) -> dict[str, Any]:
        """
        Safely score anomalies. Returns UNAVAILABLE if artifact is missing.
        """
        if not ANOMALY_ARTIFACT.exists():
            return {
                "available": False,
                "status": "UNAVAILABLE",
                "error": f"Model artifact '{ANOMALY_ARTIFACT.name}' is not available on this system.",
                "scores": None,
                "flags": None,
            }
        try:
            import joblib
            if self._cached_anomaly_model is None:
                self._cached_anomaly_model = joblib.load(ANOMALY_ARTIFACT)
            bundle = self._cached_anomaly_model
            features = bundle["feature_columns"]
            missing = [c for c in features if c not in records.columns]
            if missing:
                return {
                    "available": False,
                    "status": "INCOMPATIBLE_FEATURES",
                    "error": f"Missing required features: {missing}",
                    "scores": None,
                    "flags": None,
                }
            pipeline = bundle["pipeline"]
            scores = (-pipeline.decision_function(records[features])).tolist()
            flags = (pipeline.predict(records[features]) == -1).astype(int).tolist()
            return {
                "available": True,
                "status": "SUCCESS",
                "scores": scores,
                "flags": flags,
            }
        except Exception as exc:
            return {
                "available": False,
                "status": "ERROR",
                "error": str(exc),
                "scores": None,
                "flags": None,
            }


ml_service = MLService()
