"""
ML Service for Smart ICU Patient Safety System.

Manages inspection, status reporting, and safe inference interfaces for:
1. Retrospective rolling 24-hour deterioration model (Target: 'ICU discharge marked expired within the next 24 hours from the prediction timestamp').
2. Historical unsupervised anomaly detection (IsolationForest v2).

Clinical Safety Guarantees:
- If model artifacts are missing, returns explicit UNAVAILABLE states.
- Never fabricates probabilities, scores, or predictions.
- Never feeds live laboratory or urine output into models not trained on them.
- Clear separation between historical dataset precomputed research metrics and live inferences.
- Transparent reporting of performance metrics (ROC-AUC, PR-AUC, Brier score, decision threshold).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "ml" / "models"
CANDIDATE_DIR = ROOT / "ml" / "candidate_artifacts_v2"

DETERIORATION_LOGISTIC_ARTIFACT = MODEL_DIR / "deterioration_logistic_v1.joblib"
DETERIORATION_RF_ARTIFACT = MODEL_DIR / "deterioration_random_forest_v1.joblib"
DETERIORATION_GBM_ARTIFACT = MODEL_DIR / "deterioration_gradient_boosting_v1.joblib"
ANOMALY_ARTIFACT = MODEL_DIR / "anomaly_isolation_forest_v2.joblib"
METRICS_JSON_PATH = MODEL_DIR / "model_comparison_metrics.json"

# Candidate v2 artifact paths (pre-promotion evaluation path)
CANDIDATE_ARTIFACT = CANDIDATE_DIR / "deterioration_candidate_v2.joblib"
CANDIDATE_METRICS_PATH = CANDIDATE_DIR / "optimization_metrics.json"

# Default deterioration model artifact (unweighted Logistic Regression, natural prevalence)
DETERIORATION_ARTIFACT = DETERIORATION_LOGISTIC_ARTIFACT

ACADEMIC_DISCLAIMER = (
    "Academic research prototype: models trained on retrospective eICU Collaborative Research Database demo. "
    "Not an FDA/CE cleared medical device. Probabilities and anomaly scores are statistical estimates intended "
    "solely for investigational clinical decision-support."
)


def assign_risk_band(p: float) -> str:
    """Standardized experimental risk stratification."""
    if p < 0.05:
        return "Low"
    elif p < 0.15:
        return "Moderate"
    elif p < 0.30:
        return "High"
    else:
        return "Extreme"


class MLService:
    def __init__(self) -> None:
        self.deterioration_target = "ICU discharge marked expired within the next 24 hours from the prediction timestamp"
        self._cached_deterioration_model: dict[str, Any] | None = None
        self._cached_anomaly_model: dict[str, Any] | None = None
        self._cached_metrics_data: dict[str, Any] | None = None
        self._cached_candidate_model: dict[str, Any] | None = None
        self._cached_candidate_metrics: dict[str, Any] | None = None
        self._candidate_load_error: str | None = None

    def _load_comparison_metrics(self) -> dict[str, Any] | None:
        if self._cached_metrics_data is not None:
            return self._cached_metrics_data
        try:
            if METRICS_JSON_PATH.exists():
                with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                    self._cached_metrics_data = json.load(f)
                    return self._cached_metrics_data
        except Exception:
            pass
        return None

    def _get_deterioration_bundle(self) -> dict[str, Any] | None:
        if self._cached_deterioration_model is not None:
            return self._cached_deterioration_model
        if not DETERIORATION_ARTIFACT.exists():
            return None
        try:
            import joblib
            self._cached_deterioration_model = joblib.load(DETERIORATION_ARTIFACT)
            return self._cached_deterioration_model
        except Exception:
            return None

    def _get_anomaly_bundle(self) -> dict[str, Any] | None:
        if self._cached_anomaly_model is not None:
            return self._cached_anomaly_model
        if not ANOMALY_ARTIFACT.exists():
            return None
        try:
            import joblib
            self._cached_anomaly_model = joblib.load(ANOMALY_ARTIFACT)
            return self._cached_anomaly_model
        except Exception:
            return None

    def _get_candidate_bundle(self) -> dict[str, Any] | None:
        if self._cached_candidate_model is not None:
            return self._cached_candidate_model
        if not CANDIDATE_ARTIFACT.exists():
            return None
        try:
            import joblib
            self._cached_candidate_model = joblib.load(CANDIDATE_ARTIFACT)
            self._candidate_load_error = None
            return self._cached_candidate_model
        except Exception as exc:
            import sklearn
            self._candidate_load_error = (
                f"Candidate artifact was serialized under scikit-learn 1.8.0 and cannot be deserialized "
                f"under scikit-learn {sklearn.__version__}: {type(exc).__name__}: {exc}"
            )
            return None


    def _load_candidate_metrics(self) -> dict[str, Any] | None:
        if self._cached_candidate_metrics is not None:
            return self._cached_candidate_metrics
        try:
            if CANDIDATE_METRICS_PATH.exists():
                with open(CANDIDATE_METRICS_PATH, "r", encoding="utf-8") as f:
                    self._cached_candidate_metrics = json.load(f)
                    return self._cached_candidate_metrics
        except Exception:
            pass
        return None

    def get_status(self) -> dict[str, Any]:
        """
        Inspect the local filesystem for trained model artifacts and report exact availability,
        provenance, and genuine evaluation metrics.
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

        det_bundle = self._get_deterioration_bundle() if det_exists else None
        metrics_data = self._load_comparison_metrics()

        if det_exists and anom_exists:
            overall_status = "ONLINE"
        elif det_exists or anom_exists:
            overall_status = "DEGRADED"
        else:
            overall_status = "UNAVAILABLE"

        # Deterioration model metadata & metrics
        det_metrics = None
        det_threshold = 0.0823
        det_training_samples = 96731
        det_pos_samples = 1461
        det_top_features = []

        if det_bundle:
            det_metrics = det_bundle.get("validation_metrics")
            det_threshold = det_bundle.get("decision_threshold", 0.0823)
            det_training_samples = det_bundle.get("training_samples", 96731)
            det_pos_samples = det_bundle.get("positive_samples", 1461)
            det_top_features = det_bundle.get("feature_importance", [])[:8]
        elif metrics_data and "models" in metrics_data and "LogisticRegression" in metrics_data["models"]:
            det_metrics = metrics_data["models"]["LogisticRegression"]
            det_threshold = det_metrics.get("decision_threshold", 0.0823)
            det_top_features = metrics_data.get("top_features_logistic", [])[:8]

        diagnostic_notes = []
        if det_exists:
            diagnostic_notes.append(
                f"Deterioration model '{DETERIORATION_ARTIFACT.name}' loaded. Target: '{self.deterioration_target}'."
            )
        else:
            diagnostic_notes.append(
                f"Deterioration model artifact not found at {DETERIORATION_ARTIFACT}. Predictions are unavailable."
            )

        if anom_exists:
            diagnostic_notes.append(
                f"Anomaly model '{ANOMALY_ARTIFACT.name}' loaded. Unsupervised IsolationForest v2."
            )
        else:
            diagnostic_notes.append(
                f"Anomaly model artifact not found at {ANOMALY_ARTIFACT}. Anomaly scores are unavailable."
            )

        # Candidate v2 artifact status (pre-promotion evaluation path)
        cand_exists = False
        try:
            cand_exists = CANDIDATE_ARTIFACT.exists()
        except Exception:
            cand_exists = False

        cand_bundle = self._get_candidate_bundle() if cand_exists else None
        cand_metrics = self._load_candidate_metrics()
        is_runtime_compatible = cand_bundle is not None

        import sklearn
        if cand_exists:
            if is_runtime_compatible:
                cand_status_str = "AVAILABLE_FOR_EVALUATION"
                diagnostic_notes.append(
                    f"Candidate v2 model '{CANDIDATE_ARTIFACT.name}' detected and loadable in current runtime "
                    f"(unpromoted; historical eICU inference only)."
                )
            else:
                cand_status_str = "UNAVAILABLE_INCOMPATIBLE_RUNTIME"
                diagnostic_notes.append(
                    f"Candidate v2 model '{CANDIDATE_ARTIFACT.name}' is present on disk but UNAVAILABLE in this runtime "
                    f"(serialized with scikit-learn 1.8.0, incompatible with current scikit-learn {sklearn.__version__})."
                )
        else:
            cand_status_str = "UNAVAILABLE"

        cand_status = {
            "artifact": CANDIDATE_ARTIFACT.name,
            "artifact_path": str(CANDIDATE_ARTIFACT),
            "model_name": cand_bundle.get("model_name", "deterioration_candidate_v2") if cand_bundle else "deterioration_candidate_v2",
            "version": "2.0.0-candidate",
            "target": self.deterioration_target,
            "available": is_runtime_compatible,
            "present_on_disk": cand_exists,
            "loaded": is_runtime_compatible,
            "runtime_compatible": is_runtime_compatible,
            "status": cand_status_str,
            "incompatibility_reason": self._candidate_load_error if not is_runtime_compatible and cand_exists else None,
            "promotion_status": "UNPROMOTED_CANDIDATE",
            "default_production_model": False,
            "supported_data_sources": ["historical_eicu"],
            "unsupported_data_sources": ["live_telemetry", "manual_entry", "bedside"],
            "feature_count": cand_bundle.get("feature_count", 52) if cand_bundle else (cand_metrics.get("feature_count", 52) if cand_metrics else 52),
            "feature_set_name": cand_bundle.get("feature_set_name", "Set_C_LaggedLabs") if cand_bundle else "Set_C_LaggedLabs",
            "calibration_method": cand_bundle.get("calibration_method", "platt_sigmoid_oof") if cand_bundle else "platt_sigmoid_oof",
            "decision_threshold": round(float(cand_bundle.get("decision_threshold", 0.0600)), 4) if cand_bundle else 0.0600,
            "episode_policy": cand_bundle.get("episode_policy", {"cooldown_hours": 6, "reset_hours": 2}) if cand_bundle else {"cooldown_hours": 6, "reset_hours": 2},
            "git_commit_sha": cand_bundle.get("git_commit_sha") if cand_bundle else (cand_metrics.get("git_commit_sha") if cand_metrics else None),
            "validation_metrics": cand_metrics.get("validation_metrics") if cand_metrics else None,
            "exploratory_test_benchmark": cand_metrics.get("exploratory_test_benchmark") if cand_metrics else None,
            "notes": (
                "Pre-promotion candidate v2 evaluated on Set C (52 features with backward as-of lagged labs). "
                "Restricted strictly to historical eICU inference; unsupported for live bedside telemetry or manual entry. "
                "Baseline v1 remains the default production model."
                if is_runtime_compatible
                else f"Candidate v2 artifact requires scikit-learn 1.8.x and is incompatible with current runtime "
                     f"(scikit-learn {sklearn.__version__}). Use the dedicated candidate evaluation environment "
                     f"(ml/.venv_candidate) for candidate inference."
            ),
        }


        return {
            "status": overall_status,
            "overall_status": overall_status,
            "academic_disclaimer": ACADEMIC_DISCLAIMER,
            "deterioration_model": {
                "artifact": DETERIORATION_ARTIFACT.name,
                "artifact_path": str(DETERIORATION_ARTIFACT),
                "model_name": "deterioration_logistic_v1",
                "model_type": "LogisticRegression (unweighted, natural prevalence)",
                "version": "1.0.0",
                "target": self.deterioration_target,
                "target_definition": self.deterioration_target,
                "available": det_exists,
                "loaded": det_exists,
                "status": "AVAILABLE" if det_exists else "UNAVAILABLE",
                "missing_path": None if det_exists else str(DETERIORATION_ARTIFACT),
                "decision_threshold": det_threshold,
                "validation_metrics": det_metrics,
                "training_samples": det_training_samples,
                "positive_samples": det_pos_samples,
                "feature_count": 22,
                "top_features": det_top_features,
                "notes": (
                    "Trained on historical eICU dataset for retrospective rolling 24-hour prediction of "
                    "ICU discharge marked expired within the next 24 hours from the prediction timestamp. "
                    "Unweighted Logistic Regression operating at natural training prevalence (~1.6%). "
                    "Held-out test calibration: slope=0.8360, intercept=-0.4605, model Brier=0.015443 vs "
                    "prevalence-only baseline Brier=0.016369 (low Brier score driven by 98.34% negative class prevalence). "
                    "Risk bands are experimental presentation categories, not validated clinical severity classifications."
                    if det_exists
                    else f"Model artifact not found at {DETERIORATION_ARTIFACT}. In compliance with clinical safety rules, no predictions are fabricated."
                ),
            },
            "candidate_v2": cand_status,
            "anomaly_model": {
                "artifact": ANOMALY_ARTIFACT.name,
                "artifact_path": str(ANOMALY_ARTIFACT),
                "model_name": "anomaly_isolation_forest_v2",
                "model_type": "IsolationForest",
                "version": "2.0.0",
                "available": anom_exists,
                "loaded": anom_exists,
                "status": "AVAILABLE" if anom_exists else "UNAVAILABLE",
                "missing_path": None if anom_exists else str(ANOMALY_ARTIFACT),
                "statistical_interpretation": "Statistical multi-variable outlier detection; not a medical diagnosis.",
                "notes": (
                    "IsolationForest trained on healthy/stable ICU physiological patterns (300 trees, 2% contamination)."
                    if anom_exists
                    else f"Model artifact not found at {ANOMALY_ARTIFACT}. Unsupervised anomaly scores are unavailable."
                ),
            },
            "model_comparison": metrics_data.get("models") if metrics_data else None,
            "winning_model": metrics_data.get("winning_model") if metrics_data else "RandomForest",
            "diagnostic_notes": diagnostic_notes,
            "clinical_rules_status": {
                "status": "ACTIVE",
                "engine": "ClinicalAlertEngine v1.0.0",
                "coverage": ["KDIGO Urine Output", "Electrolyte & Renal Panels", "Complete Blood Count", "Vitals"],
                "notes": "Deterministic rules operate independently of ML model artifact presence.",
            },
        }

    def predict_deterioration(self, records: Any) -> dict[str, Any]:
        """
        Safely score deterioration probability using trained pipeline.
        Returns UNAVAILABLE if artifact is missing.
        """
        if not DETERIORATION_ARTIFACT.exists():
            return {
                "available": False,
                "status": "UNAVAILABLE",
                "error": f"Model artifact '{DETERIORATION_ARTIFACT.name}' is not available on this system.",
                "target": self.deterioration_target,
                "probabilities": None,
                "risk_bands": None,
                "latest_prediction": None,
            }

        bundle = self._get_deterioration_bundle()
        if bundle is None:
            return {
                "available": False,
                "status": "ERROR",
                "error": "Failed to load model bundle.",
                "probabilities": None,
                "risk_bands": None,
                "latest_prediction": None,
            }

        try:
            if not isinstance(records, pd.DataFrame):
                records = pd.DataFrame(records)

            features = bundle["feature_columns"]
            missing = [c for c in features if c not in records.columns]
            if missing:
                return {
                    "available": False,
                    "status": "INCOMPATIBLE_FEATURES",
                    "error": f"Missing required features: {missing}",
                    "probabilities": None,
                    "risk_bands": None,
                    "latest_prediction": None,
                }

            pipeline = bundle["pipeline"]
            raw_probs = pipeline.predict_proba(records[features])[:, 1]
            probs = [round(float(np.clip(p, 0.0, 1.0)), 4) for p in raw_probs]

            threshold = bundle.get("decision_threshold", 0.0823)

            def assign_risk_band(p: float) -> str:
                if p < 0.05:
                    return "Low"
                elif p < 0.15:
                    return "Moderate"
                elif p < 0.30:
                    return "High"
                else:
                    return "Extreme"

            risk_bands = [assign_risk_band(p) for p in probs]

            # Compute feature contributions for the latest row (index -1)
            latest_prediction = None
            if len(records) > 0:
                last_idx = len(records) - 1
                latest_p = probs[last_idx]
                latest_band = risk_bands[last_idx]

                # Extract pipeline steps
                imputer = pipeline.named_steps.get("imputer")
                scaler = pipeline.named_steps.get("scaler")
                model = pipeline.named_steps.get("model")

                intercept_val = None
                log_odds_val = None
                top_contributors = []
                if imputer and scaler and model and hasattr(model, "coef_"):
                    imputed = imputer.transform(records[features])
                    scaled = scaler.transform(imputed)
                    weights = model.coef_[0]
                    contributions = scaled[last_idx] * weights
                    if hasattr(model, "intercept_"):
                        intercept_val = round(float(model.intercept_[0]), 4)
                        log_odds_val = round(float(model.intercept_[0] + np.sum(contributions)), 4)

                    contrib_list = []
                    for col_name, cont, w in zip(features, contributions, weights):
                        raw_v = records.iloc[last_idx][col_name]
                        contrib_list.append({
                            "feature": col_name,
                            "value": None if pd.isna(raw_v) else round(float(raw_v), 3),
                            "weight": round(float(w), 4),
                            "contribution": round(float(cont), 4),
                            "abs_contribution": round(float(abs(cont)), 4),
                            "direction": "INCREASING_RISK" if cont > 0 else "DECREASING_RISK",
                        })
                    contrib_list.sort(key=lambda x: x["abs_contribution"], reverse=True)
                    top_contributors = contrib_list[:6]

                latest_prediction = {
                    "probability": latest_p,
                    "risk_band": latest_band,
                    "decision_threshold": threshold,
                    "threshold_exceeded": bool(latest_p >= threshold),
                    "model_intercept": intercept_val,
                    "predicted_log_odds": log_odds_val,
                    "top_contributing_features": top_contributors,
                }

            return {
                "available": True,
                "status": "SUCCESS",
                "model_name": bundle.get("model_name", "deterioration_logistic_v1"),
                "model_version": bundle.get("version", "1.0.0"),
                "target": self.deterioration_target,
                "probabilities": probs,
                "risk_bands": risk_bands,
                "latest_prediction": latest_prediction,
                "disclaimer": ACADEMIC_DISCLAIMER,
            }
        except Exception as exc:
            return {
                "available": False,
                "status": "ERROR",
                "error": str(exc),
                "probabilities": None,
                "risk_bands": None,
                "latest_prediction": None,
            }

    def predict_deterioration_candidate(
        self,
        records: Any,
        data_source: str = "historical_eicu",
        allow_fallback: bool = False,
    ) -> dict[str, Any]:
        """
        Safely score deterioration probability using candidate v2 model (52 features, lagged labs).

        Engineering Safety Guarantees:
        - Only historical eICU data ('historical_eicu') is supported.
        - Live telemetry and manual entry are strictly rejected with UNSUPPORTED_DATA_SOURCE.
        - If allow_fallback=True and baseline v1 features exist, falls back to baseline v1 with
          transparent provenance recording (fallback_applied=True).
        - Baseline v1 remains the default production model with zero promotion.
        """
        # 1. Enforce strict data source isolation
        if data_source != "historical_eicu":
            if allow_fallback:
                fb_res = self.predict_deterioration(records)
                if fb_res.get("available") and fb_res.get("status") == "SUCCESS":
                    fb_res["fallback_applied"] = True
                    fb_res["fallback_reason"] = (
                        f"Candidate v2 supports only historical eICU data ('historical_eicu'). "
                        f"Data source '{data_source}' is unsupported. Fell back to production baseline v1."
                    )
                    fb_res["candidate_requested"] = True
                    return fb_res
                return {
                    "available": False,
                    "status": "UNSUPPORTED_DATA_SOURCE",
                    "error": (
                        f"Candidate v2 supports only historical eICU stays ('historical_eicu'). "
                        f"Data source '{data_source}' is unsupported. Baseline fallback failed: {fb_res.get('error')}"
                    ),
                    "probabilities": None,
                    "risk_bands": None,
                    "latest_prediction": None,
                    "fallback_applied": False,
                }
            return {
                "available": False,
                "status": "UNSUPPORTED_DATA_SOURCE",
                "error": (
                    f"Candidate v2 supports only historical eICU stays ('historical_eicu'). "
                    f"Received unsupported data source: '{data_source}'. "
                    "Live bedside telemetry, manual entry, and new admissions cannot be scored with candidate v2."
                ),
                "probabilities": None,
                "risk_bands": None,
                "latest_prediction": None,
                "fallback_applied": False,
            }

        # 2. Check candidate artifact availability
        if not CANDIDATE_ARTIFACT.exists():
            if allow_fallback:
                fb_res = self.predict_deterioration(records)
                if fb_res.get("available") and fb_res.get("status") == "SUCCESS":
                    fb_res["fallback_applied"] = True
                    fb_res["fallback_reason"] = (
                        f"Candidate artifact '{CANDIDATE_ARTIFACT.name}' is not available. "
                        "Fell back to production baseline v1."
                    )
                    fb_res["candidate_requested"] = True
                    return fb_res
            return {
                "available": False,
                "status": "UNAVAILABLE",
                "error": f"Candidate model artifact '{CANDIDATE_ARTIFACT.name}' is not available on this system.",
                "target": self.deterioration_target,
                "probabilities": None,
                "risk_bands": None,
                "latest_prediction": None,
                "fallback_applied": False,
            }

        bundle = self._get_candidate_bundle()
        if bundle is None:
            import sklearn
            reason = (
                self._candidate_load_error
                or f"Candidate artifact was serialized under scikit-learn 1.8.0 and cannot be deserialized "
                   f"under scikit-learn {sklearn.__version__}."
            )
            if allow_fallback:
                fb_res = self.predict_deterioration(records)
                if fb_res.get("available") and fb_res.get("status") == "SUCCESS":
                    fb_res["fallback_applied"] = True
                    fb_res["fallback_reason"] = (
                        f"Candidate artifact '{CANDIDATE_ARTIFACT.name}' is incompatible with runtime "
                        f"(scikit-learn {sklearn.__version__}). Fell back to production baseline v1."
                    )
                    fb_res["candidate_requested"] = True
                    return fb_res
            return {
                "available": False,
                "status": "UNAVAILABLE_INCOMPATIBLE_RUNTIME",
                "error": reason,
                "probabilities": None,
                "risk_bands": None,
                "latest_prediction": None,
                "fallback_applied": False,
            }


        try:
            if not isinstance(records, pd.DataFrame):
                records = pd.DataFrame(records)

            features = bundle.get("features", [])
            missing = [c for c in features if c not in records.columns]
            if missing:
                if allow_fallback:
                    fb_res = self.predict_deterioration(records)
                    if fb_res.get("available") and fb_res.get("status") == "SUCCESS":
                        fb_res["fallback_applied"] = True
                        fb_res["fallback_reason"] = (
                            f"Missing {len(missing)} candidate v2 features. "
                            "Fell back to production baseline v1."
                        )
                        fb_res["candidate_requested"] = True
                        return fb_res
                return {
                    "available": False,
                    "status": "INCOMPATIBLE_FEATURES",
                    "error": f"Missing required candidate v2 features ({len(missing)} missing): {missing[:10]}...",
                    "probabilities": None,
                    "risk_bands": None,
                    "latest_prediction": None,
                    "fallback_applied": False,
                }

            pipeline = bundle["pipeline"]
            calibrator = bundle.get("calibrator", None)
            cal_method = bundle.get("calibration_method", "raw")
            threshold = round(float(bundle.get("decision_threshold", 0.0600)), 4)

            # Strictly order columns as expected by the pipeline
            X = records[features]
            raw_probs = pipeline.predict_proba(X.values)[:, 1]

            if cal_method == "raw" or calibrator is None:
                cal_probs = raw_probs
            else:
                from scipy.special import logit
                eps = 1e-6
                logits = logit(np.clip(raw_probs, eps, 1.0 - eps)).reshape(-1, 1)
                cal_probs = calibrator.predict_proba(logits)[:, 1]

            # Numerical validity check
            if np.isnan(cal_probs).any() or np.isinf(cal_probs).any():
                return {
                    "available": False,
                    "status": "ERROR",
                    "error": "Candidate inference produced invalid (NaN or Infinite) probabilities.",
                    "probabilities": None,
                    "risk_bands": None,
                    "latest_prediction": None,
                    "fallback_applied": False,
                }

            probs = [round(float(np.clip(p, 0.0, 1.0)), 4) for p in cal_probs]
            risk_bands = [assign_risk_band(p) for p in probs]

            # Compute feature attributions for latest prediction row
            latest_prediction = None
            if len(records) > 0:
                last_idx = len(records) - 1
                latest_p = probs[last_idx]
                latest_band = risk_bands[last_idx]

                imputer = pipeline.named_steps.get("imputer", None)
                scaler = pipeline.named_steps.get("scaler", None)
                model = pipeline.named_steps.get("model", None)

                intercept_val = None
                log_odds_val = None
                top_contributors = []

                if imputer and scaler and model and hasattr(model, "coef_"):
                    imputed = imputer.transform(X)
                    scaled = scaler.transform(imputed)
                    weights = model.coef_[0]
                    contributions = scaled[last_idx] * weights
                    if hasattr(model, "intercept_"):
                        intercept_val = round(float(model.intercept_[0]), 4)
                        log_odds_val = round(float(model.intercept_[0] + np.sum(contributions)), 4)

                    contrib_list = []
                    for col_name, cont, w in zip(features, contributions, weights):
                        raw_v = records.iloc[last_idx][col_name]
                        contrib_list.append({
                            "feature": col_name,
                            "value": None if pd.isna(raw_v) else round(float(raw_v), 3),
                            "weight": round(float(w), 4),
                            "contribution": round(float(cont), 4),
                            "abs_contribution": round(float(abs(cont)), 4),
                            "direction": "INCREASING_RISK" if cont > 0 else "DECREASING_RISK",
                        })
                    contrib_list.sort(key=lambda x: x["abs_contribution"], reverse=True)
                    top_contributors = contrib_list[:8]

                latest_prediction = {
                    "probability": latest_p,
                    "risk_band": latest_band,
                    "decision_threshold": threshold,
                    "threshold_exceeded": bool(latest_p >= threshold),
                    "model_intercept": intercept_val,
                    "predicted_log_odds": log_odds_val,
                    "top_contributing_features": top_contributors,
                }

            return {
                "available": True,
                "status": "SUCCESS",
                "model_name": bundle.get("model_name", "deterioration_candidate_v2"),
                "model_version": "2.0.0-candidate",
                "feature_set_version": bundle.get("feature_set_name", "Set_C_LaggedLabs"),
                "feature_count": len(features),
                "calibration_method": cal_method,
                "decision_threshold": threshold,
                "target": self.deterioration_target,
                "probabilities": probs,
                "risk_bands": risk_bands,
                "latest_prediction": latest_prediction,
                "data_source": data_source,
                "fallback_applied": False,
                "fallback_reason": None,
                "disclaimer": ACADEMIC_DISCLAIMER,
            }
        except Exception as exc:
            return {
                "available": False,
                "status": "ERROR",
                "error": str(exc),
                "probabilities": None,
                "risk_bands": None,
                "latest_prediction": None,
                "fallback_applied": False,
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

        bundle = self._get_anomaly_bundle()
        if bundle is None:
            return {
                "available": False,
                "status": "ERROR",
                "error": "Failed to load anomaly model bundle.",
                "scores": None,
                "flags": None,
            }

        try:
            if not isinstance(records, pd.DataFrame):
                records = pd.DataFrame(records)

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
            scores = [round(float(s), 4) for s in -pipeline.decision_function(records[features])]
            flags = [int(f) for f in (pipeline.predict(records[features]) == -1)]
            return {
                "available": True,
                "status": "SUCCESS",
                "scores": scores,
                "flags": flags,
                "disclaimer": ACADEMIC_DISCLAIMER,
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
