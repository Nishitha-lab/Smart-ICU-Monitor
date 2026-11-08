"""
Level 3: Historical eICU Population-Level Anomaly Detection Engine.

Compares patient measurements against empirical population distributions derived from
the 139,432 patient-hour eICU dataset (data/processed/eicu_population_reference_v1.json),
and provides multivariate Isolation Forest scoring using anomaly_isolation_forest_v2.joblib.

Terminology strictly enforced:
Identified outliers are designated as HISTORICAL_POPULATION_ANOMALY with explicit phrasing:
"Unusual relative to historical eICU population reference distribution".
They are strictly distinct from acute clinical deterioration and data transmission errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal
import joblib
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_PATH = ROOT / "data" / "processed" / "eicu_population_reference_v1.json"
DEFAULT_MODEL_PATH = ROOT / "ml" / "models" / "anomaly_isolation_forest_v2.joblib"

CANONICAL_METRIC_MAP: dict[str, str] = {
    # Vitals
    "heart rate": "heart_rate",
    "heart_rate": "heart_rate",
    "pulse": "heart_rate",
    "bpm": "heart_rate",
    "oxygen saturation": "spo2",
    "oxygen_saturation": "spo2",
    "spo2": "spo2",
    "respiratory rate": "respiration",
    "respiration": "respiration",
    "breaths/min": "respiration",
    "systolic blood pressure": "systolic_bp",
    "systolic_bp": "systolic_bp",
    "sbp": "systolic_bp",
    "diastolic blood pressure": "diastolic_bp",
    "diastolic_bp": "diastolic_bp",
    "dbp": "diastolic_bp",
    "mean blood pressure": "mean_bp",
    "mean arterial pressure": "mean_bp",
    "mean_bp": "mean_bp",
    "map": "mean_bp",
    # Labs
    "bun": "BUN",
    "blood urea nitrogen": "BUN",
    "creatinine": "creatinine",
    "blood glucose": "glucose",
    "glucose": "glucose",
    "hemoglobin": "hemoglobin",
    "potassium": "potassium",
    "sodium": "sodium",
    # Fluids
    "total intake": "total_intake_ml",
    "total_intake_ml": "total_intake_ml",
    "total output": "total_output_ml",
    "total_output_ml": "total_output_ml",
    "net fluid balance": "net_fluid_balance_ml",
    "net_fluid_balance_ml": "net_fluid_balance_ml",
    "urine output": "urine_output_ml",
    "urine_output": "urine_output_ml",
    "urine_output_ml": "urine_output_ml",
}

DISCLAIMER_TEXT = (
    "Academic Research Prototype: Evaluated relative to retrospective eICU population "
    "reference distributions. Not a clinical diagnosis and not validated for acute deterioration triage."
)


class PopulationComparisonResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metric: str
    canonical_metric: str
    measured_value: float
    reference_mean: float
    reference_std: float
    reference_median: float
    reference_iqr: float
    reference_p05: float
    reference_p25: float
    reference_p75: float
    reference_p95: float
    percentile_rank: float
    robust_z_score: float
    is_anomaly: bool
    severity: Literal["NORMAL", "LOW", "MODERATE", "HIGH", "CRITICAL"]
    anomaly_type: str
    explanation: str
    clinical_validation_recommendation: str
    disclaimer: str = DISCLAIMER_TEXT


class IsolationForestEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    available: bool
    anomaly_score: float | None = None
    anomaly_flag: int | None = None
    is_anomaly: bool = False
    model_name: str = "anomaly_isolation_forest_v2"
    model_version: str = "2.0.0"
    missing_features: list[str] = Field(default_factory=list)
    explanation: str = ""
    disclaimer: str = DISCLAIMER_TEXT


class HistoricalReferenceEngine:
    def __init__(
        self,
        reference_path: Path | None = None,
        model_path: Path | None = None,
    ) -> None:
        self.reference_path = Path(reference_path or DEFAULT_REFERENCE_PATH)
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self._reference_data: dict[str, Any] | None = None
        self._model_bundle: dict[str, Any] | None = None
        self._load_reference_data()
        self._load_model_bundle()

    def _load_reference_data(self) -> None:
        if self.reference_path.exists():
            try:
                with open(self.reference_path, "r", encoding="utf-8") as f:
                    self._reference_data = json.load(f)
            except Exception:
                self._reference_data = None

    def _load_model_bundle(self) -> None:
        if self.model_path.exists():
            try:
                loaded = joblib.load(self.model_path)
                if isinstance(loaded, dict) and "pipeline" in loaded:
                    self._model_bundle = loaded
            except Exception:
                self._model_bundle = None

    @property
    def is_reference_available(self) -> bool:
        return self._reference_data is not None and "metrics" in self._reference_data

    @property
    def is_model_available(self) -> bool:
        return self._model_bundle is not None

    def get_metadata(self) -> dict[str, Any]:
        return {
            "version": self._reference_data.get("version", "1.0.0") if self._reference_data else "unavailable",
            "source_dataset": self._reference_data.get("source_dataset", "unknown") if self._reference_data else "unavailable",
            "total_stays": self._reference_data.get("total_stays", 0) if self._reference_data else 0,
            "total_patient_hours": self._reference_data.get("total_patient_hours", 0) if self._reference_data else 0,
            "reference_metrics_count": len(self._reference_data.get("metrics", {})) if self._reference_data else 0,
            "isolation_forest_available": self.is_model_available,
        }

    def canonicalize_metric(self, metric: str) -> str | None:
        cleaned = metric.strip().lower()
        return CANONICAL_METRIC_MAP.get(cleaned)

    def estimate_percentile(self, metric_stats: dict[str, Any], value: float) -> float:
        """
        Estimate percentile rank from reference distribution quantiles.
        Uses piecewise linear interpolation across key quantiles.
        """
        anchors = [
            (0.0, float(metric_stats.get("min", value))),
            (1.0, float(metric_stats.get("p01", value))),
            (5.0, float(metric_stats.get("p05", value))),
            (10.0, float(metric_stats.get("p10", value))),
            (25.0, float(metric_stats.get("p25", value))),
            (50.0, float(metric_stats.get("p50", value))),
            (75.0, float(metric_stats.get("p75", value))),
            (90.0, float(metric_stats.get("p90", value))),
            (95.0, float(metric_stats.get("p95", value))),
            (99.0, float(metric_stats.get("p99", value))),
            (100.0, float(metric_stats.get("max", value))),
        ]

        if value <= anchors[0][1]:
            return 0.1
        if value >= anchors[-1][1]:
            return 99.9

        for i in range(len(anchors) - 1):
            p_lo, val_lo = anchors[i]
            p_hi, val_hi = anchors[i + 1]
            if val_lo <= value <= val_hi:
                if val_hi == val_lo:
                    return p_lo
                frac = (value - val_lo) / (val_hi - val_lo)
                return round(p_lo + frac * (p_hi - p_lo), 1)

        return 50.0

    def evaluate_value(
        self,
        metric: str,
        value: float,
    ) -> PopulationComparisonResult | None:
        """
        Compare a single metric value against the eICU reference population.
        """
        if not self.is_reference_available or self._reference_data is None:
            return None

        canonical = self.canonicalize_metric(metric)
        if not canonical:
            return None

        metrics_dict = self._reference_data.get("metrics", {})
        stats = metrics_dict.get(canonical)
        if not stats:
            return None

        median = float(stats["median"])
        iqr = float(stats["iqr"])
        mean = float(stats["mean"])
        std = float(stats["std"])

        # Robust z-score using IQR (1.349 * IQR is standard normal equivalent)
        if iqr > 0:
            robust_z = (value - median) / (iqr / 1.349)
        elif std > 0:
            robust_z = (value - mean) / std
        else:
            robust_z = 0.0

        percentile = self.estimate_percentile(stats, value)

        abs_z = abs(robust_z)
        is_anomaly = abs_z >= 3.0 or percentile <= 1.0 or percentile >= 99.0

        if abs_z >= 4.0 or percentile <= 0.5 or percentile >= 99.5:
            severity = "CRITICAL"
        elif abs_z >= 3.0 or percentile <= 1.0 or percentile >= 99.0:
            severity = "HIGH"
        elif abs_z >= 2.5 or percentile <= 2.5 or percentile >= 97.5:
            severity = "MODERATE"
        elif abs_z >= 2.0 or percentile <= 5.0 or percentile >= 95.0:
            severity = "LOW"
        else:
            severity = "NORMAL"

        if is_anomaly:
            anomaly_type = "POPULATION_STATISTICAL_OUTLIER"
            direction = "extremely elevated" if robust_z > 0 else "extremely depressed"
            explanation = (
                f"Unusual relative to historical eICU population reference distribution: "
                f"{metric} of {value:g} is {direction} (eICU percentile: {percentile:.1f}%, "
                f"robust z-score: {robust_z:+.2f}, population median: {median:g}, IQR: {iqr:g})."
            )
            recommendation = (
                f"Verify measurement {metric} against bedside monitors and clinical presentation. "
                "Confirm sensor calibration and check for physiological stressors."
            )
        else:
            anomaly_type = "NONE"
            explanation = (
                f"{metric} of {value:g} falls within standard historical eICU distribution "
                f"(eICU percentile: {percentile:.1f}%, robust z-score: {robust_z:+.2f})."
            )
            recommendation = "Continue standard clinical observation."

        return PopulationComparisonResult(
            metric=metric,
            canonical_metric=canonical,
            measured_value=value,
            reference_mean=mean,
            reference_std=std,
            reference_median=median,
            reference_iqr=iqr,
            reference_p05=float(stats.get("p05", 0)),
            reference_p25=float(stats.get("p25", 0)),
            reference_p75=float(stats.get("p75", 0)),
            reference_p95=float(stats.get("p95", 0)),
            percentile_rank=percentile,
            robust_z_score=round(robust_z, 3),
            is_anomaly=is_anomaly,
            severity=severity,
            anomaly_type=anomaly_type,
            explanation=explanation,
            clinical_validation_recommendation=recommendation,
        )

    def evaluate_isolation_forest(
        self,
        features: dict[str, float | None],
    ) -> IsolationForestEvaluationResult:
        """
        Multivariate anomaly score evaluation using trained Isolation Forest v2.
        Requires the 22 feature columns.
        """
        if not self.is_model_available or self._model_bundle is None:
            return IsolationForestEvaluationResult(
                available=False,
                explanation="Isolation Forest model artifact not loaded.",
            )

        pipeline = self._model_bundle.get("pipeline")
        expected_cols = self._model_bundle.get("feature_columns") or []

        missing = [col for col in expected_cols if col not in features or features[col] is None]
        if missing:
            return IsolationForestEvaluationResult(
                available=True,
                missing_features=missing,
                explanation=(
                    f"Multivariate Isolation Forest requires all 22 feature inputs; "
                    f"{len(missing)} features unavailable ({', '.join(missing[:5])})."
                ),
            )

        import numpy as np

        row = np.array([[float(features[col]) for col in expected_cols]])
        try:
            raw_scores = pipeline.score_samples(row)
            preds = pipeline.predict(row)
            score = float(raw_scores[0])
            flag = int(preds[0])
            is_anomaly = flag == -1

            explanation = (
                "Unusual relative to historical eICU multivariate distribution "
                f"(Isolation Forest anomaly score: {score:.4f}, flag: {flag})."
                if is_anomaly
                else f"Multivariate physiological pattern consistent with eICU baseline (score: {score:.4f})."
            )

            return IsolationForestEvaluationResult(
                available=True,
                anomaly_score=round(score, 4),
                anomaly_flag=flag,
                is_anomaly=is_anomaly,
                missing_features=[],
                explanation=explanation,
            )
        except Exception as err:
            return IsolationForestEvaluationResult(
                available=True,
                explanation=f"Error evaluating Isolation Forest: {err}",
            )


historical_reference_engine = HistoricalReferenceEngine()
