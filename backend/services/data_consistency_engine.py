"""
Level 1: Cross-Source Clinical Data Consistency Engine.

Compares concurrent ESP32 telemetry readings (Heart Rate, SpO2, Fluid Volume)
with manually entered clinical observations and laboratory results for the same
patient and ICU admission within a configurable temporal window (default: +-30 minutes).

Enforces strict clinical distinction:
- Urine output represents renal elimination and is NEVER compared to IV fluid reservoir volume.
- Neither source is arbitrarily designated as "correct".
- Discrepancy classifications:
  CONSISTENT, MINOR_DISCREPANCY, SIGNIFICANT_DISCREPANCY, CRITICAL_DISCREPANCY,
  INSUFFICIENT_TEMPORAL_OVERLAP, INSUFFICIENT_DATA, INCOMPATIBLE_UNITS, INCOMPATIBLE_METRIC.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

DiscrepancyLevel = Literal[
    "CONSISTENT",
    "MINOR_DISCREPANCY",
    "SIGNIFICANT_DISCREPANCY",
    "CRITICAL_DISCREPANCY",
    "INSUFFICIENT_TEMPORAL_OVERLAP",
    "INSUFFICIENT_DATA",
    "INCOMPATIBLE_UNITS",
    "INCOMPATIBLE_METRIC",
]


class MetricTolerance(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metric_name: str
    telemetry_unit: str
    clinical_units: list[str]
    minor_threshold: float
    significant_threshold: float
    critical_threshold: float
    max_time_window_seconds: int = 1800  # 30 minutes


DEFAULT_TOLERANCES: dict[str, MetricTolerance] = {
    "heart_rate": MetricTolerance(
        metric_name="Heart Rate",
        telemetry_unit="bpm",
        clinical_units=["bpm", "beats/min"],
        minor_threshold=10.0,
        significant_threshold=20.0,
        critical_threshold=30.0,
        max_time_window_seconds=1800,
    ),
    "spo2": MetricTolerance(
        metric_name="Oxygen Saturation",
        telemetry_unit="%",
        clinical_units=["%"],
        minor_threshold=3.0,
        significant_threshold=5.0,
        critical_threshold=8.0,
        max_time_window_seconds=1800,
    ),
    "fluid": MetricTolerance(
        metric_name="IV Fluid Volume",
        telemetry_unit="mL",
        clinical_units=["mL", "ml"],
        minor_threshold=50.0,
        significant_threshold=150.0,
        critical_threshold=300.0,
        max_time_window_seconds=1800,
    ),
}


class DiscrepancyResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metric: str
    canonical_metric: str
    status: DiscrepancyLevel
    telemetry_value: float | None = None
    telemetry_time: str | None = None
    clinical_value: float | None = None
    clinical_time: str | None = None
    clinical_record_id: str | None = None
    clinical_measurement_name: str | None = None
    time_difference_seconds: float | None = None
    absolute_difference: float | None = None
    relative_difference_pct: float | None = None
    explanation: str
    clinical_rationale: str
    recommended_action: str
    requires_alert: bool = False
    alert_severity: Literal["INFO", "LOW", "MODERATE", "HIGH", "CRITICAL"] = "INFO"


def _parse_iso(timestamp: str | datetime | None) -> datetime | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime):
        return timestamp if timestamp.utcoffset() else timestamp.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(timestamp).strip())
        return dt if dt.utcoffset() else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


class DataConsistencyEngine:
    def __init__(
        self,
        tolerances: dict[str, MetricTolerance] | None = None,
    ) -> None:
        self.tolerances = tolerances or DEFAULT_TOLERANCES

    def get_tolerance(self, metric: str) -> MetricTolerance | None:
        return self.tolerances.get(metric.lower())

    def evaluate_measurement_pair(
        self,
        canonical_metric: str,
        telemetry_value: float | None,
        telemetry_time: str | datetime | None,
        clinical_value: float | None,
        clinical_time: str | datetime | None,
        clinical_unit: str | None = None,
        clinical_record_id: str | None = None,
        clinical_measurement_name: str | None = None,
        category: str | None = None,
    ) -> DiscrepancyResult:
        """
        Compare one telemetry reading with one clinical record.
        """
        metric_key = canonical_metric.lower()

        # Strict isolation of urine output from fluid volume
        if category == "urine_output" or (clinical_measurement_name and "urine" in clinical_measurement_name.lower()):
            if metric_key == "fluid":
                return DiscrepancyResult(
                    metric="IV Fluid Volume vs Urine Output",
                    canonical_metric="fluid",
                    status="INCOMPATIBLE_METRIC",
                    telemetry_value=telemetry_value,
                    clinical_value=clinical_value,
                    clinical_record_id=clinical_record_id,
                    clinical_measurement_name=clinical_measurement_name,
                    explanation=(
                        "Urine output represents renal elimination and cannot be compared "
                        "against IV fluid reservoir telemetry. Renal excretion and IV infusion are distinct physiological streams."
                    ),
                    clinical_rationale="Physiologically distinct parameters: excretion vs intake.",
                    recommended_action="Do not attempt to correlate urine output directly with IV reservoir level.",
                    requires_alert=False,
                    alert_severity="INFO",
                )

        tolerance = self.tolerances.get(metric_key)
        if not tolerance:
            return DiscrepancyResult(
                metric=canonical_metric,
                canonical_metric=canonical_metric,
                status="INCOMPATIBLE_METRIC",
                explanation=f"Metric '{canonical_metric}' is not supported for cross-source consistency checking.",
                clinical_rationale="Unsupported metric mapping.",
                recommended_action="Verify metric configuration.",
            )

        # Check for missing values
        if telemetry_value is None or clinical_value is None:
            return DiscrepancyResult(
                metric=tolerance.metric_name,
                canonical_metric=metric_key,
                status="INSUFFICIENT_DATA",
                telemetry_value=telemetry_value,
                clinical_value=clinical_value,
                clinical_record_id=clinical_record_id,
                clinical_measurement_name=clinical_measurement_name,
                explanation=(
                    f"Insufficient data for {tolerance.metric_name}: "
                    f"Telemetry value is {'present' if telemetry_value is not None else 'missing'}; "
                    f"Clinical value is {'present' if clinical_value is not None else 'missing'}."
                ),
                clinical_rationale="Both sources must have valid numeric readings to establish consistency.",
                recommended_action="Verify that both telemetry sensors and bedside clinical entries are active.",
                requires_alert=False,
                alert_severity="INFO",
            )

        # Check units
        if clinical_unit and tolerance.clinical_units:
            norm_unit = clinical_unit.strip().lower()
            allowed = [u.lower() for u in tolerance.clinical_units]
            if norm_unit not in allowed:
                return DiscrepancyResult(
                    metric=tolerance.metric_name,
                    canonical_metric=metric_key,
                    status="INCOMPATIBLE_UNITS",
                    telemetry_value=telemetry_value,
                    clinical_value=clinical_value,
                    clinical_record_id=clinical_record_id,
                    clinical_measurement_name=clinical_measurement_name,
                    explanation=(
                        f"Unit incompatibility for {tolerance.metric_name}: "
                        f"Clinical record has unit '{clinical_unit}', but expected {tolerance.clinical_units}. "
                        "Automatic unit conversion is prohibited."
                    ),
                    clinical_rationale="Unit discrepancies prevent reliable numerical comparison.",
                    recommended_action="Check unit of entry in the clinical record.",
                    requires_alert=True,
                    alert_severity="LOW",
                )

        # Temporal overlap validation
        dt_telem = _parse_iso(telemetry_time)
        dt_clin = _parse_iso(clinical_time)

        if not dt_telem or not dt_clin:
            return DiscrepancyResult(
                metric=tolerance.metric_name,
                canonical_metric=metric_key,
                status="INSUFFICIENT_DATA",
                telemetry_value=telemetry_value,
                clinical_value=clinical_value,
                clinical_record_id=clinical_record_id,
                clinical_measurement_name=clinical_measurement_name,
                explanation="Valid timestamps for both telemetry and clinical record are required.",
                clinical_rationale="Missing timestamp prevents temporal correlation.",
                recommended_action="Verify recording timestamps.",
                requires_alert=False,
            )

        time_delta_sec = abs((dt_telem - dt_clin).total_seconds())

        if time_delta_sec > tolerance.max_time_window_seconds:
            window_min = tolerance.max_time_window_seconds // 60
            delta_min = round(time_delta_sec / 60, 1)
            return DiscrepancyResult(
                metric=tolerance.metric_name,
                canonical_metric=metric_key,
                status="INSUFFICIENT_TEMPORAL_OVERLAP",
                telemetry_value=telemetry_value,
                telemetry_time=dt_telem.isoformat(),
                clinical_value=clinical_value,
                clinical_time=dt_clin.isoformat(),
                clinical_record_id=clinical_record_id,
                clinical_measurement_name=clinical_measurement_name,
                time_difference_seconds=round(time_delta_sec, 1),
                explanation=(
                    f"Temporal gap of {delta_min} minutes exceeds allowable matching window "
                    f"of {window_min} minutes between telemetry ({dt_telem.strftime('%H:%M:%S')}) "
                    f"and clinical observation ({dt_clin.strftime('%H:%M:%S')})."
                ),
                clinical_rationale=(
                    "Physiological parameters fluctuate over time; measurements separated by "
                    f"more than {window_min} minutes cannot be reliably compared."
                ),
                recommended_action="Record a fresh bedside observation or await current telemetry.",
                requires_alert=False,
                alert_severity="INFO",
            )

        # Compute numerical differences
        abs_diff = abs(telemetry_value - clinical_value)
        mean_val = (abs(telemetry_value) + abs(clinical_value)) / 2.0
        rel_diff_pct = (abs_diff / mean_val * 100.0) if mean_val > 1e-6 else 0.0

        # Discrepancy evaluation against thresholds
        if abs_diff > tolerance.critical_threshold:
            status: DiscrepancyLevel = "CRITICAL_DISCREPANCY"
            severity = "CRITICAL"
            requires_alert = True
        elif abs_diff > tolerance.significant_threshold:
            status = "SIGNIFICANT_DISCREPANCY"
            severity = "HIGH"
            requires_alert = True
        elif abs_diff > tolerance.minor_threshold:
            status = "MINOR_DISCREPANCY"
            severity = "MODERATE"
            requires_alert = True
        else:
            status = "CONSISTENT"
            severity = "INFO"
            requires_alert = False

        # Rationale & Recommendations
        if metric_key == "heart_rate":
            rationale = (
                f"Pulse oximeter plethysmograph ({telemetry_value:g} bpm) vs manual/arterial pulse ({clinical_value:g} bpm). "
                "Discrepancies often arise from sensor motion artifact, peripheral vasoconstriction, arrhythmia, or rapid physiological shifts."
            )
            recommendations = {
                "CONSISTENT": "Telemetry and bedside heart rate observations are consistent.",
                "MINOR_DISCREPANCY": "Inspect pulse oximeter finger clip placement and waveform; confirm radial pulse.",
                "SIGNIFICANT_DISCREPANCY": "Check patient perfusion, reposition oximeter probe, and verify heart rhythm via ECG monitor.",
                "CRITICAL_DISCREPANCY": "Urgent bedside verification required: perform 12-lead ECG, assess hemodynamics, check probe signal quality.",
            }
        elif metric_key == "spo2":
            rationale = (
                f"ESP32 optical sensor ({telemetry_value:g}%) vs manual observation/ABG ({clinical_value:g}%). "
                "Discrepancies may be caused by peripheral hypoperfusion, ambient light interference, nail polish, or acute desaturation."
            )
            recommendations = {
                "CONSISTENT": "Telemetry and bedside oxygen saturation observations are consistent.",
                "MINOR_DISCREPANCY": "Ensure sensor is clean and shielded from strong ambient light; verify finger warmth.",
                "SIGNIFICANT_DISCREPANCY": "Check airway patency, oxygen delivery system, and probe placement; consider arterial blood gas confirmation.",
                "CRITICAL_DISCREPANCY": "Immediate clinical response: assess respiratory effort, auscultate lungs, check oxygen supply, obtain arterial blood gas.",
            }
        elif metric_key == "fluid":
            rationale = (
                f"ESP32 container sensor ({telemetry_value:g} mL) vs charted intake ({clinical_value:g} mL). "
                "Discrepancies may stem from pump flow rate changes, container level calibration, tubing dead space, or charting lag."
            )
            recommendations = {
                "CONSISTENT": "Container telemetry and charted fluid volume are consistent.",
                "MINOR_DISCREPANCY": "Verify infusion pump counter against container markings.",
                "SIGNIFICANT_DISCREPANCY": "Inspect IV line for occlusion or infiltration; check infusion pump programming.",
                "CRITICAL_DISCREPANCY": "Immediate inspection: verify IV line patency, container integrity, and total delivered fluid volume.",
            }
        else:
            rationale = "Cross-source parameter comparison."
            recommendations = {
                "CONSISTENT": "Values are consistent.",
                "MINOR_DISCREPANCY": "Inspect source devices and monitor.",
                "SIGNIFICANT_DISCREPANCY": "Verify bedside parameters.",
                "CRITICAL_DISCREPANCY": "Immediate clinical verification recommended.",
            }

        rec = recommendations.get(status, "Verify bedside reading.")
        diff_str = f"{abs_diff:.1f} {tolerance.telemetry_unit}"
        delta_min = round(time_delta_sec / 60, 1)

        if status == "CONSISTENT":
            explanation = (
                f"{tolerance.metric_name} is consistent across sources (telemetry: {telemetry_value:g} {tolerance.telemetry_unit}, "
                f"clinical: {clinical_value:g} {clinical_unit or tolerance.telemetry_unit}, "
                f"delta: {diff_str}, time gap: {delta_min}m)."
            )
        else:
            explanation = (
                f"{status.replace('_', ' ').title()} in {tolerance.metric_name}: "
                f"Telemetry reading ({telemetry_value:g} {tolerance.telemetry_unit}) differs from "
                f"clinical record ({clinical_value:g} {clinical_unit or tolerance.telemetry_unit}) "
                f"by {diff_str} ({rel_diff_pct:.1f}% relative difference) over a {delta_min}m window. "
                "Neither source is assumed correct."
            )

        return DiscrepancyResult(
            metric=tolerance.metric_name,
            canonical_metric=metric_key,
            status=status,
            telemetry_value=telemetry_value,
            telemetry_time=dt_telem.isoformat(),
            clinical_value=clinical_value,
            clinical_time=dt_clin.isoformat(),
            clinical_record_id=clinical_record_id,
            clinical_measurement_name=clinical_measurement_name,
            time_difference_seconds=round(time_delta_sec, 1),
            absolute_difference=round(abs_diff, 2),
            relative_difference_pct=round(rel_diff_pct, 2),
            explanation=explanation,
            clinical_rationale=rationale,
            recommended_action=rec,
            requires_alert=requires_alert,
            alert_severity=severity,
        )


data_consistency_engine = DataConsistencyEngine()
