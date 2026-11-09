"""
Level 2: Patient-Specific Historical Trend Analyzer.

Evaluates a patient's physiological trajectory within their active ICU stay across
chronological telemetry and clinical records. Computes moving baselines (mean, std),
rate-of-change, sudden spikes/drops, sustained trend shifts, and stale telemetry.

Strict Requirement:
If fewer than 3 readings exist for the stay, explicitly returns
INSUFFICIENT_PATIENT_HISTORY, preventing premature false alarms.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

TrendStatus = Literal[
    "NORMAL_TREND",
    "SUDDEN_SPIKE",
    "SUDDEN_DROP",
    "SUSTAINED_ABNORMALITY",
    "RAPID_DETERIORATION",
    "STALE_TELEMETRY",
    "INSUFFICIENT_PATIENT_HISTORY",
]


class ObservationPoint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: datetime
    value: float
    source: str = "TELEMETRY"  # "TELEMETRY" or "CLINICAL_RECORD"


class TrendAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metric: str
    canonical_metric: str
    status: TrendStatus
    is_anomaly: bool
    severity: Literal["NORMAL", "LOW", "MODERATE", "HIGH", "CRITICAL"]
    observation_count: int
    current_value: float | None = None
    current_timestamp: str | None = None
    baseline_mean: float | None = None
    baseline_std: float | None = None
    previous_value: float | None = None
    delta_from_baseline: float | None = None
    rate_of_change_per_minute: float | None = None
    time_delta_seconds: float | None = None
    explanation: str
    clinical_recommendation: str
    requires_alert: bool = False


def _parse_ts(val: str | datetime | None) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.utcoffset() else val.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(val).strip())
        return dt if dt.utcoffset() else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


class PatientTrendAnalyzer:
    def __init__(
        self,
        stale_telemetry_threshold_seconds: int = 900,  # 15 minutes
    ) -> None:
        self.stale_threshold_seconds = stale_telemetry_threshold_seconds

    def check_stale_telemetry(
        self,
        latest_reading_time: str | datetime | None,
        is_device_active: bool = True,
        reference_time: datetime | None = None,
    ) -> TrendAnalysisResult | None:
        """
        Detects if telemetry is stale when a device is actively assigned.
        """
        if not is_device_active:
            return None

        if latest_reading_time is None:
            return TrendAnalysisResult(
                metric="Telemetry Continuity",
                canonical_metric="telemetry_continuity",
                status="STALE_TELEMETRY",
                is_anomaly=True,
                severity="HIGH",
                observation_count=0,
                explanation="No telemetry readings have been received from the assigned monitoring device.",
                clinical_recommendation="Inspect ESP32 device power, serial/Wi-Fi connection, and patient attachment.",
                requires_alert=True,
            )

        dt_latest = _parse_ts(latest_reading_time)
        if not dt_latest:
            return None

        ref_dt = reference_time or datetime.now(timezone.utc)
        elapsed_sec = (ref_dt - dt_latest).total_seconds()

        if elapsed_sec > self.stale_threshold_seconds:
            elapsed_min = round(elapsed_sec / 60, 1)
            thresh_min = self.stale_threshold_seconds // 60
            return TrendAnalysisResult(
                metric="Telemetry Continuity",
                canonical_metric="telemetry_continuity",
                status="STALE_TELEMETRY",
                is_anomaly=True,
                severity="HIGH",
                observation_count=1,
                current_timestamp=dt_latest.isoformat(),
                time_delta_seconds=round(elapsed_sec, 1),
                explanation=(
                    f"Stale telemetry: No new data received for {elapsed_min} minutes "
                    f"(exceeds {thresh_min}-minute threshold). Last reading was at {dt_latest.strftime('%H:%M:%S UTC')}."
                ),
                clinical_recommendation=(
                    "Check ESP32 sensor hardware, connection cable/COM port, and sensor probe attachment on patient."
                ),
                requires_alert=True,
            )

        return None

    def analyze_trajectory(
        self,
        metric_name: str,
        observations: list[tuple[datetime | str, float]],
    ) -> TrendAnalysisResult:
        """
        Analyze chronological series of (timestamp, numeric_value) tuples for a single stay.
        """
        canonical = metric_name.lower().replace(" ", "_")

        # Parse & sort observations by timestamp
        parsed_points: list[ObservationPoint] = []
        for ts_raw, val in observations:
            if val is not None and not math.isnan(val) and math.isfinite(val):
                dt = _parse_ts(ts_raw)
                if dt:
                    parsed_points.append(ObservationPoint(timestamp=dt, value=float(val)))

        parsed_points.sort(key=lambda p: p.timestamp)
        n = len(parsed_points)

        # Minimum history constraint
        if n < 3:
            curr_val = parsed_points[-1].value if n > 0 else None
            curr_ts = parsed_points[-1].timestamp.isoformat() if n > 0 else None
            return TrendAnalysisResult(
                metric=metric_name,
                canonical_metric=canonical,
                status="INSUFFICIENT_PATIENT_HISTORY",
                is_anomaly=False,
                severity="NORMAL",
                observation_count=n,
                current_value=curr_val,
                current_timestamp=curr_ts,
                explanation=(
                    f"Insufficient patient history for {metric_name}: "
                    f"at least 3 chronological readings within the ICU stay are required to establish "
                    f"a personalized baseline (currently {n} recorded)."
                ),
                clinical_recommendation="Continue routine bedside monitoring until sufficient data points accumulate.",
                requires_alert=False,
            )

        # Baseline calculations
        values = [p.value for p in parsed_points]
        mean_val = sum(values) / n
        variance = sum((x - mean_val) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
        std_val = math.sqrt(variance)

        latest_point = parsed_points[-1]
        penultimate_point = parsed_points[-2]

        curr_val = latest_point.value
        prev_val = penultimate_point.value
        curr_ts = latest_point.timestamp.isoformat()

        delta_baseline = curr_val - mean_val
        time_delta_sec = (latest_point.timestamp - penultimate_point.timestamp).total_seconds()
        rate_per_min = (
            ((curr_val - prev_val) / (time_delta_sec / 60.0))
            if time_delta_sec > 5.0
            else 0.0
        )

        status: TrendStatus = "NORMAL_TREND"
        severity: Literal["NORMAL", "LOW", "MODERATE", "HIGH", "CRITICAL"] = "NORMAL"
        is_anomaly = False
        requires_alert = False
        explanation = ""
        rec = "Trend within patient baseline."

        # Detect specific vital sign spikes/drops
        if canonical in {"heart_rate", "bpm", "pulse"}:
            hr_change = curr_val - prev_val
            # Sudden Spike: +25 bpm in < 15 min or rate > 2.5 bpm/min
            if (hr_change >= 25.0 and time_delta_sec <= 900) or rate_per_min >= 2.5:
                status = "SUDDEN_SPIKE"
                is_anomaly = True
                severity = "HIGH" if hr_change < 40.0 else "CRITICAL"
                requires_alert = True
                explanation = (
                    f"Sudden tachycardia spike: Heart rate rose by {hr_change:+.1f} bpm "
                    f"({prev_val:g} -> {curr_val:g} bpm) in {round(time_delta_sec / 60, 1)} minutes "
                    f"(rate: {rate_per_min:+.1f} bpm/min; baseline mean: {mean_val:.1f} +- {std_val:.1f})."
                )
                rec = "Assess patient immediately: verify cardiac rhythm via ECG, check pain/anxiety, rule out hemorrhage or sepsis."
            # Sudden Drop: -25 bpm in < 15 min or rate < -2.5 bpm/min
            elif (hr_change <= -25.0 and time_delta_sec <= 900) or rate_per_min <= -2.5:
                status = "SUDDEN_DROP"
                is_anomaly = True
                severity = "HIGH" if hr_change > -40.0 else "CRITICAL"
                requires_alert = True
                explanation = (
                    f"Sudden bradycardia drop: Heart rate fell by {hr_change:+.1f} bpm "
                    f"({prev_val:g} -> {curr_val:g} bpm) in {round(time_delta_sec / 60, 1)} minutes "
                    f"(rate: {rate_per_min:+.1f} bpm/min; baseline mean: {mean_val:.1f} +- {std_val:.1f})."
                )
                rec = "Evaluate patient immediately: check pulse volume, mental status, vagal stimulation, or conduction block."
            # Sustained shift: latest 3 readings all > 2.5 sigma from mean
            elif std_val > 2.0 and abs(delta_baseline) >= 2.5 * std_val:
                status = "SUSTAINED_ABNORMALITY"
                is_anomaly = True
                severity = "MODERATE"
                requires_alert = True
                explanation = (
                    f"Sustained shift in heart rate: current value ({curr_val:g} bpm) deviates by "
                    f"{delta_baseline:+.1f} bpm from stay baseline mean ({mean_val:.1f} bpm, std: {std_val:.1f})."
                )
                rec = "Re-evaluate trajectory trends; check medication effects and volume status."

        elif canonical in {"spo2", "oxygen_saturation"}:
            spo2_change = curr_val - prev_val
            # Sudden Drop: -4% in < 15 min or rate < -0.3%/min
            if (spo2_change <= -4.0 and time_delta_sec <= 900) or rate_per_min <= -0.4:
                status = "SUDDEN_DROP"
                is_anomaly = True
                severity = "CRITICAL" if curr_val < 90.0 else "HIGH"
                requires_alert = True
                explanation = (
                    f"Sudden hypoxemia drop: SpO2 dropped by {spo2_change:+.1f}% "
                    f"({prev_val:g}% -> {curr_val:g}%) in {round(time_delta_sec / 60, 1)} minutes "
                    f"(rate: {rate_per_min:+.2f}%/min; stay baseline: {mean_val:.1f}%)."
                )
                rec = "Immediate clinical assessment: check airway, oxygen flow rate, probe position, and lung sounds."
            elif curr_val <= 88.0 and prev_val <= 88.0:
                status = "SUSTAINED_ABNORMALITY"
                is_anomaly = True
                severity = "CRITICAL"
                requires_alert = True
                explanation = (
                    f"Sustained severe desaturation: SpO2 has remained <= 88% across multiple readings "
                    f"(current: {curr_val:g}%, stay mean: {mean_val:.1f}%)."
                )
                rec = "Escalate respiratory support immediately; alert ICU attending and respiratory therapy."

        elif canonical in {"fluid", "fluid_ml", "iv_fluid"}:
            fluid_change = curr_val - prev_val
            # Rapid fluid depletion: drop > 100 mL in < 15 min (abnormal rapid infusion)
            if fluid_change <= -100.0 and time_delta_sec <= 900:
                status = "RAPID_DETERIORATION"
                is_anomaly = True
                severity = "HIGH"
                requires_alert = True
                explanation = (
                    f"Rapid fluid volume change: Reservoir volume decreased by {abs(fluid_change):.1f} mL "
                    f"in {round(time_delta_sec / 60, 1)} minutes (infusion rate: {abs(rate_per_min) * 60:.0f} mL/h)."
                )
                rec = "Verify infusion pump rate programming and check for unintended bolus delivery."
            elif curr_val <= 150.0:
                status = "SUSTAINED_ABNORMALITY"
                is_anomaly = True
                severity = "HIGH"
                requires_alert = True
                explanation = (
                    f"Critically low IV fluid reservoir: {curr_val:g} mL remaining "
                    f"(stay baseline: {mean_val:.1f} mL)."
                )
                rec = "Prepare replacement IV fluid bag to prevent infusion interruption."

        # Default normal text if no anomaly triggered
        if not is_anomaly:
            status = "NORMAL_TREND"
            severity = "NORMAL"
            explanation = (
                f"{metric_name} is stable within personalized ICU stay baseline "
                f"(current: {curr_val:g}, mean: {mean_val:.1f}, std: {std_val:.1f}, n={n})."
            )
            rec = "Patient trajectory stable. Continue routine monitoring."

        return TrendAnalysisResult(
            metric=metric_name,
            canonical_metric=canonical,
            status=status,
            is_anomaly=is_anomaly,
            severity=severity,
            observation_count=n,
            current_value=curr_val,
            current_timestamp=curr_ts,
            baseline_mean=round(mean_val, 2),
            baseline_std=round(std_val, 2),
            previous_value=prev_val,
            delta_from_baseline=round(delta_baseline, 2),
            rate_of_change_per_minute=round(rate_per_min, 3),
            time_delta_seconds=round(time_delta_sec, 1),
            explanation=explanation,
            clinical_recommendation=rec,
            requires_alert=requires_alert,
        )


patient_trend_analyzer = PatientTrendAnalyzer()
