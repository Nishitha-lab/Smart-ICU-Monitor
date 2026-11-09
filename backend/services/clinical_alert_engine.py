"""
Clinical Alert Engine for Smart ICU Patient Safety System.

Implements deterministic, clinically defensible rules for:
1. Urine output interpretation (KDIGO criteria, duration-dependent thresholds, weight-adjusted rates).
2. Laboratory result interpretation (canonical test registry, reference ranges, critical thresholds).
3. Clinical observations (vitals: temperature, respiration, blood pressure).

Clinical Sources:
- KDIGO Clinical Practice Guideline for Acute Kidney Injury (Kidney Int Suppl. 2012;2:1-138).
- Tietz Textbook of Clinical Chemistry and Molecular Diagnostics, 6th Ed.
- Surviving Sepsis Campaign International Guidelines 2021.
- American Heart Association / ACC Blood Pressure Guidelines.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


InterpretationStatus = Literal[
    "NORMAL",
    "LOW",
    "HIGH",
    "CRITICAL_LOW",
    "CRITICAL_HIGH",
    "UNINTERPRETABLE",
]

AlertSeverity = Literal[
    "INFO",
    "LOW",
    "MODERATE",
    "HIGH",
    "CRITICAL",
]


class ClinicalInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: InterpretationStatus
    severity: AlertSeverity | None = None
    category: str
    measurement_name: str
    canonical_test_id: str
    measured_value: float | str
    unit: str | None
    reference_range: str | None
    critical_threshold: str | None = None
    reference_source: str
    explanation: str
    requires_alert: bool = False
    evaluation_version: str = "1.0.0"
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    details: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# CANONICAL LAB REGISTRY
# ============================================================

LAB_REGISTRY: dict[str, dict[str, Any]] = {
    "POTASSIUM": {
        "display_name": "Potassium",
        "synonyms": ["potassium", "k", "k+", "serum potassium", "plasma potassium"],
        "category": "electrolyte",
        "standard_unit": "mmol/L",
        "acceptable_units": ["mmol/L", "mEq/L"],
        "normal_min": 3.5,
        "normal_max": 5.0,
        "critical_low": 2.8,
        "critical_high": 6.0,
        "source": "Tietz Textbook of Clinical Chemistry / Mayo Clinic Laboratories",
        "notes": "Hyperkalemia > 6.0 mmol/L risks lethal cardiac arrhythmias; hypokalemia < 2.8 mmol/L risks ventricular irritability.",
    },
    "SODIUM": {
        "display_name": "Sodium",
        "synonyms": ["sodium", "na", "na+", "serum sodium"],
        "category": "electrolyte",
        "standard_unit": "mmol/L",
        "acceptable_units": ["mmol/L", "mEq/L"],
        "normal_min": 135.0,
        "normal_max": 145.0,
        "critical_low": 120.0,
        "critical_high": 160.0,
        "source": "Tietz Textbook of Clinical Chemistry",
        "notes": "Critical hyponatremia (< 120 mmol/L) risks cerebral edema and seizure; severe hypernatremia (> 160 mmol/L) risks osmotic demyelination.",
    },
    "CHLORIDE": {
        "display_name": "Chloride",
        "synonyms": ["chloride", "cl", "cl-", "serum chloride"],
        "category": "electrolyte",
        "standard_unit": "mmol/L",
        "acceptable_units": ["mmol/L", "mEq/L"],
        "normal_min": 96.0,
        "normal_max": 106.0,
        "critical_low": 80.0,
        "critical_high": 125.0,
        "source": "Tietz Textbook of Clinical Chemistry",
        "notes": "Key extracellular anion; extreme levels reflect severe acid-base disturbance or fluid imbalance.",
    },
    "BICARBONATE": {
        "display_name": "Bicarbonate",
        "synonyms": ["bicarbonate", "hco3", "hco3-", "total co2", "co2"],
        "category": "electrolyte",
        "standard_unit": "mmol/L",
        "acceptable_units": ["mmol/L", "mEq/L"],
        "normal_min": 22.0,
        "normal_max": 29.0,
        "critical_low": 10.0,
        "critical_high": 40.0,
        "source": "Tietz Textbook of Clinical Chemistry",
        "notes": "Critical low (< 10 mmol/L) indicates profound metabolic acidosis.",
    },
    "CREATININE": {
        "display_name": "Creatinine",
        "synonyms": ["creatinine", "creat", "cr", "serum creatinine"],
        "category": "renal",
        "standard_unit": "mg/dL",
        "acceptable_units": ["mg/dL"],
        "normal_min": 0.6,
        "normal_max": 1.2,
        "critical_low": None,
        "critical_high": 4.0,
        "source": "KDIGO Clinical Practice Guideline for AKI",
        "notes": "Marked elevation (>= 4.0 mg/dL) represents KDIGO Stage 3 acute kidney failure or severe end-stage renal disease.",
    },
    "BUN": {
        "display_name": "BUN",
        "synonyms": ["bun", "blood urea nitrogen", "urea nitrogen"],
        "category": "renal",
        "standard_unit": "mg/dL",
        "acceptable_units": ["mg/dL"],
        "normal_min": 7.0,
        "normal_max": 20.0,
        "critical_low": None,
        "critical_high": 60.0,
        "source": "Tietz Textbook of Clinical Chemistry",
        "notes": "BUN > 60 mg/dL reflects severe azotemia, dehydration, or gastrointestinal bleeding.",
    },
    "UREA": {
        "display_name": "Urea",
        "synonyms": ["urea", "serum urea"],
        "category": "renal",
        "standard_unit": "mg/dL",
        "acceptable_units": ["mg/dL", "mmol/L"],
        "normal_min": 15.0,
        "normal_max": 45.0,
        "critical_low": None,
        "critical_high": 100.0,
        "source": "Tietz Textbook of Clinical Chemistry",
        "notes": "Severe uremia risks uremic pericarditis and encephalopathy.",
    },
    "HEMOGLOBIN": {
        "display_name": "Hemoglobin",
        "synonyms": ["hemoglobin", "haemoglobin", "hgb", "hb"],
        "category": "hematology",
        "standard_unit": "g/dL",
        "acceptable_units": ["g/dL"],
        "normal_min": 12.0,
        "normal_max": 17.5,
        "critical_low": 7.0,
        "critical_high": 20.0,
        "source": "Surviving Sepsis / AABB Transfusion Guidelines",
        "notes": "Hemoglobin < 7.0 g/dL is the standard ICU restrictive transfusion trigger.",
    },
    "HEMATOCRIT": {
        "display_name": "Hematocrit",
        "synonyms": ["hematocrit", "haematocrit", "hct"],
        "category": "hematology",
        "standard_unit": "%",
        "acceptable_units": ["%"],
        "normal_min": 36.0,
        "normal_max": 50.0,
        "critical_low": 20.0,
        "critical_high": 60.0,
        "source": "Tietz Textbook of Clinical Chemistry",
        "notes": "Critical low reflects severe anemia or hemorrhage; critical high indicates hemoconcentration or polycythemia.",
    },
    "WBC": {
        "display_name": "White blood cell count",
        "synonyms": ["white blood cell count", "wbc", "leukocyte count", "white count"],
        "category": "hematology",
        "standard_unit": "K/uL",
        "acceptable_units": ["K/uL", "x10^3/uL", "10^3/uL", "10^9/L", "/uL"],
        "normal_min": 4.0,
        "normal_max": 11.0,
        "critical_low": 2.0,
        "critical_high": 30.0,
        "source": "Tietz Textbook of Clinical Chemistry",
        "notes": "Critical low (< 2.0 K/uL) indicates profound neutropenia; critical high (> 30.0 K/uL) suggests severe sepsis or leukemoid reaction.",
    },
    "PLATELETS": {
        "display_name": "Platelet count",
        "synonyms": ["platelet count", "platelets", "plt"],
        "category": "hematology",
        "standard_unit": "K/uL",
        "acceptable_units": ["K/uL", "x10^3/uL", "10^3/uL", "10^9/L", "/uL"],
        "normal_min": 150.0,
        "normal_max": 450.0,
        "critical_low": 20.0,
        "critical_high": 1000.0,
        "source": "AABB Platelet Transfusion Guidelines",
        "notes": "Platelets < 20 K/uL risks spontaneous life-threatening hemorrhage.",
    },
    "GLUCOSE": {
        "display_name": "Blood glucose",
        "synonyms": ["blood glucose", "glucose", "glu", "fasting glucose", "random glucose", "blood sugar"],
        "category": "metabolic",
        "standard_unit": "mg/dL",
        "acceptable_units": ["mg/dL"],
        "normal_min": 70.0,
        "normal_max": 140.0,
        "critical_low": 50.0,
        "critical_high": 400.0,
        "source": "American Diabetes Association / Critical Care Guidelines",
        "notes": "Glucose < 50 mg/dL causes neuroglycopenia and loss of consciousness; glucose > 400 mg/dL indicates hyperosmolar or ketoacidotic emergency.",
    },
    "CALCIUM": {
        "display_name": "Calcium",
        "synonyms": ["calcium", "ca", "serum calcium", "total calcium"],
        "category": "metabolic",
        "standard_unit": "mg/dL",
        "acceptable_units": ["mg/dL"],
        "normal_min": 8.5,
        "normal_max": 10.5,
        "critical_low": 6.5,
        "critical_high": 13.0,
        "source": "Tietz Textbook of Clinical Chemistry",
        "notes": "Critical hypocalcemia (< 6.5 mg/dL) risks tetany and arrhythmia; critical hypercalcemia (> 13.0 mg/dL) causes renal failure and coma.",
    },
    "MAGNESIUM": {
        "display_name": "Magnesium",
        "synonyms": ["magnesium", "mg", "serum magnesium"],
        "category": "metabolic",
        "standard_unit": "mg/dL",
        "acceptable_units": ["mg/dL"],
        "normal_min": 1.7,
        "normal_max": 2.2,
        "critical_low": 1.0,
        "critical_high": 4.5,
        "source": "Tietz Textbook of Clinical Chemistry",
        "notes": "Hypomagnesemia predisposes to refractory hypokalemia and ventricular arrhythmias (Torsades).",
    },
    "LACTATE": {
        "display_name": "Lactate",
        "synonyms": ["lactate", "lactic acid", "serum lactate", "plasma lactate"],
        "category": "metabolic",
        "standard_unit": "mmol/L",
        "acceptable_units": ["mmol/L", "mg/dL"],
        "normal_min": 0.5,
        "normal_max": 2.0,
        "critical_low": None,
        "critical_high": 4.0,
        "source": "Surviving Sepsis Campaign Guidelines 2021",
        "notes": "Lactate >= 4.0 mmol/L defines severe tissue hypoperfusion and septic shock requiring immediate resuscitation.",
    },
}


def find_canonical_test(name: str) -> tuple[str, dict[str, Any]] | None:
    norm = name.strip().lower()
    for test_id, info in LAB_REGISTRY.items():
        if norm == info["display_name"].lower():
            return test_id, info
        for syn in info["synonyms"]:
            if norm == syn.lower():
                return test_id, info
    return None


def parse_custom_reference_range(ref_str: str | None) -> tuple[float | None, float | None]:
    """Parse custom range string such as '3.5 - 5.0', '< 1.2', '> 10'."""
    if not ref_str:
        return None, None
    s = ref_str.strip()

    range_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[-–—to]+\s*([0-9]+(?:\.[0-9]+)?)", s, re.IGNORECASE)
    if range_match:
        try:
            return float(range_match.group(1)), float(range_match.group(2))
        except ValueError:
            pass

    lt_match = re.search(r"<\s*([0-9]+(?:\.[0-9]+)?)", s)
    if lt_match:
        try:
            return None, float(lt_match.group(1))
        except ValueError:
            pass

    gt_match = re.search(r">\s*([0-9]+(?:\.[0-9]+)?)", s)
    if gt_match:
        try:
            return float(gt_match.group(1)), None
        except ValueError:
            pass

    return None, None


def parse_duration_hours(period_str: str | None) -> float | None:
    """
    Parse duration in hours from string like '1h', '6 hours', '12h', '120 min', '2 hr', '24h'.
    """
    if not period_str:
        return None
    s = period_str.strip().lower()

    try:
        val = float(s)
        if val > 0:
            return val
    except ValueError:
        pass

    min_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:min|mins|minute|minutes|m\b)", s)
    if min_match:
        mins = float(min_match.group(1))
        return round(mins / 60.0, 4) if mins > 0 else None

    hour_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:h|hr|hrs|hour|hours)", s)
    if hour_match:
        hrs = float(hour_match.group(1))
        return hrs if hrs > 0 else None

    day_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:d|day|days)", s)
    if day_match:
        days = float(day_match.group(1))
        return round(days * 24.0, 4) if days > 0 else None

    return None


class ClinicalAlertEngine:
    """
    Modular engine evaluating clinical records for urine output, laboratory results, and vitals.
    """

    VERSION = "1.0.0"

    # ------------------------------------------------------------
    # 1. URINE OUTPUT EVALUATION
    # ------------------------------------------------------------
    def evaluate_urine_output(
        self,
        volume_ml: float,
        unit: str,
        duration_hours: float | None = None,
        patient_weight_kg: float | None = None,
        observation_period: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> ClinicalInterpretation:
        """
        Evaluate urine output using KDIGO AKI guidelines.
        Requires volume in mL. Resolves duration from duration_hours, observation_period, or start/end times.
        """
        if unit.strip().lower() != "ml":
            return ClinicalInterpretation(
                status="UNINTERPRETABLE",
                severity=None,
                category="renal",
                measurement_name="Urine output",
                canonical_test_id="URINE_OUTPUT",
                measured_value=volume_ml,
                unit=unit,
                reference_range=None,
                reference_source="KDIGO Clinical Practice Guideline for AKI (2012)",
                explanation=f"Urine output unit '{unit}' is unsupported; volume must be recorded in mL.",
                requires_alert=False,
                evaluation_version=self.VERSION,
            )

        if volume_ml < 0:
            return ClinicalInterpretation(
                status="UNINTERPRETABLE",
                severity=None,
                category="renal",
                measurement_name="Urine output",
                canonical_test_id="URINE_OUTPUT",
                measured_value=volume_ml,
                unit="mL",
                reference_range=None,
                reference_source="KDIGO Clinical Practice Guideline for AKI (2012)",
                explanation="Urine output volume cannot be negative.",
                requires_alert=False,
                evaluation_version=self.VERSION,
            )

        eff_duration: float | None = None
        if duration_hours is not None and duration_hours > 0:
            eff_duration = duration_hours
        elif observation_period:
            eff_duration = parse_duration_hours(observation_period)
        elif start_time and end_time:
            delta_sec = (end_time - start_time).total_seconds()
            if delta_sec > 0:
                eff_duration = round(delta_sec / 3600.0, 4)

        if eff_duration is None or eff_duration <= 0:
            return ClinicalInterpretation(
                status="UNINTERPRETABLE",
                severity=None,
                category="renal",
                measurement_name="Urine output",
                canonical_test_id="URINE_OUTPUT",
                measured_value=volume_ml,
                unit="mL",
                reference_range=None,
                reference_source="KDIGO Clinical Practice Guideline for AKI (2012)",
                explanation=(
                    f"Recorded urine output of {volume_ml:g} mL cannot be evaluated because measurement duration "
                    "is unavailable or zero. Provide observation duration (e.g., '1h', '6 hours') to evaluate rates."
                ),
                requires_alert=False,
                evaluation_version=self.VERSION,
                details={"volume_ml": volume_ml, "duration_available": False},
            )

        hourly_rate = round(volume_ml / eff_duration, 2)

        eff_weight: float | None = None
        if patient_weight_kg is not None and patient_weight_kg > 0:
            eff_weight = patient_weight_kg

        weight_adj_rate: float | None = None
        if eff_weight:
            weight_adj_rate = round(volume_ml / (eff_weight * eff_duration), 3)

        details: dict[str, Any] = {
            "volume_ml": volume_ml,
            "duration_hours": eff_duration,
            "hourly_rate_ml_h": hourly_rate,
            "weight_kg": eff_weight,
            "weight_adjusted_ml_kg_h": weight_adj_rate,
        }

        if eff_weight:
            rate_disp = f"{weight_adj_rate} mL/kg/h ({hourly_rate} mL/h over {eff_duration:g}h, weight: {eff_weight:g} kg)"
            ref_range_disp = "0.5 - 2.0 mL/kg/h"

            if volume_ml == 0 and eff_duration >= 12.0:
                return ClinicalInterpretation(
                    status="CRITICAL_LOW",
                    severity="CRITICAL",
                    category="renal",
                    measurement_name="Urine output",
                    canonical_test_id="URINE_OUTPUT",
                    measured_value=weight_adj_rate,
                    unit="mL/kg/h",
                    reference_range=ref_range_disp,
                    critical_threshold="Anuria for >= 12 hours (KDIGO Stage 3)",
                    reference_source="KDIGO Clinical Practice Guideline for AKI (2012)",
                    explanation=(
                        f"CRITICAL ANURIA (KDIGO Stage 3) — 0 mL urine output recorded over {eff_duration:g} hours. "
                        "Immediate nephrology assessment and catheter patency check required."
                    ),
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                    details=details,
                )

            if weight_adj_rate < 0.3 and eff_duration >= 24.0:
                return ClinicalInterpretation(
                    status="CRITICAL_LOW",
                    severity="CRITICAL",
                    category="renal",
                    measurement_name="Urine output",
                    canonical_test_id="URINE_OUTPUT",
                    measured_value=weight_adj_rate,
                    unit="mL/kg/h",
                    reference_range=ref_range_disp,
                    critical_threshold="< 0.3 mL/kg/h for >= 24 hours (KDIGO Stage 3)",
                    reference_source="KDIGO Clinical Practice Guideline for AKI (2012)",
                    explanation=(
                        f"CRITICAL OLIGURIA (KDIGO Stage 3) — measured {rate_disp}. "
                        "Sustained severe oliguria over 24+ hours indicates acute kidney injury stage 3."
                    ),
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                    details=details,
                )

            if weight_adj_rate < 0.5 and eff_duration >= 12.0:
                return ClinicalInterpretation(
                    status="LOW",
                    severity="HIGH",
                    category="renal",
                    measurement_name="Urine output",
                    canonical_test_id="URINE_OUTPUT",
                    measured_value=weight_adj_rate,
                    unit="mL/kg/h",
                    reference_range=ref_range_disp,
                    critical_threshold="< 0.5 mL/kg/h for >= 12 hours (KDIGO Stage 2)",
                    reference_source="KDIGO Clinical Practice Guideline for AKI (2012)",
                    explanation=(
                        f"SEVERE OLIGURIA (KDIGO Stage 2) — measured {rate_disp}. "
                        "Sustained low urine output over 12+ hours meets KDIGO Stage 2 criteria."
                    ),
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                    details=details,
                )

            if weight_adj_rate < 0.5 and eff_duration >= 6.0:
                return ClinicalInterpretation(
                    status="LOW",
                    severity="HIGH",
                    category="renal",
                    measurement_name="Urine output",
                    canonical_test_id="URINE_OUTPUT",
                    measured_value=weight_adj_rate,
                    unit="mL/kg/h",
                    reference_range=ref_range_disp,
                    critical_threshold="< 0.5 mL/kg/h for >= 6 hours (KDIGO Stage 1)",
                    reference_source="KDIGO Clinical Practice Guideline for AKI (2012)",
                    explanation=(
                        f"SUSTAINED OLIGURIA (KDIGO Stage 1) — measured {rate_disp}. "
                        "Sustained low urine output over >= 6 hours meets KDIGO Stage 1 criteria."
                    ),
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                    details=details,
                )

            if weight_adj_rate < 0.5:
                return ClinicalInterpretation(
                    status="LOW",
                    severity="MODERATE",
                    category="renal",
                    measurement_name="Urine output",
                    canonical_test_id="URINE_OUTPUT",
                    measured_value=weight_adj_rate,
                    unit="mL/kg/h",
                    reference_range=ref_range_disp,
                    critical_threshold="< 0.5 mL/kg/h",
                    reference_source="KDIGO Clinical Practice Guideline for AKI (2012)",
                    explanation=(
                        f"LOW URINE OUTPUT (Screening Warning) — measured {rate_disp}. "
                        "Duration is < 6 hours; this is a screening warning, not a sustained KDIGO diagnostic criterion. "
                        "Monitor output trends over next hours."
                    ),
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                    details=details,
                )

            if weight_adj_rate > 2.5:
                return ClinicalInterpretation(
                    status="HIGH",
                    severity="LOW",
                    category="renal",
                    measurement_name="Urine output",
                    canonical_test_id="URINE_OUTPUT",
                    measured_value=weight_adj_rate,
                    unit="mL/kg/h",
                    reference_range=ref_range_disp,
                    critical_threshold="> 2.5 mL/kg/h",
                    reference_source="General ICU Monitoring Heuristic (Unverified Provenance)",
                    explanation=(
                        f"POLYURIA — measured {rate_disp}. Unusually high urine output (monitoring heuristic; "
                        "not a KDIGO AKI staging criterion). Monitor for dehydration, post-obstructive diuresis, "
                        "or diabetes insipidus."
                    ),
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                    details=details,
                )

            return ClinicalInterpretation(
                status="NORMAL",
                severity=None,
                category="renal",
                measurement_name="Urine output",
                canonical_test_id="URINE_OUTPUT",
                measured_value=weight_adj_rate,
                unit="mL/kg/h",
                reference_range=ref_range_disp,
                reference_source="KDIGO Clinical Practice Guideline for AKI (2012)",
                explanation=f"NORMAL URINE OUTPUT — measured {rate_disp}. Within physiological limits.",
                requires_alert=False,
                evaluation_version=self.VERSION,
                details=details,
            )

        else:
            ref_range_disp = "30 - 150 mL/h (unadjusted; weight unavailable)"

            if volume_ml == 0 and eff_duration >= 6.0:
                return ClinicalInterpretation(
                    status="CRITICAL_LOW",
                    severity="CRITICAL",
                    category="renal",
                    measurement_name="Urine output",
                    canonical_test_id="URINE_OUTPUT",
                    measured_value=hourly_rate,
                    unit="mL/h",
                    reference_range=ref_range_disp,
                    critical_threshold="0 mL over >= 6 hours",
                    reference_source="Standard ICU Volumetric Screening (KDIGO unadjusted fallback)",
                    explanation=(
                        f"CRITICAL ANURIA — 0 mL recorded over {eff_duration:g} hours. "
                        "Patient weight is unavailable; weight-adjusted KDIGO criteria cannot be calculated. "
                        "Immediate clinical evaluation required."
                    ),
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                    details=details,
                )

            if hourly_rate < 30.0:
                severity: AlertSeverity = "HIGH" if eff_duration >= 6.0 else "MODERATE"
                return ClinicalInterpretation(
                    status="LOW",
                    severity=severity,
                    category="renal",
                    measurement_name="Urine output",
                    canonical_test_id="URINE_OUTPUT",
                    measured_value=hourly_rate,
                    unit="mL/h",
                    reference_range=ref_range_disp,
                    critical_threshold="< 30 mL/h",
                    reference_source="Standard ICU Volumetric Screening (KDIGO unadjusted fallback)",
                    explanation=(
                        f"LOW URINE OUTPUT RATE — measured {hourly_rate} mL/h over {eff_duration:g} hours. "
                        "Patient weight is unavailable; weight-adjusted KDIGO assessment cannot be completed. "
                        "Screening rate is below the 30 mL/h adult threshold."
                    ),
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                    details=details,
                )

            if hourly_rate > 200.0:
                return ClinicalInterpretation(
                    status="HIGH",
                    severity="LOW",
                    category="renal",
                    measurement_name="Urine output",
                    canonical_test_id="URINE_OUTPUT",
                    measured_value=hourly_rate,
                    unit="mL/h",
                    reference_range=ref_range_disp,
                    critical_threshold="> 200 mL/h",
                    reference_source="Standard ICU Volumetric Screening (KDIGO unadjusted fallback)",
                    explanation=(
                        f"HIGH URINE OUTPUT RATE (Polyuria) — measured {hourly_rate} mL/h over {eff_duration:g} hours. "
                        "Patient weight is unavailable; monitor fluid balance."
                    ),
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                    details=details,
                )

            return ClinicalInterpretation(
                status="NORMAL",
                severity=None,
                category="renal",
                measurement_name="Urine output",
                canonical_test_id="URINE_OUTPUT",
                measured_value=hourly_rate,
                unit="mL/h",
                reference_range=ref_range_disp,
                reference_source="Standard ICU Volumetric Screening (KDIGO unadjusted fallback)",
                explanation=(
                    f"NORMAL URINE OUTPUT RATE — measured {hourly_rate} mL/h over {eff_duration:g} hours. "
                    "Patient weight is unavailable; weight-adjusted criteria not assessed."
                ),
                requires_alert=False,
                evaluation_version=self.VERSION,
                details=details,
            )

    # ------------------------------------------------------------
    # 2. LABORATORY RESULT EVALUATION
    # ------------------------------------------------------------
    def evaluate_lab_result(
        self,
        measurement_name: str,
        value: float | str,
        unit: str | None,
        reference_range_str: str | None = None,
        patient_sex: str | None = None,
        patient_age: int | None = None,
    ) -> ClinicalInterpretation:
        """
        Evaluate numeric laboratory result against canonical lab registry and custom reference ranges.
        """
        numeric_val: float
        if isinstance(value, (int, float)):
            numeric_val = float(value)
        else:
            try:
                numeric_val = float(str(value).strip())
            except ValueError:
                return ClinicalInterpretation(
                    status="UNINTERPRETABLE",
                    severity=None,
                    category="other_lab",
                    measurement_name=measurement_name,
                    canonical_test_id="UNKNOWN",
                    measured_value=value,
                    unit=unit,
                    reference_range=reference_range_str,
                    reference_source="None",
                    explanation=f"Non-numeric laboratory value '{value}' cannot be clinically evaluated.",
                    requires_alert=False,
                    evaluation_version=self.VERSION,
                )

        canonical = find_canonical_test(measurement_name)
        if not canonical:
            custom_min, custom_max = parse_custom_reference_range(reference_range_str)
            if custom_min is not None or custom_max is not None:
                is_low = custom_min is not None and numeric_val < custom_min
                is_high = custom_max is not None and numeric_val > custom_max
                if is_low:
                    return ClinicalInterpretation(
                        status="LOW",
                        severity="MODERATE",
                        category="other_lab",
                        measurement_name=measurement_name,
                        canonical_test_id="CUSTOM_LAB",
                        measured_value=numeric_val,
                        unit=unit,
                        reference_range=reference_range_str,
                        reference_source="User-Provided Reference Range",
                        explanation=f"{measurement_name} is LOW ({numeric_val} {unit or ''}) based on custom reference range {reference_range_str}.",
                        requires_alert=True,
                        evaluation_version=self.VERSION,
                    )
                if is_high:
                    return ClinicalInterpretation(
                        status="HIGH",
                        severity="MODERATE",
                        category="other_lab",
                        measurement_name=measurement_name,
                        canonical_test_id="CUSTOM_LAB",
                        measured_value=numeric_val,
                        unit=unit,
                        reference_range=reference_range_str,
                        reference_source="User-Provided Reference Range",
                        explanation=f"{measurement_name} is HIGH ({numeric_val} {unit or ''}) based on custom reference range {reference_range_str}.",
                        requires_alert=True,
                        evaluation_version=self.VERSION,
                    )
                return ClinicalInterpretation(
                    status="NORMAL",
                    severity=None,
                    category="other_lab",
                    measurement_name=measurement_name,
                    canonical_test_id="CUSTOM_LAB",
                    measured_value=numeric_val,
                    unit=unit,
                    reference_range=reference_range_str,
                    reference_source="User-Provided Reference Range",
                    explanation=f"{measurement_name} ({numeric_val} {unit or ''}) is within user-provided reference range {reference_range_str}.",
                    requires_alert=False,
                    evaluation_version=self.VERSION,
                )

            return ClinicalInterpretation(
                status="UNINTERPRETABLE",
                severity=None,
                category="other_lab",
                measurement_name=measurement_name,
                canonical_test_id="UNKNOWN",
                measured_value=numeric_val,
                unit=unit,
                reference_range=None,
                reference_source="None",
                explanation=(
                    f"Laboratory test '{measurement_name}' is not in the standard reference registry, and no valid "
                    "reference range was provided. Result cannot be classified."
                ),
                requires_alert=False,
                evaluation_version=self.VERSION,
            )

        test_id, info = canonical

        unit_str = (unit or "").strip()
        acceptable_units = info["acceptable_units"]

        if unit_str == "/uL" and test_id in ("WBC", "PLATELETS"):
            numeric_val = round(numeric_val / 1000.0, 2)
            unit_str = "K/uL"
        elif test_id == "LACTATE" and unit_str == "mg/dL":
            # 1 mmol/L lactate = 9.008 mg/dL
            numeric_val = round(numeric_val / 9.008, 2)
            unit_str = "mmol/L"
        elif test_id == "UREA" and unit_str == "mmol/L":
            # 1 mmol/L urea = 6.006 mg/dL
            numeric_val = round(numeric_val * 6.006, 2)
            unit_str = "mg/dL"

        if not unit_str or (unit_str not in acceptable_units and unit_str != "K/uL"):
            return ClinicalInterpretation(
                status="UNINTERPRETABLE",
                severity=None,
                category=info["category"],
                measurement_name=info["display_name"],
                canonical_test_id=test_id,
                measured_value=numeric_val,
                unit=unit,
                reference_range=f"{info['normal_min']} - {info['normal_max']} {info['standard_unit']}",
                reference_source=info["source"],
                explanation=(
                    f"Unsupported unit '{unit}' for {info['display_name']}. "
                    f"Expected unit: {info['standard_unit']} (accepted: {', '.join(acceptable_units)})."
                ),
                requires_alert=False,
                evaluation_version=self.VERSION,
            )

        norm_min: float = info["normal_min"]
        norm_max: float = info["normal_max"]
        crit_low: float | None = info["critical_low"]
        crit_high: float | None = info["critical_high"]

        if test_id == "HEMOGLOBIN" and patient_sex:
            sex_norm = patient_sex.strip().lower()
            if sex_norm == "male":
                norm_min, norm_max = 13.5, 17.5
            elif sex_norm == "female":
                norm_min, norm_max = 12.0, 15.5

        if test_id == "HEMATOCRIT" and patient_sex:
            sex_norm = patient_sex.strip().lower()
            if sex_norm == "male":
                norm_min, norm_max = 41.0, 50.0
            elif sex_norm == "female":
                norm_min, norm_max = 36.0, 44.0

        custom_min, custom_max = parse_custom_reference_range(reference_range_str)
        if custom_min is not None:
            norm_min = custom_min
        if custom_max is not None:
            norm_max = custom_max

        ref_range_display = f"{norm_min} - {norm_max} {unit_str}"

        if crit_low is not None and numeric_val < crit_low:
            return ClinicalInterpretation(
                status="CRITICAL_LOW",
                severity="CRITICAL",
                category=info["category"],
                measurement_name=info["display_name"],
                canonical_test_id=test_id,
                measured_value=numeric_val,
                unit=unit_str,
                reference_range=ref_range_display,
                critical_threshold=f"< {crit_low} {unit_str}",
                reference_source=info["source"],
                explanation=(
                    f"CRITICAL LOW — {info['display_name']}: {numeric_val} {unit_str}. "
                    f"Critical threshold: < {crit_low} {unit_str} ({info['source']}). "
                    f"Clinical implications: {info['notes']} Prompt clinical review and intervention required."
                ),
                requires_alert=True,
                evaluation_version=self.VERSION,
                details={"crit_low": crit_low, "norm_min": norm_min, "norm_max": norm_max},
            )

        if crit_high is not None and numeric_val >= crit_high:
            return ClinicalInterpretation(
                status="CRITICAL_HIGH",
                severity="CRITICAL",
                category=info["category"],
                measurement_name=info["display_name"],
                canonical_test_id=test_id,
                measured_value=numeric_val,
                unit=unit_str,
                reference_range=ref_range_display,
                critical_threshold=f">= {crit_high} {unit_str}",
                reference_source=info["source"],
                explanation=(
                    f"CRITICAL HIGH — {info['display_name']}: {numeric_val} {unit_str}. "
                    f"Critical threshold: >= {crit_high} {unit_str} ({info['source']}). "
                    f"Clinical implications: {info['notes']} Prompt clinical review and intervention required."
                ),
                requires_alert=True,
                evaluation_version=self.VERSION,
                details={"crit_high": crit_high, "norm_min": norm_min, "norm_max": norm_max},
            )

        if numeric_val < norm_min:
            return ClinicalInterpretation(
                status="LOW",
                severity="MODERATE",
                category=info["category"],
                measurement_name=info["display_name"],
                canonical_test_id=test_id,
                measured_value=numeric_val,
                unit=unit_str,
                reference_range=ref_range_display,
                critical_threshold=f"< {crit_low} {unit_str}" if crit_low else None,
                reference_source=info["source"],
                explanation=(
                    f"LOW — {info['display_name']}: {numeric_val} {unit_str} is below normal reference range ({ref_range_display}). "
                    "Review trend and patient clinical status."
                ),
                requires_alert=True,
                evaluation_version=self.VERSION,
                details={"norm_min": norm_min, "norm_max": norm_max},
            )

        if numeric_val > norm_max:
            return ClinicalInterpretation(
                status="HIGH",
                severity="MODERATE",
                category=info["category"],
                measurement_name=info["display_name"],
                canonical_test_id=test_id,
                measured_value=numeric_val,
                unit=unit_str,
                reference_range=ref_range_display,
                critical_threshold=f">= {crit_high} {unit_str}" if crit_high else None,
                reference_source=info["source"],
                explanation=(
                    f"HIGH — {info['display_name']}: {numeric_val} {unit_str} exceeds normal reference range ({ref_range_display}). "
                    "Review trend and patient clinical status."
                ),
                requires_alert=True,
                evaluation_version=self.VERSION,
                details={"norm_min": norm_min, "norm_max": norm_max},
            )

        return ClinicalInterpretation(
            status="NORMAL",
            severity=None,
            category=info["category"],
            measurement_name=info["display_name"],
            canonical_test_id=test_id,
            measured_value=numeric_val,
            unit=unit_str,
            reference_range=ref_range_display,
            reference_source=info["source"],
            explanation=f"NORMAL — {info['display_name']}: {numeric_val} {unit_str} is within reference range ({ref_range_display}).",
            requires_alert=False,
            evaluation_version=self.VERSION,
            details={"norm_min": norm_min, "norm_max": norm_max},
        )

    # ------------------------------------------------------------
    # 3. CLINICAL OBSERVATIONS / VITALS EVALUATION
    # ------------------------------------------------------------
    def evaluate_clinical_observation(
        self,
        measurement_name: str,
        value: float | str,
        unit: str | None,
    ) -> ClinicalInterpretation:
        """
        Evaluate bedside clinical vitals (temperature, heart rate, respiration, blood pressure).
        Narrative notes and consciousness assessments are preserved as INFO/NORMAL without false alarms.
        """
        norm_name = measurement_name.strip()

        if not isinstance(value, (int, float)):
            try:
                num_val = float(str(value).strip())
            except ValueError:
                return ClinicalInterpretation(
                    status="NORMAL",
                    severity=None,
                    category="clinical_observation",
                    measurement_name=measurement_name,
                    canonical_test_id="NARRATIVE_OBSERVATION",
                    measured_value=str(value),
                    unit=unit,
                    reference_range="Descriptive note",
                    reference_source="Bedside Clinical Assessment",
                    explanation=f"Recorded bedside clinical observation: '{value}'.",
                    requires_alert=False,
                    evaluation_version=self.VERSION,
                )
        else:
            num_val = float(value)

        # Body Temperature
        if "temperature" in norm_name.lower():
            if unit != "°C":
                return ClinicalInterpretation(
                    status="UNINTERPRETABLE",
                    severity=None,
                    category="clinical_observation",
                    measurement_name="Body temperature",
                    canonical_test_id="BODY_TEMPERATURE",
                    measured_value=num_val,
                    unit=unit,
                    reference_range="36.0 - 37.5 °C",
                    reference_source="NICE Clinical Guidelines for Fever & Sepsis",
                    explanation=f"Body temperature unit '{unit}' is unsupported; expected °C.",
                    requires_alert=False,
                    evaluation_version=self.VERSION,
                )
            if num_val > 40.0:
                return ClinicalInterpretation(
                    status="CRITICAL_HIGH",
                    severity="CRITICAL",
                    category="clinical_observation",
                    measurement_name="Body temperature",
                    canonical_test_id="BODY_TEMPERATURE",
                    measured_value=num_val,
                    unit="°C",
                    reference_range="36.0 - 37.5 °C",
                    critical_threshold="> 40.0 °C",
                    reference_source="NICE Clinical Guidelines for Fever & Sepsis",
                    explanation=f"CRITICAL HYPERTHERMIA — Body temperature: {num_val} °C. High risk of seizure or malignant hyperthermia.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            if num_val < 35.0:
                return ClinicalInterpretation(
                    status="CRITICAL_LOW",
                    severity="CRITICAL",
                    category="clinical_observation",
                    measurement_name="Body temperature",
                    canonical_test_id="BODY_TEMPERATURE",
                    measured_value=num_val,
                    unit="°C",
                    reference_range="36.0 - 37.5 °C",
                    critical_threshold="< 35.0 °C",
                    reference_source="NICE Clinical Guidelines for Sepsis",
                    explanation=f"CRITICAL HYPOTHERMIA — Body temperature: {num_val} °C. Severe hypothermia risks coagulopathy and arrhythmia.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            if num_val > 38.0:
                return ClinicalInterpretation(
                    status="HIGH",
                    severity="MODERATE",
                    category="clinical_observation",
                    measurement_name="Body temperature",
                    canonical_test_id="BODY_TEMPERATURE",
                    measured_value=num_val,
                    unit="°C",
                    reference_range="36.0 - 37.5 °C",
                    critical_threshold="> 40.0 °C",
                    reference_source="NICE Clinical Guidelines for Fever & Sepsis",
                    explanation=f"FEVER — Body temperature: {num_val} °C. Monitor for infectious or inflammatory etiology.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            if num_val < 36.0:
                return ClinicalInterpretation(
                    status="LOW",
                    severity="LOW",
                    category="clinical_observation",
                    measurement_name="Body temperature",
                    canonical_test_id="BODY_TEMPERATURE",
                    measured_value=num_val,
                    unit="°C",
                    reference_range="36.0 - 37.5 °C",
                    critical_threshold="< 35.0 °C",
                    reference_source="NICE Clinical Guidelines for Sepsis",
                    explanation=f"MILD HYPOTHERMIA — Body temperature: {num_val} °C.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            return ClinicalInterpretation(
                status="NORMAL",
                severity=None,
                category="clinical_observation",
                measurement_name="Body temperature",
                canonical_test_id="BODY_TEMPERATURE",
                measured_value=num_val,
                unit="°C",
                reference_range="36.0 - 37.5 °C",
                reference_source="NICE Clinical Guidelines for Fever & Sepsis",
                explanation=f"NORMAL TEMPERATURE — {num_val} °C is within normal limits.",
                requires_alert=False,
                evaluation_version=self.VERSION,
            )

        # Respiratory rate
        if "respiratory" in norm_name.lower():
            if num_val > 35:
                return ClinicalInterpretation(
                    status="CRITICAL_HIGH",
                    severity="CRITICAL",
                    category="clinical_observation",
                    measurement_name="Respiratory rate",
                    canonical_test_id="RESPIRATORY_RATE",
                    measured_value=num_val,
                    unit=unit or "breaths/min",
                    reference_range="12 - 20 breaths/min",
                    critical_threshold="> 35 breaths/min",
                    reference_source="Surviving Sepsis / NEWS2 Criteria",
                    explanation=f"CRITICAL TACHYPNEA — Respiratory rate: {num_val:g} breaths/min. Severe respiratory distress or impending failure.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            if num_val < 8:
                return ClinicalInterpretation(
                    status="CRITICAL_LOW",
                    severity="CRITICAL",
                    category="clinical_observation",
                    measurement_name="Respiratory rate",
                    canonical_test_id="RESPIRATORY_RATE",
                    measured_value=num_val,
                    unit=unit or "breaths/min",
                    reference_range="12 - 20 breaths/min",
                    critical_threshold="< 8 breaths/min",
                    reference_source="NEWS2 Criteria",
                    explanation=f"CRITICAL BRADYPNEA — Respiratory rate: {num_val:g} breaths/min. Respiratory depression or arrest risk.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            if num_val > 24:
                return ClinicalInterpretation(
                    status="HIGH",
                    severity="HIGH" if num_val > 30 else "MODERATE",
                    category="clinical_observation",
                    measurement_name="Respiratory rate",
                    canonical_test_id="RESPIRATORY_RATE",
                    measured_value=num_val,
                    unit=unit or "breaths/min",
                    reference_range="12 - 20 breaths/min",
                    critical_threshold="> 35 breaths/min",
                    reference_source="NEWS2 Criteria",
                    explanation=f"TACHYPNEA — Respiratory rate: {num_val:g} breaths/min is elevated.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            if num_val < 12:
                return ClinicalInterpretation(
                    status="LOW",
                    severity="LOW",
                    category="clinical_observation",
                    measurement_name="Respiratory rate",
                    canonical_test_id="RESPIRATORY_RATE",
                    measured_value=num_val,
                    unit=unit or "breaths/min",
                    reference_range="12 - 20 breaths/min",
                    critical_threshold="< 8 breaths/min",
                    reference_source="NEWS2 Criteria",
                    explanation=f"LOW RESPIRATORY RATE — {num_val:g} breaths/min.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            return ClinicalInterpretation(
                status="NORMAL",
                severity=None,
                category="clinical_observation",
                measurement_name="Respiratory rate",
                canonical_test_id="RESPIRATORY_RATE",
                measured_value=num_val,
                unit=unit or "breaths/min",
                reference_range="12 - 20 breaths/min",
                reference_source="NEWS2 Criteria",
                explanation=f"NORMAL RESPIRATION — {num_val:g} breaths/min is within physiological limits.",
                requires_alert=False,
                evaluation_version=self.VERSION,
            )

        # Systolic blood pressure
        if "systolic" in norm_name.lower():
            if num_val >= 180:
                return ClinicalInterpretation(
                    status="CRITICAL_HIGH",
                    severity="CRITICAL",
                    category="clinical_observation",
                    measurement_name="Systolic blood pressure",
                    canonical_test_id="SYSTOLIC_BP",
                    measured_value=num_val,
                    unit=unit or "mmHg",
                    reference_range="90 - 130 mmHg",
                    critical_threshold=">= 180 mmHg",
                    reference_source="AHA/ACC Hypertension Guidelines",
                    explanation=f"HYPERTENSIVE CRISIS — Systolic BP: {num_val:g} mmHg. Risk of acute end-organ damage.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            if num_val < 80:
                return ClinicalInterpretation(
                    status="CRITICAL_LOW",
                    severity="CRITICAL",
                    category="clinical_observation",
                    measurement_name="Systolic blood pressure",
                    canonical_test_id="SYSTOLIC_BP",
                    measured_value=num_val,
                    unit=unit or "mmHg",
                    reference_range="90 - 130 mmHg",
                    critical_threshold="< 80 mmHg",
                    reference_source="Surviving Sepsis / Shock Guidelines",
                    explanation=f"CRITICAL HYPOTENSION — Systolic BP: {num_val:g} mmHg. Immediate vasopressor or volume evaluation required.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            if num_val < 90:
                return ClinicalInterpretation(
                    status="LOW",
                    severity="HIGH",
                    category="clinical_observation",
                    measurement_name="Systolic blood pressure",
                    canonical_test_id="SYSTOLIC_BP",
                    measured_value=num_val,
                    unit=unit or "mmHg",
                    reference_range="90 - 130 mmHg",
                    critical_threshold="< 80 mmHg",
                    reference_source="Shock Guidelines",
                    explanation=f"HYPOTENSION — Systolic BP: {num_val:g} mmHg is below the standard 90 mmHg perfusion threshold.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            if num_val > 140:
                return ClinicalInterpretation(
                    status="HIGH",
                    severity="MODERATE",
                    category="clinical_observation",
                    measurement_name="Systolic blood pressure",
                    canonical_test_id="SYSTOLIC_BP",
                    measured_value=num_val,
                    unit=unit or "mmHg",
                    reference_range="90 - 130 mmHg",
                    critical_threshold=">= 180 mmHg",
                    reference_source="AHA/ACC Hypertension Guidelines",
                    explanation=f"ELEVATED SYSTOLIC BP — {num_val:g} mmHg exceeds 130 mmHg baseline.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            return ClinicalInterpretation(
                status="NORMAL",
                severity=None,
                category="clinical_observation",
                measurement_name="Systolic blood pressure",
                canonical_test_id="SYSTOLIC_BP",
                measured_value=num_val,
                unit=unit or "mmHg",
                reference_range="90 - 130 mmHg",
                reference_source="AHA/ACC Hypertension Guidelines",
                explanation=f"NORMAL BLOOD PRESSURE — Systolic BP: {num_val:g} mmHg is within normal limits.",
                requires_alert=False,
                evaluation_version=self.VERSION,
            )

        # Diastolic blood pressure
        if "diastolic" in norm_name.lower():
            if num_val >= 120:
                return ClinicalInterpretation(
                    status="CRITICAL_HIGH",
                    severity="CRITICAL",
                    category="clinical_observation",
                    measurement_name="Diastolic blood pressure",
                    canonical_test_id="DIASTOLIC_BP",
                    measured_value=num_val,
                    unit=unit or "mmHg",
                    reference_range="60 - 85 mmHg",
                    critical_threshold=">= 120 mmHg",
                    reference_source="AHA/ACC Hypertension Guidelines",
                    explanation=f"CRITICAL DIASTOLIC HYPERTENSION — Diastolic BP: {num_val:g} mmHg.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            if num_val < 50:
                return ClinicalInterpretation(
                    status="LOW",
                    severity="MODERATE",
                    category="clinical_observation",
                    measurement_name="Diastolic blood pressure",
                    canonical_test_id="DIASTOLIC_BP",
                    measured_value=num_val,
                    unit=unit or "mmHg",
                    reference_range="60 - 85 mmHg",
                    critical_threshold="< 50 mmHg",
                    reference_source="AHA/ACC Guidelines",
                    explanation=f"LOW DIASTOLIC BP — Diastolic BP: {num_val:g} mmHg.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            if num_val > 90:
                return ClinicalInterpretation(
                    status="HIGH",
                    severity="LOW",
                    category="clinical_observation",
                    measurement_name="Diastolic blood pressure",
                    canonical_test_id="DIASTOLIC_BP",
                    measured_value=num_val,
                    unit=unit or "mmHg",
                    reference_range="60 - 85 mmHg",
                    critical_threshold=">= 120 mmHg",
                    reference_source="AHA/ACC Guidelines",
                    explanation=f"ELEVATED DIASTOLIC BP — {num_val:g} mmHg.",
                    requires_alert=True,
                    evaluation_version=self.VERSION,
                )
            return ClinicalInterpretation(
                status="NORMAL",
                severity=None,
                category="clinical_observation",
                measurement_name="Diastolic blood pressure",
                canonical_test_id="DIASTOLIC_BP",
                measured_value=num_val,
                unit=unit or "mmHg",
                reference_range="60 - 85 mmHg",
                reference_source="AHA/ACC Guidelines",
                explanation=f"NORMAL DIASTOLIC BP — {num_val:g} mmHg is within normal limits.",
                requires_alert=False,
                evaluation_version=self.VERSION,
            )

        return ClinicalInterpretation(
            status="NORMAL",
            severity=None,
            category="clinical_observation",
            measurement_name=measurement_name,
            canonical_test_id="GENERAL_OBSERVATION",
            measured_value=num_val,
            unit=unit,
            reference_range="Physiological observation",
            reference_source="Clinical Observation Record",
            explanation=f"Recorded clinical observation for {measurement_name}: {num_val} {unit or ''}.",
            requires_alert=False,
            evaluation_version=self.VERSION,
        )


clinical_alert_engine = ClinicalAlertEngine()
