"""Unit tests for Clinical Alert Engine (KDIGO urine output, laboratory registry, vitals)."""

from datetime import datetime, timezone
import pytest

from services.clinical_alert_engine import ClinicalAlertEngine, clinical_alert_engine


def test_urine_output_normal_with_weight():
    # 60 kg patient produces 120 mL over 2 hours = 1.0 mL/kg/h
    interp = clinical_alert_engine.evaluate_urine_output(
        volume_ml=120.0,
        unit="mL",
        duration_hours=2.0,
        patient_weight_kg=60.0,
    )
    assert interp.status == "NORMAL"
    assert interp.requires_alert is False
    assert interp.details["weight_adjusted_ml_kg_h"] == 1.0
    assert "0.5 - 2.0 mL/kg/h" in interp.reference_range


def test_urine_output_screening_oliguria_short_duration():
    # 70 kg patient produces 50 mL over 2 hours = 0.357 mL/kg/h (< 6h duration)
    interp = clinical_alert_engine.evaluate_urine_output(
        volume_ml=50.0,
        unit="mL",
        duration_hours=2.0,
        patient_weight_kg=70.0,
    )
    assert interp.status == "LOW"
    assert interp.severity == "MODERATE"
    assert interp.requires_alert is True
    assert "Screening Warning" in interp.explanation
    assert "not a sustained KDIGO" in interp.explanation


def test_urine_output_kdigo_stage_1_sustained_6_hours():
    # 60 kg patient produces 150 mL over 6 hours = 0.417 mL/kg/h (>= 6h)
    interp = clinical_alert_engine.evaluate_urine_output(
        volume_ml=150.0,
        unit="mL",
        duration_hours=6.0,
        patient_weight_kg=60.0,
    )
    assert interp.status == "LOW"
    assert interp.severity == "HIGH"
    assert interp.requires_alert is True
    assert "KDIGO Stage 1" in interp.explanation


def test_urine_output_kdigo_stage_2_sustained_12_hours():
    # 60 kg patient produces 300 mL over 12 hours = 0.417 mL/kg/h (>= 12h)
    interp = clinical_alert_engine.evaluate_urine_output(
        volume_ml=300.0,
        unit="mL",
        duration_hours=12.0,
        patient_weight_kg=60.0,
    )
    assert interp.status == "LOW"
    assert interp.severity == "HIGH"
    assert interp.requires_alert is True
    assert "KDIGO Stage 2" in interp.explanation


def test_urine_output_kdigo_stage_3_severe_24_hours():
    # 70 kg patient produces 350 mL over 24 hours = 0.208 mL/kg/h (< 0.3 over 24h)
    interp = clinical_alert_engine.evaluate_urine_output(
        volume_ml=350.0,
        unit="mL",
        duration_hours=24.0,
        patient_weight_kg=70.0,
    )
    assert interp.status == "CRITICAL_LOW"
    assert interp.severity == "CRITICAL"
    assert interp.requires_alert is True
    assert "KDIGO Stage 3" in interp.explanation


def test_urine_output_kdigo_stage_3_anuria_12_hours():
    # 0 mL over 12 hours
    interp = clinical_alert_engine.evaluate_urine_output(
        volume_ml=0.0,
        unit="mL",
        duration_hours=12.0,
        patient_weight_kg=70.0,
    )
    assert interp.status == "CRITICAL_LOW"
    assert interp.severity == "CRITICAL"
    assert interp.requires_alert is True
    assert "CRITICAL ANURIA" in interp.explanation


def test_urine_output_polyuria():
    # 60 kg patient produces 400 mL over 2 hours = 3.33 mL/kg/h
    interp = clinical_alert_engine.evaluate_urine_output(
        volume_ml=400.0,
        unit="mL",
        duration_hours=2.0,
        patient_weight_kg=60.0,
    )
    assert interp.status == "HIGH"
    assert interp.requires_alert is True
    assert "POLYURIA" in interp.explanation


def test_urine_output_missing_weight():
    # Weight unavailable: calculates volumetric rate mL/h
    interp = clinical_alert_engine.evaluate_urine_output(
        volume_ml=40.0,
        unit="mL",
        duration_hours=2.0,
        patient_weight_kg=None,
    )
    assert interp.status == "LOW"
    assert interp.details["weight_kg"] is None
    assert "Patient weight is unavailable" in interp.explanation
    assert interp.details["hourly_rate_ml_h"] == 20.0


def test_urine_output_missing_duration():
    # Duration missing or not parseable
    interp = clinical_alert_engine.evaluate_urine_output(
        volume_ml=100.0,
        unit="mL",
        duration_hours=None,
        observation_period=None,
    )
    assert interp.status == "UNINTERPRETABLE"
    assert interp.requires_alert is False
    assert "duration is unavailable" in interp.explanation


def test_urine_output_parse_observation_period():
    # "6 hours" string parsed to 6.0
    interp = clinical_alert_engine.evaluate_urine_output(
        volume_ml=300.0,
        unit="mL",
        observation_period="6 hours",
        patient_weight_kg=50.0,
    )
    assert interp.status == "NORMAL"
    assert interp.details["duration_hours"] == 6.0


def test_urine_output_negative_volume():
    interp = clinical_alert_engine.evaluate_urine_output(
        volume_ml=-50.0,
        unit="mL",
        duration_hours=2.0,
    )
    assert interp.status == "UNINTERPRETABLE"
    assert "cannot be negative" in interp.explanation


def test_urine_output_invalid_unit():
    interp = clinical_alert_engine.evaluate_urine_output(
        volume_ml=100.0,
        unit="L",
        duration_hours=2.0,
    )
    assert interp.status == "UNINTERPRETABLE"
    assert "unsupported" in interp.explanation


# ------------------------------------------------------------
# LABORATORY TESTS
# ------------------------------------------------------------

def test_lab_potassium_normal():
    interp = clinical_alert_engine.evaluate_lab_result("Potassium", 4.2, "mmol/L")
    assert interp.status == "NORMAL"
    assert interp.requires_alert is False


def test_lab_potassium_low():
    interp = clinical_alert_engine.evaluate_lab_result("Potassium", 3.1, "mmol/L")
    assert interp.status == "LOW"
    assert interp.severity == "MODERATE"
    assert interp.requires_alert is True


def test_lab_potassium_critical_high():
    interp = clinical_alert_engine.evaluate_lab_result("Potassium", 6.5, "mmol/L")
    assert interp.status == "CRITICAL_HIGH"
    assert interp.severity == "CRITICAL"
    assert interp.requires_alert is True
    assert "CRITICAL HIGH" in interp.explanation
    assert interp.critical_threshold == ">= 6.0 mmol/L"


def test_lab_potassium_critical_low():
    interp = clinical_alert_engine.evaluate_lab_result("Potassium", 2.4, "mmol/L")
    assert interp.status == "CRITICAL_LOW"
    assert interp.severity == "CRITICAL"
    assert interp.requires_alert is True
    assert "CRITICAL LOW" in interp.explanation


def test_lab_creatinine_critical_high():
    interp = clinical_alert_engine.evaluate_lab_result("Creatinine", 4.5, "mg/dL")
    assert interp.status == "CRITICAL_HIGH"
    assert interp.severity == "CRITICAL"
    assert interp.requires_alert is True
    assert "KDIGO Stage 3" in interp.explanation


def test_lab_glucose_critical_low():
    interp = clinical_alert_engine.evaluate_lab_result("Blood glucose", 38, "mg/dL")
    assert interp.status == "CRITICAL_LOW"
    assert interp.severity == "CRITICAL"
    assert interp.requires_alert is True
    assert "neuroglycopenia" in interp.explanation


def test_lab_glucose_critical_high():
    interp = clinical_alert_engine.evaluate_lab_result("Blood glucose", 450, "mg/dL")
    assert interp.status == "CRITICAL_HIGH"
    assert interp.severity == "CRITICAL"
    assert interp.requires_alert is True


def test_lab_lactate_critical_high():
    interp = clinical_alert_engine.evaluate_lab_result("Lactate", 4.2, "mmol/L")
    assert interp.status == "CRITICAL_HIGH"
    assert interp.severity == "CRITICAL"
    assert interp.requires_alert is True
    assert "tissue hypoperfusion" in interp.explanation


def test_lab_hemoglobin_critical_low():
    interp = clinical_alert_engine.evaluate_lab_result("Hemoglobin", 6.4, "g/dL")
    assert interp.status == "CRITICAL_LOW"
    assert interp.severity == "CRITICAL"
    assert interp.requires_alert is True
    assert "transfusion" in interp.explanation


def test_lab_wbc_unit_conversion():
    # 15000 /uL converted to 15.0 K/uL -> HIGH
    interp = clinical_alert_engine.evaluate_lab_result("White blood cell count", 15000, "/uL")
    assert interp.status == "HIGH"
    assert interp.measured_value == 15.0
    assert interp.unit == "K/uL"


def test_lab_custom_reference_range():
    # Custom test not in registry, but clinician provided "10 - 20"
    interp = clinical_alert_engine.evaluate_lab_result(
        measurement_name="Troponin I",
        value=25.0,
        unit="ng/mL",
        reference_range_str="10 - 20",
    )
    assert interp.status == "HIGH"
    assert interp.requires_alert is True
    assert "User-Provided Reference Range" in interp.reference_source


def test_lab_unsupported_test_without_range():
    interp = clinical_alert_engine.evaluate_lab_result(
        measurement_name="Unregistered Novel Biomarker",
        value=42.0,
        unit="ng/mL",
        reference_range_str=None,
    )
    assert interp.status == "UNINTERPRETABLE"
    assert interp.requires_alert is False
    assert "not in the standard reference registry" in interp.explanation


def test_lab_invalid_unit():
    interp = clinical_alert_engine.evaluate_lab_result("Potassium", 4.2, "mg/dL")
    assert interp.status == "UNINTERPRETABLE"
    assert "Unsupported unit" in interp.explanation


def test_lab_non_numeric_value():
    interp = clinical_alert_engine.evaluate_lab_result("Potassium", "hemolyzed", "mmol/L")
    assert interp.status == "UNINTERPRETABLE"
    assert "Non-numeric" in interp.explanation


# ------------------------------------------------------------
# CLINICAL OBSERVATIONS / VITALS
# ------------------------------------------------------------

def test_vitals_temperature_critical_hyperthermia():
    interp = clinical_alert_engine.evaluate_clinical_observation("Body temperature", 40.5, "°C")
    assert interp.status == "CRITICAL_HIGH"
    assert interp.severity == "CRITICAL"
    assert interp.requires_alert is True
    assert "CRITICAL HYPERTHERMIA" in interp.explanation


def test_vitals_respiratory_rate_critical_bradypnea():
    interp = clinical_alert_engine.evaluate_clinical_observation("Respiratory rate", 6, "breaths/min")
    assert interp.status == "CRITICAL_LOW"
    assert interp.severity == "CRITICAL"
    assert interp.requires_alert is True
    assert "CRITICAL BRADYPNEA" in interp.explanation


def test_vitals_systolic_bp_critical_hypotension():
    interp = clinical_alert_engine.evaluate_clinical_observation("Systolic blood pressure", 74, "mmHg")
    assert interp.status == "CRITICAL_LOW"
    assert interp.severity == "CRITICAL"
    assert interp.requires_alert is True
    assert "CRITICAL HYPOTENSION" in interp.explanation


def test_vitals_systolic_bp_hypertensive_crisis():
    interp = clinical_alert_engine.evaluate_clinical_observation("Systolic blood pressure", 195, "mmHg")
    assert interp.status == "CRITICAL_HIGH"
    assert interp.severity == "CRITICAL"
    assert interp.requires_alert is True
    assert "HYPERTENSIVE CRISIS" in interp.explanation


def test_vitals_narrative_note():
    interp = clinical_alert_engine.evaluate_clinical_observation("Clinical note", "Patient resting comfortably.", None)
    assert interp.status == "NORMAL"
    assert interp.requires_alert is False


def test_lab_lactate_unit_conversion_mg_dl():
    # 10.0 mg/dL / 9.008 = ~1.11 mmol/L -> within normal limits (0.5 - 2.0 mmol/L)
    interp = clinical_alert_engine.evaluate_lab_result("Lactate", 10.0, "mg/dL")
    assert interp.status == "NORMAL"
    assert interp.measured_value == 1.11
    assert interp.unit == "mmol/L"
    assert interp.requires_alert is False

    # 45.0 mg/dL / 9.008 = ~5.00 mmol/L -> CRITICAL_HIGH (>= 4.0 mmol/L)
    crit = clinical_alert_engine.evaluate_lab_result("Lactate", 45.0, "mg/dL")
    assert crit.status == "CRITICAL_HIGH"
    assert crit.severity == "CRITICAL"
    assert crit.measured_value == 5.0
    assert crit.unit == "mmol/L"
    assert crit.requires_alert is True


def test_lab_urea_unit_conversion_mmol_l():
    # 5.0 mmol/L * 6.006 = 30.03 mg/dL -> within normal limits (15 - 45 mg/dL)
    interp = clinical_alert_engine.evaluate_lab_result("Urea", 5.0, "mmol/L")
    assert interp.status == "NORMAL"
    assert interp.measured_value == 30.03
    assert interp.unit == "mg/dL"
    assert interp.requires_alert is False

    # 20.0 mmol/L * 6.006 = 120.12 mg/dL -> CRITICAL_HIGH (>= 100 mg/dL)
    crit = clinical_alert_engine.evaluate_lab_result("Urea", 20.0, "mmol/L")
    assert crit.status == "CRITICAL_HIGH"
    assert crit.severity == "CRITICAL"
    assert crit.measured_value == 120.12
    assert crit.unit == "mg/dL"
    assert crit.requires_alert is True


def test_urine_output_missing_weight_reference_source_labeled_properly():
    interp = clinical_alert_engine.evaluate_urine_output(
        volume_ml=40.0,
        unit="mL",
        duration_hours=2.0,
        patient_weight_kg=None,
    )
    assert interp.reference_source == "Standard ICU Volumetric Screening (KDIGO unadjusted fallback)"
    assert "Patient weight is unavailable" in interp.explanation
