"""
Unit tests for Level 1 Cross-Source Clinical Data Consistency Engine.
"""

from datetime import datetime, timezone, timedelta
import pytest
from services.data_consistency_engine import DataConsistencyEngine, DEFAULT_TOLERANCES


@pytest.fixture
def engine():
    return DataConsistencyEngine()


def test_heart_rate_consistent(engine):
    now = datetime.now(timezone.utc)
    res = engine.evaluate_measurement_pair(
        canonical_metric="heart_rate",
        telemetry_value=82.0,
        telemetry_time=now.isoformat(),
        clinical_value=85.0,
        clinical_time=(now - timedelta(minutes=2)).isoformat(),
        clinical_unit="bpm",
    )
    assert res.status == "CONSISTENT"
    assert res.requires_alert is False
    assert res.alert_severity == "INFO"
    assert res.absolute_difference == 3.0


def test_heart_rate_minor_discrepancy(engine):
    now = datetime.now(timezone.utc)
    res = engine.evaluate_measurement_pair(
        canonical_metric="heart_rate",
        telemetry_value=80.0,
        telemetry_time=now.isoformat(),
        clinical_value=95.0,  # delta = 15 bpm (> 10 bpm, <= 20 bpm)
        clinical_time=(now - timedelta(minutes=5)).isoformat(),
        clinical_unit="bpm",
    )
    assert res.status == "MINOR_DISCREPANCY"
    assert res.requires_alert is True
    assert res.alert_severity == "MODERATE"
    assert res.absolute_difference == 15.0


def test_heart_rate_significant_discrepancy(engine):
    now = datetime.now(timezone.utc)
    res = engine.evaluate_measurement_pair(
        canonical_metric="heart_rate",
        telemetry_value=80.0,
        telemetry_time=now.isoformat(),
        clinical_value=105.0,  # delta = 25 bpm (> 20 bpm, <= 30 bpm)
        clinical_time=(now - timedelta(minutes=5)).isoformat(),
        clinical_unit="bpm",
    )
    assert res.status == "SIGNIFICANT_DISCREPANCY"
    assert res.requires_alert is True
    assert res.alert_severity == "HIGH"
    assert res.absolute_difference == 25.0


def test_heart_rate_critical_discrepancy(engine):
    now = datetime.now(timezone.utc)
    res = engine.evaluate_measurement_pair(
        canonical_metric="heart_rate",
        telemetry_value=80.0,
        telemetry_time=now.isoformat(),
        clinical_value=125.0,  # delta = 45 bpm (> 30 bpm)
        clinical_time=(now - timedelta(minutes=5)).isoformat(),
        clinical_unit="bpm",
    )
    assert res.status == "CRITICAL_DISCREPANCY"
    assert res.requires_alert is True
    assert res.alert_severity == "CRITICAL"
    assert "Urgent bedside verification" in res.recommended_action


def test_spo2_discrepancies(engine):
    now = datetime.now(timezone.utc)
    # Consistent: delta = 1% (<= 3%)
    r_con = engine.evaluate_measurement_pair("spo2", 98.0, now, 97.0, now, "%")
    assert r_con.status == "CONSISTENT"
    assert r_con.requires_alert is False

    # Minor: delta = 4% (> 3%, <= 5%)
    r_min = engine.evaluate_measurement_pair("spo2", 98.0, now, 94.0, now, "%")
    assert r_min.status == "MINOR_DISCREPANCY"
    assert r_min.alert_severity == "MODERATE"

    # Significant: delta = 6% (> 5%, <= 8%)
    r_sig = engine.evaluate_measurement_pair("spo2", 98.0, now, 92.0, now, "%")
    assert r_sig.status == "SIGNIFICANT_DISCREPANCY"
    assert r_sig.alert_severity == "HIGH"

    # Critical: delta = 10% (> 8%)
    r_crit = engine.evaluate_measurement_pair("spo2", 98.0, now, 88.0, now, "%")
    assert r_crit.status == "CRITICAL_DISCREPANCY"
    assert r_crit.alert_severity == "CRITICAL"


def test_urine_output_isolated_from_fluid(engine):
    now = datetime.now(timezone.utc)
    res = engine.evaluate_measurement_pair(
        canonical_metric="fluid",
        telemetry_value=500.0,
        telemetry_time=now,
        clinical_value=200.0,
        clinical_time=now,
        clinical_unit="mL",
        clinical_measurement_name="Urine output",
        category="urine_output",
    )
    assert res.status == "INCOMPATIBLE_METRIC"
    assert "Urine output represents renal elimination" in res.explanation
    assert res.requires_alert is False


def test_temporal_overlap_exceeded(engine):
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(minutes=40)  # > 30 minutes
    res = engine.evaluate_measurement_pair(
        canonical_metric="heart_rate",
        telemetry_value=80.0,
        telemetry_time=now.isoformat(),
        clinical_value=82.0,
        clinical_time=old_time.isoformat(),
        clinical_unit="bpm",
    )
    assert res.status == "INSUFFICIENT_TEMPORAL_OVERLAP"
    assert res.requires_alert is False
    assert "Temporal gap of 40.0 minutes exceeds allowable matching window" in res.explanation


def test_insufficient_data(engine):
    now = datetime.now(timezone.utc)
    res = engine.evaluate_measurement_pair(
        canonical_metric="heart_rate",
        telemetry_value=None,
        telemetry_time=now.isoformat(),
        clinical_value=80.0,
        clinical_time=now.isoformat(),
        clinical_unit="bpm",
    )
    assert res.status == "INSUFFICIENT_DATA"
    assert res.requires_alert is False


def test_incompatible_units(engine):
    now = datetime.now(timezone.utc)
    res = engine.evaluate_measurement_pair(
        canonical_metric="heart_rate",
        telemetry_value=80.0,
        telemetry_time=now.isoformat(),
        clinical_value=80.0,
        clinical_time=now.isoformat(),
        clinical_unit="mmHg",  # wrong unit
    )
    assert res.status == "INCOMPATIBLE_UNITS"
    assert res.requires_alert is True
