"""
Unit tests for Level 3 Historical eICU Population-Level Anomaly Detection Engine.
"""

import pytest
from services.historical_reference_engine import HistoricalReferenceEngine


@pytest.fixture
def engine():
    return HistoricalReferenceEngine()


def test_reference_metadata_loaded(engine):
    meta = engine.get_metadata()
    assert meta["version"] == "1.0.0"
    assert meta["total_patient_hours"] == 139432
    assert meta["total_stays"] == 2419
    assert meta["reference_metrics_count"] >= 10


def test_normal_heart_rate_evaluation(engine):
    # Median is ~84.4, IQR ~24.5
    res = engine.evaluate_value("Heart rate", 82.0)
    assert res is not None
    assert res.is_anomaly is False
    assert res.severity == "NORMAL"
    assert res.anomaly_type == "NONE"
    assert -1.0 <= res.robust_z_score <= 1.0
    assert 30.0 <= res.percentile_rank <= 60.0


def test_extreme_tachycardia_population_anomaly(engine):
    # HR 155 is far beyond p99 (130)
    res = engine.evaluate_value("Heart rate", 155.0)
    assert res is not None
    assert res.is_anomaly is True
    assert res.severity in ("HIGH", "CRITICAL")
    assert res.anomaly_type == "POPULATION_STATISTICAL_OUTLIER"
    assert "Unusual relative to historical eICU population reference distribution" in res.explanation
    assert res.percentile_rank >= 99.0
    assert res.robust_z_score >= 3.0


def test_extreme_hypoxemia_population_anomaly(engine):
    # SpO2 75 is far below p01 (86)
    res = engine.evaluate_value("Oxygen saturation", 75.0)
    assert res is not None
    assert res.is_anomaly is True
    assert res.severity in ("HIGH", "CRITICAL")
    assert res.percentile_rank <= 1.0
    assert "Unusual relative to historical eICU population reference distribution" in res.explanation


def test_isolation_forest_partial_and_complete(engine):
    if not engine.is_model_available:
        pytest.skip("Isolation Forest model artifact not deployed.")

    # Incomplete features
    partial = {"heart_rate": 80.0, "spo2": 98.0}
    r_partial = engine.evaluate_isolation_forest(partial)
    assert r_partial.available is True
    assert len(r_partial.missing_features) > 0
    assert "requires all 22 feature inputs" in r_partial.explanation

    # Complete features
    complete = {
        "heart_rate": 85.0,
        "spo2": 97.0,
        "respiration": 20.0,
        "systolic_bp": 120.0,
        "diastolic_bp": 70.0,
        "mean_bp": 85.0,
        "heart_rate_change_1h": 0.0,
        "spo2_change_1h": 0.0,
        "respiration_change_1h": 0.0,
        "systolic_bp_change_1h": 0.0,
        "diastolic_bp_change_1h": 0.0,
        "mean_bp_change_1h": 0.0,
        "heart_rate_rolling_mean_3h": 85.0,
        "spo2_rolling_mean_3h": 97.0,
        "respiration_rolling_mean_3h": 20.0,
        "mean_bp_rolling_mean_3h": 85.0,
        "net_fluid_balance_ml": 0.0,
        "urine_output_ml": 50.0,
        "active_infusion_count": 0.0,
        "vasoactive_support": 0.0,
        "clinical_warning_count": 0.0,
        "clinical_severe_count": 0.0,
    }
    r_complete = engine.evaluate_isolation_forest(complete)
    assert r_complete.available is True
    assert len(r_complete.missing_features) == 0
    assert r_complete.anomaly_score is not None
    assert r_complete.anomaly_flag in (-1, 1)
