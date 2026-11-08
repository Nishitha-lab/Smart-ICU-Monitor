"""
Unit tests for Level 2 Patient-Specific Historical Trend Analyzer.
"""

from datetime import datetime, timezone, timedelta
import pytest
from services.patient_trend_analyzer import PatientTrendAnalyzer


@pytest.fixture
def analyzer():
    return PatientTrendAnalyzer(stale_telemetry_threshold_seconds=900)


def test_insufficient_patient_history(analyzer):
    now = datetime.now(timezone.utc)
    # 0 observations
    r0 = analyzer.analyze_trajectory("Heart Rate", [])
    assert r0.status == "INSUFFICIENT_PATIENT_HISTORY"
    assert r0.observation_count == 0
    assert r0.is_anomaly is False

    # 1 observation
    r1 = analyzer.analyze_trajectory("Heart Rate", [(now, 80.0)])
    assert r1.status == "INSUFFICIENT_PATIENT_HISTORY"
    assert r1.observation_count == 1
    assert r1.is_anomaly is False

    # 2 observations
    r2 = analyzer.analyze_trajectory("Heart Rate", [(now - timedelta(minutes=5), 80.0), (now, 82.0)])
    assert r2.status == "INSUFFICIENT_PATIENT_HISTORY"
    assert r2.observation_count == 2
    assert r2.is_anomaly is False
    assert "at least 3 chronological readings" in r2.explanation


def test_stable_trajectory_normal_trend(analyzer):
    now = datetime.now(timezone.utc)
    obs = [
        (now - timedelta(minutes=30), 75.0),
        (now - timedelta(minutes=20), 78.0),
        (now - timedelta(minutes=10), 76.0),
        (now, 77.0),
    ]
    res = analyzer.analyze_trajectory("Heart Rate", obs)
    assert res.status == "NORMAL_TREND"
    assert res.is_anomaly is False
    assert res.severity == "NORMAL"
    assert res.observation_count == 4
    assert res.baseline_mean == 76.5


def test_sudden_tachycardia_spike(analyzer):
    now = datetime.now(timezone.utc)
    obs = [
        (now - timedelta(minutes=30), 75.0),
        (now - timedelta(minutes=20), 78.0),
        (now - timedelta(minutes=10), 80.0),
        (now, 115.0),  # +35 bpm jump in 10 minutes
    ]
    res = analyzer.analyze_trajectory("Heart Rate", obs)
    assert res.status == "SUDDEN_SPIKE"
    assert res.is_anomaly is True
    assert res.severity in ("HIGH", "CRITICAL")
    assert res.requires_alert is True
    assert "Sudden tachycardia spike" in res.explanation


def test_sudden_bradycardia_drop(analyzer):
    now = datetime.now(timezone.utc)
    obs = [
        (now - timedelta(minutes=30), 85.0),
        (now - timedelta(minutes=20), 82.0),
        (now - timedelta(minutes=10), 80.0),
        (now, 48.0),  # -32 bpm drop in 10 minutes
    ]
    res = analyzer.analyze_trajectory("Heart Rate", obs)
    assert res.status == "SUDDEN_DROP"
    assert res.is_anomaly is True
    assert res.severity in ("HIGH", "CRITICAL")
    assert res.requires_alert is True
    assert "Sudden bradycardia drop" in res.explanation


def test_sudden_spo2_desaturation(analyzer):
    now = datetime.now(timezone.utc)
    obs = [
        (now - timedelta(minutes=30), 98.0),
        (now - timedelta(minutes=20), 97.0),
        (now - timedelta(minutes=10), 96.0),
        (now, 90.0),  # -6% drop in 10 minutes
    ]
    res = analyzer.analyze_trajectory("Oxygen Saturation", obs)
    assert res.status == "SUDDEN_DROP"
    assert res.is_anomaly is True
    assert res.requires_alert is True
    assert "Sudden hypoxemia drop" in res.explanation


def test_stale_telemetry_detection(analyzer):
    now = datetime.now(timezone.utc)
    # Stale: 25 minutes ago (> 15 min threshold)
    stale_ts = now - timedelta(minutes=25)
    r_stale = analyzer.check_stale_telemetry(stale_ts, is_device_active=True, reference_time=now)
    assert r_stale is not None
    assert r_stale.status == "STALE_TELEMETRY"
    assert r_stale.is_anomaly is True
    assert r_stale.severity == "HIGH"
    assert "Stale telemetry: No new data received for 25.0 minutes" in r_stale.explanation

    # Fresh: 5 minutes ago (<= 15 min threshold)
    fresh_ts = now - timedelta(minutes=5)
    r_fresh = analyzer.check_stale_telemetry(fresh_ts, is_device_active=True, reference_time=now)
    assert r_fresh is None

    # Inactive device ignores stale check
    r_inactive = analyzer.check_stale_telemetry(stale_ts, is_device_active=False, reference_time=now)
    assert r_inactive is None
