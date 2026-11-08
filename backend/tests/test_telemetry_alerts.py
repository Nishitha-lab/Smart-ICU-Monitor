"""Tests for ESP32 telemetry alert generation, quality masking, and patient isolation."""

from datetime import datetime, timezone
import pytest
from pathlib import Path

from services.patient_service import (
    PatientStore,
    PatientCreate,
    IcuStayCreate,
    DeviceCreate,
    DeviceAssign,
)


@pytest.fixture
def store(tmp_path: Path) -> PatientStore:
    db = tmp_path / "test_patient_telem.sqlite3"
    s = PatientStore(db)
    s.initialize()
    return s


def test_telemetry_generates_critical_fluid_alert(store: PatientStore):
    # Setup patient, stay, device, assignment
    pat = store.create_patient(PatientCreate(name="Test Patient 1", weight_kg=70.0))
    stay = store.create_stay(pat["id"], IcuStayCreate(admission_datetime="2026-11-08T10:00:00Z"))
    store.register_device(DeviceCreate(device_id="ESP32-FLUID", device_name="Device 1", connection_type="serial"))
    store.assign_device("ESP32-FLUID", pat["id"], stay["id"])

    # Ingest reading with critical fluid <= 200 mL
    now = "2026-11-08T11:00:00Z"
    reading = store.ingest_reading("ESP32-FLUID", {
        "fluid_ml": 180.0,
        "bpm": 75.0,
        "spo2": 98.0,
        "device_status": "NORMAL",
        "reading_time": now,
    })
    assert reading["patient_id"] == pat["id"]
    assert reading["stay_id"] == stay["id"]

    # Verify critical fluid alert was generated
    alerts = store.alert_manager.list_alerts(stay_id=stay["id"])
    fluid_alerts = [a for a in alerts if a["measurement_name"] == "IV Fluid Level"]
    assert len(fluid_alerts) == 1
    assert fluid_alerts[0]["severity"] == "CRITICAL"
    assert fluid_alerts[0]["status"] == "ACTIVE"
    assert "180" in fluid_alerts[0]["explanation"]


def test_telemetry_generates_hypoxemia_alert(store: PatientStore):
    pat = store.create_patient(PatientCreate(name="Test Patient 2"))
    stay = store.create_stay(pat["id"], IcuStayCreate(admission_datetime="2026-11-08T10:00:00Z"))
    store.register_device(DeviceCreate(device_id="ESP32-SPO2", device_name="Device 2", connection_type="serial"))
    store.assign_device("ESP32-SPO2", pat["id"], stay["id"])

    # SpO2 = 86% -> CRITICAL alert
    now = "2026-11-08T11:00:00Z"
    store.ingest_reading("ESP32-SPO2", {
        "fluid_ml": 800.0,
        "bpm": 80.0,
        "spo2": 86.0,
        "reading_time": now,
    })

    alerts = store.alert_manager.list_alerts(stay_id=stay["id"])
    spo2_alerts = [a for a in alerts if a["measurement_name"] == "SpO2"]
    assert len(spo2_alerts) == 1
    assert spo2_alerts[0]["severity"] == "CRITICAL"


def test_invalid_sensor_telemetry_does_not_generate_alert(store: PatientStore):
    pat = store.create_patient(PatientCreate(name="Test Patient 3"))
    stay = store.create_stay(pat["id"], IcuStayCreate(admission_datetime="2026-11-08T10:00:00Z"))
    store.register_device(DeviceCreate(device_id="ESP32-INVALID", device_name="Device 3", connection_type="serial"))
    store.assign_device("ESP32-INVALID", pat["id"], stay["id"])

    # Out of range BPM and SpO2 (e.g. sensor disconnected)
    now = "2026-11-08T11:00:00Z"
    reading = store.ingest_reading("ESP32-INVALID", {
        "fluid_ml": 800.0,
        "bpm": 0.0,      # Invalid BPM (< 40)
        "spo2": 0.0,     # Invalid SpO2 (< 70)
        "reading_time": now,
    })
    # Validity masks invalid readings to None
    assert reading["bpm"] is None
    assert reading["spo2"] is None

    # No false physiological bradycardia or hypoxemia alerts should be generated!
    alerts = store.alert_manager.list_alerts(stay_id=stay["id"])
    physio_alerts = [a for a in alerts if a["measurement_name"] in ("Heart Rate", "SpO2")]
    assert len(physio_alerts) == 0


def test_unassigned_telemetry_does_not_attribute_or_alert(store: PatientStore):
    # Device exists but is NOT assigned to any patient/stay
    store.register_device(DeviceCreate(device_id="ESP32-UNASSIGNED", device_name="Unassigned", connection_type="serial"))

    now = "2026-11-08T11:00:00Z"
    reading = store.ingest_reading("ESP32-UNASSIGNED", {
        "fluid_ml": 100.0,  # Would be critical if assigned
        "bpm": 35.0,
        "spo2": 80.0,
        "reading_time": now,
    })
    assert reading["patient_id"] is None
    assert reading["stay_id"] is None

    # No patient alerts exist
    alerts = store.alert_manager.list_alerts()
    patient_alerts = [a for a in alerts if a["patient_id"] is not None]
    assert len(patient_alerts) == 0
