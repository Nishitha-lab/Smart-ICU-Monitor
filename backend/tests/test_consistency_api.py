"""
Integration tests for Unified Clinical Data Consistency & Intelligence Engine API endpoints.
"""

from datetime import datetime, timezone, timedelta
import pytest
from starlette.testclient import TestClient

import main
from services.patient_service import PatientCreate, IcuStayCreate, DeviceCreate, DeviceAssign
from services.clinical_records import ClinicalRecordCreate


@pytest.fixture
def client(tmp_path, monkeypatch):
    test_db = tmp_path / "test_consistency_api.sqlite3"
    monkeypatch.setattr(main, "CLINICAL_DB_FILE", test_db)
    monkeypatch.setattr(main.patient_store, "database_path", test_db)
    monkeypatch.setattr(main.clinical_record_store, "database_path", test_db)
    monkeypatch.setattr(main.alert_manager, "database_path", test_db)
    monkeypatch.setattr(main.clinical_intelligence, "database_path", test_db)
    monkeypatch.setattr(main.clinical_intelligence, "alert_manager", main.alert_manager)
    monkeypatch.setattr(main.patient_store, "alert_manager", main.alert_manager)

    main.patient_store.initialize()
    main.clinical_record_store.initialize()
    main.alert_manager.initialize()
    main.clinical_intelligence.initialize()

    with TestClient(main.app) as c:
        yield c


def test_consistency_status_endpoint(client):
    res = client.get("/api/consistency/status")
    assert res.status_code == 200
    data = res.json()
    assert data["engine_status"] == "ONLINE"
    assert "reference_population" in data
    assert "tolerances" in data
    assert "heart_rate" in data["tolerances"]
    assert "spo2" in data["tolerances"]
    assert "fluid" in data["tolerances"]


def test_nonexistent_patient_returns_404(client):
    res = client.get("/api/consistency/patient/99999")
    assert res.status_code == 404
    assert "Patient 99999 not found" in res.json()["detail"]


def test_consistency_evaluation_and_alert_lifecycle(client):
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # 1. Create Patient
    pat = main.patient_store.create_patient(
        PatientCreate(name="Consistency Test Patient", age=65, sex="male")
    )
    pat_id = pat["id"]

    # 2. Admit to ICU
    stay = main.patient_store.create_stay(
        pat_id,
        IcuStayCreate(bed_number="Bed-10", ward_or_unit="ICU-A", admission_datetime=now_iso),
    )
    stay_id = stay["id"]

    # 3. Register & Assign Device
    dev_id = "ESP32-TEST-001"
    main.patient_store.register_device(
        DeviceCreate(device_id=dev_id, device_name="Test Monitor", connection_type="serial")
    )
    main.patient_store.assign_device(dev_id, pat_id, stay_id)

    # 4. Ingest Telemetry: HR = 80 bpm, SpO2 = 98%, Fluid = 500 mL
    read_time = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
    main.patient_store.ingest_reading(
        dev_id,
        {
            "bpm": 80.0,
            "spo2": 98.0,
            "fluid_ml": 500.0,
            "reading_time": read_time,
            "connection_status": "CONNECTED",
        },
    )

    # 5. Add Bedside Clinical Record: HR = 125 bpm (Critical discrepancy: delta 45 bpm)
    rec_res = client.post(
        "/api/clinical-records",
        json={
            "patient_id": pat_id,
            "stay_id": stay_id,
            "category": "clinical_observation",
            "measurement_name": "Heart rate",
            "value": 125.0,
            "unit": "bpm",
            "measurement_time": read_time,
            "entered_by": "Nurse Kelly",
        },
    )
    assert rec_res.status_code == 201

    # 6. Query consistency evaluation for patient
    eval_res = client.get(f"/api/consistency/patient/{pat_id}")
    assert eval_res.status_code == 200
    report = eval_res.json()
    assert report["patient_id"] == pat_id
    assert report["stay_id"] == stay_id

    # Verify Level 1 discrepancy
    l1_hr = next(r for r in report["level1_cross_source"] if r["canonical_metric"] == "heart_rate")
    assert l1_hr["status"] == "CRITICAL_DISCREPANCY"
    assert l1_hr["absolute_difference"] == 45.0
    assert l1_hr["requires_alert"] is True

    # 7. Check consistency alerts
    alerts_res = client.get("/api/consistency/alerts?status=ACTIVE")
    assert alerts_res.status_code == 200
    alerts = alerts_res.json()["alerts"]
    assert len(alerts) >= 1
    target_alert = next(a for a in alerts if a["source"] == "CROSS_SOURCE_DISCREPANCY")
    assert target_alert["severity"] == "CRITICAL"
    alert_id = target_alert["alert_id"]

    # 8. Alert Deduplication: Evaluating again should NOT produce duplicate active alert
    client.post(f"/api/consistency/evaluate/{pat_id}")
    alerts_after = client.get("/api/consistency/alerts?status=ACTIVE").json()["alerts"]
    assert len(alerts_after) == len(alerts)

    # 9. Acknowledge Alert
    ack_res = client.post(
        f"/api/consistency/alerts/{alert_id}/acknowledge",
        json={"acknowledged_by": "Dr. Davis", "note": "Checked patient telemetry clip."},
    )
    assert ack_res.status_code == 200
    assert ack_res.json()["status"] == "ACKNOWLEDGED"

    # 10. Resolve Alert
    res_res = client.post(
        f"/api/consistency/alerts/{alert_id}/resolve",
        json={"resolved_by": "Dr. Davis", "note": "Sensor clip repositioned; reading normal."},
    )
    assert res_res.status_code == 200
    assert res_res.json()["status"] == "RESOLVED"

    # 11. Check patient consistency history
    hist_res = client.get(f"/api/consistency/patient/{pat_id}/history")
    assert hist_res.status_code == 200
    hist = hist_res.json()
    assert any(c["metric"] == "Heart Rate" for c in hist["consistency_checks"])
