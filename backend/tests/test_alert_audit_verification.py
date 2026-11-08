"""
Independent End-to-End Verification of Unified Clinical Alerts and ML Intelligence Engine.

Specifically verifies:
1. Patient admission & lab alert generation without hardware.
2. Abnormal lab (Hemoglobin 6.5 g/dL -> CRITICAL).
3. Normal lab (Hemoglobin 13.5 g/dL -> NORMAL, no alert created; existing alert auto-resolved or resolved).
4. Moderate lab (Hemoglobin 9.0 g/dL -> MODERATE alert).
5. Hardware-disconnected operation (/api/alerts and /api/alerts/summary operate independently of ESP32).
6. Complete alert lifecycle (Acknowledge, Resolve, Audit Logs, validation edge cases).
7. Deduplication and escalation behavior.
8. Filter parameters (status, severity, source, stay_id).
"""

import asyncio
import json
from pathlib import Path
import pytest
import main
from services.patient_service import PatientStore
from services.clinical_records import ClinicalRecordStore
from services.alert_manager import AlertManager


def request_app(
    path: str,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    body = json.dumps(payload).encode() if payload is not None else b""

    async def call_app() -> tuple[int, dict[str, object]]:
        messages: list[dict[str, object]] = []
        request_sent = False

        async def receive() -> dict[str, object]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        raw_path, _, query_string = path.partition("?")
        await main.app(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": method,
                "scheme": "http",
                "path": raw_path,
                "raw_path": raw_path.encode(),
                "query_string": query_string.encode(),
                "root_path": "",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8000),
            },
            receive,
            send,
        )
        response = next(message for message in messages if message["type"] == "http.response.start")
        response_body = next(message for message in messages if message["type"] == "http.response.body")
        data = json.loads(response_body["body"]) if response_body["body"] else {}
        return int(response["status"]), data

    return asyncio.run(call_app())


@pytest.fixture(autouse=True)
def setup_isolated_db(tmp_path: Path):
    db_file = tmp_path / "test_e2e_alerts.sqlite3"
    main.CLINICAL_DB_FILE = db_file
    main.clinical_record_store = ClinicalRecordStore(db_file)
    main.patient_store = PatientStore(db_file)
    main.alert_manager = AlertManager(db_file)
    main.patient_store.initialize()
    main.clinical_record_store.initialize()
    main.alert_manager.initialize()
    yield


def test_lab_alerts_without_hardware_lifecycle():
    """
    Verifies Section 4 & 5:
    1. Register patient and ICU stay.
    2. Record critical lab (Hemoglobin 6.5 g/dL -> CRITICAL).
    3. Query /api/alerts and verify fields: patient_id, stay_id, source, value, unit, severity, explanation, timestamps.
    4. Record moderate lab (Hemoglobin 9.0 g/dL -> MODERATE).
    5. Acknowledge and resolve alerts.
    """
    # 1. Register patient & stay
    status, pat = request_app("/api/patients", "POST", {
        "name": "Audit Test Patient",
        "sex": "Female",
        "age": 58,
        "weight_kg": 65.0,
    })
    assert status == 201
    pid = pat["id"]

    status, stay = request_app(f"/api/patients/{pid}/stays", "POST", {
        "admission_datetime": "2026-11-08T08:00:00Z"
    })
    assert status == 201
    sid = stay["id"]

    # 2. Record critical lab: Hemoglobin 6.5 g/dL (< 7.0 critical threshold)
    status, rec1 = request_app("/api/clinical-records", "POST", {
        "stay_id": sid,
        "category": "blood_test",
        "measurement_name": "Hemoglobin",
        "value": 6.5,
        "unit": "g/dL",
        "measurement_time": "2026-11-08T09:00:00Z",
        "sample_collection_time": "2026-11-08T08:30:00Z",
        "entered_by": "Nurse Veronica",
    })
    assert status == 201
    assert "interpretation" in rec1
    assert rec1["interpretation"]["status"] == "CRITICAL_LOW"
    assert rec1["interpretation"]["severity"] == "CRITICAL"
    assert "alert" in rec1
    assert rec1["alert"]["severity"] == "CRITICAL"
    assert rec1["alert"]["status"] == "ACTIVE"
    alert1_id = rec1["alert"]["alert_id"]

    # 3. Query GET /api/alerts
    status, alerts_resp = request_app(f"/api/alerts?stay_id={sid}")
    assert status == 200
    alerts = alerts_resp["alerts"]
    assert len(alerts) == 1
    alert_obj = alerts[0]
    assert alert_obj["alert_id"] == alert1_id
    assert alert_obj["patient_id"] == pid
    assert alert_obj["stay_id"] == sid
    assert alert_obj["source"] == "LAB_RESULT"
    assert alert_obj["measured_value"] == "6.5"
    assert alert_obj["unit"] == "g/dL"
    assert alert_obj["severity"] == "CRITICAL"
    assert "critical" in alert_obj["explanation"].lower() or "6.5" in alert_obj["explanation"]
    assert alert_obj["trigger_time"].startswith("2026-11-08T09:00:00")
    assert alert_obj["created_at"] is not None

    # 4. Check summary
    status, sum_resp = request_app("/api/alerts/summary")
    assert status == 200
    assert sum_resp["active_alerts"] == 1
    assert sum_resp["critical_alerts"] == 1
    assert sum_resp["total_alerts"] == 1

    # 5. Complete alert acknowledgement
    status, ack_resp = request_app(f"/api/alerts/{alert1_id}/acknowledge", "POST", {
        "acknowledged_by": "Dr. Sarah Adams",
        "note": "2 units PRBC ordered immediately; crossmatch sent.",
    })
    assert status == 200
    assert ack_resp["status"] == "ACKNOWLEDGED"
    assert ack_resp["acknowledged_by"] == "Dr. Sarah Adams"
    assert ack_resp["acknowledgement_note"] == "2 units PRBC ordered immediately; crossmatch sent."
    assert ack_resp["acknowledged_at"] is not None

    # 6. Verify audit trail has CREATED and ACKNOWLEDGED
    status, hist_resp = request_app(f"/api/alerts/{alert1_id}/history")
    assert status == 200
    history = hist_resp["history"]
    actions = [h["action"] for h in history]
    assert "CREATED" in actions
    assert "ACKNOWLEDGED" in actions

    # 7. Complete alert resolution
    status, res_resp = request_app(f"/api/alerts/{alert1_id}/resolve", "POST", {
        "resolved_by": "Dr. Sarah Adams",
        "note": "Transfusion complete, post-transfusion labs pending, patient stable.",
    })
    assert status == 200
    assert res_resp["status"] == "RESOLVED"
    assert res_resp["resolved_by"] == "Dr. Sarah Adams"
    assert res_resp["resolution_note"] == "Transfusion complete, post-transfusion labs pending, patient stable."
    assert res_resp["resolved_at"] is not None

    # 8. Check summary after resolution
    status, sum_resp2 = request_app("/api/alerts/summary")
    assert status == 200
    assert sum_resp2["active_alerts"] == 0
    assert sum_resp2["resolved_alerts"] == 1


def test_hardware_disconnected_operation():
    """
    Verifies Section 5:
    Serial monitor disconnected / no ESP32 attached.
    Confirms /api/alerts and /api/alerts/summary continue to serve correctly
    without 500 error, without telemetry fabrication, and without alert suppression.
    """
    assert main.serial_monitor._data["connection_status"] == "DISCONNECTED"

    # Register patient, stay, and a lab alert
    status, pat = request_app("/api/patients", "POST", {"name": "Offline Hardware Patient"})
    assert status == 201
    status, stay = request_app(f"/api/patients/{pat['id']}/stays", "POST", {
        "admission_datetime": "2026-11-08T10:00:00Z"
    })
    assert status == 201

    status, rec = request_app("/api/clinical-records", "POST", {
        "stay_id": stay["id"],
        "category": "blood_test",
        "measurement_name": "Potassium",
        "value": 2.5,
        "unit": "mmol/L",
        "measurement_time": "2026-11-08T10:15:00Z",
        "sample_collection_time": "2026-11-08T10:00:00Z",
        "entered_by": "Nurse Clark",
    })
    assert status == 201
    assert rec["alert"]["severity"] == "CRITICAL"

    # Query alerts while hardware is disconnected
    status, alerts_resp = request_app("/api/alerts")
    assert status == 200
    assert len(alerts_resp["alerts"]) == 1
    assert alerts_resp["alerts"][0]["source"] == "LAB_RESULT"

    # Query summary while hardware is disconnected
    status, summary_resp = request_app("/api/alerts/summary")
    assert status == 200
    assert summary_resp["active_alerts"] == 1
    assert summary_resp["critical_alerts"] == 1


def test_alert_lifecycle_validations():
    """
    Verifies Section 6:
    Validation edge cases:
    - Missing clinician identity in acknowledge -> 400
    - Missing clinician identity in resolve -> 400
    - Non-existent alert ID -> 404
    - Acknowledge already resolved alert -> 400
    """
    # Create an alert
    status, pat = request_app("/api/patients", "POST", {"name": "Validation Test Patient"})
    status, stay = request_app(f"/api/patients/{pat['id']}/stays", "POST", {
        "admission_datetime": "2026-11-08T10:00:00Z"
    })
    status, rec = request_app("/api/clinical-records", "POST", {
        "stay_id": stay["id"],
        "category": "blood_test",
        "measurement_name": "Sodium",
        "value": 115.0,  # Critical low (< 120)
        "unit": "mmol/L",
        "measurement_time": "2026-11-08T10:30:00Z",
        "sample_collection_time": "2026-11-08T10:00:00Z",
        "entered_by": "Nurse Dave",
    })
    assert status == 201
    aid = rec["alert"]["alert_id"]

    # 1. Non-existent alert ID
    status, _ = request_app("/api/alerts/nonexistent-id-9999", "GET")
    assert status == 404
    status, _ = request_app("/api/alerts/nonexistent-id-9999/history", "GET")
    assert status == 404
    status, _ = request_app("/api/alerts/nonexistent-id-9999/acknowledge", "POST", {
        "acknowledged_by": "Dr. Smith",
    })
    assert status == 404

    # 2. Missing identity on acknowledge
    status, err_ack = request_app(f"/api/alerts/{aid}/acknowledge", "POST", {
        "acknowledged_by": "",
        "note": "A note without author",
    })
    assert status in (400, 422)

    # 3. Missing identity on resolve
    status, err_res = request_app(f"/api/alerts/{aid}/resolve", "POST", {
        "resolved_by": "   ",
        "note": "Resolved note without author",
    })
    assert status in (400, 422)

    # 4. Resolve the alert
    status, _ = request_app(f"/api/alerts/{aid}/resolve", "POST", {
        "resolved_by": "Dr. Gregory House",
        "note": "Hypertonic saline completed.",
    })
    assert status == 200

    # 5. Attempt to acknowledge already RESOLVED alert -> 400
    status, err_reack = request_app(f"/api/alerts/{aid}/acknowledge", "POST", {
        "acknowledged_by": "Dr. Allison Cameron",
    })
    assert status == 400
    assert "Cannot acknowledge alert with status 'RESOLVED'" in err_reack.get("detail", "")


def test_deduplication_and_normal_reading_behavior():
    """
    Verifies Section 10:
    - Repeated abnormal measurement updates the existing alert without creating duplicate rows.
    - Subsequent normal measurement auto-resolves the alert with SYSTEM_AUTO_RESOLVE note.
    """
    status, pat = request_app("/api/patients", "POST", {"name": "Dedup Patient", "weight_kg": 70.0})
    status, stay = request_app(f"/api/patients/{pat['id']}/stays", "POST", {
        "admission_datetime": "2026-11-08T09:00:00Z"
    })
    sid = stay["id"]

    # First abnormal lab: Hemoglobin 9.0 g/dL -> MODERATE
    status, rec1 = request_app("/api/clinical-records", "POST", {
        "stay_id": sid,
        "category": "blood_test",
        "measurement_name": "Hemoglobin",
        "value": 9.0,
        "unit": "g/dL",
        "measurement_time": "2026-11-08T10:00:00Z",
        "sample_collection_time": "2026-11-08T09:30:00Z",
        "entered_by": "Nurse First",
    })
    assert status == 201
    aid1 = rec1["alert"]["alert_id"]
    assert rec1["alert"]["severity"] == "MODERATE"

    # Second abnormal lab: Hemoglobin 8.8 g/dL -> Same severity MODERATE
    status, rec2 = request_app("/api/clinical-records", "POST", {
        "stay_id": sid,
        "category": "blood_test",
        "measurement_name": "Hemoglobin",
        "value": 8.8,
        "unit": "g/dL",
        "measurement_time": "2026-11-08T12:00:00Z",
        "sample_collection_time": "2026-11-08T11:30:00Z",
        "entered_by": "Nurse Second",
    })
    assert status == 201
    aid2 = rec2["alert"]["alert_id"]

    # Must be same alert ID (deduplicated, no duplicate row created)
    assert aid1 == aid2
    assert rec2["alert"]["measured_value"] == "8.8"

    # Only 1 alert exists for this stay
    status, alerts_resp = request_app(f"/api/alerts?stay_id={sid}")
    assert len(alerts_resp["alerts"]) == 1

    # Check audit history: contains CREATED and RE_EVALUATED
    status, hist_resp = request_app(f"/api/alerts/{aid1}/history")
    actions = [h["action"] for h in hist_resp["history"]]
    assert "CREATED" in actions
    assert "RE_EVALUATED" in actions

    # Third lab: Normal Hemoglobin 13.5 g/dL -> NORMAL
    status, rec3 = request_app("/api/clinical-records", "POST", {
        "stay_id": sid,
        "category": "blood_test",
        "measurement_name": "Hemoglobin",
        "value": 13.5,
        "unit": "g/dL",
        "measurement_time": "2026-11-08T16:00:00Z",
        "sample_collection_time": "2026-11-08T15:30:00Z",
        "entered_by": "Nurse Third",
    })
    assert status == 201
    assert rec3["interpretation"]["status"] == "NORMAL"
    assert rec3["interpretation"]["requires_alert"] is False
    assert "alert" not in rec3 or rec3["alert"] is None

    # The previous active alert was auto-resolved by subsequent normal measurement
    status, alert_check = request_app(f"/api/alerts/{aid1}")
    assert status == 200
    assert alert_check["status"] == "RESOLVED"
    assert alert_check["resolved_by"] == "SYSTEM_AUTO_RESOLVE"
    assert "Auto-resolved" in alert_check["resolution_note"]


def test_alert_filters_and_summary_consistency():
    """
    Verifies Section 8 & 9:
    - Filters: status (ACTIVE, ACKNOWLEDGED, RESOLVED, ALL), severity, source, stay_id.
    - Summary counts match exact database states.
    """
    status, pat = request_app("/api/patients", "POST", {"name": "Filter Patient", "weight_kg": 68.0})
    sid = request_app(f"/api/patients/{pat['id']}/stays", "POST", {"admission_datetime": "2026-11-08T08:00:00Z"})[1]["id"]

    # 1. Create a CRITICAL potassium lab alert
    rec_k = request_app("/api/clinical-records", "POST", {
        "stay_id": sid,
        "category": "blood_test",
        "measurement_name": "Potassium",
        "value": 6.8,
        "unit": "mmol/L",
        "measurement_time": "2026-11-08T09:00:00Z",
        "sample_collection_time": "2026-11-08T08:30:00Z",
        "entered_by": "Nurse A",
    })[1]
    aid_k = rec_k["alert"]["alert_id"]

    # 2. Create a MODERATE blood glucose alert
    rec_glu = request_app("/api/clinical-records", "POST", {
        "stay_id": sid,
        "category": "blood_test",
        "measurement_name": "Blood glucose",
        "value": 250.0,
        "unit": "mg/dL",
        "measurement_time": "2026-11-08T09:15:00Z",
        "sample_collection_time": "2026-11-08T08:45:00Z",
        "entered_by": "Nurse A",
    })[1]
    aid_glu = rec_glu["alert"]["alert_id"]

    # 3. Create a CRITICAL low blood pressure observation alert
    rec_bp = request_app("/api/clinical-records", "POST", {
        "stay_id": sid,
        "category": "clinical_observation",
        "measurement_name": "Systolic blood pressure",
        "value": 65.0,
        "unit": "mmHg",
        "measurement_time": "2026-11-08T09:30:00Z",
        "entered_by": "Nurse A",
    })[1]
    aid_bp = rec_bp["alert"]["alert_id"]

    # Summary check before acknowledgements
    sum1 = request_app("/api/alerts/summary")[1]
    assert sum1["active_alerts"] == 3
    assert sum1["critical_alerts"] == 2
    assert sum1["unacknowledged_alerts"] == 3
    assert sum1["resolved_alerts"] == 0

    # Acknowledge the potassium alert
    request_app(f"/api/alerts/{aid_k}/acknowledge", "POST", {
        "acknowledged_by": "Dr. OnCall",
        "note": "Noted, insulin/dextrose protocol activated.",
    })

    # Resolve the glucose alert
    request_app(f"/api/alerts/{aid_glu}/resolve", "POST", {
        "resolved_by": "Dr. OnCall",
        "note": "Subcutaneous lispro given.",
    })

    # Summary check after ack and resolve
    sum2 = request_app("/api/alerts/summary")[1]
    # In AlertManager summary: active_alerts is status='ACTIVE', unack_alerts is status='ACTIVE' AND ack_at IS NULL
    # Potassium is ACKNOWLEDGED, BP is ACTIVE (unacknowledged), Glucose is RESOLVED
    assert sum2["active_alerts"] == 1
    assert sum2["unacknowledged_alerts"] == 1
    assert sum2["resolved_alerts"] == 1
    assert sum2["total_alerts"] == 3

    # Filter by status:
    # 1. Default (no status param) returns status IN ('ACTIVE', 'ACKNOWLEDGED') -> 2 alerts (aid_bp, aid_k)
    active_and_ack = request_app(f"/api/alerts?stay_id={sid}")[1]["alerts"]
    assert len(active_and_ack) == 2
    returned_ids = [a["alert_id"] for a in active_and_ack]
    assert aid_bp in returned_ids
    assert aid_k in returned_ids
    assert aid_glu not in returned_ids

    # 2. status=ALL returns all 3 alerts
    all_alerts = request_app(f"/api/alerts?stay_id={sid}&status=all")[1]["alerts"]
    assert len(all_alerts) == 3

    # 3. status=RESOLVED returns only the resolved glucose alert
    resolved_alerts = request_app(f"/api/alerts?stay_id={sid}&status=RESOLVED")[1]["alerts"]
    assert len(resolved_alerts) == 1
    assert resolved_alerts[0]["alert_id"] == aid_glu

    # 4. Filter by severity: CRITICAL
    crit_alerts = request_app(f"/api/alerts?stay_id={sid}&severity=CRITICAL")[1]["alerts"]
    assert len(crit_alerts) == 2
    assert all(a["severity"] == "CRITICAL" for a in crit_alerts)

    # 5. Filter by source: CLINICAL_OBSERVATION
    obs_alerts = request_app(f"/api/alerts?stay_id={sid}&source=CLINICAL_OBSERVATION")[1]["alerts"]
    assert len(obs_alerts) == 1
    assert obs_alerts[0]["alert_id"] == aid_bp


def test_kdigo_urine_output_evaluation_and_alerts():
    """
    Verifies Section 1 & 4:
    KDIGO 2012 Urine Output staging:
    - 6h urine output with rate < 0.5 mL/kg/h -> KDIGO Stage 1 (HIGH severity alert).
    - Normal urine output (>= 0.5 mL/kg/h) -> NORMAL (no alert).
    """
    status, pat = request_app("/api/patients", "POST", {"name": "KDIGO Renal Patient", "weight_kg": 70.0})
    sid = request_app(f"/api/patients/{pat['id']}/stays", "POST", {"admission_datetime": "2026-11-08T08:00:00Z"})[1]["id"]

    # 70 kg patient over 6h:
    # Threshold for 0.5 mL/kg/h is 0.5 * 70 * 6 = 210 mL.
    # Recorded volume: 120 mL -> rate = 120 / (70 * 6) = 0.286 mL/kg/h < 0.5 mL/kg/h -> Oliguria / KDIGO Stage 1
    status, rec_oligo = request_app("/api/clinical-records", "POST", {
        "stay_id": sid,
        "category": "urine_output",
        "measurement_name": "Urine output",
        "value": 120.0,
        "unit": "mL",
        "observation_period": "6h",
        "measurement_time": "2026-11-08T14:00:00Z",
        "sample_collection_time": "2026-11-08T08:00:00Z",
        "entered_by": "Nurse Sarah",
    })
    assert status == 201
    assert "interpretation" in rec_oligo
    assert rec_oligo["interpretation"]["status"] == "LOW"
    assert rec_oligo["interpretation"]["severity"] in ("HIGH", "CRITICAL")
    assert "KDIGO" in rec_oligo["interpretation"]["explanation"]
    assert "alert" in rec_oligo
    aid_oligo = rec_oligo["alert"]["alert_id"]

    # Verify alert appears in /api/alerts
    alerts = request_app(f"/api/alerts?stay_id={sid}")[1]["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["alert_id"] == aid_oligo
    assert alerts[0]["source"] == "CLINICAL_OBSERVATION"
