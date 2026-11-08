"""Integration tests for Unified Alert API and Clinical Record alert hooks using request_app."""

import asyncio
import json
import tempfile
from pathlib import Path
import pytest
import main


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
        return int(response["status"]), json.loads(response_body["body"])

    return asyncio.run(call_app())


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path: Path):
    db_file = tmp_path / "test_api_alerts.sqlite3"
    main.CLINICAL_DB_FILE = db_file
    main.clinical_record_store = main.ClinicalRecordStore(db_file)
    main.patient_store = main.PatientStore(db_file)
    main.alert_manager = main.AlertManager(db_file)
    main.patient_store.initialize()
    main.clinical_record_store.initialize()
    main.alert_manager.initialize()
    yield


def test_clinical_record_submission_generates_alert_and_interpretation():
    # Create patient and stay
    status, pat_data = request_app("/api/patients", "POST", {"name": "Alice Wonderland", "weight_kg": 60.0})
    assert status == 201
    pid = pat_data["id"]

    status, stay_data = request_app(f"/api/patients/{pid}/stays", "POST", {"admission_datetime": "2026-11-08T10:00:00Z"})
    assert status == 201
    sid = stay_data["id"]

    # Submit critical potassium lab result
    status, rec_data = request_app("/api/clinical-records", "POST", {
        "stay_id": sid,
        "category": "blood_test",
        "measurement_name": "Potassium",
        "value": 6.8,
        "unit": "mmol/L",
        "measurement_time": "2026-11-08T11:00:00Z",
        "sample_collection_time": "2026-11-08T10:30:00Z",
        "entered_by": "Nurse Kelly",
    })
    assert status == 201

    assert "interpretation" in rec_data
    assert rec_data["interpretation"]["status"] == "CRITICAL_HIGH"
    assert rec_data["interpretation"]["severity"] == "CRITICAL"
    assert "alert" in rec_data
    assert rec_data["alert"]["severity"] == "CRITICAL"
    assert rec_data["alert"]["status"] == "ACTIVE"

    # Verify alert appears in /api/alerts
    status, alerts_data = request_app(f"/api/alerts?stay_id={sid}")
    assert status == 200
    alerts = alerts_data["alerts"]
    assert len(alerts) == 1
    aid = alerts[0]["alert_id"]

    # Acknowledge alert
    status, ack_data = request_app(f"/api/alerts/{aid}/acknowledge", "POST", {
        "acknowledged_by": "Dr. House",
        "note": "Ordering repeat labs and calcium gluconate.",
    })
    assert status == 200
    assert ack_data["status"] == "ACKNOWLEDGED"

    # Check alert summary
    status, sum_data = request_app("/api/alerts/summary")
    assert status == 200
    assert sum_data["total_alerts"] >= 1

    # Resolve alert
    status, res_data = request_app(f"/api/alerts/{aid}/resolve", "POST", {
        "resolved_by": "Dr. House",
        "note": "Treatment administered, ECG stable.",
    })
    assert status == 200
    assert res_data["status"] == "RESOLVED"


def test_clinical_record_update_supersedes_alert():
    status, pat_data = request_app("/api/patients", "POST", {"name": "Bob Builder", "weight_kg": 70.0})
    pid = pat_data["id"]
    status, stay_data = request_app(f"/api/patients/{pid}/stays", "POST", {"admission_datetime": "2026-11-08T10:00:00Z"})
    sid = stay_data["id"]

    # Initial entry: low urine output (< 6h screening)
    status, rec_data = request_app("/api/clinical-records", "POST", {
        "stay_id": sid,
        "category": "urine_output",
        "measurement_name": "Urine output",
        "value": 40.0,
        "unit": "mL",
        "observation_period": "2h",
        "measurement_time": "2026-11-08T12:00:00Z",
        "entered_by": "Nurse Ann",
    })
    assert status == 201
    rid = rec_data["record_id"]

    status, alerts_data = request_app(f"/api/alerts?stay_id={sid}")
    alerts = alerts_data["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["status"] == "ACTIVE"
    old_aid = alerts[0]["alert_id"]

    # Correction: nurse realized volume was actually 180 mL (normal)
    status, put_data = request_app(f"/api/clinical-records/{rid}", "PUT", {
        "category": "urine_output",
        "measurement_name": "Urine output",
        "value": 180.0,
        "unit": "mL",
        "observation_period": "2h",
        "measurement_time": "2026-11-08T12:00:00Z",
        "entered_by": "Nurse Ann",
        "expected_version": 1,
        "change_reason": "Corrected typographical error in volume entry.",
    })
    assert status == 200
    assert put_data["interpretation"]["status"] == "NORMAL"

    # Previous alert must now be SUPERSEDED
    status, old_alert = request_app(f"/api/alerts/{old_aid}")
    assert status == 200
    assert old_alert["status"] == "SUPERSEDED"


def test_clinical_record_voiding_supersedes_alert():
    status, pat_data = request_app("/api/patients", "POST", {"name": "Charlie Chaplin"})
    pid = pat_data["id"]
    status, stay_data = request_app(f"/api/patients/{pid}/stays", "POST", {"admission_datetime": "2026-11-08T10:00:00Z"})
    sid = stay_data["id"]

    status, rec_data = request_app("/api/clinical-records", "POST", {
        "stay_id": sid,
        "category": "clinical_observation",
        "measurement_name": "Systolic blood pressure",
        "value": 70.0,
        "unit": "mmHg",
        "measurement_time": "2026-11-08T12:00:00Z",
        "entered_by": "Nurse John",
    })
    assert status == 201
    rid = rec_data["record_id"]

    status, alerts_data = request_app(f"/api/alerts?stay_id={sid}")
    alerts = alerts_data["alerts"]
    assert len(alerts) == 1
    aid = alerts[0]["alert_id"]
    assert alerts[0]["status"] == "ACTIVE"

    # Void record
    status, void_data = request_app(f"/api/clinical-records/{rid}", "DELETE", {
        "expected_version": 1,
        "voided_by": "Charge Nurse",
        "void_reason": "Duplicate reading entered for wrong bed.",
    })
    assert status == 200

    # Alert must be marked SUPERSEDED
    status, alert_check = request_app(f"/api/alerts/{aid}")
    assert status == 200
    assert alert_check["status"] == "SUPERSEDED"


def test_preview_clinical_evaluation_endpoint():
    status, data = request_app("/api/clinical-records/evaluate", "POST", {
        "category": "urine_output",
        "measurement_name": "Urine output",
        "value": 20.0,
        "unit": "mL",
        "observation_period": "6h",
        "weight_kg": 70.0,
    })
    assert status == 200
    assert data["status"] == "LOW"
    assert data["severity"] == "HIGH"
    assert "KDIGO Stage 1" in data["explanation"]
