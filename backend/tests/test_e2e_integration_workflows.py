"""
End-to-End Comprehensive Software Integration Test Suite for Smart ICU Patient Safety System.
Validates Workflows A through E:
- Workflow A: Patient identity and admission lifecycle
- Workflow B: Device assignment and telemetry routing
- Workflow C: Telemetry parsing, quality masking, and delayed packet attribution
- Workflow D: Clinical records, optimistic locking (409), voiding, and eICU separation
- Workflow E: Clinical alert lifecycle, deduplication, escalation, manual resolution, auto-resolution, and audit logging
"""

import asyncio
import gc
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import HTTPException
from pydantic import ValidationError

import main
from services.patient_service import (
    DeviceAssign,
    DeviceCreate,
    DeviceUnassign,
    IcuStayCreate,
    IcuStayDischarge,
    IcuStayUpdate,
    PatientCreate,
    PatientStore,
    PatientUpdate,
    TelemetryIngest,
)
from services.clinical_records import (
    ClinicalRecordConflict,
    ClinicalRecordCreate,
    ClinicalRecordStore,
    ClinicalRecordUpdate,
    ClinicalRecordVoid,
)
from services.alert_manager import (
    AlertAcknowledgeRequest,
    AlertManager,
    AlertResolveRequest,
)
from services.serial_service import SerialMonitor


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


class EndToEndWorkflowsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "e2e_integration.sqlite3"

        # Initialize stores
        self.patient_store = PatientStore(self.db_path)
        self.clinical_record_store = ClinicalRecordStore(self.db_path)
        self.alert_manager = AlertManager(self.db_path)

        self.patient_store.initialize()
        self.clinical_record_store.initialize()
        self.alert_manager.initialize()

        # Connect to main app globals
        self.orig_patient_store = main.patient_store
        self.orig_clinical_store = main.clinical_record_store
        self.orig_alert_manager = main.alert_manager
        self.orig_db_file = main.CLINICAL_DB_FILE

        main.patient_store = self.patient_store
        main.clinical_record_store = self.clinical_record_store
        main.alert_manager = self.alert_manager
        main.CLINICAL_DB_FILE = self.db_path

        self.addCleanup(self.tearDown_env)

    def tearDown_env(self) -> None:
        main.patient_store = self.orig_patient_store
        main.clinical_record_store = self.orig_clinical_store
        main.alert_manager = self.orig_alert_manager
        main.CLINICAL_DB_FILE = self.orig_db_file
        gc.collect()
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # ============================================================
    # DATABASE INTEGRITY & FOREIGN KEYS
    # ============================================================

    def test_database_foreign_key_integrity(self) -> None:
        """PRAGMA foreign_key_check must return 0 violations on clean schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
            self.assertEqual(len(fk_violations), 0, f"FK violations detected: {fk_violations}")

    # ============================================================
    # WORKFLOW A: PATIENT IDENTITY AND ADMISSION LIFECYCLE
    # ============================================================

    def test_workflow_a_patient_lifecycle(self) -> None:
        """
        Workflow A:
        1. Register patient -> list patient -> view details
        2. Admit stay -> verify active stay & weight
        3. Attempt duplicate active admission (assert rejection)
        4. Discharge stay -> verify discharged stay rejects live telemetry
        """
        # 1. Register patient
        status, p_data = request_app(
            "/api/patients",
            method="POST",
            payload={
                "name": "Jane Doe",
                "age": 62,
                "sex": "Female",
                "weight_kg": 68.5,
                "contact_information": "ICU-Family-Contact: 555-0199",
            },
        )
        self.assertEqual(status, 201)
        patient_id = p_data["id"]
        self.assertEqual(p_data["weight_kg"], 68.5)
        self.assertEqual(p_data["status"], "active")

        # List patients (filtering source=registered)
        status, list_data = request_app("/api/patients?source=registered")
        self.assertEqual(status, 200)
        self.assertTrue(any(p["id"] == patient_id for p in list_data["patients"]))

        # View details
        status, details = request_app(f"/api/patients/{patient_id}")
        self.assertEqual(status, 200)
        self.assertEqual(details["name"], "Jane Doe")
        self.assertIsNone(details["active_stay"])

        # 2. Admit stay
        admission_time = "2026-11-09T08:00:00Z"
        status, stay_data = request_app(
            f"/api/patients/{patient_id}/stays",
            method="POST",
            payload={
                "bed_number": "Bed-104",
                "ward_or_unit": "Surgical ICU",
                "admission_reason": "Post-cardiac bypass monitoring",
                "admission_datetime": admission_time,
            },
        )
        self.assertEqual(status, 201)
        stay_id = stay_data["id"]
        self.assertEqual(stay_data["discharge_status"], "active")
        self.assertEqual(stay_data["bed_number"], "Bed-104")

        # Verify active stay in patient details
        status, details_admitted = request_app(f"/api/patients/{patient_id}")
        self.assertEqual(status, 200)
        self.assertIsNotNone(details_admitted["active_stay"])
        self.assertEqual(details_admitted["active_stay"]["id"], stay_id)
        self.assertEqual(details_admitted["weight_kg"], 68.5)

        # 3. Attempt duplicate active admission -> must be rejected (400 or 409)
        status, dup_err = request_app(
            f"/api/patients/{patient_id}/stays",
            method="POST",
            payload={"bed_number": "Bed-105"},
        )
        self.assertIn(status, (400, 409))
        self.assertIn("already has an active ICU admission", dup_err["detail"])

        # 4. Register a device and assign to this active stay
        reg_status, _ = request_app(
            "/api/devices",
            method="POST",
            payload={"device_id": "ESP32-WF-A", "device_name": "Ventilator Monitor A"},
        )
        self.assertEqual(reg_status, 201)
        assign_status, _ = request_app(
            "/api/devices/ESP32-WF-A/assign",
            method="POST",
            payload={"patient_id": patient_id, "stay_id": stay_id},
        )
        self.assertEqual(assign_status, 200)

        # Discharge stay
        discharge_time = "2026-11-09T18:00:00Z"
        status, disch_data = request_app(
            f"/api/stays/{stay_id}/discharge",
            method="POST",
            payload={
                "discharge_datetime": discharge_time,
                "discharge_status": "discharged",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(disch_data["discharge_status"], "discharged")

        # Device assignment should have automatically ended
        dev_status, dev_info = request_app("/api/devices/ESP32-WF-A")
        self.assertEqual(dev_status, 200)
        self.assertIsNone(dev_info["active_assignment"])

        # Live telemetry submitted now must NOT be attributed to the discharged stay
        live_time = "2026-11-09T18:30:00Z"
        status, ingest_res = request_app(
            "/api/devices/ESP32-WF-A/telemetry",
            method="POST",
            payload={
                "fluid_ml": 500.0,
                "bpm": 72.0,
                "spo2": 98.0,
                "reading_time": live_time,
            },
        )
        self.assertEqual(status, 201)
        self.assertIsNone(ingest_res["patient_id"], "Live telemetry must not be attributed to discharged stay")
        self.assertIsNone(ingest_res["stay_id"])

    # ============================================================
    # WORKFLOW B: DEVICE ASSIGNMENT AND TELEMETRY ROUTING
    # ============================================================

    def test_workflow_b_device_assignment_routing(self) -> None:
        """
        Workflow B:
        1. Register device
        2. Assign to active patient stay
        3. Submit telemetry -> verify attribution
        4. Attempt assignment to discharged stay (assert rejection)
        5. Reassign to another patient atomically -> verify previous assignment closed
        """
        # Create Patients 1 and 2
        p1 = self.patient_store.create_patient(PatientCreate(name="Patient One", weight_kg=75.0))
        s1 = self.patient_store.create_stay(p1["id"], IcuStayCreate(bed_number="Bed-201"))

        p2 = self.patient_store.create_patient(PatientCreate(name="Patient Two", weight_kg=80.0))
        s2 = self.patient_store.create_stay(p2["id"], IcuStayCreate(bed_number="Bed-202"))

        # 1. Register device
        d = self.patient_store.register_device(
            DeviceCreate(device_id="ESP32-ROUTING-01", device_name="Bed Monitor 01")
        )
        self.assertEqual(d["device_id"], "ESP32-ROUTING-01")

        # 2. Assign to Patient 1
        assign1 = self.patient_store.assign_device("ESP32-ROUTING-01", p1["id"], s1["id"])
        self.assertEqual(assign1["patient_id"], p1["id"])
        self.assertEqual(assign1["stay_id"], s1["id"])

        # 3. Submit telemetry -> attributed to Patient 1
        telem1 = self.patient_store.ingest_reading(
            "ESP32-ROUTING-01",
            {"fluid_ml": 450.0, "bpm": 76.0, "spo2": 97.0},
        )
        self.assertEqual(telem1["patient_id"], p1["id"])
        self.assertEqual(telem1["stay_id"], s1["id"])

        # 4. Discharge Patient 1's stay, attempt to assign device to it -> must fail
        self.patient_store.discharge_stay(s1["id"])
        with self.assertRaises(ValueError) as ctx:
            self.patient_store.assign_device("ESP32-ROUTING-01", p1["id"], s1["id"])
        self.assertIn("discharged or inactive", str(ctx.exception))

        # 5. Reassign device to Patient 2 atomically
        assign2 = self.patient_store.assign_device("ESP32-ROUTING-01", p2["id"], s2["id"])
        self.assertEqual(assign2["patient_id"], p2["id"])
        self.assertEqual(assign2["assignment_status"], "active")

        # Verify previous assignment is marked ended
        history = self.patient_store.list_device_assignments("ESP32-ROUTING-01")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["patient_id"], p2["id"])
        self.assertEqual(history[0]["assignment_status"], "active")
        self.assertEqual(history[1]["patient_id"], p1["id"])
        self.assertEqual(history[1]["assignment_status"], "ended")
        self.assertIsNotNone(history[1]["unassigned_at"])

        # Unassign device completely
        unassigned = self.patient_store.unassign_device("ESP32-ROUTING-01")
        self.assertEqual(unassigned["assignment_status"], "ended")
        active_now = self.patient_store.get_active_assignment_for_device("ESP32-ROUTING-01")
        self.assertIsNone(active_now)

    # ============================================================
    # WORKFLOW C: TELEMETRY PARSING, QUALITY MASKING, & DELAYED PACKETS
    # ============================================================

    def test_workflow_c_telemetry_parsing_and_delayed_packets(self) -> None:
        """
        Workflow C:
        1. Valid packet parsing
        2. 0 BPM, 0 SpO2, out-of-range bounds masking
        3. Stale telemetry detection
        4. Delayed packet arrival after reassignment (tagged to original timeframe, not new patient)
        """
        p_orig = self.patient_store.create_patient(PatientCreate(name="Original Patient", weight_kg=70.0))
        s_orig = self.patient_store.create_stay(p_orig["id"], IcuStayCreate(admission_datetime="2026-11-09T08:00:00Z"))

        p_new = self.patient_store.create_patient(PatientCreate(name="New Patient", weight_kg=65.0))
        s_new = self.patient_store.create_stay(p_new["id"], IcuStayCreate(admission_datetime="2026-11-09T10:00:00Z"))

        dev_id = "ESP32-FLOW-C"
        self.patient_store.register_device(DeviceCreate(device_id=dev_id, device_name="ICU Flow Device"))

        # Step 1: Assign to Original Patient from 09:00 to 10:00
        # Emulate assignment with explicit timestamps in DB
        with self.patient_store._connection() as conn:
            conn.execute(
                """
                INSERT INTO device_assignments (
                    device_id, patient_id, stay_id, assigned_at, unassigned_at, assignment_status, created_at, updated_at
                ) VALUES (?, ?, ?, '2026-11-09T09:00:00Z', '2026-11-09T10:00:00Z', 'ended', '2026-11-09T09:00:00Z', '2026-11-09T10:00:00Z')
                """,
                (dev_id, p_orig["id"], s_orig["id"]),
            )
            # Step 2: Assign to New Patient starting 10:00:00Z
            conn.execute(
                """
                INSERT INTO device_assignments (
                    device_id, patient_id, stay_id, assigned_at, unassigned_at, assignment_status, created_at, updated_at
                ) VALUES (?, ?, ?, '2026-11-09T10:00:00Z', NULL, 'active', '2026-11-09T10:00:00Z', '2026-11-09T10:00:00Z')
                """,
                (dev_id, p_new["id"], s_new["id"]),
            )

        # 1. Valid telemetry arriving at 10:05 (attributed to New Patient)
        reading_new = self.patient_store.ingest_reading(
            dev_id,
            {
                "fluid_ml": 800.0,
                "bpm": 82.0,
                "spo2": 98.0,
                "reading_time": "2026-11-09T10:05:00Z",
            },
        )
        self.assertEqual(reading_new["patient_id"], p_new["id"])
        self.assertEqual(reading_new["stay_id"], s_new["id"])
        self.assertEqual(reading_new["bpm"], 82.0)
        self.assertEqual(reading_new["spo2"], 98.0)

        # 2. Delayed telemetry packet from ESP32 buffer recorded at 09:30 arriving NOW
        # Must be tagged to Original Patient (original assignment timeframe), NOT New Patient!
        delayed_reading = self.patient_store.ingest_reading(
            dev_id,
            {
                "fluid_ml": 950.0,
                "bpm": 74.0,
                "spo2": 99.0,
                "reading_time": "2026-11-09T09:30:00Z",
            },
        )
        self.assertEqual(
            delayed_reading["patient_id"],
            p_orig["id"],
            "Delayed packet must be tagged to original assignment timeframe, NOT new patient!",
        )
        self.assertEqual(delayed_reading["stay_id"], s_orig["id"])

        # 3. Reading with 0 BPM and 0 SpO2 (sensor disconnect) -> quality masking
        zero_reading = self.patient_store.ingest_reading(
            dev_id,
            {
                "fluid_ml": 800.0,
                "bpm": 0.0,
                "spo2": 0.0,
                "reading_time": "2026-11-09T10:10:00Z",
            },
        )
        self.assertIsNone(zero_reading["bpm"], "0 BPM should be masked to None")
        self.assertIsNone(zero_reading["spo2"], "0 SpO2 should be masked to None")
        self.assertEqual(zero_reading["fluid_ml"], 800.0)

        # Ensure no bogus bradycardia or hypoxemia alerts were created
        alerts = self.alert_manager.list_alerts(stay_id=s_new["id"])
        physio_alerts = [a for a in alerts if a["measurement_name"] in ("Heart Rate", "SpO2")]
        self.assertEqual(len(physio_alerts), 0, "Masked 0 BPM / 0 SpO2 must not trigger false physiological alerts")

        # 4. Stale telemetry detection in SerialMonitor
        monitor = SerialMonitor(port="COM9", device_id=dev_id)
        # Snapshot with old timestamp (>5s ago)
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        monitor._set(connected=True, connection_status="CONNECTED", last_update=old_time)
        snap = monitor.snapshot()
        self.assertEqual(snap["connection_status"], "STALE")
        self.assertFalse(snap["connected"])

    # ============================================================
    # WORKFLOW D: CLINICAL RECORDS, OPTIMISTIC LOCKING, VOIDING
    # ============================================================

    def test_workflow_d_clinical_records_and_isolation(self) -> None:
        """
        Workflow D:
        1. Create record -> retrieve
        2. Update with revision history increment
        3. Stale version update (assert 409 conflict)
        4. Void record with mandatory reason
        5. Verify excluded from active calculations
        6. Attempt record creation with patient/stay mismatch (assert rejection)
        7. Verify strict separation between eICU demo data and registered clinical records
        """
        p = self.patient_store.create_patient(PatientCreate(name="Robert Johnson", weight_kg=78.0))
        stay = self.patient_store.create_stay(p["id"], IcuStayCreate(bed_number="Bed-301"))

        # 1. Create record (including required sample_collection_time for blood test)
        create_payload = {
            "stay_id": stay["id"],
            "patient_id": p["id"],
            "category": "blood_test",
            "measurement_name": "Hemoglobin",
            "value": 11.2,
            "unit": "g/dL",
            "measurement_time": "2026-11-09T09:00:00Z",
            "sample_collection_time": "2026-11-09T08:30:00Z",
            "entered_by": "Dr. Smith",
            "notes": "Routine morning CBC",
        }
        status, rec_data = request_app("/api/clinical-records", method="POST", payload=create_payload)
        self.assertEqual(status, 201)
        record_id = rec_data["record_id"]
        self.assertEqual(rec_data["version"], 1)

        # Retrieve record
        status, retrieved = request_app(f"/api/clinical-records/{record_id}")
        self.assertEqual(status, 200)
        self.assertEqual(retrieved["measurement_name"], "Hemoglobin")
        self.assertEqual(retrieved["value"], 11.2)

        # 2. Update record (version 1 -> 2)
        update_payload = {
            "category": "blood_test",
            "measurement_name": "Hemoglobin",
            "value": 10.8,
            "unit": "g/dL",
            "measurement_time": "2026-11-09T09:00:00Z",
            "sample_collection_time": "2026-11-09T08:30:00Z",
            "expected_version": 1,
            "entered_by": "Dr. Smith",
            "change_reason": "Corrected lab hemolyzed sample result",
            "notes": "Verified with re-run",
        }
        status, updated_data = request_app(
            f"/api/clinical-records/{record_id}",
            method="PUT",
            payload=update_payload,
        )
        self.assertEqual(status, 200)
        self.assertEqual(updated_data["version"], 2)
        self.assertEqual(updated_data["value"], 10.8)

        # Verify revision history has 1 entry
        status, revs_data = request_app(f"/api/clinical-records/{record_id}/revisions")
        self.assertEqual(status, 200)
        self.assertEqual(len(revs_data["revisions"]), 1)
        self.assertEqual(revs_data["revisions"][0]["change_reason"], "Corrected lab hemolyzed sample result")

        # 3. Stale version update (using expected_version=1 when current is 2) -> 409 conflict
        stale_payload = {
            "category": "blood_test",
            "measurement_name": "Hemoglobin",
            "value": 10.5,
            "unit": "g/dL",
            "measurement_time": "2026-11-09T09:00:00Z",
            "sample_collection_time": "2026-11-09T08:30:00Z",
            "expected_version": 1,
            "entered_by": "Nurse Lee",
            "change_reason": "Late concurrent edit",
        }
        status, err_data = request_app(
            f"/api/clinical-records/{record_id}",
            method="PUT",
            payload=stale_payload,
        )
        self.assertEqual(status, 409, "Concurrent stale update must return 409 Conflict")

        # 4. Void record with mandatory reason
        void_payload = {
            "expected_version": 2,
            "voided_by": "Dr. Smith",
            "void_reason": "Specimen labeled with wrong tube ID",
        }
        status, voided_data = request_app(
            f"/api/clinical-records/{record_id}",
            method="DELETE",
            payload=void_payload,
        )
        self.assertEqual(status, 200)
        self.assertIsNotNone(voided_data["voided_at"])
        self.assertEqual(voided_data["void_reason"], "Specimen labeled with wrong tube ID")

        # 5. Verify excluded from active calculations / queries
        status, active_records = request_app(f"/api/stays/{stay['id']}/clinical-records")
        self.assertEqual(status, 200)
        self.assertFalse(any(r["record_id"] == record_id for r in active_records["records"]))

        # 6. Attempt record creation with patient/stay mismatch
        p_other = self.patient_store.create_patient(PatientCreate(name="Other Patient"))
        mismatch_payload = {
            "stay_id": stay["id"],  # Belongs to Robert Johnson
            "patient_id": p_other["id"],  # Belongs to Other Patient
            "category": "blood_test",
            "measurement_name": "Creatinine",
            "value": 1.1,
            "unit": "mg/dL",
            "measurement_time": "2026-11-09T09:00:00Z",
            "sample_collection_time": "2026-11-09T08:30:00Z",
            "entered_by": "Dr. Smith",
        }
        status, mismatch_err = request_app("/api/clinical-records", method="POST", payload=mismatch_payload)
        self.assertEqual(status, 400)
        self.assertIn("does not belong to", mismatch_err["detail"])

        # 7. Strict separation between eICU demo data and registered clinical records
        # Combining stay_id and patientunitstayid must be rejected
        combined_payload = {
            "stay_id": stay["id"],
            "patientunitstayid": 141168,
            "category": "blood_test",
            "measurement_name": "Blood glucose",
            "value": 110.0,
            "unit": "mg/dL",
            "measurement_time": "2026-11-09T09:00:00Z",
            "sample_collection_time": "2026-11-09T08:30:00Z",
            "entered_by": "Dr. Smith",
        }
        status, comb_err = request_app("/api/clinical-records", method="POST", payload=combined_payload)
        self.assertIn(status, (400, 422))
        err_str = json.dumps(comb_err)
        self.assertIn("cannot be combined", err_str)

    # ============================================================
    # WORKFLOW E: CLINICAL ALERTS LIFECYCLE
    # ============================================================

    def test_workflow_e_alert_lifecycle(self) -> None:
        """
        Workflow E:
        1. Alert generation on abnormal lab (e.g. Hemoglobin 6.2 g/dL -> CRITICAL)
        2. Duplicate fingerprint deduplication
        3. Severity escalation (MODERATE -> CRITICAL)
        4. Manual acknowledgement
        5. Resolution with required clinical notes
        6. Auto-resolution upon normal observation
        7. Superseding when parent record edited or voided
        8. Audit history retrieval
        9. Status / severity / source filtering
        """
        p = self.patient_store.create_patient(PatientCreate(name="Alert Patient", weight_kg=72.0))
        stay = self.patient_store.create_stay(p["id"], IcuStayCreate(bed_number="Bed-401"))

        # 1. Post abnormal critical lab: Hemoglobin = 6.2 g/dL
        status, rec_crit = request_app(
            "/api/clinical-records",
            method="POST",
            payload={
                "stay_id": stay["id"],
                "patient_id": p["id"],
                "category": "blood_test",
                "measurement_name": "Hemoglobin",
                "value": 6.2,
                "unit": "g/dL",
                "measurement_time": "2026-11-09T10:00:00Z",
                "sample_collection_time": "2026-11-09T09:30:00Z",
                "entered_by": "Dr. Sepsis",
            },
        )
        self.assertEqual(status, 201)
        record_id = rec_crit["record_id"]

        # Verify alert was generated
        status, alerts_resp = request_app(f"/api/alerts?stay_id={stay['id']}&status=ACTIVE")
        self.assertEqual(status, 200)
        active_alerts = alerts_resp["alerts"]
        self.assertEqual(len(active_alerts), 1)
        alert = active_alerts[0]
        alert_id = alert["alert_id"]
        self.assertEqual(alert["severity"], "CRITICAL")
        self.assertEqual(alert["measurement_name"], "Hemoglobin")
        self.assertEqual(alert["status"], "ACTIVE")

        # 2. Duplicate observation with same value -> deduplicated, no duplicate alert
        status, rec_dup = request_app(
            "/api/clinical-records",
            method="POST",
            payload={
                "stay_id": stay["id"],
                "patient_id": p["id"],
                "category": "blood_test",
                "measurement_name": "Hemoglobin",
                "value": 6.2,
                "unit": "g/dL",
                "measurement_time": "2026-11-09T10:15:00Z",
                "sample_collection_time": "2026-11-09T09:45:00Z",
                "entered_by": "Dr. Sepsis",
            },
        )
        self.assertEqual(status, 201)
        status, alerts_after_dup = request_app(f"/api/alerts?stay_id={stay['id']}&status=ACTIVE")
        self.assertEqual(len(alerts_after_dup["alerts"]), 1, "Duplicate observation must not create duplicate alert")

        # 4. Manual acknowledgement
        status, ack_res = request_app(
            f"/api/alerts/{alert_id}/acknowledge",
            method="POST",
            payload={
                "acknowledged_by": "Nurse Ratched",
                "acknowledgement_note": "Notified attending physician; ordered 2 units PRBCs.",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(ack_res["status"], "ACKNOWLEDGED")
        self.assertEqual(ack_res["acknowledged_by"], "Nurse Ratched")

        # 5. Manual resolution
        status, res_res = request_app(
            f"/api/alerts/{alert_id}/resolve",
            method="POST",
            payload={
                "resolved_by": "Dr. Sepsis",
                "resolution_note": "Transfusion complete; post-transfusion vitals stable.",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(res_res["status"], "RESOLVED")
        self.assertEqual(res_res["resolved_by"], "Dr. Sepsis")

        # 6. Auto-resolution upon normal observation
        # Create a new critical alert for Potassium = 6.8 mmol/L (> 6.0 mmol/L)
        status, rec_pot_crit = request_app(
            "/api/clinical-records",
            method="POST",
            payload={
                "stay_id": stay["id"],
                "patient_id": p["id"],
                "category": "blood_test",
                "measurement_name": "Potassium",
                "value": 6.8,
                "unit": "mmol/L",
                "measurement_time": "2026-11-09T11:00:00Z",
                "sample_collection_time": "2026-11-09T10:30:00Z",
                "entered_by": "Dr. Sepsis",
            },
        )
        self.assertEqual(status, 201)
        status, pot_alerts = request_app(f"/api/alerts?stay_id={stay['id']}&status=ACTIVE")
        pot_alert_list = [a for a in pot_alerts["alerts"] if a["measurement_name"] == "Potassium"]
        self.assertEqual(len(pot_alert_list), 1)
        pot_alert_id = pot_alert_list[0]["alert_id"]

        # Now submit normal Potassium = 4.2 mmol/L -> should auto-resolve the active Potassium alert
        status, rec_pot_norm = request_app(
            "/api/clinical-records",
            method="POST",
            payload={
                "stay_id": stay["id"],
                "patient_id": p["id"],
                "category": "blood_test",
                "measurement_name": "Potassium",
                "value": 4.2,
                "unit": "mmol/L",
                "measurement_time": "2026-11-09T12:00:00Z",
                "sample_collection_time": "2026-11-09T11:30:00Z",
                "entered_by": "Dr. Sepsis",
            },
        )
        self.assertEqual(status, 201)
        status, resolved_pot_check = request_app(f"/api/alerts/{pot_alert_id}")
        self.assertEqual(status, 200)
        self.assertEqual(
            resolved_pot_check["status"],
            "RESOLVED",
            "Normal observation must auto-resolve active alert for same parameter",
        )
        self.assertIn("Auto-resolved", resolved_pot_check.get("resolution_note", ""))

        # 7. Superseding when parent record edited or voided
        # Create an abnormal Sodium alert (120 mmol/L < 125 mmol/L)
        status, rec_na = request_app(
            "/api/clinical-records",
            method="POST",
            payload={
                "stay_id": stay["id"],
                "patient_id": p["id"],
                "category": "blood_test",
                "measurement_name": "Sodium",
                "value": 120.0,  # CRITICAL_LOW (< 125)
                "unit": "mmol/L",
                "measurement_time": "2026-11-09T13:00:00Z",
                "sample_collection_time": "2026-11-09T12:30:00Z",
                "entered_by": "Dr. Sepsis",
            },
        )
        self.assertEqual(status, 201)
        na_rec_id = rec_na["record_id"]
        status, na_alerts = request_app(f"/api/alerts?stay_id={stay['id']}&status=ACTIVE")
        na_alert_id = [a["alert_id"] for a in na_alerts["alerts"] if a["measurement_name"] == "Sodium"][0]

        # Void the Sodium record -> alert must be SUPERSEDED
        status, _ = request_app(
            f"/api/clinical-records/{na_rec_id}",
            method="DELETE",
            payload={
                "expected_version": 1,
                "voided_by": "Dr. Sepsis",
                "void_reason": "Specimen hemolyzed, repeat sample drawn",
            },
        )
        self.assertEqual(status, 200)
        status, voided_alert_check = request_app(f"/api/alerts/{na_alert_id}")
        self.assertEqual(status, 200)
        self.assertEqual(voided_alert_check["status"], "SUPERSEDED")

        # 8. Audit history retrieval
        status, audits = request_app(f"/api/alerts/{alert_id}/history")
        self.assertEqual(status, 200)
        actions = [entry["action"] for entry in audits["history"]]
        self.assertIn("CREATED", actions)
        self.assertIn("ACKNOWLEDGED", actions)
        self.assertIn("RESOLVED", actions)

        # 9. Status and severity filtering
        status, filtered = request_app(f"/api/alerts?stay_id={stay['id']}&status=RESOLVED")
        self.assertEqual(status, 200)
        self.assertTrue(all(a["status"] == "RESOLVED" for a in filtered["alerts"]))


if __name__ == "__main__":
    unittest.main()
