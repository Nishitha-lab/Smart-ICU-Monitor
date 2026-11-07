import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    ClinicalRecordCreate,
    ClinicalRecordStore,
)


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


class PatientManagementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_patient_mgmt.sqlite3"
        self.store = PatientStore(self.db_path)
        self.store.initialize()

        self.orig_patient_store = main.patient_store
        self.orig_clinical_store = main.clinical_record_store
        main.patient_store = self.store
        main.clinical_record_store = ClinicalRecordStore(self.db_path)
        main.clinical_record_store.initialize()

        self.addCleanup(self.cleanup_env)

    def cleanup_env(self) -> None:
        main.patient_store = self.orig_patient_store
        main.clinical_record_store = self.orig_clinical_store
        self.temp_dir.cleanup()

    # ============================================================
    # PATIENT TESTS
    # ============================================================

    def test_create_patient_successfully(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Alice Smith", age=45, sex="female"))
        self.assertIsNotNone(p["id"])
        self.assertEqual(p["name"], "Alice Smith")
        self.assertEqual(p["age"], 45)
        self.assertEqual(p["sex"], "female")
        self.assertEqual(p["status"], "active")
        self.assertTrue(p["patient_code"].startswith("P"))

    def test_reject_missing_or_blank_patient_name(self) -> None:
        with self.assertRaises(ValidationError):
            PatientCreate(name="")
        with self.assertRaises(ValidationError):
            PatientCreate(name="   ")

    def test_reject_invalid_age(self) -> None:
        with self.assertRaises(ValidationError):
            PatientCreate(name="Bob", age=-1)
        with self.assertRaises(ValidationError):
            PatientCreate(name="Bob", age=150)

    def test_generate_unique_patient_codes(self) -> None:
        p1 = self.store.create_patient(PatientCreate(name="Patient One"))
        p2 = self.store.create_patient(PatientCreate(name="Patient Two"))
        self.assertNotEqual(p1["patient_code"], p2["patient_code"])
        self.assertEqual(p1["patient_code"], "P001")
        self.assertEqual(p2["patient_code"], "P002")

    def test_reject_duplicate_patient_code(self) -> None:
        self.store.create_patient(PatientCreate(name="First", patient_code="CUSTOM-01"))
        with self.assertRaises(ValueError):
            self.store.create_patient(PatientCreate(name="Second", patient_code="CUSTOM-01"))

    def test_search_and_filter_patients(self) -> None:
        self.store.create_patient(PatientCreate(name="John Doe", patient_code="P100"))
        self.store.create_patient(PatientCreate(name="Jane Roe", patient_code="P200"))

        # Search by name
        res = self.store.list_patients(search="Jane")
        self.assertEqual(res["total"], 1)
        self.assertEqual(res["patients"][0]["name"], "Jane Roe")

        # Search by code
        res2 = self.store.list_patients(search="P100")
        self.assertEqual(res2["total"], 1)
        self.assertEqual(res2["patients"][0]["name"], "John Doe")

    def test_retrieve_and_update_patient_details(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Charlie", age=30))
        details = self.store.get_patient(p["id"])
        self.assertIsNotNone(details)
        self.assertEqual(details["name"], "Charlie")

        updated = self.store.update_patient(p["id"], PatientUpdate(age=31, contact_information="555-0199"))
        self.assertEqual(updated["age"], 31)
        self.assertEqual(updated["contact_information"], "555-0199")

    def test_archive_patient(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Dave"))
        archived = self.store.archive_patient(p["id"])
        self.assertEqual(archived["status"], "archived")

        active_list = self.store.list_patients(status="active")
        self.assertEqual(active_list["total"], 0)

        archived_list = self.store.list_patients(status="archived")
        self.assertEqual(archived_list["total"], 1)

    def test_prevent_deletion_of_patient_with_associated_records(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Eve"))
        stay = self.store.create_stay(p["id"], IcuStayCreate(bed_number="ICU-10"))

        with self.assertRaises(ValueError) as err:
            self.store.delete_patient(p["id"])
        self.assertIn("associated ICU stays", str(err.exception))

    # ============================================================
    # ICU STAY TESTS
    # ============================================================

    def test_create_stay_for_registered_patient(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Frank"))
        stay = self.store.create_stay(p["id"], IcuStayCreate(bed_number="Bed-A", ward_or_unit="Trauma ICU"))
        self.assertIsNotNone(stay["id"])
        self.assertEqual(stay["patient_id"], p["id"])
        self.assertEqual(stay["discharge_status"], "active")
        self.assertTrue(stay["stay_code"].startswith("S"))

    def test_reject_stay_for_invalid_patient(self) -> None:
        with self.assertRaises(KeyError):
            self.store.create_stay(999999, IcuStayCreate(bed_number="Bed-X"))

    def test_reject_duplicate_active_admission(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Grace"))
        self.store.create_stay(p["id"], IcuStayCreate(bed_number="Bed-1"))

        with self.assertRaises(ValueError) as err:
            self.store.create_stay(p["id"], IcuStayCreate(bed_number="Bed-2"))
        self.assertIn("already has an active ICU admission", str(err.exception))

    def test_create_second_stay_after_discharge(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Heidi"))
        stay1 = self.store.create_stay(p["id"], IcuStayCreate(bed_number="Bed-1"))

        # Discharge stay 1
        discharged = self.store.discharge_stay(stay1["id"])
        self.assertEqual(discharged["discharge_status"], "discharged")
        self.assertIsNotNone(discharged["discharge_datetime"])

        # Admit stay 2
        stay2 = self.store.create_stay(p["id"], IcuStayCreate(bed_number="Bed-2"))
        self.assertEqual(stay2["discharge_status"], "active")
        self.assertNotEqual(stay1["id"], stay2["id"])

        # History preserves both stays
        stays = self.store.list_stays_for_patient(p["id"])
        self.assertEqual(len(stays), 2)

    def test_prevent_discharge_of_already_discharged_stay(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Ivan"))
        stay = self.store.create_stay(p["id"], IcuStayCreate(bed_number="Bed-1"))
        self.store.discharge_stay(stay["id"])

        with self.assertRaises(ValueError) as err:
            self.store.discharge_stay(stay["id"])
        self.assertIn("already discharged", str(err.exception))

    # ============================================================
    # DEVICE ASSIGNMENT TESTS
    # ============================================================

    def test_register_device(self) -> None:
        dev = self.store.register_device(DeviceCreate(device_id="ESP32-999", device_name="Device 999", port="COM10"))
        self.assertEqual(dev["device_id"], "ESP32-999")
        self.assertEqual(dev["device_name"], "Device 999")

        # Reject duplicate registration
        with self.assertRaises(ValueError):
            self.store.register_device(DeviceCreate(device_id="ESP32-999", device_name="Duplicate"))

    def test_assign_device_to_active_patient_and_stay(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Judy"))
        stay = self.store.create_stay(p["id"], IcuStayCreate(bed_number="ICU-03"))

        assignment = self.store.assign_device("ESP32-001", p["id"], stay["id"])
        self.assertEqual(assignment["device_id"], "ESP32-001")
        self.assertEqual(assignment["patient_id"], p["id"])
        self.assertEqual(assignment["stay_id"], stay["id"])
        self.assertEqual(assignment["assignment_status"], "active")

        active = self.store.get_active_assignment_for_device("ESP32-001")
        self.assertIsNotNone(active)
        self.assertEqual(active["patient_id"], p["id"])

    def test_reject_assignment_to_discharged_stay(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Kevin"))
        stay = self.store.create_stay(p["id"], IcuStayCreate(bed_number="ICU-04"))
        self.store.discharge_stay(stay["id"])

        with self.assertRaises(ValueError) as err:
            self.store.assign_device("ESP32-001", p["id"], stay["id"])
        self.assertIn("discharged or inactive", str(err.exception))

    def test_reject_assignment_to_mismatched_patient_stay(self) -> None:
        p1 = self.store.create_patient(PatientCreate(name="Patient 1"))
        stay1 = self.store.create_stay(p1["id"], IcuStayCreate(bed_number="ICU-05"))

        p2 = self.store.create_patient(PatientCreate(name="Patient 2"))

        with self.assertRaises(ValueError) as err:
            self.store.assign_device("ESP32-001", p2["id"], stay1["id"])
        self.assertIn("does not belong to the selected patient", str(err.exception))

    def test_reassignment_closes_previous_assignment_and_preserves_history(self) -> None:
        p1 = self.store.create_patient(PatientCreate(name="Patient A"))
        s1 = self.store.create_stay(p1["id"], IcuStayCreate(bed_number="Bed-A"))

        p2 = self.store.create_patient(PatientCreate(name="Patient B"))
        s2 = self.store.create_stay(p2["id"], IcuStayCreate(bed_number="Bed-B"))

        # Assign to P1
        self.store.assign_device("ESP32-001", p1["id"], s1["id"])
        active1 = self.store.get_active_assignment_for_device("ESP32-001")
        self.assertEqual(active1["patient_id"], p1["id"])

        # Reassign to P2
        self.store.assign_device("ESP32-001", p2["id"], s2["id"])
        active2 = self.store.get_active_assignment_for_device("ESP32-001")
        self.assertEqual(active2["patient_id"], p2["id"])

        # Verify history has 2 records, previous closed
        history = self.store.list_device_assignments("ESP32-001")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["patient_id"], p2["id"])
        self.assertEqual(history[0]["assignment_status"], "active")
        self.assertEqual(history[1]["patient_id"], p1["id"])
        self.assertEqual(history[1]["assignment_status"], "ended")
        self.assertIsNotNone(history[1]["unassigned_at"])

    def test_unassign_device(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Laura"))
        stay = self.store.create_stay(p["id"], IcuStayCreate(bed_number="Bed-L"))
        self.store.assign_device("ESP32-001", p["id"], stay["id"])

        unassigned = self.store.unassign_device("ESP32-001", reason="Patient moved to general ward")
        self.assertEqual(unassigned["assignment_status"], "ended")
        self.assertIsNotNone(unassigned["unassigned_at"])

        active = self.store.get_active_assignment_for_device("ESP32-001")
        self.assertIsNone(active)

    # ============================================================
    # TELEMETRY ATTRIBUTION TESTS (CRITICAL)
    # ============================================================

    def test_telemetry_attributed_to_assigned_patient(self) -> None:
        p_a = self.store.create_patient(PatientCreate(name="Patient A", patient_code="P001"))
        s_a = self.store.create_stay(p_a["id"], IcuStayCreate(bed_number="Bed-101"))
        self.store.assign_device("ESP32-001", p_a["id"], s_a["id"])

        # Ingest reading
        reading = self.store.ingest_reading("ESP32-001", {
            "fluid_ml": 450.0,
            "bpm": 72.0,
            "spo2": 98.0,
            "device_status": "NORMAL",
        })
        self.assertEqual(reading["patient_id"], p_a["id"])
        self.assertEqual(reading["stay_id"], s_a["id"])

        # Patient telemetry query returns it
        telem = self.store.get_patient_telemetry(p_a["id"])
        self.assertEqual(len(telem), 1)
        self.assertEqual(telem[0]["fluid_ml"], 450.0)

    def test_telemetry_switches_attribution_on_reassignment(self) -> None:
        p_a = self.store.create_patient(PatientCreate(name="Patient A", patient_code="P001"))
        s_a = self.store.create_stay(p_a["id"], IcuStayCreate(bed_number="Bed-A"))
        p_b = self.store.create_patient(PatientCreate(name="Patient B", patient_code="P002"))
        s_b = self.store.create_stay(p_b["id"], IcuStayCreate(bed_number="Bed-B"))

        # Assign to A
        self.store.assign_device("ESP32-001", p_a["id"], s_a["id"])
        self.store.ingest_reading("ESP32-001", {"fluid_ml": 500.0, "bpm": 70.0, "spo2": 97.0})

        # Reassign to B
        self.store.assign_device("ESP32-001", p_b["id"], s_b["id"])
        self.store.ingest_reading("ESP32-001", {"fluid_ml": 300.0, "bpm": 85.0, "spo2": 99.0})

        telem_a = self.store.get_patient_telemetry(p_a["id"])
        telem_b = self.store.get_patient_telemetry(p_b["id"])

        self.assertEqual(len(telem_a), 1)
        self.assertEqual(telem_a[0]["fluid_ml"], 500.0)

        self.assertEqual(len(telem_b), 1)
        self.assertEqual(telem_b[0]["fluid_ml"], 300.0)

    def test_unassigned_device_readings_not_attached_to_any_patient(self) -> None:
        # Ingest reading when no assignment exists
        reading = self.store.ingest_reading("ESP32-001", {"fluid_ml": 250.0, "bpm": 80.0, "spo2": 96.0})
        self.assertIsNone(reading["patient_id"])
        self.assertIsNone(reading["stay_id"])
        self.assertIsNone(reading["assignment_id"])

        # Create patient
        p = self.store.create_patient(PatientCreate(name="Patient C"))
        telem = self.store.get_patient_telemetry(p["id"])
        self.assertEqual(len(telem), 0, "Unassigned reading must NEVER leak into a patient record")

    def test_reading_after_unassignment_stops_attributing_to_previous_patient(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Patient D"))
        s = self.store.create_stay(p["id"], IcuStayCreate(bed_number="Bed-D"))
        self.store.assign_device("ESP32-001", p["id"], s["id"])

        # Reading while assigned
        self.store.ingest_reading("ESP32-001", {"fluid_ml": 400.0, "bpm": 75.0, "spo2": 98.0})

        # Unassign
        self.store.unassign_device("ESP32-001")

        # Reading after unassign
        reading_after = self.store.ingest_reading("ESP32-001", {"fluid_ml": 390.0, "bpm": 74.0, "spo2": 98.0})
        self.assertIsNone(reading_after["patient_id"])

        telem = self.store.get_patient_telemetry(p["id"])
        self.assertEqual(len(telem), 1, "Only reading during active assignment should belong to patient")
        self.assertEqual(telem[0]["fluid_ml"], 400.0)

    def test_delayed_reading_policy(self) -> None:
        """Readings with timestamp older than current assignment start are not attributed to current patient."""
        p = self.store.create_patient(PatientCreate(name="Patient E"))
        s = self.store.create_stay(p["id"], IcuStayCreate(bed_number="Bed-E"))
        self.store.assign_device("ESP32-001", p["id"], s["id"])

        # Timestamp from 10 minutes BEFORE assignment
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        reading = self.store.ingest_reading("ESP32-001", {
            "fluid_ml": 900.0,
            "bpm": 60.0,
            "spo2": 95.0,
            "reading_time": old_time,
        })
        self.assertIsNone(reading["patient_id"], "Delayed reading prior to assignment start must not be attributed")

    def test_backend_assignment_determines_attribution_regardless_of_frontend(self) -> None:
        """
        Critical safety test:
        Even if frontend requests or views Patient A, the backend telemetry attribution
        depends purely on device assignment in the DB (which is assigned to Patient B).
        """
        p_a = self.store.create_patient(PatientCreate(name="Patient A", patient_code="P001"))
        s_a = self.store.create_stay(p_a["id"], IcuStayCreate(bed_number="Bed-A"))

        p_b = self.store.create_patient(PatientCreate(name="Patient B", patient_code="P002"))
        s_b = self.store.create_stay(p_b["id"], IcuStayCreate(bed_number="Bed-B"))

        # Device is assigned to Patient B!
        self.store.assign_device("ESP32-001", p_b["id"], s_b["id"])

        # Sensor reading arrives
        self.store.ingest_reading("ESP32-001", {"fluid_ml": 333.0, "bpm": 88.0, "spo2": 99.0})

        # Query Patient A: must be empty
        telem_a = self.store.get_patient_telemetry(p_a["id"])
        self.assertEqual(len(telem_a), 0)

        # Query Patient B: contains reading
        telem_b = self.store.get_patient_telemetry(p_b["id"])
        self.assertEqual(len(telem_b), 1)
        self.assertEqual(telem_b[0]["fluid_ml"], 333.0)

    # ============================================================
    # CLINICAL RECORD INTEGRATION TESTS
    # ============================================================

    def test_clinical_record_stored_against_patient_and_stay(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Patient Clinical"))
        stay = self.store.create_stay(p["id"], IcuStayCreate(bed_number="ICU-CR"))

        now_iso = datetime.now(timezone.utc).isoformat()
        rec = main.clinical_record_store.create(ClinicalRecordCreate(
            category="urine_output",
            measurement_name="Urine output",
            value=150.0,
            unit="mL",
            measurement_time=now_iso,
            entered_by="Nurse Joy",
            stay_id=stay["id"],
            patient_id=p["id"],
        ))
        self.assertEqual(rec["stay_id"], stay["id"])
        self.assertEqual(rec["patient_id"], p["id"])

        # Fetch records for patient
        patient_records = main.clinical_record_store.list_for_patient_id(p["id"])
        self.assertEqual(len(patient_records), 1)
        self.assertEqual(patient_records[0]["value"], 150.0)

    def test_patients_clinical_records_are_isolated(self) -> None:
        p1 = self.store.create_patient(PatientCreate(name="Patient 1"))
        s1 = self.store.create_stay(p1["id"], IcuStayCreate(bed_number="Bed-1"))

        p2 = self.store.create_patient(PatientCreate(name="Patient 2"))
        s2 = self.store.create_stay(p2["id"], IcuStayCreate(bed_number="Bed-2"))

        now_iso = datetime.now(timezone.utc).isoformat()
        main.clinical_record_store.create(ClinicalRecordCreate(
            category="urine_output",
            measurement_name="Urine output",
            value=200.0,
            unit="mL",
            measurement_time=now_iso,
            entered_by="Nurse 1",
            stay_id=s1["id"],
            patient_id=p1["id"],
        ))

        p1_records = main.clinical_record_store.list_for_patient_id(p1["id"])
        p2_records = main.clinical_record_store.list_for_patient_id(p2["id"])

        self.assertEqual(len(p1_records), 1)
        self.assertEqual(len(p2_records), 0)

    def test_patient_timeline_chronological_events(self) -> None:
        p = self.store.create_patient(PatientCreate(name="Timeline Patient"))
        stay = self.store.create_stay(p["id"], IcuStayCreate(bed_number="TL-1"))
        self.store.assign_device("ESP32-001", p["id"], stay["id"])
        self.store.ingest_reading("ESP32-001", {"fluid_ml": 400.0, "bpm": 70.0, "spo2": 98.0})

        now_iso = datetime.now(timezone.utc).isoformat()
        main.clinical_record_store.create(ClinicalRecordCreate(
            category="urine_output",
            measurement_name="Urine output",
            value=100.0,
            unit="mL",
            measurement_time=now_iso,
            entered_by="Nurse T",
            stay_id=stay["id"],
            patient_id=p["id"],
        ))

        timeline = self.store.get_patient_timeline(p["id"])
        self.assertGreaterEqual(timeline["events_count"], 4)
        event_types = [e["event_type"] for e in timeline["events"]]
        self.assertIn("admission", event_types)
        self.assertIn("device_assignment", event_types)
        self.assertIn("telemetry", event_types)
        self.assertIn("clinical_record", event_types)


    # ============================================================
    # HTTP API END-TO-END TESTS
    # ============================================================

    def test_api_patient_lifecycle_and_device_attribution(self) -> None:
        # 1. Create Patient
        status, patient = request_app("/api/patients", method="POST", payload={"name": "George", "age": 52, "sex": "male"})
        self.assertEqual(status, 201)
        patient_id = patient["id"]
        patient_code = patient["patient_code"]

        # 2. Admit to ICU
        status, stay = request_app(f"/api/patients/{patient_id}/stays", method="POST", payload={"bed_number": "ICU-E2E"})
        self.assertEqual(status, 201)
        stay_id = stay["id"]

        # 3. Assign Device
        status, assignment = request_app("/api/devices/ESP32-001/assign", method="POST", payload={"patient_id": patient_id, "stay_id": stay_id})
        self.assertEqual(status, 200)

        # 4. Inject Telemetry via synthetic API endpoint
        status, telem = request_app("/api/devices/ESP32-001/telemetry", method="POST", payload={"fluid_ml": 480.0, "bpm": 76.0, "spo2": 98.0})
        self.assertEqual(status, 201)
        self.assertEqual(telem["patient_id"], patient_id)

        # 5. Get Patient Telemetry
        status, patient_telem = request_app(f"/api/patients/{patient_code}/telemetry", method="GET")
        self.assertEqual(status, 200)
        self.assertEqual(len(patient_telem["telemetry"]), 1)
        self.assertEqual(patient_telem["telemetry"][0]["fluid_ml"], 480.0)

        # 6. Discharge Stay
        status, disch = request_app(f"/api/stays/{stay_id}/discharge", method="POST")
        self.assertEqual(status, 200)
        self.assertEqual(disch["discharge_status"], "discharged")

        # 7. Subsequent telemetry should now be unassigned
        status, telem2 = request_app("/api/devices/ESP32-001/telemetry", method="POST", payload={"fluid_ml": 470.0, "bpm": 76.0, "spo2": 98.0})
        self.assertEqual(status, 201)
        self.assertIsNone(telem2["patient_id"])


if __name__ == "__main__":
    unittest.main()
