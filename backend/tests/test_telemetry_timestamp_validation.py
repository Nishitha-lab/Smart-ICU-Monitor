"""
Regression tests for Telemetry Timestamp Validation (Priority 1).
Verifies:
- Valid ISO-8601 string accepted (HTTP 201)
- Valid ISO-8601 with offset accepted (HTTP 201)
- Valid ISO-8601 with non-UTC timezone accepted (HTTP 201)
- Missing timestamp accepted (HTTP 201), defaults to current time
- Explicit null timestamp accepted (HTTP 201), defaults to current time
- Malformed string rejected (HTTP 422, NOT 500)
- Invalid date (e.g. Feb 31) rejected (HTTP 422)
- Invalid time (e.g. hour 25) rejected (HTTP 422)
- Empty string rejected (HTTP 422)
- Whitespace-only string rejected (HTTP 422)
- Non-string invalid type (integer/boolean) rejected (HTTP 422)
- Direct service-layer calls with invalid timestamps raise ValueError
- Database isolation: DB is NOT updated when a malformed timestamp is rejected
"""

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from starlette.testclient import TestClient

import main
from services.patient_service import PatientStore, TelemetryIngest
from services.clinical_records import ClinicalRecordStore


class TelemetryTimestampValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_telemetry_ts.sqlite3"
        self.store = PatientStore(self.db_path)
        self.store.initialize()

        self.orig_patient_store = main.patient_store
        self.orig_clinical_store = main.clinical_record_store
        main.patient_store = self.store
        main.clinical_record_store = ClinicalRecordStore(self.db_path)
        main.clinical_record_store.initialize()

        self.client = TestClient(main.app)
        self.addCleanup(self.cleanup_env)

    def cleanup_env(self) -> None:
        main.patient_store = self.orig_patient_store
        main.clinical_record_store = self.orig_clinical_store
        self.temp_dir.cleanup()

    # ============================================================
    # HTTP API VALIDATION TESTS (HTTP 201 vs HTTP 422)
    # ============================================================

    def test_valid_iso8601_utc_z_accepted(self) -> None:
        payload = {
            "bpm": 75.0,
            "spo2": 98.0,
            "reading_time": "2026-03-29T14:30:00Z",
        }
        response = self.client.post("/api/devices/DEV-001/telemetry", json=payload)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["reading_time"], "2026-03-29T14:30:00+00:00")

    def test_valid_iso8601_with_utc_offset_accepted(self) -> None:
        payload = {
            "bpm": 80.0,
            "spo2": 99.0,
            "reading_time": "2026-03-29T14:30:00+00:00",
        }
        response = self.client.post("/api/devices/DEV-001/telemetry", json=payload)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["reading_time"], "2026-03-29T14:30:00+00:00")

    def test_valid_iso8601_with_non_utc_offset_normalized(self) -> None:
        payload = {
            "bpm": 82.0,
            "spo2": 97.0,
            "reading_time": "2026-03-29T20:00:00+05:30",
        }
        response = self.client.post("/api/devices/DEV-001/telemetry", json=payload)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["reading_time"], "2026-03-29T14:30:00+00:00")

    def test_missing_reading_time_defaults_to_current_time(self) -> None:
        payload = {
            "bpm": 70.0,
            "spo2": 96.0,
        }
        before = datetime.now(timezone.utc)
        response = self.client.post("/api/devices/DEV-001/telemetry", json=payload)
        after = datetime.now(timezone.utc)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("reading_time", data)
        parsed = datetime.fromisoformat(data["reading_time"])
        self.assertTrue(before <= parsed <= after)

    def test_explicit_null_reading_time_defaults_to_current_time(self) -> None:
        payload = {
            "bpm": 70.0,
            "spo2": 96.0,
            "reading_time": None,
        }
        before = datetime.now(timezone.utc)
        response = self.client.post("/api/devices/DEV-001/telemetry", json=payload)
        after = datetime.now(timezone.utc)
        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertIn("reading_time", data)
        parsed = datetime.fromisoformat(data["reading_time"])
        self.assertTrue(before <= parsed <= after)

    def test_malformed_string_rejected_422(self) -> None:
        payload = {
            "bpm": 75.0,
            "spo2": 98.0,
            "reading_time": "not-a-timestamp",
        }
        response = self.client.post("/api/devices/DEV-001/telemetry", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("detail", response.json())

    def test_invalid_date_february_31_rejected_422(self) -> None:
        payload = {
            "bpm": 75.0,
            "spo2": 98.0,
            "reading_time": "2026-02-31T12:00:00Z",
        }
        response = self.client.post("/api/devices/DEV-001/telemetry", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_invalid_time_hour_25_rejected_422(self) -> None:
        payload = {
            "bpm": 75.0,
            "spo2": 98.0,
            "reading_time": "2026-03-29T25:00:00Z",
        }
        response = self.client.post("/api/devices/DEV-001/telemetry", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_empty_string_rejected_422(self) -> None:
        payload = {
            "bpm": 75.0,
            "spo2": 98.0,
            "reading_time": "",
        }
        response = self.client.post("/api/devices/DEV-001/telemetry", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_whitespace_string_rejected_422(self) -> None:
        payload = {
            "bpm": 75.0,
            "spo2": 98.0,
            "reading_time": "    ",
        }
        response = self.client.post("/api/devices/DEV-001/telemetry", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_non_string_integer_rejected_422(self) -> None:
        payload = {
            "bpm": 75.0,
            "spo2": 98.0,
            "reading_time": 1711722600,
        }
        response = self.client.post("/api/devices/DEV-001/telemetry", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_non_string_boolean_rejected_422(self) -> None:
        payload = {
            "bpm": 75.0,
            "spo2": 98.0,
            "reading_time": True,
        }
        response = self.client.post("/api/devices/DEV-001/telemetry", json=payload)
        self.assertEqual(response.status_code, 422)

    # ============================================================
    # DIRECT SERVICE-LAYER VALIDATION TESTS
    # ============================================================

    def test_direct_service_layer_malformed_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.store.ingest_reading("DEV-RAW", {"bpm": 70, "reading_time": "bad-iso-timestamp"})
        self.assertIn("Invalid timestamp format", str(ctx.exception))

    def test_direct_service_layer_empty_string_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.store.ingest_reading("DEV-RAW", {"bpm": 70, "reading_time": ""})
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_direct_service_layer_invalid_type_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.store.ingest_reading("DEV-RAW", {"bpm": 70, "reading_time": 987654321})
        self.assertIn("Invalid timestamp type", str(ctx.exception))

    def test_direct_service_layer_none_succeeds(self) -> None:
        result = self.store.ingest_reading("DEV-RAW", {"bpm": 70, "reading_time": None})
        self.assertIsNotNone(result)
        self.assertIn("reading_time", result)
        self.assertEqual(result["bpm"], 70)

    # ============================================================
    # DATABASE ISOLATION VERIFICATION
    # ============================================================

    def test_database_not_modified_on_rejected_telemetry(self) -> None:
        with self.store._connection() as conn:
            dev_count_before = conn.execute("SELECT COUNT(*) FROM devices").fetchone()[0]
            rec_count_before = conn.execute("SELECT COUNT(*) FROM telemetry_records").fetchone()[0]

        # 1. Rejected via API (Pydantic / route layer)
        res_api = self.client.post(
            "/api/devices/DEV-REJECT-1/telemetry",
            json={"bpm": 90, "reading_time": "invalid-datetime-string"},
        )
        self.assertEqual(res_api.status_code, 422)

        # 2. Rejected via direct service call
        with self.assertRaises(ValueError):
            self.store.ingest_reading(
                "DEV-REJECT-2",
                {"bpm": 90, "reading_time": "2026-02-31T12:00:00Z"},
            )

        with self.store._connection() as conn:
            dev_count_after = conn.execute("SELECT COUNT(*) FROM devices").fetchone()[0]
            rec_count_after = conn.execute("SELECT COUNT(*) FROM telemetry_records").fetchone()[0]

        self.assertEqual(dev_count_before, dev_count_after, "devices table must not be modified on rejected reading")
        self.assertEqual(rec_count_before, rec_count_after, "telemetry_records table must not be modified on rejected reading")


if __name__ == "__main__":
    unittest.main()
