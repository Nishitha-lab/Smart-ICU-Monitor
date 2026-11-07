"""
Regression tests for:
1. Registered clinical record creation with patient_id and stay_id, with patientunitstayid NULL.
2. Historical clinical record creation with patientunitstayid, without a registered stay_id.
3. Rejection of mismatched patient and stay.
4. Rejection of nonexistent patient/stay references.
5. Existing historical records surviving migration.
6. Fresh database initialization.
7. Ambiguous numeric identifiers requiring explicit source where necessary.
8. Invalid source query parameter rejection.
9. Serial callback exception logging and successful processing of the next reading.
"""

import asyncio
import json
import logging
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
from fastapi import HTTPException

import main
from services.clinical_records import (
    ClinicalRecordCreate,
    ClinicalRecordStore,
)
from services.patient_service import (
    IcuStayCreate,
    PatientCreate,
    PatientStore,
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
        response = next(m for m in messages if m["type"] == "http.response.start")
        response_body = next(m for m in messages if m["type"] == "http.response.body")
        return int(response["status"]), json.loads(response_body["body"])

    return asyncio.run(call_app())


def make_mock_df() -> pd.DataFrame:
    cols = [
        "patientunitstayid", "hour", "heart_rate", "spo2", "respiration",
        "systolic_bp", "diastolic_bp", "mean_bp", "total_intake_ml", "total_output_ml",
        "dialysis_ml", "reported_net_ml", "net_fluid_balance_ml", "urine_output_ml",
        "active_infusion_count", "vasoactive_support", "anomaly_score", "ml_anomaly",
        "risk_probability", "risk_band", "clinical_rule_severity_v2",
        "clinical_rule_explanation_v2", "final_alert_priority", "alert_category",
        "final_alert_explanation",
    ]
    data = [
        {c: 0 for c in cols} | {"patientunitstayid": 101, "hour": 1.0, "final_alert_priority": "normal", "risk_band": "low", "clinical_rule_severity_v2": "normal", "clinical_rule_explanation_v2": "", "alert_category": "normal", "final_alert_explanation": "", "ml_anomaly": False},
        {c: 0 for c in cols} | {"patientunitstayid": 202, "hour": 1.0, "final_alert_priority": "high", "risk_band": "high", "clinical_rule_severity_v2": "high", "clinical_rule_explanation_v2": "", "alert_category": "single-signal", "final_alert_explanation": "", "ml_anomaly": False},
    ]
    return pd.DataFrame(data)


class IdentifierIntegrityAndCorrectionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_ident.sqlite3"

        self.previous_df = main.df
        self.previous_store = main.clinical_record_store
        self.previous_pstore = main.patient_store

        main.df = make_mock_df()
        main.patient_store = PatientStore(self.db_path)
        main.patient_store.initialize()
        main.clinical_record_store = ClinicalRecordStore(self.db_path)
        main.clinical_record_store.initialize()

    def tearDown(self) -> None:
        main.df = self.previous_df
        main.clinical_record_store = self.previous_store
        main.patient_store = self.previous_pstore
        try:
            self.temp_dir.cleanup()
        except (PermissionError, OSError):
            pass

    # ========================================================
    # 1. Registered record creation: patient_id & stay_id, patientunitstayid NULL
    # ========================================================

    def test_registered_record_creation_leaves_patientunitstayid_null(self) -> None:
        pat = main.patient_store.create_patient(PatientCreate(name="Registered Jane"))
        stay = main.patient_store.create_stay(pat["id"], IcuStayCreate(bed_number="BED-01"))

        now = datetime.now(timezone.utc)
        record_create = ClinicalRecordCreate(
            patient_id=pat["id"],
            stay_id=stay["id"],
            category="urine_output",
            measurement_name="Urine output",
            value=135.0,
            unit="mL",
            measurement_time=now,
            entered_by="Nurse Registered",
            observation_period="1 hour",
        )

        created = main.create_clinical_record(record_create)
        self.assertEqual(created["patient_id"], pat["id"])
        self.assertEqual(created["stay_id"], stay["id"])
        self.assertIsNone(created["patientunitstayid"])

        # Check SQLite directly: patientunitstayid must be strictly NULL
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT patientunitstayid, patient_id, stay_id FROM clinical_records WHERE record_id = ?",
            (created["record_id"],),
        ).fetchone()
        conn.close()

        self.assertIsNone(row[0], "patientunitstayid must be NULL for a registered record in SQLite.")
        self.assertEqual(row[1], pat["id"])
        self.assertEqual(row[2], stay["id"])

    # ========================================================
    # 2. Historical record creation: patientunitstayid only, no registered stay_id
    # ========================================================

    def test_historical_record_creation_leaves_stay_id_and_patient_id_null(self) -> None:
        now = datetime.now(timezone.utc)
        record_create = ClinicalRecordCreate(
            patientunitstayid=101,  # Historical eICU stay from df
            category="urine_output",
            measurement_name="Urine output",
            value=150.0,
            unit="mL",
            measurement_time=now,
            entered_by="Nurse Historical",
            observation_period="1 hour",
        )

        created = main.create_clinical_record(record_create)
        self.assertEqual(created["patientunitstayid"], 101)
        self.assertIsNone(created["patient_id"])
        self.assertIsNone(created["stay_id"])

        # Check SQLite directly: stay_id and patient_id must be strictly NULL
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT patientunitstayid, patient_id, stay_id FROM clinical_records WHERE record_id = ?",
            (created["record_id"],),
        ).fetchone()
        conn.close()

        self.assertEqual(row[0], 101)
        self.assertIsNone(row[1], "patient_id must be NULL for historical record.")
        self.assertIsNone(row[2], "stay_id must be NULL for historical record.")

    # ========================================================
    # 3. Rejection of mismatched patient and stay
    # ========================================================

    def test_rejection_of_mismatched_patient_and_stay(self) -> None:
        pat_a = main.patient_store.create_patient(PatientCreate(name="Patient A"))
        stay_a = main.patient_store.create_stay(pat_a["id"], IcuStayCreate(bed_number="BED-A"))

        pat_b = main.patient_store.create_patient(PatientCreate(name="Patient B"))
        stay_b = main.patient_store.create_stay(pat_b["id"], IcuStayCreate(bed_number="BED-B"))

        now = datetime.now(timezone.utc).isoformat()
        # Attempt to save a record for Stay B with Patient A
        status, body = request_app(
            "/api/clinical-records",
            "POST",
            {
                "patient_id": pat_a["id"],
                "stay_id": stay_b["id"],  # Mismatch!
                "category": "urine_output",
                "measurement_name": "Urine output",
                "value": 100.0,
                "unit": "mL",
                "measurement_time": now,
                "entered_by": "Nurse Mismatch",
                "observation_period": "1 hour",
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("does not belong to the specified patient", body.get("detail", ""))

    # ========================================================
    # 4. Rejection of nonexistent patient/stay references
    # ========================================================

    def test_rejection_of_nonexistent_stay_reference(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        status, body = request_app(
            "/api/clinical-records",
            "POST",
            {
                "stay_id": 99999,  # Nonexistent stay
                "category": "urine_output",
                "measurement_name": "Urine output",
                "value": 100.0,
                "unit": "mL",
                "measurement_time": now,
                "entered_by": "Nurse Ghost",
                "observation_period": "1 hour",
            },
        )
        self.assertEqual(status, 404)
        self.assertIn("not found", body.get("detail", "").lower())

    def test_rejection_of_nonexistent_patient_reference_with_valid_stay(self) -> None:
        pat = main.patient_store.create_patient(PatientCreate(name="Real Patient"))
        stay = main.patient_store.create_stay(pat["id"], IcuStayCreate(bed_number="BED-01"))

        now = datetime.now(timezone.utc).isoformat()
        status, body = request_app(
            "/api/clinical-records",
            "POST",
            {
                "patient_id": 99999,  # Nonexistent patient
                "stay_id": stay["id"],
                "category": "urine_output",
                "measurement_name": "Urine output",
                "value": 100.0,
                "unit": "mL",
                "measurement_time": now,
                "entered_by": "Nurse Ghost",
                "observation_period": "1 hour",
            },
        )
        self.assertEqual(status, 404)
        self.assertIn("not found", body.get("detail", "").lower())

    # ========================================================
    # 5. Existing historical records surviving migration
    # ========================================================

    def test_existing_historical_records_survive_migration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_file = Path(td) / "old_schema.sqlite3"
            conn = sqlite3.connect(db_file)
            # Create old unmigrated table: patientunitstayid INTEGER NOT NULL, no patient_id or stay_id
            conn.execute(
                """
                CREATE TABLE clinical_records (
                    record_id TEXT PRIMARY KEY,
                    patientunitstayid INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    measurement_name TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    unit TEXT,
                    measurement_time TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    entered_by TEXT NOT NULL,
                    source TEXT NOT NULL,
                    notes TEXT,
                    observation_period TEXT,
                    reference_range TEXT,
                    sample_collection_time TEXT,
                    result_time TEXT,
                    updated_at TEXT,
                    voided_at TEXT,
                    voided_by TEXT,
                    void_reason TEXT
                )
                """
            )
            # Insert historical records into the unmigrated table
            conn.execute(
                """
                INSERT INTO clinical_records (
                    record_id, patientunitstayid, category, measurement_name, value_json, unit,
                    measurement_time, entry_time, version, entered_by, source, notes
                ) VALUES (
                    'hist-old-001', 101, 'urine_output', 'Urine output', '250.0', 'mL',
                    '2026-11-08T00:00:00Z', '2026-11-08T00:00:00Z', 1, 'Nurse Old', 'MANUAL_ENTRY', 'Pre-migration historical record'
                )
                """
            )
            conn.commit()
            conn.close()

            # Now run initialize() which performs the migration
            store = ClinicalRecordStore(db_file)
            store.initialize()

            # Verify the record survived and has correct structure
            rec = store.get("hist-old-001")
            self.assertIsNotNone(rec)
            self.assertEqual(rec["record_id"], "hist-old-001")
            self.assertEqual(rec["patientunitstayid"], 101)
            self.assertEqual(rec["value"], 250.0)
            self.assertIsNone(rec["patient_id"])
            self.assertIsNone(rec["stay_id"])
            self.assertEqual(rec["notes"], "Pre-migration historical record")

            # Verify table schema: patientunitstayid is nullable
            conn = sqlite3.connect(db_file)
            table_info = conn.execute("PRAGMA table_info(clinical_records)").fetchall()
            col_dict = {row[1]: row for row in table_info}
            self.assertEqual(col_dict["patientunitstayid"][3], 0, "patientunitstayid must be nullable after migration.")
            conn.close()

    # ========================================================
    # 6. Fresh database initialization
    # ========================================================

    def test_fresh_database_initialization_schema_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_file = Path(td) / "fresh.sqlite3"
            store = ClinicalRecordStore(db_file)
            store.initialize()

            conn = sqlite3.connect(db_file)
            # Check PRAGMA table_info
            table_info = conn.execute("PRAGMA table_info(clinical_records)").fetchall()
            col_dict = {row[1]: row for row in table_info}

            self.assertIn("patientunitstayid", col_dict)
            self.assertEqual(col_dict["patientunitstayid"][3], 0, "patientunitstayid must be nullable in fresh DB.")
            self.assertIn("patient_id", col_dict)
            self.assertIn("stay_id", col_dict)

            # Check foreign keys
            fk_list = conn.execute("PRAGMA foreign_key_list(clinical_records)").fetchall()
            fk_tables = {row[2] for row in fk_list}
            self.assertIn("patients", fk_tables)
            self.assertIn("icu_stays", fk_tables)

            # Check foreign key check
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
            self.assertEqual(violations, [])
            conn.close()

    # ========================================================
    # 7. Ambiguous numeric identifiers requiring explicit source
    # ========================================================

    def test_ambiguous_numeric_identifier_requires_explicit_source(self) -> None:
        # Create a registered patient whose ID matches an eICU patientunitstayid
        # main.df has stay 202
        conn = sqlite3.connect(self.db_path)
        now = datetime.now(timezone.utc).isoformat()
        # Explicitly insert registered patient with ID 202
        conn.execute(
            """
            INSERT INTO patients (id, patient_code, name, created_at, updated_at)
            VALUES (202, 'P202', 'Ambiguous Patient', ?, ?)
            """,
            (now, now),
        )
        conn.commit()
        conn.close()

        # Query /api/patients/202 without source: must raise HTTP 400 Bad Request
        status, body = request_app("/api/patients/202", "GET")
        self.assertEqual(status, 400)
        self.assertIn("ambiguous", body.get("detail", "").lower())
        self.assertIn("source=registered", body.get("detail", "").lower())

        # Query /api/patients/202?source=registered: returns registered patient
        status_reg, body_reg = request_app("/api/patients/202?source=registered", "GET")
        self.assertEqual(status_reg, 200)
        self.assertEqual(body_reg["patient_code"], "P202")

        # Query /api/patients/202?source=historical: returns historical stay history
        status_hist, body_hist = request_app("/api/patients/202?source=historical", "GET")
        self.assertEqual(status_hist, 200)
        self.assertEqual(body_hist["patientunitstayid"], 202)

        # Query /api/patients/202/clinical-records without source: must raise HTTP 400 Bad Request
        status_rec, body_rec = request_app("/api/patients/202/clinical-records", "GET")
        self.assertEqual(status_rec, 400)
        self.assertIn("ambiguous", body_rec.get("detail", "").lower())

    # ========================================================
    # 8. Invalid source query parameter rejection
    # ========================================================

    def test_invalid_source_query_parameter_rejection(self) -> None:
        status_pat, _ = request_app("/api/patients/101?source=unsupported_source", "GET")
        self.assertEqual(status_pat, 422)

        status_cr, _ = request_app("/api/patients/101/clinical-records?source=invalid", "GET")
        self.assertEqual(status_cr, 422)

        status_stay, _ = request_app("/api/stays/101/clinical-records?source=bogus", "GET")
        self.assertEqual(status_stay, 422)

    # ========================================================
    # 9. Serial callback exception logging and processing next reading
    # ========================================================

    def test_serial_callback_exception_logging_and_subsequent_reading_processed(self) -> None:
        monitor = SerialMonitor(port="test-port")

        call_count = 0
        received_readings = []

        def failing_then_succeeding_callback(device_id: str, data: dict[str, object]) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated transient database error on reading 1")
            received_readings.append((device_id, data))

        monitor.set_on_reading(failing_then_succeeding_callback)

        # Reading 1: triggers callback exception
        with self.assertLogs("services.serial_service", level="ERROR") as log_capture:
            res1 = monitor.simulate_reading(fluid_ml=100.0, bpm=75.0, spo2=98.0)
            self.assertEqual(res1["fluid_ml"], 100.0)
            self.assertIn("Error in simulated reading callback", log_capture.output[0])

        # Reading 2: subsequent reading must succeed
        res2 = monitor.simulate_reading(fluid_ml=120.0, bpm=78.0, spo2=99.0)
        self.assertEqual(len(received_readings), 1)
        self.assertEqual(received_readings[0][1]["fluid_ml"], 120.0)
        self.assertEqual(monitor.snapshot()["fluid_ml"], 120.0)


if __name__ == "__main__":
    unittest.main()
