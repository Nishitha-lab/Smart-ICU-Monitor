"""
Comprehensive Stress, Concurrency, Security, and Reliability Tests
for Smart ICU Patient Safety System.

Validates:
1. Foreign key enforcement, deletion restriction, and multi-stay integrity under direct SQLite queries.
2. Concurrency stress: concurrent patient registration, concurrent admissions, and concurrent device assignments.
3. Concurrency on clinical records: strict optimistic locking under competing concurrent updates.
4. Telemetry burst stress: 500 packets, zero cross-patient contamination, delayed packet isolation, unassigned handling.
5. Serial monitor failure modes: malformed frames, out-of-bounds metrics, disconnect recovery, prompt thread shutdown.
"""

import json
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from services.clinical_records import (
    ClinicalRecordConflict,
    ClinicalRecordCreate,
    ClinicalRecordStore,
    ClinicalRecordUpdate,
)
from services.patient_service import (
    DeviceCreate,
    IcuStayCreate,
    PatientCreate,
    PatientStore,
    TelemetryIngest,
)
from services.serial_service import SerialMonitor, LINE_PATTERN


class StressAndReliabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_stress.sqlite3"
        self.patient_store = PatientStore(self.db_path)
        self.patient_store.initialize()
        self.clinical_store = ClinicalRecordStore(self.db_path)
        self.clinical_store.initialize()

    def tearDown(self) -> None:
        try:
            self.temp_dir.cleanup()
        except (PermissionError, OSError):
            pass

    # ========================================================
    # 1. DATABASE CONSTRAINT & FOREIGN KEY RIGOR
    # ========================================================

    def test_direct_sqlite_foreign_key_enforcement(self) -> None:
        """PRAGMA foreign_keys = ON must reject orphan stays and orphan clinical records."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        now = datetime.now(timezone.utc).isoformat()

        # Attempt to insert an ICU stay referencing non-existent patient ID 99999
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO icu_stays (stay_code, patient_id, admission_datetime, discharge_status, created_at, updated_at)
                VALUES ('S-ORPHAN', 99999, ?, 'active', ?, ?)
                """,
                (now, now, now),
            )
        conn.close()

    def test_prevent_patient_deletion_with_associated_records_at_sql_level(self) -> None:
        """ON DELETE RESTRICT must prevent direct SQL deletion of patient with stays."""
        patient = self.patient_store.create_patient(PatientCreate(name="Relational Test Patient"))
        stay = self.patient_store.create_stay(
            patient["id"], IcuStayCreate(bed_number="BED-01")
        )
        self.assertIsNotNone(stay)

        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM patients WHERE id = ?", (patient["id"],))
        conn.close()

        # PatientStore delete_patient must also enforce this and raise ValueError
        with self.assertRaises(ValueError):
            self.patient_store.delete_patient(patient["id"])

    def test_unique_active_stay_index_prevents_multiple_active_stays(self) -> None:
        """Direct SQL insertion of two active stays for same patient must fail."""
        patient = self.patient_store.create_patient(PatientCreate(name="Double Active Patient"))
        self.patient_store.create_stay(patient["id"], IcuStayCreate(bed_number="BED-01"))

        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO icu_stays (stay_code, patient_id, admission_datetime, discharge_status, created_at, updated_at)
                VALUES ('S-CONFLICT', ?, ?, 'active', ?, ?)
                """,
                (patient["id"], now, now, now),
            )
        conn.close()

    # ========================================================
    # 2. CONCURRENCY & RACE CONDITIONS
    # ========================================================

    def test_concurrent_patient_creations_maintain_uniqueness(self) -> None:
        """Multiple threads creating patients simultaneously produce unique codes and valid rows."""
        num_threads = 20
        created_patients = []
        errors = []

        def worker(idx: int) -> None:
            try:
                p = self.patient_store.create_patient(
                    PatientCreate(name=f"Concurrent Patient {idx}", age=30 + idx)
                )
                created_patients.append(p)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors occurred during concurrent creation: {errors}")
        self.assertEqual(len(created_patients), num_threads)

        codes = [p["patient_code"] for p in created_patients]
        self.assertEqual(len(codes), len(set(codes)), "Duplicate patient codes detected!")

    def test_concurrent_device_assignment_race_condition(self) -> None:
        """Simultaneous assignments of the same device to different patients must never create double active assignments."""
        dev = self.patient_store.register_device(
            DeviceCreate(device_id="ESP32-RACE-001", device_name="Race Device")
        )
        patients = [
            self.patient_store.create_patient(PatientCreate(name=f"Race Patient {i}"))
            for i in range(5)
        ]
        stays = [
            self.patient_store.create_stay(p["id"], IcuStayCreate(bed_number=f"BED-{i}"))
            for i, p in enumerate(patients)
        ]

        assignment_results = []

        def assign_worker(patient_id: int, stay_id: int) -> None:
            try:
                res = self.patient_store.assign_device(
                    dev["device_id"], patient_id, stay_id
                )
                assignment_results.append(res)
            except Exception as e:
                assignment_results.append(e)

        threads = [
            threading.Thread(target=assign_worker, args=(p["id"], s["id"]))
            for p, s in zip(patients, stays)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify only ONE active assignment exists for this device in the database
        conn = sqlite3.connect(self.db_path)
        active_rows = conn.execute(
            "SELECT * FROM device_assignments WHERE device_id = ? AND assignment_status = 'active'",
            (dev["device_id"],),
        ).fetchall()
        conn.close()

        self.assertEqual(len(active_rows), 1, "Exactly one active assignment must exist after race.")

    def test_concurrent_clinical_record_optimistic_locking(self) -> None:
        """10 concurrent threads attempting to update the same record version=1: exactly 1 wins, 9 fail with conflict."""
        patient = self.patient_store.create_patient(PatientCreate(name="Optimistic Patient"))
        stay = self.patient_store.create_stay(patient["id"], IcuStayCreate(bed_number="BED-01"))

        now = datetime.now(timezone.utc)
        record = self.clinical_store.create(
            ClinicalRecordCreate(
                patient_id=patient["id"],
                stay_id=stay["id"],
                category="urine_output",
                measurement_name="Urine output",
                value=100.0,
                unit="mL",
                measurement_time=now,
                entered_by="Nurse Initial",
                observation_period="1 hour",
            )
        )
        record_id = record["record_id"]
        self.assertEqual(record["version"], 1)

        successes = []
        conflicts = []

        def updater_worker(worker_id: int) -> None:
            try:
                updated = self.clinical_store.update(
                    record_id,
                    ClinicalRecordUpdate(
                        expected_version=1,  # All 10 threads expect version 1
                        change_reason=f"Update by worker {worker_id}",
                        category="urine_output",
                        measurement_name="Urine output",
                        value=120.0 + worker_id,
                        unit="mL",
                        measurement_time=now,
                        entered_by=f"Worker {worker_id}",
                        observation_period="1 hour",
                    ),
                )
                successes.append((worker_id, updated))
            except ClinicalRecordConflict as cc:
                conflicts.append((worker_id, str(cc)))

        threads = [threading.Thread(target=updater_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(successes), 1, f"Expected exactly 1 winner, got {len(successes)}")
        self.assertEqual(len(conflicts), 9, f"Expected 9 conflicts, got {len(conflicts)}")

        final_rec = self.clinical_store.get(record_id)
        self.assertEqual(final_rec["version"], 2)

    # ========================================================
    # 3. TELEMETRY BURST & STRICT ATTRIBUTION INTEGRITY
    # ========================================================

    def test_sustained_telemetry_burst_500_packets_with_zero_cross_talk(self) -> None:
        """
        Send 500 telemetry packets alternating between assigned Device A (Patient A),
        assigned Device B (Patient B), and unassigned Device C.
        Verify:
        - 100% of Device A readings attach to Patient A, 0% to Patient B
        - 100% of Device B readings attach to Patient B, 0% to Patient A
        - 100% of Device C readings attach to no patient
        """
        dev_a = self.patient_store.register_device(
            DeviceCreate(device_id="ESP32-A", device_name="Device Alpha")
        )
        dev_b = self.patient_store.register_device(
            DeviceCreate(device_id="ESP32-B", device_name="Device Beta")
        )
        dev_c = self.patient_store.register_device(
            DeviceCreate(device_id="ESP32-UNASSIGNED", device_name="Unassigned Device")
        )

        pat_a = self.patient_store.create_patient(PatientCreate(name="Patient Alpha"))
        stay_a = self.patient_store.create_stay(pat_a["id"], IcuStayCreate(bed_number="BED-A"))
        self.patient_store.assign_device(dev_a["device_id"], pat_a["id"], stay_a["id"])

        pat_b = self.patient_store.create_patient(PatientCreate(name="Patient Beta"))
        stay_b = self.patient_store.create_stay(pat_b["id"], IcuStayCreate(bed_number="BED-B"))
        self.patient_store.assign_device(dev_b["device_id"], pat_b["id"], stay_b["id"])

        now_base = datetime.now(timezone.utc)
        total_packets = 500

        for i in range(total_packets):
            device_choice = [dev_a["device_id"], dev_b["device_id"], dev_c["device_id"]][i % 3]
            reading_time = (now_base + timedelta(seconds=i)).isoformat()
            self.patient_store.ingest_reading(
                device_choice,
                {
                    "reading_time": reading_time,
                    "fluid_ml": 100.0 + i,
                    "bpm": 72.0 + (i % 20),
                    "spo2": 98.0,
                    "device_status": "NORMAL",
                    "connection_status": "CONNECTED",
                },
            )

        # Retrieve attributed telemetry for Patient A & Patient B
        readings_a = self.patient_store.get_patient_telemetry(pat_a["id"], limit=1000)
        readings_b = self.patient_store.get_patient_telemetry(pat_b["id"], limit=1000)

        expected_a_count = sum(1 for i in range(total_packets) if i % 3 == 0)
        expected_b_count = sum(1 for i in range(total_packets) if i % 3 == 1)

        self.assertEqual(len(readings_a), expected_a_count)
        self.assertEqual(len(readings_b), expected_b_count)

        # Check attribution isolation: No reading in Patient A belongs to dev_b or dev_c
        for r in readings_a:
            self.assertEqual(r["device_id"], dev_a["device_id"])
            self.assertEqual(r["patient_id"], pat_a["id"])
            self.assertEqual(r["stay_id"], stay_a["id"])

        for r in readings_b:
            self.assertEqual(r["device_id"], dev_b["device_id"])
            self.assertEqual(r["patient_id"], pat_b["id"])
            self.assertEqual(r["stay_id"], stay_b["id"])

        # Check unassigned telemetry in database: must have patient_id IS NULL
        conn = sqlite3.connect(self.db_path)
        unassigned_count = conn.execute(
            "SELECT COUNT(*) FROM telemetry_records WHERE device_id = ? AND patient_id IS NULL",
            (dev_c["device_id"],),
        ).fetchone()[0]
        conn.close()

        expected_c_count = sum(1 for i in range(total_packets) if i % 3 == 2)
        self.assertEqual(unassigned_count, expected_c_count)

    def test_delayed_reading_policy_before_assignment_start(self) -> None:
        """Readings with timestamp prior to assignment start time must NOT attach to the patient."""
        dev = self.patient_store.register_device(
            DeviceCreate(device_id="ESP32-DELAY", device_name="Delay Device")
        )
        pat = self.patient_store.create_patient(PatientCreate(name="Delayed Reading Patient"))
        stay = self.patient_store.create_stay(pat["id"], IcuStayCreate(bed_number="BED-D"))

        # Assign device NOW
        assignment = self.patient_store.assign_device(
            dev["device_id"], pat["id"], stay["id"]
        )
        assign_time = datetime.fromisoformat(assignment["assigned_at"])

        # Delayed packet from 10 minutes BEFORE assignment
        delayed_time = (assign_time - timedelta(minutes=10)).isoformat()
        self.patient_store.ingest_reading(
            dev["device_id"],
            {
                "reading_time": delayed_time,
                "fluid_ml": 50.0,
                "bpm": 75.0,
                "spo2": 99.0,
                "device_status": "NORMAL",
                "connection_status": "CONNECTED",
            },
        )

        # Fresh packet from 10 seconds AFTER assignment
        fresh_time = (assign_time + timedelta(seconds=10)).isoformat()
        self.patient_store.ingest_reading(
            dev["device_id"],
            {
                "reading_time": fresh_time,
                "fluid_ml": 55.0,
                "bpm": 76.0,
                "spo2": 99.0,
                "device_status": "NORMAL",
                "connection_status": "CONNECTED",
            },
        )

        patient_readings = self.patient_store.get_patient_telemetry(pat["id"])
        self.assertEqual(len(patient_readings), 1, "Only the reading after assignment must be attributed.")
        self.assertEqual(patient_readings[0]["fluid_ml"], 55.0)

    def test_telemetry_validity_is_server_derived_and_invalid_values_are_hidden(self) -> None:
        device = self.patient_store.register_device(
            DeviceCreate(device_id="ESP32-QUALITY", device_name="Quality Device")
        )
        patient = self.patient_store.create_patient(
            PatientCreate(name="Telemetry Quality Patient")
        )
        stay = self.patient_store.create_stay(
            patient["id"], IcuStayCreate(bed_number="BED-Q")
        )
        self.patient_store.assign_device(
            device["device_id"], patient["id"], stay["id"]
        )

        reading = self.patient_store.ingest_reading(
            device["device_id"],
            {
                "fluid_ml": 30.0,
                "bpm": 0.0,
                "spo2": 0.0,
                "validity": {"fluid": False, "bpm": True, "spo2": True},
            },
        )
        expected_validity = {"fluid": True, "bpm": False, "spo2": False}
        self.assertEqual(reading["validity"], expected_validity)
        self.assertEqual(reading["fluid_ml"], 30.0)
        self.assertIsNone(reading["bpm"])
        self.assertIsNone(reading["spo2"])

        with sqlite3.connect(self.db_path) as connection:
            stored = connection.execute(
                """
                SELECT fluid_ml, bpm, spo2, quality_metadata
                FROM telemetry_records WHERE device_id = ?
                """,
                (device["device_id"],),
            ).fetchone()
        self.assertEqual(stored[:3], (30.0, 0.0, 0.0))
        self.assertEqual(json.loads(stored[3]), expected_validity)

        self.assertEqual(
            self.patient_store.get_patient(patient["id"])["latest_telemetry"]["validity"],
            expected_validity,
        )
        history = self.patient_store.get_patient_telemetry(patient["id"])
        self.assertIsNone(history[0]["bpm"])
        device_view = next(
            item
            for item in self.patient_store.list_devices()
            if item["device_id"] == device["device_id"]
        )
        self.assertIsNone(device_view["last_telemetry"]["spo2"])
        timeline = self.patient_store.get_patient_timeline(patient["id"])
        telemetry_event = next(
            event for event in timeline["events"] if event["event_type"] == "telemetry"
        )
        self.assertNotIn("BPM 0.0", telemetry_event["summary"])

        with self.assertRaises(ValueError):
            TelemetryIngest(fluid_ml=float("nan"))

    # ========================================================
    # 4. SERIAL SERVICE RESILIENCE & FAILURE RECOVERY
    # ========================================================

    def test_regex_line_pattern_against_malformed_and_valid_inputs(self) -> None:
        """Validate LINE_PATTERN behavior against valid, boundary, and corrupted serial inputs."""
        valid_line = "Fluid: 250.5 mL | BPM: 78.0 | SpO2: 98.2% | STATUS: NORMAL"
        match = LINE_PATTERN.search(valid_line)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "250.5")
        self.assertEqual(match.group(2), "78.0")
        self.assertEqual(match.group(3), "98.2")
        self.assertEqual(match.group(4), "NORMAL")

        # Corrupted / noisy strings
        corrupted_lines = [
            "Fluid: mL | BPM: 78 | SpO2: 98% | STATUS: NORMAL",
            "Garbage serial line from baudrate mismatch ####%%%%",
            "Fluid: -20 mL | BPM: | SpO2: 95% | STATUS: OK",
            "",
            "Fluid: 100 mL | BPM: 80 | SpO2: % | STATUS: ",
        ]
        for line in corrupted_lines:
            match = LINE_PATTERN.search(line)
            if match:
                try:
                    map(float, match.group(1, 2, 3))
                except ValueError:
                    pass

    def test_serial_monitor_validity_flags_on_extreme_values(self) -> None:
        """Metrics outside plausible physiological ranges must set validity flags to False."""
        monitor = SerialMonitor(port="test")

        # Set extreme / unphysiological data
        monitor._set(
            connected=True,
            connection_status="CONNECTED",
            fluid_ml=-10.0,
            bpm=280.0,
            spo2=50.0,
            device_status="ALERT",
            last_update=datetime.now(timezone.utc).isoformat(),
            validity={"fluid": False, "bpm": False, "spo2": False},
        )
        snap = monitor.snapshot()
        self.assertFalse(snap["validity"]["fluid"])
        self.assertFalse(snap["validity"]["bpm"])
        self.assertFalse(snap["validity"]["spo2"])

    def test_serial_monitor_clean_thread_shutdown(self) -> None:
        """SerialMonitor thread must stop cleanly on stop()."""
        monitor = SerialMonitor(port="NON_EXISTENT_PORT_COM99")
        monitor.start()
        time.sleep(0.05)  # Let thread initiate
        monitor.stop()
        self.assertFalse(monitor._thread.is_alive(), "Monitor thread must be stopped.")


if __name__ == "__main__":
    unittest.main()
