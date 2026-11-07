import asyncio
import json
import sqlite3
import threading
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException
from pydantic import ValidationError

import main
from services.clinical_records import (
    ClinicalRecordConflict,
    ClinicalRecordCreate,
    ClinicalRecordDatabaseBusy,
    ClinicalRecordStore,
    ClinicalRecordUpdate,
    ClinicalRecordVoid,
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


def urine_record(
    patient_id: int = 101,
    measured_at: datetime | None = None,
    value: float = 125.0,
    entered_by: str = "Nurse A",
) -> ClinicalRecordCreate:
    timestamp = measured_at or datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
    return ClinicalRecordCreate(
        patientunitstayid=patient_id,
        category="urine_output",
        measurement_name="Urine output",
        value=value,
        unit="mL",
        measurement_time=timestamp,
        entered_by=entered_by,
        observation_period="1 hour",
        notes=None,
    )


class ClinicalRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.previous_df = main.df
        self.previous_store = main.clinical_record_store
        main.df = pd.DataFrame({"patientunitstayid": [101, 202]})
        main.clinical_record_store = ClinicalRecordStore(
            Path(self.temp_dir.name) / "clinical.sqlite3"
        )
        main.clinical_record_store.initialize()
        self.addCleanup(self.restore_globals)

    def restore_globals(self) -> None:
        main.df = self.previous_df
        main.clinical_record_store = self.previous_store

    def test_create_record_sets_source_and_server_entry_time(self) -> None:
        record = main.create_clinical_record(urine_record())

        self.assertRegex(record["record_id"], r"^[0-9a-f-]{36}$")
        self.assertEqual(record["patientunitstayid"], 101)
        self.assertEqual(record["category"], "urine_output")
        self.assertEqual(record["value"], 125.0)
        self.assertEqual(record["unit"], "mL")
        self.assertEqual(record["source"], "MANUAL_ENTRY")
        self.assertEqual(record["entered_by"], "Nurse A")
        self.assertIsNotNone(datetime.fromisoformat(record["entry_time"]).tzinfo)
        self.assertIsNone(record["voided_at"])

    def test_missing_fields_and_invalid_numeric_values_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ClinicalRecordCreate.model_validate({"patientunitstayid": 101})

        with self.assertRaises(ValidationError):
            urine_record(value=float("nan"))

        with self.assertRaises(ValidationError):
            urine_record(value=-1)

        with self.assertRaises(ValidationError):
            ClinicalRecordCreate(
                **{
                    **urine_record().model_dump(),
                    "value": "not numeric",
                }
            )
        with self.assertRaises(ValidationError):
            urine_record(value=1_000_000_000_001)

    def test_timestamps_require_timezones_and_return_normalized_utc(self) -> None:
        naive_timestamp = datetime(2026, 11, 7, 12)
        with self.assertRaises(ValidationError):
            ClinicalRecordCreate(
                **{
                    **urine_record().model_dump(),
                    "measurement_time": naive_timestamp,
                }
            )

        measured_at = datetime.fromisoformat("2026-11-07T17:30:00+05:30")
        record = main.create_clinical_record(urine_record(measured_at=measured_at))
        self.assertEqual(record["measurement_time"], "2026-11-07T12:00:00+00:00")

    def test_lab_sample_and_result_timestamps_require_timezones_and_order(self) -> None:
        fields = {
            "patientunitstayid": 101,
            "category": "other_lab",
            "measurement_name": "Custom test",
            "value": 2.5,
            "unit": "unit",
            "measurement_time": datetime(2026, 11, 7, 10, tzinfo=timezone.utc),
            "sample_collection_time": datetime(2026, 11, 7, 11, tzinfo=timezone.utc),
            "result_time": datetime(2026, 11, 7, 12, tzinfo=timezone.utc),
            "entered_by": "Operator",
        }
        for field in ("sample_collection_time", "result_time"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ClinicalRecordCreate(**{**fields, field: datetime(2026, 11, 7, 11)})

        with self.assertRaises(ValidationError):
            ClinicalRecordCreate(
                **{
                    **fields,
                    "sample_collection_time": datetime(2026, 11, 7, 13, tzinfo=timezone.utc),
                }
            )

        custom = ClinicalRecordCreate(**fields)
        record = main.create_clinical_record(custom)
        self.assertEqual(record["measurement_time"], "2026-11-07T10:00:00+00:00")
        self.assertEqual(record["sample_collection_time"], "2026-11-07T11:00:00+00:00")
        self.assertEqual(record["result_time"], "2026-11-07T12:00:00+00:00")

    def test_text_value_has_a_technical_length_limit(self) -> None:
        observation = {
            "patientunitstayid": 101,
            "category": "clinical_observation",
            "measurement_name": "Clinical note",
            "value": "x" * 5001,
            "measurement_time": datetime(2026, 11, 7, 12, tzinfo=timezone.utc),
            "entered_by": "Operator",
        }
        with self.assertRaises(ValidationError):
            ClinicalRecordCreate(**observation)

    def test_units_are_validated_without_conversion(self) -> None:
        with self.assertRaises(ValidationError):
            ClinicalRecordCreate(
                **{
                    **urine_record().model_dump(),
                    "unit": "L",
                }
            )

    def test_nonexistent_patient_id_is_rejected(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            main.create_clinical_record(urine_record(patient_id=999))

        self.assertEqual(raised.exception.status_code, 404)

    def test_patient_specific_retrieval_keeps_records_separate(self) -> None:
        main.create_clinical_record(urine_record(patient_id=101))
        main.create_clinical_record(urine_record(patient_id=202))

        result = main.get_patient_clinical_records(101)

        self.assertEqual(result["patientunitstayid"], 101)
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["patientunitstayid"], 101)

    def test_numeric_patient_reference_can_explicitly_select_registered_or_historical_source(self) -> None:
        registered_patient = {"id": 101, "patient_code": "P101"}
        registered_record = {"record_id": "registered-record"}
        historical_record = {"record_id": "historical-record"}

        with (
            patch.object(main.patient_store, "get_patient", return_value=registered_patient),
            patch.object(
                main.clinical_record_store,
                "list_for_patient_id",
                return_value=[registered_record],
            ) as registered_query,
            patch.object(
                main.clinical_record_store,
                "list_for_patient",
                return_value=[historical_record],
            ) as historical_query,
        ):
            registered_status, registered_body = request_app(
                "/api/patients/101/clinical-records?source=registered&stay_id=7"
            )
            historical_status, historical_body = request_app(
                "/api/patients/101/clinical-records?source=historical"
            )

        self.assertEqual(registered_status, 200)
        self.assertEqual(registered_body["records"], [registered_record])
        registered_query.assert_called_once()
        self.assertEqual(registered_query.call_args.kwargs["stay_id"], 7)
        self.assertEqual(historical_status, 200)
        self.assertEqual(historical_body["records"], [historical_record])
        historical_query.assert_called_once()

    def test_category_and_date_filters(self) -> None:
        first_time = datetime(2026, 11, 7, 10, tzinfo=timezone.utc)
        second_time = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
        main.create_clinical_record(
            urine_record(measured_at=first_time, value=80)
        )
        main.create_clinical_record(
            urine_record(measured_at=second_time, value=100)
        )
        blood_test = ClinicalRecordCreate(
            patientunitstayid=101,
            category="blood_test",
            measurement_name="Hemoglobin",
            value=12.4,
            unit="g/dL",
            measurement_time=second_time,
            sample_collection_time=second_time,
            result_time=second_time + timedelta(minutes=30),
            entered_by="Nurse B",
        )
        main.create_clinical_record(blood_test)

        result = main.get_patient_clinical_records(
            101,
            category="urine_output",
            start_time=first_time,
            end_time=first_time,
        )

        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["value"], 80.0)

    def test_blood_and_other_labs_preserve_collection_and_result_times(self) -> None:
        collected = datetime(2026, 11, 7, 10, tzinfo=timezone.utc)
        resulted = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
        hemoglobin = ClinicalRecordCreate(
            patientunitstayid=101,
            category="blood_test",
            measurement_name="Hemoglobin",
            value=12.4,
            unit="g/dL",
            measurement_time=collected,
            sample_collection_time=collected,
            result_time=resulted,
            entered_by="Operator",
        )
        custom_lab = ClinicalRecordCreate(
            patientunitstayid=101,
            category="other_lab",
            measurement_name="Lactate",
            value=1.8,
            unit="mmol/L",
            measurement_time=collected,
            sample_collection_time=collected,
            result_time=resulted,
            entered_by="Operator",
            reference_range="0.5-2.2",
        )

        blood_record = main.create_clinical_record(hemoglobin)
        custom_record = main.create_clinical_record(custom_lab)

        self.assertNotEqual(
            blood_record["sample_collection_time"],
            blood_record["result_time"],
        )
        self.assertEqual(custom_record["measurement_name"], "Lactate")
        self.assertEqual(custom_record["reference_range"], "0.5-2.2")
        self.assertEqual(custom_record["unit"], "mmol/L")

    def test_narrative_clinical_observation_and_numeric_observation(self) -> None:
        measured_at = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
        narrative = ClinicalRecordCreate(
            patientunitstayid=101,
            category="clinical_observation",
            measurement_name="Consciousness / general observation",
            value="Responds to verbal stimuli.",
            measurement_time=measured_at,
            entered_by="Operator",
            notes="Observed during routine rounding.",
        )
        temperature = ClinicalRecordCreate(
            patientunitstayid=101,
            category="clinical_observation",
            measurement_name="Body temperature",
            value=37.1,
            unit="°C",
            measurement_time=measured_at,
            entered_by="Operator",
        )

        narrative_record = main.create_clinical_record(narrative)
        temperature_record = main.create_clinical_record(temperature)

        self.assertEqual(narrative_record["value"], "Responds to verbal stimuli.")
        self.assertIsNone(narrative_record["unit"])
        self.assertEqual(temperature_record["value"], 37.1)
        self.assertEqual(temperature_record["unit"], "°C")

    def test_patient_stay_search_returns_only_existing_ids(self) -> None:
        result = main.get_clinical_record_patient_stays(query="20", limit=10)

        self.assertEqual(result, {"patient_ids": [202]})
        with self.assertRaises(HTTPException) as raised:
            main.get_clinical_record_patient_stays(query="not-an-id", limit=10)
        self.assertEqual(raised.exception.status_code, 400)

    def test_naive_or_reversed_date_filters_are_rejected(self) -> None:
        with self.assertRaises(HTTPException) as naive:
            main.get_patient_clinical_records(
                101,
                start_time=datetime(2026, 11, 7, 10),
            )
        self.assertEqual(naive.exception.status_code, 422)

        with self.assertRaises(HTTPException) as reversed_range:
            main.get_patient_clinical_records(
                101,
                start_time=datetime(2026, 11, 7, 12, tzinfo=timezone.utc),
                end_time=datetime(2026, 11, 7, 10, tzinfo=timezone.utc),
            )
        self.assertEqual(reversed_range.exception.status_code, 422)

    def test_update_preserves_previous_version(self) -> None:
        created = main.create_clinical_record(urine_record())
        update_fields = urine_record(
            value=150,
            entered_by="Nurse B",
        ).model_dump(exclude={"patientunitstayid"})
        update = ClinicalRecordUpdate(
            **{
                **update_fields,
                "expected_version": created["version"],
                "change_reason": "Corrected transcription.",
            }
        )

        updated = main.update_clinical_record(created["record_id"], update)

        self.assertEqual(updated["record_id"], created["record_id"])
        self.assertEqual(updated["value"], 150.0)
        self.assertEqual(created["version"], 1)
        self.assertEqual(updated["version"], 2)
        self.assertIsNotNone(updated["updated_at"])
        revisions = main.get_clinical_record_revisions(created["record_id"])
        self.assertEqual(revisions["revisions"][0]["previous_record"]["value"], 125.0)
        self.assertEqual(revisions["revisions"][0]["previous_record"]["version"], 1)
        self.assertEqual(
            revisions["revisions"][0]["change_reason"],
            "Corrected transcription.",
        )
        self.assertEqual(len(revisions["revisions"]), 1)

    def test_stale_update_returns_conflict_without_adding_revision(self) -> None:
        created = main.create_clinical_record(urine_record())
        current_update = ClinicalRecordUpdate(
            **urine_record(value=150).model_dump(exclude={"patientunitstayid"}),
            expected_version=1,
            change_reason="First correction.",
        )
        main.update_clinical_record(created["record_id"], current_update)
        stale_update = ClinicalRecordUpdate(
            **urine_record(value=175).model_dump(exclude={"patientunitstayid"}),
            expected_version=1,
            change_reason="Stale correction.",
        )

        with self.assertRaises(HTTPException) as raised:
            main.update_clinical_record(created["record_id"], stale_update)

        self.assertEqual(raised.exception.status_code, 409)
        current = main.get_clinical_record(created["record_id"])
        self.assertEqual(current["value"], 150.0)
        self.assertEqual(current["version"], 2)
        self.assertEqual(
            len(main.get_clinical_record_revisions(created["record_id"])["revisions"]),
            1,
        )

    def test_two_competing_updates_only_apply_one_correction(self) -> None:
        created = main.create_clinical_record(urine_record())
        barrier = threading.Barrier(2)
        outcomes: list[dict[str, object] | int] = []
        outcome_lock = threading.Lock()

        def compete(value: float) -> None:
            update = ClinicalRecordUpdate(
                **urine_record(value=value).model_dump(exclude={"patientunitstayid"}),
                expected_version=created["version"],
                change_reason=f"Competing correction {value}.",
            )
            barrier.wait()
            try:
                outcome: dict[str, object] | int = main.update_clinical_record(
                    created["record_id"],
                    update,
                )
            except HTTPException as error:
                outcome = error.status_code
            with outcome_lock:
                outcomes.append(outcome)

        threads = [
            threading.Thread(target=compete, args=(150.0,)),
            threading.Thread(target=compete, args=(175.0,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        successful = [item for item in outcomes if isinstance(item, dict)]
        conflicts = [item for item in outcomes if item == 409]
        self.assertEqual(len(successful), 1)
        self.assertEqual(len(conflicts), 1)
        current = main.get_clinical_record(created["record_id"])
        self.assertEqual(current["version"], 2)
        self.assertEqual(
            len(main.get_clinical_record_revisions(created["record_id"])["revisions"]),
            1,
        )

    def test_failed_record_update_rolls_back_revision_insert(self) -> None:
        created = main.create_clinical_record(urine_record())
        with main.clinical_record_store._connection() as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_record_update
                BEFORE UPDATE ON clinical_records
                BEGIN
                    SELECT RAISE(ABORT, 'forced update failure');
                END
                """
            )
        update = ClinicalRecordUpdate(
            **urine_record(value=150).model_dump(exclude={"patientunitstayid"}),
            expected_version=1,
            change_reason="This transaction must roll back.",
        )

        with self.assertRaises(sqlite3.IntegrityError):
            main.update_clinical_record(created["record_id"], update)

        current = main.get_clinical_record(created["record_id"])
        self.assertEqual(current["value"], 125.0)
        self.assertEqual(current["version"], 1)
        self.assertEqual(
            main.get_clinical_record_revisions(created["record_id"])["revisions"],
            [],
        )

    def test_existing_database_migrates_version_without_losing_records(self) -> None:
        created = main.create_clinical_record(urine_record())
        with main.clinical_record_store._connection() as connection:
            connection.execute("ALTER TABLE clinical_records DROP COLUMN version")

        main.clinical_record_store.initialize()
        migrated = main.get_clinical_record(created["record_id"])

        self.assertEqual(migrated["version"], 1)
        self.assertEqual(migrated["value"], created["value"])
        self.assertEqual(migrated["patientunitstayid"], created["patientunitstayid"])

    def test_http_validation_conflict_and_database_busy_statuses(self) -> None:
        invalid_status, invalid_body = request_app(
            "/api/clinical-records",
            "POST",
            {},
        )
        self.assertEqual(invalid_status, 422)
        self.assertIn("detail", invalid_body)

        invalid_patient_payload = {
            **urine_record(patient_id=999).model_dump(mode="json"),
        }
        missing_stay_status, _ = request_app(
            "/api/clinical-records",
            "POST",
            invalid_patient_payload,
        )
        self.assertEqual(missing_stay_status, 404)

        invalid_timestamp_payload = {
            **invalid_patient_payload,
            "patientunitstayid": 101,
            "measurement_time": "not-a-timestamp",
        }
        invalid_timestamp_status, _ = request_app(
            "/api/clinical-records",
            "POST",
            invalid_timestamp_payload,
        )
        self.assertEqual(invalid_timestamp_status, 422)

        created = main.create_clinical_record(urine_record())
        update_payload = {
            **urine_record(value=150).model_dump(mode="json", exclude={"patientunitstayid"}),
            "expected_version": 1,
            "change_reason": "HTTP correction.",
        }
        successful_status, successful_body = request_app(
            f"/api/clinical-records/{created['record_id']}",
            "PUT",
            update_payload,
        )
        self.assertEqual(successful_status, 200)
        self.assertEqual(successful_body["version"], 2)

        conflict_payload = {
            **update_payload,
            "value": 175,
        }
        conflict_status, conflict_body = request_app(
            f"/api/clinical-records/{created['record_id']}",
            "PUT",
            conflict_payload,
        )
        self.assertEqual(conflict_status, 409)
        self.assertIn("Reload the latest record", conflict_body["detail"])

        with patch.object(
            main.clinical_record_store,
            "update",
            side_effect=ClinicalRecordDatabaseBusy("Database busy."),
        ):
            busy_status, busy_body = request_app(
                f"/api/clinical-records/{created['record_id']}",
                "PUT",
                update_payload | {"expected_version": 2},
            )
        self.assertEqual(busy_status, 503)
        self.assertEqual(busy_body["detail"], "Database busy.")

        stale_void_status, _ = request_app(
            f"/api/clinical-records/{created['record_id']}",
            "DELETE",
            {
                "expected_version": 1,
                "voided_by": "Operator",
                "void_reason": "Stale void.",
            },
        )
        self.assertEqual(stale_void_status, 409)
        void_status, void_body = request_app(
            f"/api/clinical-records/{created['record_id']}",
            "DELETE",
            {
                "expected_version": 2,
                "voided_by": "Operator",
                "void_reason": "HTTP smoke void.",
            },
        )
        self.assertEqual(void_status, 200)
        self.assertTrue(void_body["voided"])
        self.assertEqual(void_body["version"], 3)

    def test_sqlite_lock_is_reported_as_retryable_database_busy(self) -> None:
        created = main.create_clinical_record(urine_record())
        lock_connection = sqlite3.connect(
            main.clinical_record_store.database_path,
            timeout=0,
        )
        lock_connection.execute("BEGIN EXCLUSIVE")
        update = ClinicalRecordUpdate(
            **urine_record(value=150).model_dump(exclude={"patientunitstayid"}),
            expected_version=1,
            change_reason="Verify lock handling.",
        )
        try:
            with patch.object(
                main.clinical_record_store,
                "_connect",
                return_value=sqlite3.connect(
                    main.clinical_record_store.database_path,
                    timeout=0.01,
                ),
            ):
                with self.assertRaises(ClinicalRecordDatabaseBusy):
                    main.clinical_record_store.update(created["record_id"], update)
        finally:
            lock_connection.rollback()
            lock_connection.close()
        self.assertEqual(main.get_clinical_record(created["record_id"])["version"], 1)

    def test_repeated_measurements_get_distinct_record_ids(self) -> None:
        first = main.create_clinical_record(urine_record())
        second = main.create_clinical_record(urine_record())

        self.assertNotEqual(first["record_id"], second["record_id"])
        self.assertEqual(
            len(main.get_patient_clinical_records(101)["records"]),
            2,
        )

    def test_void_is_audited_and_not_a_hard_delete(self) -> None:
        created = main.create_clinical_record(urine_record())
        with self.assertRaises(HTTPException) as stale_void:
            main.void_clinical_record(
                created["record_id"],
                ClinicalRecordVoid(
                    expected_version=2,
                    voided_by="Nurse B",
                    void_reason="Stale void.",
                ),
            )
        self.assertEqual(stale_void.exception.status_code, 409)

        voided = main.void_clinical_record(
            created["record_id"],
            ClinicalRecordVoid(
                expected_version=created["version"],
                voided_by="Nurse B",
                void_reason="Duplicate entry.",
            ),
        )

        self.assertTrue(voided["voided"])
        self.assertEqual(voided["version"], 2)
        self.assertEqual(main.get_patient_clinical_records(101)["records"], [])
        retained = main.get_patient_clinical_records(101, include_voided=True)
        self.assertEqual(len(retained["records"]), 1)
        self.assertEqual(retained["records"][0]["void_reason"], "Duplicate entry.")
        self.assertEqual(len(main.get_clinical_record_revisions(created["record_id"])["revisions"]), 1)

    def test_empty_history_and_unknown_record(self) -> None:
        self.assertEqual(main.get_patient_clinical_records(101)["records"], [])

        with self.assertRaises(HTTPException) as raised:
            main.get_clinical_record("00000000-0000-0000-0000-000000000001")
        self.assertEqual(raised.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
