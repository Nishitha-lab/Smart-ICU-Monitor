from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ClinicalCategory = Literal[
    "urine_output",
    "blood_test",
    "other_lab",
    "clinical_observation",
]

BLOOD_TEST_UNITS = {
    "Hemoglobin": "g/dL",
    "Creatinine": "mg/dL",
    "BUN": "mg/dL",
    "Sodium": "mmol/L",
    "Potassium": "mmol/L",
    "Blood glucose": "mg/dL",
}

CLINICAL_OBSERVATION_UNITS = {
    "Body temperature": "°C",
    "Respiratory rate": "breaths/min",
    "Systolic blood pressure": "mmHg",
    "Diastolic blood pressure": "mmHg",
    "Consciousness / general observation": None,
    "Clinical note": None,
}

MAX_TECHNICAL_VALUE = 1_000_000_000_000
MAX_NARRATIVE_LENGTH = 5_000


class ClinicalRecordConflict(Exception):
    pass


class ClinicalRecordDatabaseBusy(Exception):
    pass


class ClinicalRecordFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: ClinicalCategory
    measurement_name: Annotated[str, Field(min_length=1, max_length=120)]
    value: float | str
    unit: Annotated[str | None, Field(max_length=40)] = None
    measurement_time: datetime
    entered_by: Annotated[str, Field(min_length=1, max_length=120)]
    notes: Annotated[str | None, Field(max_length=5000)] = None
    observation_period: Annotated[str | None, Field(max_length=80)] = None
    reference_range: Annotated[str | None, Field(max_length=120)] = None
    sample_collection_time: datetime | None = None
    result_time: datetime | None = None

    @field_validator("measurement_name", "entered_by")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be blank.")
        return value

    @field_validator(
        "unit",
        "notes",
        "observation_period",
        "reference_range",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("value", mode="before")
    @classmethod
    def reject_boolean_value(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("Boolean values are not valid measurements.")
        return value

    @model_validator(mode="after")
    def validate_measurement(self) -> ClinicalRecordFields:
        timestamps = {
            "measurement time": self.measurement_time,
            "sample collection time": self.sample_collection_time,
            "result time": self.result_time,
        }
        for label, value in timestamps.items():
            if value is not None and value.utcoffset() is None:
                raise ValueError(f"{label} must include a timezone.")

        if (
            self.sample_collection_time is not None
            and self.result_time is not None
            and self.sample_collection_time > self.result_time
        ):
            raise ValueError(
                "Sample collection time cannot be after result time."
            )

        if isinstance(self.value, (int, float)) and not (
            float("-inf") < float(self.value) < float("inf")
        ):
            raise ValueError("Measurement value must be finite.")
        if isinstance(self.value, (int, float)) and abs(self.value) > MAX_TECHNICAL_VALUE:
            raise ValueError(
                f"Numeric value exceeds the technical input limit of {MAX_TECHNICAL_VALUE:g}; "
                "this limit is not a clinical reference range."
            )
        if isinstance(self.value, str) and len(self.value) > MAX_NARRATIVE_LENGTH:
            raise ValueError(
                f"Text value cannot exceed {MAX_NARRATIVE_LENGTH} characters."
            )

        if self.category in {"urine_output", "blood_test", "other_lab"}:
            if not isinstance(self.value, (int, float)):
                raise ValueError("This measurement requires a numeric value.")

        if self.category == "urine_output":
            if self.measurement_name != "Urine output":
                raise ValueError("Urine output requires the measurement name 'Urine output'.")
            if self.unit != "mL":
                raise ValueError("Urine output unit must be mL; unit conversion is not performed.")
            if self.value < 0:
                raise ValueError("Urine volume cannot be negative.")

        if self.category == "blood_test":
            expected_unit = BLOOD_TEST_UNITS.get(self.measurement_name)
            if expected_unit is None:
                raise ValueError("Choose a supported blood test or use Other Laboratory Results.")
            if self.unit != expected_unit:
                raise ValueError(
                    f"{self.measurement_name} unit must be {expected_unit}; unit conversion is not performed."
                )
            if self.value < 0:
                raise ValueError("Blood test result cannot be negative.")
            if self.sample_collection_time is None:
                raise ValueError("Blood test sample collection time is required.")

        if self.category == "other_lab":
            if not self.unit:
                raise ValueError("A unit is required for laboratory results.")
            if self.sample_collection_time is None:
                raise ValueError("Laboratory sample collection time is required.")

        if self.category == "clinical_observation":
            expected_unit = CLINICAL_OBSERVATION_UNITS.get(self.measurement_name)
            if expected_unit is None and self.measurement_name not in CLINICAL_OBSERVATION_UNITS:
                raise ValueError("Choose a supported clinical observation.")
            if self.unit != expected_unit:
                raise ValueError(
                    f"{self.measurement_name} unit must be "
                    f"{expected_unit or 'omitted'}; unit conversion is not performed."
                )
            if expected_unit is None:
                if not isinstance(self.value, str) or not self.value.strip():
                    raise ValueError("This observation requires non-empty text.")
            elif not isinstance(self.value, (int, float)):
                raise ValueError("This observation requires a numeric value.")
            elif self.value < 0:
                raise ValueError("Clinical measurement cannot be negative.")

        return self


class ClinicalRecordCreate(ClinicalRecordFields):
    patientunitstayid: Annotated[int | None, Field(gt=0)] = None
    patient_id: Annotated[int | None, Field(gt=0)] = None
    stay_id: Annotated[int | None, Field(gt=0)] = None

    @model_validator(mode="after")
    def validate_reference(self) -> ClinicalRecordCreate:
        if self.patientunitstayid is not None and (self.stay_id is not None or self.patient_id is not None):
            raise ValueError(
                "Historical identifier 'patientunitstayid' cannot be combined with registered 'stay_id' or 'patient_id'."
            )
        if self.patientunitstayid is None and self.stay_id is None:
            raise ValueError(
                "Either patientunitstayid (for historical eICU records) or stay_id (for registered ICU admissions) must be provided."
            )
        return self


class ClinicalRecordUpdate(ClinicalRecordFields):
    expected_version: Annotated[int, Field(ge=1)]
    change_reason: Annotated[str, Field(min_length=1, max_length=500)]
    patient_id: Annotated[int | None, Field(gt=0)] = None
    stay_id: Annotated[int | None, Field(gt=0)] = None

    @field_validator("change_reason")
    @classmethod
    def strip_change_reason(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("A correction reason is required.")
        return value


class ClinicalRecordVoid(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: Annotated[int, Field(ge=1)]
    voided_by: Annotated[str, Field(min_length=1, max_length=120)]
    void_reason: Annotated[str, Field(min_length=1, max_length=500)]

    @field_validator("voided_by", "void_reason")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be blank.")
        return value


def _utc_iso(value: datetime) -> str:
    if value.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone.")
    return value.astimezone(timezone.utc).isoformat()


class ClinicalRecordStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.execute("PRAGMA journal_mode = WAL;")
        connection.execute("PRAGMA busy_timeout = 5000;")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        except sqlite3.OperationalError as error:
            if "locked" in str(error).lower() or "busy" in str(error).lower():
                raise ClinicalRecordDatabaseBusy(
                    "Clinical records are temporarily busy; retry the request."
                ) from error
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_code TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    age INTEGER CHECK (age IS NULL OR (age >= 0 AND age <= 130)),
                    sex TEXT CHECK (sex IS NULL OR sex IN ('male', 'female', 'other', 'unknown', 'Male', 'Female', 'Other', 'Unknown')),
                    date_of_birth TEXT,
                    contact_information TEXT,
                    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
                    weight_kg REAL CHECK (weight_kg IS NULL OR (weight_kg > 0 AND weight_kg <= 500)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_patients_code ON patients (patient_code);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_patients_status ON patients (status);"
            )
            table_info = connection.execute("PRAGMA table_info(patients)").fetchall()
            col_names = {row["name"] for row in table_info}
            if "weight_kg" not in col_names:
                try:
                    connection.execute("ALTER TABLE patients ADD COLUMN weight_kg REAL;")
                except sqlite3.OperationalError:
                    pass

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS icu_stays (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stay_code TEXT NOT NULL UNIQUE,
                    patient_id INTEGER NOT NULL REFERENCES patients (id) ON DELETE RESTRICT,
                    admission_datetime TEXT NOT NULL,
                    discharge_datetime TEXT,
                    bed_number TEXT,
                    ward_or_unit TEXT,
                    admission_reason TEXT,
                    discharge_status TEXT NOT NULL DEFAULT 'active' CHECK (discharge_status IN ('active', 'discharged', 'transferred')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_icu_stays_patient ON icu_stays (patient_id);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_icu_stays_code ON icu_stays (stay_code);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_icu_stays_status ON icu_stays (discharge_status);"
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_active_stay_per_patient
                ON icu_stays (patient_id)
                WHERE discharge_status = 'active';
                """
            )

            # Check clinical_records table schema
            table_check = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='clinical_records'"
            ).fetchone()

            if not table_check:
                connection.execute(
                    """
                    CREATE TABLE clinical_records (
                        record_id TEXT PRIMARY KEY,
                        patientunitstayid INTEGER,
                        patient_id INTEGER REFERENCES patients(id) ON DELETE RESTRICT,
                        stay_id INTEGER REFERENCES icu_stays(id) ON DELETE RESTRICT,
                        category TEXT NOT NULL CHECK (
                            category IN (
                                'urine_output', 'blood_test',
                                'other_lab', 'clinical_observation'
                            )
                        ),
                        measurement_name TEXT NOT NULL,
                        value_json TEXT NOT NULL,
                        unit TEXT,
                        measurement_time TEXT NOT NULL,
                        entry_time TEXT NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
                        entered_by TEXT NOT NULL,
                        source TEXT NOT NULL CHECK (source = 'MANUAL_ENTRY'),
                        notes TEXT,
                        observation_period TEXT,
                        reference_range TEXT,
                        sample_collection_time TEXT,
                        result_time TEXT,
                        updated_at TEXT,
                        voided_at TEXT,
                        voided_by TEXT,
                        void_reason TEXT,
                        CHECK (patientunitstayid IS NOT NULL OR stay_id IS NOT NULL)
                    )
                    """
                )
            else:
                table_info = connection.execute(
                    "PRAGMA table_info(clinical_records)"
                ).fetchall()
                col_map = {row["name"]: dict(row) for row in table_info}

                pus_col = col_map.get("patientunitstayid", {})
                is_pus_not_null = pus_col.get("notnull", 0) == 1
                has_patient_id = "patient_id" in col_map
                has_stay_id = "stay_id" in col_map

                fk_list = connection.execute(
                    "PRAGMA foreign_key_list(clinical_records)"
                ).fetchall()
                fk_tables = {row["table"] for row in fk_list}
                has_fks = "patients" in fk_tables and "icu_stays" in fk_tables

                if "version" not in col_map or is_pus_not_null or not has_patient_id or not has_stay_id or not has_fks:
                    connection.execute("PRAGMA foreign_keys = OFF;")
                    connection.execute(
                        """
                        CREATE TABLE clinical_records_migration_tmp (
                            record_id TEXT PRIMARY KEY,
                            patientunitstayid INTEGER,
                            patient_id INTEGER REFERENCES patients(id) ON DELETE RESTRICT,
                            stay_id INTEGER REFERENCES icu_stays(id) ON DELETE RESTRICT,
                            category TEXT NOT NULL CHECK (
                                category IN (
                                    'urine_output', 'blood_test',
                                    'other_lab', 'clinical_observation'
                                )
                            ),
                            measurement_name TEXT NOT NULL,
                            value_json TEXT NOT NULL,
                            unit TEXT,
                            measurement_time TEXT NOT NULL,
                            entry_time TEXT NOT NULL,
                            version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
                            entered_by TEXT NOT NULL,
                            source TEXT NOT NULL CHECK (source = 'MANUAL_ENTRY'),
                            notes TEXT,
                            observation_period TEXT,
                            reference_range TEXT,
                            sample_collection_time TEXT,
                            result_time TEXT,
                            updated_at TEXT,
                            voided_at TEXT,
                            voided_by TEXT,
                            void_reason TEXT,
                            CHECK (patientunitstayid IS NOT NULL OR stay_id IS NOT NULL)
                        );
                        """
                    )
                    col_names = set(col_map.keys())
                    p_id_expr = "patient_id" if "patient_id" in col_names else "NULL"
                    s_id_expr = "stay_id" if "stay_id" in col_names else "NULL"
                    v_expr = "version" if "version" in col_names else "1"

                    connection.execute(
                        f"""
                        INSERT INTO clinical_records_migration_tmp (
                            record_id, patientunitstayid, patient_id, stay_id, category, measurement_name,
                            value_json, unit, measurement_time, entry_time, version, entered_by,
                            source, notes, observation_period, reference_range,
                            sample_collection_time, result_time, updated_at, voided_at, voided_by, void_reason
                        )
                        SELECT
                            record_id, patientunitstayid, {p_id_expr}, {s_id_expr}, category, measurement_name,
                            value_json, unit, measurement_time, entry_time, {v_expr}, entered_by,
                            source, notes, observation_period, reference_range,
                            sample_collection_time, result_time, updated_at, voided_at, voided_by, void_reason
                        FROM clinical_records;
                        """
                    )
                    connection.execute("DROP TABLE clinical_records;")
                    connection.execute(
                        "ALTER TABLE clinical_records_migration_tmp RENAME TO clinical_records;"
                    )
                    connection.execute("PRAGMA foreign_keys = ON;")

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS clinical_records_patient_time
                ON clinical_records (patientunitstayid, measurement_time)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS clinical_records_patient_id
                ON clinical_records (patient_id, measurement_time)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS clinical_records_stay_id
                ON clinical_records (stay_id, measurement_time)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS clinical_record_revisions (
                    revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL,
                    revised_at TEXT NOT NULL,
                    revised_by TEXT NOT NULL,
                    change_reason TEXT NOT NULL,
                    previous_record_json TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> dict[str, object]:
        record = dict(row)
        record["value"] = json.loads(record.pop("value_json"))
        record["voided"] = record["voided_at"] is not None
        if "patient_id" not in record:
            record["patient_id"] = None
        if "stay_id" not in record:
            record["stay_id"] = None
        return record

    @staticmethod
    def _values(fields: ClinicalRecordFields) -> tuple[object, ...]:
        return (
            fields.category,
            fields.measurement_name,
            json.dumps(fields.value, allow_nan=False),
            fields.unit,
            _utc_iso(fields.measurement_time),
            fields.entered_by,
            fields.notes,
            fields.observation_period,
            fields.reference_range,
            _utc_iso(fields.sample_collection_time)
            if fields.sample_collection_time
            else None,
            _utc_iso(fields.result_time) if fields.result_time else None,
        )

    def create(self, fields: ClinicalRecordCreate) -> dict[str, object]:
        record_id = str(uuid.uuid4())
        entry_time = datetime.now(timezone.utc).isoformat()
        values = self._values(fields)

        stay_id = fields.stay_id
        patient_id = fields.patient_id
        patientunitstayid = fields.patientunitstayid

        if stay_id is not None and patientunitstayid is not None:
            raise ValueError(
                "Historical identifier 'patientunitstayid' cannot be combined with registered 'stay_id'."
            )

        with self._connection() as connection:
            if stay_id is not None:
                stay_row = connection.execute(
                    "SELECT id, patient_id, discharge_status FROM icu_stays WHERE id = ?",
                    (stay_id,),
                ).fetchone()
                if not stay_row:
                    raise KeyError(f"ICU stay '{stay_id}' not found.")
                if stay_row["discharge_status"] != "active":
                    raise ValueError("Cannot add clinical records to a discharged ICU stay.")

                if patient_id is not None:
                    pat_row = connection.execute(
                        "SELECT id FROM patients WHERE id = ?", (patient_id,)
                    ).fetchone()
                    if not pat_row:
                        raise KeyError(f"Patient '{patient_id}' not found.")
                    if stay_row["patient_id"] != patient_id:
                        raise ValueError(f"ICU stay '{stay_id}' does not belong to patient '{patient_id}'.")
                else:
                    patient_id = stay_row["patient_id"]

                # Registered record: ensure patientunitstayid is NULL (never substitute stay_id!)
                patientunitstayid = None

            elif patientunitstayid is not None:
                # Historical eICU record: registered fields must be None
                patient_id = None
                stay_id = None
            else:
                raise ValueError("Either patientunitstayid or stay_id must be provided.")

            try:
                connection.execute(
                    """
                    INSERT INTO clinical_records (
                        record_id, patientunitstayid, patient_id, stay_id, category, measurement_name,
                        value_json, unit, measurement_time, entry_time, entered_by,
                        source, notes, observation_period, reference_range,
                        sample_collection_time, result_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'MANUAL_ENTRY', ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id,
                        patientunitstayid,
                        patient_id,
                        stay_id,
                        *values[:5],
                        entry_time,
                        *values[5:],
                    ),
                )
            except sqlite3.IntegrityError as err:
                raise ValueError(f"Invalid clinical record reference: {err}") from err
            row = connection.execute(
                "SELECT * FROM clinical_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return self._record(row)

    def get(self, record_id: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM clinical_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return self._record(row) if row else None

    def list_for_patient(
        self,
        patientunitstayid: int,
        category: ClinicalCategory | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        include_voided: bool = False,
    ) -> list[dict[str, object]]:
        conditions = ["patientunitstayid = ?"]
        parameters: list[object] = [patientunitstayid]
        if category:
            conditions.append("category = ?")
            parameters.append(category)
        if start_time:
            conditions.append("measurement_time >= ?")
            parameters.append(_utc_iso(start_time))
        if end_time:
            conditions.append("measurement_time <= ?")
            parameters.append(_utc_iso(end_time))
        if not include_voided:
            conditions.append("voided_at IS NULL")
        query = (
            "SELECT * FROM clinical_records WHERE "
            + " AND ".join(conditions)
            + " ORDER BY measurement_time DESC, entry_time DESC"
        )
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._record(row) for row in rows]

    def list_for_patient_id(
        self,
        patient_id: int,
        category: ClinicalCategory | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        include_voided: bool = False,
        stay_id: int | None = None,
    ) -> list[dict[str, object]]:
        conditions = ["patient_id = ?"]
        parameters: list[object] = [patient_id]
        if stay_id:
            conditions.append("stay_id = ?")
            parameters.append(stay_id)
        if category:
            conditions.append("category = ?")
            parameters.append(category)
        if start_time:
            conditions.append("measurement_time >= ?")
            parameters.append(_utc_iso(start_time))
        if end_time:
            conditions.append("measurement_time <= ?")
            parameters.append(_utc_iso(end_time))
        if not include_voided:
            conditions.append("voided_at IS NULL")
        query = (
            "SELECT * FROM clinical_records WHERE "
            + " AND ".join(conditions)
            + " ORDER BY measurement_time DESC, entry_time DESC"
        )
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._record(row) for row in rows]

    def list_for_stay_id(
        self,
        stay_id: int,
        category: ClinicalCategory | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        include_voided: bool = False,
    ) -> list[dict[str, object]]:
        conditions = ["stay_id = ?"]
        parameters: list[object] = [stay_id]
        if category:
            conditions.append("category = ?")
            parameters.append(category)
        if start_time:
            conditions.append("measurement_time >= ?")
            parameters.append(_utc_iso(start_time))
        if end_time:
            conditions.append("measurement_time <= ?")
            parameters.append(_utc_iso(end_time))
        if not include_voided:
            conditions.append("voided_at IS NULL")
        query = (
            "SELECT * FROM clinical_records WHERE "
            + " AND ".join(conditions)
            + " ORDER BY measurement_time DESC, entry_time DESC"
        )
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._record(row) for row in rows]

    def update(
        self,
        record_id: str,
        fields: ClinicalRecordUpdate,
    ) -> dict[str, object] | None:
        revised_at = datetime.now(timezone.utc).isoformat()
        values = self._values(fields)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM clinical_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if row is None or row["voided_at"] is not None:
                return None
            if row["version"] != fields.expected_version:
                raise ClinicalRecordConflict(
                    "This record has changed since it was loaded. "
                    "Reload the latest record before correcting it."
                )
            previous = self._record(row)
            connection.execute(
                """
                INSERT INTO clinical_record_revisions (
                    record_id, revised_at, revised_by, change_reason,
                    previous_record_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    revised_at,
                    fields.entered_by,
                    fields.change_reason,
                    json.dumps(previous, allow_nan=False),
                ),
            )
            cursor = connection.execute(
                """
                UPDATE clinical_records SET
                    category = ?, measurement_name = ?, value_json = ?,
                    unit = ?, measurement_time = ?, entered_by = ?, notes = ?,
                    observation_period = ?, reference_range = ?,
                    sample_collection_time = ?, result_time = ?, updated_at = ?,
                    version = version + 1
                WHERE record_id = ? AND version = ? AND voided_at IS NULL
                """,
                (*values, revised_at, record_id, fields.expected_version),
            )
            if cursor.rowcount != 1:
                raise ClinicalRecordConflict(
                    "This record has changed since it was loaded. "
                    "Reload the latest record before correcting it."
                )
            updated = connection.execute(
                "SELECT * FROM clinical_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return self._record(updated)

    def void(
        self,
        record_id: str,
        expected_version: int,
        voided_by: str,
        void_reason: str,
    ) -> dict[str, object] | None:
        voided_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM clinical_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                return None
            if row["voided_at"] is None:
                if row["version"] != expected_version:
                    raise ClinicalRecordConflict(
                        "This record has changed since it was loaded. "
                        "Reload the latest record before voiding it."
                    )
                previous = self._record(row)
                connection.execute(
                    """
                    INSERT INTO clinical_record_revisions (
                        record_id, revised_at, revised_by, change_reason,
                        previous_record_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record_id,
                        voided_at,
                        voided_by,
                        void_reason,
                        json.dumps(previous, allow_nan=False),
                    ),
                )
                cursor = connection.execute(
                    """
                    UPDATE clinical_records SET
                        voided_at = ?, voided_by = ?, void_reason = ?,
                        version = version + 1
                    WHERE record_id = ? AND version = ? AND voided_at IS NULL
                    """,
                    (
                        voided_at,
                        voided_by,
                        void_reason,
                        record_id,
                        expected_version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ClinicalRecordConflict(
                        "This record has changed since it was loaded. "
                        "Reload the latest record before voiding it."
                    )
            voided = connection.execute(
                "SELECT * FROM clinical_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return self._record(voided)

    def revisions(self, record_id: str) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT revision_id, record_id, revised_at, revised_by,
                       change_reason, previous_record_json
                FROM clinical_record_revisions
                WHERE record_id = ?
                ORDER BY revision_id
                """,
                (record_id,),
            ).fetchall()
        return [
            {
                "revision_id": row["revision_id"],
                "record_id": row["record_id"],
                "revised_at": row["revised_at"],
                "revised_by": row["revised_by"],
                "change_reason": row["change_reason"],
                "previous_record": json.loads(row["previous_record_json"]),
            }
            for row in rows
        ]
