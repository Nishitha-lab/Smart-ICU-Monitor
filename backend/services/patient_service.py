from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utc_iso(value: datetime | str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return datetime.now(timezone.utc).isoformat()
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as err:
            raise ValueError(f"Invalid timestamp format '{value}': {err}") from err
        if parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    if value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


# ============================================================
# PYDANTIC MODELS
# ============================================================

class PatientCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=120)]
    patient_code: Annotated[str | None, Field(max_length=40)] = None
    age: Annotated[int | None, Field(ge=0, le=130)] = None
    sex: Literal["male", "female", "other", "unknown", "Male", "Female", "Other", "Unknown"] | None = None
    date_of_birth: Annotated[str | None, Field(max_length=20)] = None
    contact_information: Annotated[str | None, Field(max_length=255)] = None

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Patient name cannot be blank.")
        return value

    @field_validator("contact_information", "date_of_birth")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("patient_code")
    @classmethod
    def normalize_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value.upper() if value else None

    @field_validator("date_of_birth")
    @classmethod
    def validate_dob(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date_of_birth must be formatted as YYYY-MM-DD.")
        return value


class PatientUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str | None, Field(min_length=1, max_length=120)] = None
    age: Annotated[int | None, Field(ge=0, le=130)] = None
    sex: Literal["male", "female", "other", "unknown", "Male", "Female", "Other", "Unknown"] | None = None
    date_of_birth: Annotated[str | None, Field(max_length=20)] = None
    contact_information: Annotated[str | None, Field(max_length=255)] = None
    status: Literal["active", "archived"] | None = None

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Patient name cannot be blank.")
        return value

    @field_validator("contact_information", "date_of_birth")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("date_of_birth")
    @classmethod
    def validate_dob(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date_of_birth must be formatted as YYYY-MM-DD.")
        return value


class IcuStayCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stay_code: Annotated[str | None, Field(max_length=40)] = None
    admission_datetime: datetime | str | None = None
    bed_number: Annotated[str | None, Field(max_length=40)] = None
    ward_or_unit: Annotated[str | None, Field(max_length=80)] = None
    admission_reason: Annotated[str | None, Field(max_length=500)] = None

    @field_validator("bed_number", "ward_or_unit", "admission_reason")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("stay_code")
    @classmethod
    def normalize_stay_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value.upper() if value else None


class IcuStayUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bed_number: Annotated[str | None, Field(max_length=40)] = None
    ward_or_unit: Annotated[str | None, Field(max_length=80)] = None
    admission_reason: Annotated[str | None, Field(max_length=500)] = None

    @field_validator("bed_number", "ward_or_unit", "admission_reason")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class IcuStayDischarge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discharge_datetime: datetime | str | None = None
    discharge_status: Literal["discharged", "transferred"] = "discharged"


class DeviceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: Annotated[str, Field(min_length=1, max_length=60)]
    device_name: Annotated[str, Field(min_length=1, max_length=120)]
    connection_type: Annotated[str, Field(max_length=40)] = "serial"
    port: Annotated[str | None, Field(max_length=60)] = None

    @field_validator("device_name")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("This field cannot be blank.")
        return value

    @field_validator("device_id")
    @classmethod
    def normalize_device_id(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("Device ID cannot be blank.")
        return value

    @field_validator("port")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class DeviceAssign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: int | str
    stay_id: int | str


class DeviceUnassign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Annotated[str | None, Field(max_length=500)] = None


class TelemetryIngest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fluid_ml: Annotated[float | None, Field(allow_inf_nan=False)] = None
    bpm: Annotated[float | None, Field(allow_inf_nan=False)] = None
    spo2: Annotated[float | None, Field(allow_inf_nan=False)] = None
    device_status: Annotated[str | None, Field(max_length=60)] = "NORMAL"
    connection_status: Annotated[str, Field(max_length=40)] = "CONNECTED"
    reading_time: datetime | str | None = None
    validity: dict[str, bool] | None = None


# ============================================================
# PATIENT & DEVICE STORE (SQLITE)
# ============================================================

class PatientStore:
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
        finally:
            connection.close()

    @staticmethod
    def _telemetry_validity(reading: dict[str, Any]) -> dict[str, bool]:
        values = {
            "fluid": reading.get("fluid_ml"),
            "bpm": reading.get("bpm"),
            "spo2": reading.get("spo2"),
        }

        def numeric(value: object) -> float | None:
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                return float(value)
            return None

        fluid = numeric(values["fluid"])
        bpm = numeric(values["bpm"])
        spo2 = numeric(values["spo2"])
        return {
            "fluid": fluid is not None and fluid >= 0,
            "bpm": bpm is not None and 40 <= bpm <= 200,
            "spo2": spo2 is not None and 70 <= spo2 <= 100,
        }

    @staticmethod
    def _mask_invalid_telemetry(reading: dict[str, Any]) -> dict[str, Any]:
        metadata = reading.get("quality_metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        validity = {
            name: metadata.get(name) is True
            for name in ("fluid", "bpm", "spo2")
        }
        for name, field in (
            ("fluid", "fluid_ml"),
            ("bpm", "bpm"),
            ("spo2", "spo2"),
        ):
            if not validity[name]:
                reading[field] = None
        reading["quality_metadata"] = metadata
        reading["validity"] = validity
        return reading

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            # Patients table
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

            # ICU Stays table
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
                "CREATE INDEX IF NOT EXISTS idx_icu_stays_status ON icu_stays (discharge_status);"
            )
            # Enforce at most one active stay per patient at the database level
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_active_patient_stay
                ON icu_stays (patient_id)
                WHERE discharge_status = 'active';
                """
            )

            # Devices table
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS devices (
                    device_id TEXT PRIMARY KEY,
                    device_name TEXT NOT NULL,
                    connection_type TEXT NOT NULL DEFAULT 'serial',
                    port TEXT,
                    status TEXT NOT NULL DEFAULT 'disconnected' CHECK (status IN ('connected', 'disconnected', 'monitoring', 'error')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            # Device Assignments table
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS device_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL REFERENCES devices (device_id) ON DELETE RESTRICT,
                    patient_id INTEGER NOT NULL REFERENCES patients (id) ON DELETE RESTRICT,
                    stay_id INTEGER NOT NULL REFERENCES icu_stays (id) ON DELETE RESTRICT,
                    assigned_at TEXT NOT NULL,
                    unassigned_at TEXT,
                    assignment_status TEXT NOT NULL DEFAULT 'active' CHECK (assignment_status IN ('active', 'ended')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_device_assignments_device ON device_assignments (device_id, assignment_status);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_device_assignments_patient ON device_assignments (patient_id);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_device_assignments_stay ON device_assignments (stay_id);"
            )
            # Enforce that no device has multiple active assignments simultaneously
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_active_device_assignment
                ON device_assignments (device_id)
                WHERE assignment_status = 'active';
                """
            )

            # Telemetry Records table
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT NOT NULL REFERENCES devices (device_id) ON DELETE RESTRICT,
                    patient_id INTEGER REFERENCES patients (id) ON DELETE RESTRICT,
                    stay_id INTEGER REFERENCES icu_stays (id) ON DELETE RESTRICT,
                    assignment_id INTEGER REFERENCES device_assignments (id) ON DELETE SET NULL,
                    reading_time TEXT NOT NULL,
                    ingestion_time TEXT NOT NULL,
                    fluid_ml REAL,
                    bpm REAL,
                    spo2 REAL,
                    device_status TEXT,
                    connection_status TEXT NOT NULL,
                    quality_metadata TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_patient_time ON telemetry_records (patient_id, reading_time);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_stay_time ON telemetry_records (stay_id, reading_time);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_device_time ON telemetry_records (device_id, reading_time);"
            )

            # Seed default ESP32 device if not present
            now_iso = datetime.now(timezone.utc).isoformat()
            connection.execute(
                """
                INSERT OR IGNORE INTO devices (
                    device_id, device_name, connection_type, port, status, created_at, updated_at
                ) VALUES ('ESP32-001', 'ICU Monitoring Device 1', 'serial', 'COM9', 'disconnected', ?, ?)
                """,
                (now_iso, now_iso),
            )

    # ------------------------------------------------------------
    # PATIENT OPERATIONS
    # ------------------------------------------------------------

    def _next_patient_code(self, connection: sqlite3.Connection) -> str:
        cursor = connection.execute(
            "SELECT patient_code FROM patients WHERE patient_code GLOB 'P[0-9]*'"
        )
        highest = 0
        for (code,) in cursor.fetchall():
            suffix = code[1:]
            if suffix.isdigit():
                highest = max(highest, int(suffix))
        return f"P{highest + 1:03d}"

    def create_patient(self, fields: PatientCreate) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            code = fields.patient_code
            if code is None:
                code = self._next_patient_code(connection)
            else:
                existing = connection.execute(
                    "SELECT id FROM patients WHERE UPPER(patient_code) = UPPER(?)", (code,)
                ).fetchone()
                if existing:
                    raise ValueError(f"Patient code '{code}' already exists.")

            try:
                cursor = connection.execute(
                    """
                    INSERT INTO patients (
                        patient_code, name, age, sex, date_of_birth, contact_information,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (
                        code,
                        fields.name,
                        fields.age,
                        fields.sex,
                        fields.date_of_birth,
                        fields.contact_information,
                        now_iso,
                        now_iso,
                    ),
                )
            except sqlite3.IntegrityError as err:
                raise ValueError(f"Patient code '{code}' already exists.") from err

            patient_id = cursor.lastrowid
            row = connection.execute(
                "SELECT * FROM patients WHERE id = ?", (patient_id,)
            ).fetchone()
        return dict(row)

    def resolve_patient_id(self, patient_ref: int | str, connection: sqlite3.Connection | None = None) -> int | None:
        def _lookup(conn: sqlite3.Connection) -> int | None:
            try:
                if isinstance(patient_ref, int) or (isinstance(patient_ref, str) and patient_ref.isdigit()):
                    row = conn.execute("SELECT id FROM patients WHERE id = ?", (int(patient_ref),)).fetchone()
                    if row:
                        return row["id"]
                row = conn.execute("SELECT id FROM patients WHERE UPPER(patient_code) = UPPER(?)", (str(patient_ref),)).fetchone()
                return row["id"] if row else None
            except sqlite3.OperationalError:
                return None

        if connection:
            return _lookup(connection)
        with self._connection() as conn:
            return _lookup(conn)

    def get_patient(self, patient_ref: int | str) -> dict[str, Any] | None:
        try:
            with self._connection() as connection:
                patient_id = self.resolve_patient_id(patient_ref, connection)
                if not patient_id:
                    return None
                patient_row = connection.execute(
                    "SELECT * FROM patients WHERE id = ?", (patient_id,)
                ).fetchone()
                if not patient_row:
                    return None

                patient = dict(patient_row)

                # Fetch stays
                stays_rows = connection.execute(
                    """
                    SELECT * FROM icu_stays
                    WHERE patient_id = ?
                    ORDER BY admission_datetime DESC
                    """,
                    (patient_id,),
                ).fetchall()
                stays = [dict(s) for s in stays_rows]
                patient["stays"] = stays

                # Active stay
                active_stay = next((s for s in stays if s["discharge_status"] == "active"), None)
                patient["active_stay"] = active_stay

                # Active device assignment
                active_device = None
                if active_stay:
                    assign_row = connection.execute(
                        """
                        SELECT a.*, d.device_name, d.status as device_connection_status
                        FROM device_assignments a
                        JOIN devices d ON a.device_id = d.device_id
                        WHERE a.patient_id = ? AND a.stay_id = ? AND a.assignment_status = 'active'
                        """,
                        (patient_id, active_stay["id"]),
                    ).fetchone()
                    if assign_row:
                        active_device = dict(assign_row)
                patient["active_device"] = active_device

                # Latest telemetry reading
                latest_telemetry = None
                telem_row = connection.execute(
                    """
                    SELECT * FROM telemetry_records
                    WHERE patient_id = ?
                    ORDER BY reading_time DESC
                    LIMIT 1
                    """,
                    (patient_id,),
                ).fetchone()
                if telem_row:
                    latest_telemetry = self._mask_invalid_telemetry(dict(telem_row))
                patient["latest_telemetry"] = latest_telemetry

                return patient
        except sqlite3.OperationalError:
            return None

    def list_patients(
        self,
        search: str | None = None,
        status: str | None = "all",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        conditions: list[str] = []
        params: list[Any] = []

        if status and status != "all":
            conditions.append("p.status = ?")
            params.append(status)

        if search:
            search_pattern = f"%{search.strip()}%"
            conditions.append("(p.name LIKE ? OR p.patient_code LIKE ?)")
            params.extend([search_pattern, search_pattern])

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._connection() as connection:
            count_query = f"SELECT COUNT(*) FROM patients p {where_clause}"
            total = connection.execute(count_query, params).fetchone()[0]

            query = f"""
                SELECT p.*,
                       s.id as active_stay_id,
                       s.stay_code as active_stay_code,
                       s.bed_number as active_bed_number,
                       s.ward_or_unit as active_ward,
                       s.admission_datetime as active_admission_datetime,
                       s.discharge_status as active_stay_status,
                       a.device_id as active_device_id,
                       d.device_name as active_device_name,
                       d.status as active_device_status
                FROM patients p
                LEFT JOIN icu_stays s ON p.id = s.patient_id AND s.discharge_status = 'active'
                LEFT JOIN device_assignments a ON s.id = a.stay_id AND a.assignment_status = 'active'
                LEFT JOIN devices d ON a.device_id = d.device_id
                {where_clause}
                ORDER BY p.id DESC
                LIMIT ? OFFSET ?
            """
            rows = connection.execute(query, [*params, limit, offset]).fetchall()

            patients = []
            for row in rows:
                p = dict(row)
                # Structure active_stay
                if p.get("active_stay_id"):
                    p["active_stay"] = {
                        "id": p.pop("active_stay_id"),
                        "stay_code": p.pop("active_stay_code"),
                        "bed_number": p.pop("active_bed_number"),
                        "ward_or_unit": p.pop("active_ward"),
                        "admission_datetime": p.pop("active_admission_datetime"),
                        "discharge_status": p.pop("active_stay_status"),
                    }
                else:
                    p["active_stay"] = None
                    p.pop("active_stay_id", None)
                    p.pop("active_stay_code", None)
                    p.pop("active_bed_number", None)
                    p.pop("active_ward", None)
                    p.pop("active_admission_datetime", None)
                    p.pop("active_stay_status", None)

                # Structure active_device
                if p.get("active_device_id"):
                    p["active_device"] = {
                        "device_id": p.pop("active_device_id"),
                        "device_name": p.pop("active_device_name"),
                        "status": p.pop("active_device_status"),
                    }
                else:
                    p["active_device"] = None
                    p.pop("active_device_id", None)
                    p.pop("active_device_name", None)
                    p.pop("active_device_status", None)

                patients.append(p)

        return {"total": total, "patients": patients}

    def update_patient(self, patient_ref: int | str, fields: PatientUpdate) -> dict[str, Any] | None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            patient_id = self.resolve_patient_id(patient_ref, connection)
            if not patient_id:
                return None

            # If archiving, discharge active stay and end active device assignments
            if fields.status == "archived":
                active_stay = connection.execute(
                    "SELECT id FROM icu_stays WHERE patient_id = ? AND discharge_status = 'active'",
                    (patient_id,),
                ).fetchone()
                if active_stay:
                    stay_id = active_stay["id"]
                    connection.execute(
                        """
                        UPDATE icu_stays
                        SET discharge_status = 'discharged', discharge_datetime = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (now_iso, now_iso, stay_id),
                    )
                    connection.execute(
                        """
                        UPDATE device_assignments
                        SET assignment_status = 'ended', unassigned_at = ?, updated_at = ?
                        WHERE stay_id = ? AND assignment_status = 'active'
                        """,
                        (now_iso, now_iso, stay_id),
                    )

            updates: list[str] = ["updated_at = ?"]
            params: list[Any] = [now_iso]

            if fields.name is not None:
                updates.append("name = ?")
                params.append(fields.name)
            if fields.age is not None:
                updates.append("age = ?")
                params.append(fields.age)
            if fields.sex is not None:
                updates.append("sex = ?")
                params.append(fields.sex)
            if fields.date_of_birth is not None:
                updates.append("date_of_birth = ?")
                params.append(fields.date_of_birth)
            if fields.contact_information is not None:
                updates.append("contact_information = ?")
                params.append(fields.contact_information)
            if fields.status is not None:
                updates.append("status = ?")
                params.append(fields.status)

            params.append(patient_id)
            connection.execute(
                f"UPDATE patients SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            row = connection.execute(
                "SELECT * FROM patients WHERE id = ?", (patient_id,)
            ).fetchone()
            return dict(row) if row else None

    def archive_patient(self, patient_ref: int | str) -> dict[str, Any] | None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            patient_id = self.resolve_patient_id(patient_ref, connection)
            if not patient_id:
                return None

            # Check if there is an active stay; if so, discharge it and unassign device
            active_stay = connection.execute(
                "SELECT id FROM icu_stays WHERE patient_id = ? AND discharge_status = 'active'",
                (patient_id,),
            ).fetchone()
            if active_stay:
                stay_id = active_stay["id"]
                connection.execute(
                    """
                    UPDATE icu_stays
                    SET discharge_status = 'discharged', discharge_datetime = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (now_iso, now_iso, stay_id),
                )
                connection.execute(
                    """
                    UPDATE device_assignments
                    SET assignment_status = 'ended', unassigned_at = ?, updated_at = ?
                    WHERE stay_id = ? AND assignment_status = 'active'
                    """,
                    (now_iso, now_iso, stay_id),
                )

            connection.execute(
                "UPDATE patients SET status = 'archived', updated_at = ? WHERE id = ?",
                (now_iso, patient_id),
            )
            row = connection.execute(
                "SELECT * FROM patients WHERE id = ?", (patient_id,)
            ).fetchone()
            return dict(row) if row else None

    def delete_patient(self, patient_ref: int | str) -> bool:
        """Prevent deletion if patient has any stays, clinical records, or telemetry."""
        with self._connection() as connection:
            patient_id = self.resolve_patient_id(patient_ref, connection)
            if not patient_id:
                return False

            # Check stays
            stays_count = connection.execute(
                "SELECT COUNT(*) FROM icu_stays WHERE patient_id = ?", (patient_id,)
            ).fetchone()[0]
            if stays_count > 0:
                raise ValueError("Cannot delete patient with associated ICU stays. Archive the patient instead.")

            # Check clinical records
            try:
                records_count = connection.execute(
                    "SELECT COUNT(*) FROM clinical_records WHERE patient_id = ?", (patient_id,)
                ).fetchone()[0]
                if records_count > 0:
                    raise ValueError("Cannot delete patient with associated clinical records. Archive the patient instead.")
            except sqlite3.OperationalError:
                pass

            # Check telemetry
            telem_count = connection.execute(
                "SELECT COUNT(*) FROM telemetry_records WHERE patient_id = ?", (patient_id,)
            ).fetchone()[0]
            if telem_count > 0:
                raise ValueError("Cannot delete patient with associated telemetry records. Archive the patient instead.")

            connection.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
            return True

    # ------------------------------------------------------------
    # ICU STAY OPERATIONS
    # ------------------------------------------------------------

    def _next_stay_code(self, connection: sqlite3.Connection) -> str:
        cursor = connection.execute(
            "SELECT stay_code FROM icu_stays WHERE stay_code GLOB 'S[0-9]*'"
        )
        highest = 0
        for (code,) in cursor.fetchall():
            suffix = code[1:]
            if suffix.isdigit():
                highest = max(highest, int(suffix))
        return f"S{highest + 1:03d}"

    def create_stay(self, patient_ref: int | str, fields: IcuStayCreate) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        admission_time = _utc_iso(fields.admission_datetime) if fields.admission_datetime else now_iso

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            patient_id = self.resolve_patient_id(patient_ref, connection)
            if not patient_id:
                raise KeyError(f"Patient '{patient_ref}' not found.")

            patient_row = connection.execute(
                "SELECT status FROM patients WHERE id = ?", (patient_id,)
            ).fetchone()
            if patient_row["status"] == "archived":
                raise ValueError("Cannot admit an archived patient. Reactivate the patient first.")

            # Check for existing active stay
            active_existing = connection.execute(
                "SELECT id, stay_code FROM icu_stays WHERE patient_id = ? AND discharge_status = 'active'",
                (patient_id,),
            ).fetchone()
            if active_existing:
                raise ValueError(
                    f"Patient already has an active ICU admission ({active_existing['stay_code']}). "
                    "Discharge the active admission before creating a new stay."
                )

            stay_code = fields.stay_code
            if stay_code is None:
                stay_code = self._next_stay_code(connection)
            else:
                existing_code = connection.execute(
                    "SELECT id FROM icu_stays WHERE UPPER(stay_code) = UPPER(?)", (stay_code,)
                ).fetchone()
                if existing_code:
                    raise ValueError(f"Stay code '{stay_code}' already exists.")

            try:
                cursor = connection.execute(
                    """
                    INSERT INTO icu_stays (
                        stay_code, patient_id, admission_datetime, bed_number,
                        ward_or_unit, admission_reason, discharge_status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (
                        stay_code,
                        patient_id,
                        admission_time,
                        fields.bed_number,
                        fields.ward_or_unit,
                        fields.admission_reason,
                        now_iso,
                        now_iso,
                    ),
                )
            except sqlite3.IntegrityError as err:
                if "uq_active_patient_stay" in str(err) or "patient_id" in str(err):
                    raise ValueError("Patient already has an active ICU admission. Discharge the active admission before creating a new stay.") from err
                raise ValueError(f"Stay code '{stay_code}' already exists.") from err

            stay_id = cursor.lastrowid
            row = connection.execute(
                "SELECT * FROM icu_stays WHERE id = ?", (stay_id,)
            ).fetchone()
            return dict(row)

    def resolve_stay_id(self, stay_ref: int | str, connection: sqlite3.Connection | None = None) -> int | None:
        def _lookup(conn: sqlite3.Connection) -> int | None:
            try:
                if isinstance(stay_ref, int) or (isinstance(stay_ref, str) and stay_ref.isdigit()):
                    row = conn.execute("SELECT id FROM icu_stays WHERE id = ?", (int(stay_ref),)).fetchone()
                    if row:
                        return row["id"]
                row = conn.execute("SELECT id FROM icu_stays WHERE UPPER(stay_code) = UPPER(?)", (str(stay_ref),)).fetchone()
                return row["id"] if row else None
            except sqlite3.OperationalError:
                return None

        if connection:
            return _lookup(connection)
        with self._connection() as conn:
            return _lookup(conn)

    def get_stay(self, stay_ref: int | str) -> dict[str, Any] | None:
        try:
            with self._connection() as connection:
                stay_id = self.resolve_stay_id(stay_ref, connection)
                if not stay_id:
                    return None
                row = connection.execute(
                    """
                    SELECT s.*, p.patient_code, p.name as patient_name, p.status as patient_status
                    FROM icu_stays s
                    JOIN patients p ON s.patient_id = p.id
                    WHERE s.id = ?
                    """,
                    (stay_id,),
                ).fetchone()
                if not row:
                    return None
                stay = dict(row)

                # Check assigned device
                assign_row = connection.execute(
                    """
                    SELECT a.*, d.device_name, d.status as device_connection_status
                    FROM device_assignments a
                    JOIN devices d ON a.device_id = d.device_id
                    WHERE a.stay_id = ? AND a.assignment_status = 'active'
                    """,
                    (stay_id,),
                ).fetchone()
                stay["active_device"] = dict(assign_row) if assign_row else None

                # Count clinical records
                try:
                    rec_count = connection.execute(
                        "SELECT COUNT(*) FROM clinical_records WHERE stay_id = ?", (stay_id,)
                    ).fetchone()[0]
                except sqlite3.OperationalError:
                    rec_count = 0
                stay["clinical_records_count"] = rec_count

                # Count telemetry
                try:
                    telem_count = connection.execute(
                        "SELECT COUNT(*) FROM telemetry_records WHERE stay_id = ?", (stay_id,)
                    ).fetchone()[0]
                except sqlite3.OperationalError:
                    telem_count = 0
                stay["telemetry_records_count"] = telem_count

                return stay
        except sqlite3.OperationalError:
            return None

    def list_stays_for_patient(self, patient_ref: int | str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            patient_id = self.resolve_patient_id(patient_ref, connection)
            if not patient_id:
                raise KeyError(f"Patient '{patient_ref}' not found.")
            rows = connection.execute(
                """
                SELECT s.*,
                       a.device_id as active_device_id,
                       d.device_name as active_device_name
                FROM icu_stays s
                LEFT JOIN device_assignments a ON s.id = a.stay_id AND a.assignment_status = 'active'
                LEFT JOIN devices d ON a.device_id = d.device_id
                WHERE s.patient_id = ?
                ORDER BY s.admission_datetime DESC
                """,
                (patient_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def update_stay(self, stay_ref: int | str, fields: IcuStayUpdate) -> dict[str, Any] | None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            stay_id = self.resolve_stay_id(stay_ref, connection)
            if not stay_id:
                return None

            updates = ["updated_at = ?"]
            params: list[Any] = [now_iso]

            if fields.bed_number is not None:
                updates.append("bed_number = ?")
                params.append(fields.bed_number)
            if fields.ward_or_unit is not None:
                updates.append("ward_or_unit = ?")
                params.append(fields.ward_or_unit)
            if fields.admission_reason is not None:
                updates.append("admission_reason = ?")
                params.append(fields.admission_reason)

            params.append(stay_id)
            connection.execute(
                f"UPDATE icu_stays SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            row = connection.execute("SELECT * FROM icu_stays WHERE id = ?", (stay_id,)).fetchone()
            return dict(row) if row else None

    def discharge_stay(
        self,
        stay_ref: int | str,
        discharge_datetime: datetime | str | None = None,
        discharge_status: str = "discharged",
    ) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        disch_time = _utc_iso(discharge_datetime) if discharge_datetime else now_iso

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            stay_id = self.resolve_stay_id(stay_ref, connection)
            if not stay_id:
                raise KeyError(f"ICU stay '{stay_ref}' not found.")

            row = connection.execute("SELECT * FROM icu_stays WHERE id = ?", (stay_id,)).fetchone()
            if not row:
                raise KeyError(f"ICU stay '{stay_ref}' not found.")
            if row["discharge_status"] != "active":
                raise ValueError(f"ICU stay is already discharged (status: {row['discharge_status']}).")

            # Update stay
            connection.execute(
                """
                UPDATE icu_stays
                SET discharge_status = ?, discharge_datetime = ?, updated_at = ?
                WHERE id = ?
                """,
                (discharge_status, disch_time, now_iso, stay_id),
            )

            # Automatically end any active device assignment
            connection.execute(
                """
                UPDATE device_assignments
                SET assignment_status = 'ended', unassigned_at = ?, updated_at = ?
                WHERE stay_id = ? AND assignment_status = 'active'
                """,
                (disch_time, now_iso, stay_id),
            )

            updated = connection.execute("SELECT * FROM icu_stays WHERE id = ?", (stay_id,)).fetchone()
            return dict(updated)

    # ------------------------------------------------------------
    # DEVICE OPERATIONS
    # ------------------------------------------------------------

    def register_device(self, fields: DeviceCreate) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT * FROM devices WHERE device_id = ?", (fields.device_id,)
            ).fetchone()
            if existing:
                raise ValueError(f"Device '{fields.device_id}' is already registered.")

            try:
                connection.execute(
                    """
                    INSERT INTO devices (
                        device_id, device_name, connection_type, port, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'disconnected', ?, ?)
                    """,
                    (
                        fields.device_id,
                        fields.device_name,
                        fields.connection_type,
                        fields.port,
                        now_iso,
                        now_iso,
                    ),
                )
            except sqlite3.IntegrityError as err:
                raise ValueError(f"Device '{fields.device_id}' is already registered.") from err
            row = connection.execute(
                "SELECT * FROM devices WHERE device_id = ?", (fields.device_id,)
            ).fetchone()
            return dict(row)

    def set_device_status(self, device_id: str, status: str, port: str | None = None) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            # Ensure device exists
            connection.execute(
                """
                INSERT OR IGNORE INTO devices (
                    device_id, device_name, connection_type, port, status, created_at, updated_at
                ) VALUES (?, ?, 'serial', ?, 'disconnected', ?, ?)
                """,
                (device_id, f"ICU Device {device_id}", port, now_iso, now_iso),
            )
            updates = ["status = ?", "updated_at = ?"]
            params: list[Any] = [status, now_iso]
            if port:
                updates.append("port = ?")
                params.append(port)
            params.append(device_id)
            connection.execute(f"UPDATE devices SET {', '.join(updates)} WHERE device_id = ?", params)

    def list_devices(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT d.*,
                       a.id as assignment_id,
                       a.patient_id,
                       a.stay_id,
                       a.assigned_at,
                       p.patient_code,
                       p.name as patient_name,
                       s.stay_code,
                       s.bed_number,
                       s.discharge_status as stay_status
                FROM devices d
                LEFT JOIN device_assignments a ON d.device_id = a.device_id AND a.assignment_status = 'active'
                LEFT JOIN patients p ON a.patient_id = p.id
                LEFT JOIN icu_stays s ON a.stay_id = s.id
                ORDER BY d.device_id ASC
                """
            ).fetchall()

            devices = []
            for r in rows:
                d = dict(r)
                if d.get("assignment_id"):
                    d["active_assignment"] = {
                        "id": d.pop("assignment_id"),
                        "patient_id": d.pop("patient_id"),
                        "stay_id": d.pop("stay_id"),
                        "assigned_at": d.pop("assigned_at"),
                        "patient_code": d.pop("patient_code"),
                        "patient_name": d.pop("patient_name"),
                        "stay_code": d.pop("stay_code"),
                        "bed_number": d.pop("bed_number"),
                        "stay_status": d.pop("stay_status"),
                    }
                else:
                    d["active_assignment"] = None
                    d.pop("assignment_id", None)
                    d.pop("patient_id", None)
                    d.pop("stay_id", None)
                    d.pop("assigned_at", None)
                    d.pop("patient_code", None)
                    d.pop("patient_name", None)
                    d.pop("stay_code", None)
                    d.pop("bed_number", None)
                    d.pop("stay_status", None)

                # Fetch last telemetry
                telem_row = connection.execute(
                    """
                    SELECT reading_time, fluid_ml, bpm, spo2, device_status, connection_status, quality_metadata
                    FROM telemetry_records
                    WHERE device_id = ?
                    ORDER BY reading_time DESC
                    LIMIT 1
                    """,
                    (d["device_id"],),
                ).fetchone()
                d["last_telemetry"] = (
                    self._mask_invalid_telemetry(dict(telem_row))
                    if telem_row
                    else None
                )

                devices.append(d)
            return devices

    def get_device(self, device_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT d.*,
                       a.id as assignment_id,
                       a.patient_id,
                       a.stay_id,
                       a.assigned_at,
                       p.patient_code,
                       p.name as patient_name,
                       s.stay_code,
                       s.bed_number,
                       s.discharge_status as stay_status
                FROM devices d
                LEFT JOIN device_assignments a ON d.device_id = a.device_id AND a.assignment_status = 'active'
                LEFT JOIN patients p ON a.patient_id = p.id
                LEFT JOIN icu_stays s ON a.stay_id = s.id
                WHERE d.device_id = ?
                """,
                (device_id,),
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            if d.get("assignment_id"):
                d["active_assignment"] = {
                    "id": d.pop("assignment_id"),
                    "patient_id": d.pop("patient_id"),
                    "stay_id": d.pop("stay_id"),
                    "assigned_at": d.pop("assigned_at"),
                    "patient_code": d.pop("patient_code"),
                    "patient_name": d.pop("patient_name"),
                    "stay_code": d.pop("stay_code"),
                    "bed_number": d.pop("bed_number"),
                    "stay_status": d.pop("stay_status"),
                }
            else:
                d["active_assignment"] = None
                d.pop("assignment_id", None)
                d.pop("patient_id", None)
                d.pop("stay_id", None)
                d.pop("assigned_at", None)
                d.pop("patient_code", None)
                d.pop("patient_name", None)
                d.pop("stay_code", None)
                d.pop("bed_number", None)
                d.pop("stay_status", None)

            telem_row = connection.execute(
                """
                SELECT * FROM telemetry_records
                WHERE device_id = ?
                ORDER BY reading_time DESC
                LIMIT 1
                """,
                (device_id,),
            ).fetchone()
            if telem_row:
                d["last_telemetry"] = self._mask_invalid_telemetry(dict(telem_row))
            else:
                d["last_telemetry"] = None

            return d

    # ------------------------------------------------------------
    # DEVICE ASSIGNMENT OPERATIONS
    # ------------------------------------------------------------

    def assign_device(
        self,
        device_id: str,
        patient_ref: int | str,
        stay_ref: int | str,
    ) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")

            # Validate device exists
            dev_row = connection.execute(
                "SELECT * FROM devices WHERE device_id = ?", (device_id,)
            ).fetchone()
            if not dev_row:
                raise KeyError(f"Device '{device_id}' is not registered.")

            # Validate patient exists and is active
            patient_id = self.resolve_patient_id(patient_ref, connection)
            if not patient_id:
                raise KeyError(f"Patient '{patient_ref}' not found.")
            patient_row = connection.execute(
                "SELECT * FROM patients WHERE id = ?", (patient_id,)
            ).fetchone()
            if patient_row["status"] == "archived":
                raise ValueError("Cannot assign device to an archived patient.")

            # Validate stay exists and belongs to this patient
            stay_id = self.resolve_stay_id(stay_ref, connection)
            if not stay_id:
                raise KeyError(f"ICU stay '{stay_ref}' not found.")
            stay_row = connection.execute(
                "SELECT * FROM icu_stays WHERE id = ?", (stay_id,)
            ).fetchone()
            if stay_row["patient_id"] != patient_id:
                raise ValueError("ICU stay does not belong to the selected patient.")
            if stay_row["discharge_status"] != "active":
                raise ValueError("Cannot assign device to a discharged or inactive ICU stay.")

            # Close previous active assignment for this device AND this stay
            connection.execute(
                """
                UPDATE device_assignments
                SET assignment_status = 'ended', unassigned_at = ?, updated_at = ?
                WHERE (device_id = ? OR stay_id = ?) AND assignment_status = 'active'
                """,
                (now_iso, now_iso, device_id, stay_id),
            )

            # Insert new active assignment
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO device_assignments (
                        device_id, patient_id, stay_id, assigned_at, assignment_status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (device_id, patient_id, stay_id, now_iso, now_iso, now_iso),
                )
            except sqlite3.IntegrityError as err:
                raise ValueError(f"Cannot assign device '{device_id}': active assignment constraint violation ({err}).") from err
            assignment_id = cursor.lastrowid

            # Update device status
            connection.execute(
                "UPDATE devices SET updated_at = ? WHERE device_id = ?",
                (now_iso, device_id),
            )

            row = connection.execute(
                """
                SELECT a.*, p.patient_code, p.name as patient_name, s.stay_code, s.bed_number
                FROM device_assignments a
                JOIN patients p ON a.patient_id = p.id
                JOIN icu_stays s ON a.stay_id = s.id
                WHERE a.id = ?
                """,
                (assignment_id,),
            ).fetchone()
            return dict(row)

    def unassign_device(self, device_id: str, reason: str | None = None) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT a.*, p.patient_code, p.name as patient_name, s.stay_code
                FROM device_assignments a
                JOIN patients p ON a.patient_id = p.id
                JOIN icu_stays s ON a.stay_id = s.id
                WHERE a.device_id = ? AND a.assignment_status = 'active'
                """,
                (device_id,),
            ).fetchone()
            if not active:
                raise KeyError(f"Device '{device_id}' does not have an active patient assignment.")

            connection.execute(
                """
                UPDATE device_assignments
                SET assignment_status = 'ended', unassigned_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now_iso, now_iso, active["id"]),
            )
            connection.execute(
                "UPDATE devices SET updated_at = ? WHERE device_id = ?",
                (now_iso, device_id),
            )

            result = dict(active)
            result["assignment_status"] = "ended"
            result["unassigned_at"] = now_iso
            return result

    def get_active_assignment_for_device(self, device_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT a.*, p.patient_code, p.name as patient_name, s.stay_code, s.bed_number
                FROM device_assignments a
                JOIN patients p ON a.patient_id = p.id
                JOIN icu_stays s ON a.stay_id = s.id
                WHERE a.device_id = ? AND a.assignment_status = 'active'
                """,
                (device_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_device_assignments(self, device_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT a.*, p.patient_code, p.name as patient_name, s.stay_code, s.bed_number
                FROM device_assignments a
                JOIN patients p ON a.patient_id = p.id
                JOIN icu_stays s ON a.stay_id = s.id
                WHERE a.device_id = ?
                ORDER BY a.assigned_at DESC
                """,
                (device_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------
    # TELEMETRY OPERATIONS & INGESTION
    # ------------------------------------------------------------

    def ingest_reading(self, device_id: str, reading: dict[str, Any]) -> dict[str, Any]:
        """
        Ingest an incoming telemetry reading for a device.
        Resolves active assignment strictly from backend database state.
        Never attributes readings to a patient unless an active assignment exists,
        the assigned stay is active, and the reading timestamp matches the assignment window.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        reading_time = _utc_iso(reading.get("reading_time")) if reading.get("reading_time") else now_iso

        fluid_ml = reading.get("fluid_ml")
        bpm = reading.get("bpm")
        spo2 = reading.get("spo2")
        device_status = reading.get("device_status") or "NORMAL"
        connection_status = reading.get("connection_status") or "CONNECTED"
        validity = self._telemetry_validity(reading)

        quality_metadata_json = json.dumps(validity)

        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")

            # Ensure device exists and update its status
            connection.execute(
                """
                INSERT OR IGNORE INTO devices (
                    device_id, device_name, connection_type, port, status, created_at, updated_at
                ) VALUES (?, ?, 'serial', NULL, ?, ?, ?)
                """,
                (device_id, f"ICU Device {device_id}", connection_status.lower(), now_iso, now_iso),
            )
            connection.execute(
                "UPDATE devices SET status = ?, updated_at = ? WHERE device_id = ?",
                (connection_status.lower(), now_iso, device_id),
            )

            # Check active assignment for this device
            assignment = connection.execute(
                """
                SELECT a.id, a.patient_id, a.stay_id, a.assigned_at,
                       s.discharge_status, s.patient_id as stay_patient_id
                FROM device_assignments a
                JOIN icu_stays s ON a.stay_id = s.id
                WHERE a.device_id = ? AND a.assignment_status = 'active'
                """,
                (device_id,),
            ).fetchone()

            patient_id: int | None = None
            stay_id: int | None = None
            assignment_id: int | None = None

            if assignment:
                # Validate stay is active and matches patient
                if (
                    assignment["discharge_status"] == "active"
                    and assignment["stay_patient_id"] == assignment["patient_id"]
                ):
                    # Check reading time vs assignment start: delayed readings before assignment are not attributed
                    if reading_time >= assignment["assigned_at"]:
                        patient_id = assignment["patient_id"]
                        stay_id = assignment["stay_id"]
                        assignment_id = assignment["id"]

            cursor = connection.execute(
                """
                INSERT INTO telemetry_records (
                    device_id, patient_id, stay_id, assignment_id,
                    reading_time, ingestion_time, fluid_ml, bpm, spo2,
                    device_status, connection_status, quality_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    device_id,
                    patient_id,
                    stay_id,
                    assignment_id,
                    reading_time,
                    now_iso,
                    fluid_ml,
                    bpm,
                    spo2,
                    device_status,
                    connection_status,
                    quality_metadata_json,
                ),
            )
            telem_id = cursor.lastrowid
            row = connection.execute(
                "SELECT * FROM telemetry_records WHERE id = ?", (telem_id,)
            ).fetchone()
            res = dict(row)
            res["quality_metadata"] = validity
            return self._mask_invalid_telemetry(res)

    def get_patient_telemetry(
        self,
        patient_ref: int | str,
        limit: int = 100,
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        with self._connection() as connection:
            patient_id = self.resolve_patient_id(patient_ref, connection)
            if not patient_id:
                raise KeyError(f"Patient '{patient_ref}' not found.")

            conditions = ["patient_id = ?"]
            params: list[Any] = [patient_id]

            if start_time:
                conditions.append("reading_time >= ?")
                params.append(_utc_iso(start_time))
            if end_time:
                conditions.append("reading_time <= ?")
                params.append(_utc_iso(end_time))

            query = f"""
                SELECT * FROM telemetry_records
                WHERE {' AND '.join(conditions)}
                ORDER BY reading_time DESC
                LIMIT ?
            """
            rows = connection.execute(query, [*params, limit]).fetchall()
            results = []
            for r in rows:
                results.append(self._mask_invalid_telemetry(dict(r)))
            return results

    def get_stay_telemetry(self, stay_ref: int | str, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as connection:
            stay_id = self.resolve_stay_id(stay_ref, connection)
            if not stay_id:
                raise KeyError(f"ICU stay '{stay_ref}' not found.")

            rows = connection.execute(
                """
                SELECT * FROM telemetry_records
                WHERE stay_id = ?
                ORDER BY reading_time DESC
                LIMIT ?
                """,
                (stay_id, limit),
            ).fetchall()
            results = []
            for r in rows:
                results.append(self._mask_invalid_telemetry(dict(r)))
            return results

    # ------------------------------------------------------------
    # PATIENT TIMELINE (AUDITABLE CHRONOLOGICAL HISTORY)
    # ------------------------------------------------------------

    def get_patient_timeline(self, patient_ref: int | str) -> dict[str, Any]:
        with self._connection() as connection:
            patient_id = self.resolve_patient_id(patient_ref, connection)
            if not patient_id:
                raise KeyError(f"Patient '{patient_ref}' not found.")

            patient_row = connection.execute(
                "SELECT id, patient_code, name FROM patients WHERE id = ?", (patient_id,)
            ).fetchone()

            events: list[dict[str, Any]] = []

            # 1. Admissions & Discharges
            stays = connection.execute(
                "SELECT * FROM icu_stays WHERE patient_id = ? ORDER BY admission_datetime ASC",
                (patient_id,),
            ).fetchall()
            for s in stays:
                events.append({
                    "event_type": "admission",
                    "timestamp": s["admission_datetime"],
                    "summary": f"Admitted to ICU ({s['stay_code']}) - Bed {s['bed_number'] or 'N/A'}",
                    "details": dict(s),
                    "source": "ICU Stay",
                })
                if s["discharge_datetime"]:
                    events.append({
                        "event_type": "discharge",
                        "timestamp": s["discharge_datetime"],
                        "summary": f"Discharged from ICU ({s['stay_code']}) - Status: {s['discharge_status']}",
                        "details": dict(s),
                        "source": "ICU Stay",
                    })

            # 2. Device Assignments & Unassignments
            assignments = connection.execute(
                "SELECT * FROM device_assignments WHERE patient_id = ? ORDER BY assigned_at ASC",
                (patient_id,),
            ).fetchall()
            for a in assignments:
                events.append({
                    "event_type": "device_assignment",
                    "timestamp": a["assigned_at"],
                    "summary": f"Assigned to monitoring device {a['device_id']}",
                    "details": dict(a),
                    "source": "Hardware",
                })
                if a["unassigned_at"]:
                    events.append({
                        "event_type": "device_unassignment",
                        "timestamp": a["unassigned_at"],
                        "summary": f"Unassigned from monitoring device {a['device_id']}",
                        "details": dict(a),
                        "source": "Hardware",
                    })

            # 3. Clinical Records & Revisions & Voids
            try:
                records = connection.execute(
                    """
                    SELECT r.*, s.stay_code
                    FROM clinical_records r
                    LEFT JOIN icu_stays s ON r.stay_id = s.id
                    WHERE r.patient_id = ? OR (r.stay_id IS NOT NULL AND r.stay_id IN (SELECT id FROM icu_stays WHERE patient_id = ?))
                    ORDER BY r.measurement_time ASC
                    """,
                    (patient_id, patient_id),
                ).fetchall()
                for r in records:
                    rec_dict = dict(r)
                    if rec_dict.get("value_json"):
                        try:
                            rec_dict["value"] = json.loads(rec_dict.pop("value_json"))
                        except Exception:
                            pass
                    val_str = f"{rec_dict.get('value')} {rec_dict.get('unit') or ''}".strip()
                    events.append({
                        "event_type": "clinical_record",
                        "timestamp": rec_dict["measurement_time"],
                        "summary": f"{rec_dict['category'].replace('_', ' ').title()}: {rec_dict['measurement_name']} ({val_str})",
                        "details": rec_dict,
                        "source": "Clinical Records",
                    })
                    if rec_dict.get("voided_at"):
                        events.append({
                            "event_type": "record_void",
                            "timestamp": rec_dict["voided_at"],
                            "summary": f"Voided record '{rec_dict['measurement_name']}': {rec_dict.get('void_reason')}",
                            "details": rec_dict,
                            "source": "Clinical Records",
                        })

                # Revisions
                rev_rows = connection.execute(
                    """
                    SELECT rev.*, r.measurement_name
                    FROM clinical_record_revisions rev
                    JOIN clinical_records r ON rev.record_id = r.record_id
                    WHERE r.patient_id = ? OR (r.stay_id IS NOT NULL AND r.stay_id IN (SELECT id FROM icu_stays WHERE patient_id = ?))
                    ORDER BY rev.revised_at ASC
                    """,
                    (patient_id, patient_id),
                ).fetchall()
                for rev in rev_rows:
                    rev_dict = dict(rev)
                    if rev_dict.get("previous_record_json"):
                        try:
                            rev_dict["previous_record"] = json.loads(rev_dict.pop("previous_record_json"))
                        except Exception:
                            pass
                    events.append({
                        "event_type": "record_correction",
                        "timestamp": rev_dict["revised_at"],
                        "summary": f"Correction to '{rev_dict['measurement_name']}': {rev_dict.get('change_reason')}",
                        "details": rev_dict,
                        "source": "Clinical Records",
                    })
            except sqlite3.OperationalError:
                pass

            # 4. Telemetry sample events (hourly or distinct milestones)
            telem_rows = connection.execute(
                """
                SELECT * FROM telemetry_records
                WHERE patient_id = ?
                ORDER BY reading_time DESC
                LIMIT 50
                """,
                (patient_id,),
            ).fetchall()
            for t in telem_rows:
                td = self._mask_invalid_telemetry(dict(t))
                events.append({
                    "event_type": "telemetry",
                    "timestamp": td["reading_time"],
                    "summary": f"Telemetry ({td['device_id']}): Fluid {td['fluid_ml']} mL, BPM {td['bpm']}, SpO2 {td['spo2']}%",
                    "details": td,
                    "source": "ESP32 Telemetry",
                })

            # Sort all events chronologically (newest first)
            events.sort(key=lambda e: e["timestamp"], reverse=True)

            return {
                "patient_id": patient_row["id"],
                "patient_code": patient_row["patient_code"],
                "name": patient_row["name"],
                "events_count": len(events),
                "events": events,
            }
