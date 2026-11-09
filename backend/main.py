from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
import os
from typing import Literal
from uuid import UUID
from typing import Any, Optional
from services.serial_service import serial_monitor
from services.clinical_alert_engine import clinical_alert_engine, ClinicalInterpretation
from services.alert_manager import AlertManager, AlertAcknowledgeRequest, AlertResolveRequest
from services.ml_service import ml_service
from services.clinical_records import (
    ClinicalCategory,
    ClinicalRecordConflict,
    ClinicalRecordCreate,
    ClinicalRecordDatabaseBusy,
    ClinicalRecordStore,
    ClinicalRecordUpdate,
    ClinicalRecordVoid,
)
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

from services.clinical_data_intelligence import ClinicalDataIntelligenceCoordinator
from services.data_consistency_engine import data_consistency_engine
from services.patient_trend_analyzer import patient_trend_analyzer
from services.historical_reference_engine import historical_reference_engine

import pandas as pd
from pydantic import BaseModel, Field
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "smart_icu_final_alerts.csv"
configured_clinical_db_path = os.getenv("SMART_ICU_CLINICAL_DB_PATH", "").strip()
CLINICAL_DB_FILE = (
    Path(configured_clinical_db_path)
    if configured_clinical_db_path
    else ROOT / "data" / "clinical_records.sqlite3"
)
if not CLINICAL_DB_FILE.is_absolute():
    CLINICAL_DB_FILE = ROOT / CLINICAL_DB_FILE
clinical_record_store = ClinicalRecordStore(CLINICAL_DB_FILE)
patient_store = PatientStore(CLINICAL_DB_FILE)
alert_manager = AlertManager(CLINICAL_DB_FILE)
clinical_intelligence = ClinicalDataIntelligenceCoordinator(
    database_path=CLINICAL_DB_FILE,
    consistency_engine=data_consistency_engine,
    trend_analyzer=patient_trend_analyzer,
    reference_engine=historical_reference_engine,
    alert_manager=alert_manager,
)


# ============================================================
# APP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global df
    print("Loading Smart-ICU historical dataset...")
    df = load_dataset()
    print(f"Loaded {len(df):,} patient-hour rows from {df['patientunitstayid'].nunique():,} ICU stays.")
    patient_store.initialize()
    clinical_record_store.initialize()
    alert_manager.initialize()
    clinical_intelligence.initialize()
    serial_monitor.set_on_reading(patient_store.ingest_reading)
    serial_monitor.start()
    try:
        yield
    finally:
        serial_monitor.stop()

app = FastAPI(
    title="Smart ICU Patient Safety System",
    description="Research prototype: historical eICU analytics and live ESP32 telemetry",
    version="0.4.0",
    lifespan=lifespan
)


@app.exception_handler(ClinicalRecordDatabaseBusy)
async def clinical_record_database_busy(
    request: Request,
    error: ClinicalRecordDatabaseBusy,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": str(error)},
        headers={"Retry-After": "1"},
    )


cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:80",
    "http://127.0.0.1:80",
    "http://localhost",
    "http://127.0.0.1",
]
env_cors = os.getenv("SMART_ICU_CORS_ORIGINS", "").strip()
if env_cors:
    for origin in env_cors.split(","):
        cleaned = origin.strip()
        if cleaned and cleaned not in cors_origins:
            cors_origins.append(cleaned)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATASET LOADING
# ============================================================

df = None


def load_dataset() -> pd.DataFrame:
    """
    Load the final Smart-ICU alert dataset.
    The dataset is loaded once when the backend starts.
    """

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Smart-ICU dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(
        DATA_FILE,
        low_memory=False
    )

    return data


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):

    if pd.isna(value):
        return None

    if isinstance(value, float):
        return round(value, 4)

    return value


def row_to_dict(row: pd.Series) -> dict:

    result = {}

    for key, value in row.items():
        result[key] = clean_value(value)

    return result


def require_patient(
    patientunitstayid: int
) -> pd.DataFrame:

    if df is None:
        raise HTTPException(
            status_code=503,
            detail="Historical patient dataset is not loaded.",
        )

    patient_df = df[
        df["patientunitstayid"]
        == patientunitstayid
    ].copy()

    if patient_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"ICU patient stay "
                f"{patientunitstayid} not found."
            )
        )

    return patient_df


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "system": "Smart ICU Patient Safety System",
        "status": "online",
        "version": app.version
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "dataset_loaded": df is not None,
        "rows": len(df) if df is not None else 0,
        "hardware": serial_monitor.snapshot()
    }


# ============================================================
# LIVE HARDWARE TELEMETRY
# ============================================================

@app.get("/api/hardware/telemetry")
def get_hardware_telemetry():
    """Latest parsed ESP32 reading with active assignment information."""
    snapshot = serial_monitor.snapshot()
    device_id = getattr(serial_monitor, "device_id", "ESP32-001")
    assignment = patient_store.get_active_assignment_for_device(device_id)
    if assignment:
        snapshot["assigned"] = True
        snapshot["assignment"] = assignment
    else:
        snapshot["assigned"] = False
        snapshot["assignment"] = None
    return snapshot


# ============================================================
# MANUALLY ENTERED CLINICAL RECORDS
# ============================================================

def require_existing_stay(stay_ref: int | str) -> None:
    stay = patient_store.get_stay(stay_ref)
    if stay is not None:
        return
    if df is None:
        raise HTTPException(
            status_code=503,
            detail="Historical patient dataset is not loaded.",
        )
    try:
        stay_id_int = int(stay_ref)
        if (df["patientunitstayid"] == stay_id_int).any():
            return
    except (ValueError, TypeError):
        pass
    raise HTTPException(
        status_code=404,
        detail=f"ICU patient stay {stay_ref} not found.",
    )


def validate_clinical_date_range(
    start_time: datetime | None,
    end_time: datetime | None,
) -> None:
    for label, value in (("start_time", start_time), ("end_time", end_time)):
        if value is not None and value.utcoffset() is None:
            raise HTTPException(
                status_code=422,
                detail=f"{label} must include a timezone.",
            )
    if start_time and end_time and start_time > end_time:
        raise HTTPException(
            status_code=422,
            detail="start_time cannot be after end_time.",
        )


@app.get("/api/clinical-records/patient-stays")
def get_clinical_record_patient_stays(
    query: str = Query(default="", max_length=12),
    limit: int = Query(default=25, ge=1, le=100),
):
    if df is None:
        raise HTTPException(
            status_code=503,
            detail="Historical patient dataset is not loaded.",
        )
    if query and not query.isdecimal():
        raise HTTPException(
            status_code=400,
            detail="Patient stay search must contain digits only.",
        )

    patient_ids = (
        pd.to_numeric(df["patientunitstayid"], errors="coerce")
        .dropna()
        .astype("int64")
        .drop_duplicates()
        .sort_values()
    )
    if query:
        patient_ids = patient_ids[
            patient_ids.astype(str).str.startswith(query)
        ]
    return {"patient_ids": patient_ids.head(limit).astype(int).tolist()}


def evaluate_record_for_alerts(
    record_dict: dict,
    record_obj: Any,
    patient_id: int | None,
    stay_id: int | None,
    patientunitstayid: int | None,
):
    try:
        weight_kg = None
        patient_sex = None
        patient_age = None
        if patient_id:
            pat = patient_store.get_patient(patient_id)
            if pat:
                weight_kg = pat.get("weight_kg")
                patient_sex = pat.get("sex")
                patient_age = pat.get("age")

        cat = getattr(record_obj, "category", record_dict.get("category"))
        meas_name = getattr(record_obj, "measurement_name", record_dict.get("measurement_name"))
        val = getattr(record_obj, "value", record_dict.get("value"))
        unit = getattr(record_obj, "unit", record_dict.get("unit"))
        obs_period = getattr(record_obj, "observation_period", record_dict.get("observation_period"))
        ref_range = getattr(record_obj, "reference_range", record_dict.get("reference_range"))
        meas_time = getattr(record_obj, "measurement_time", None)
        sample_time = getattr(record_obj, "sample_collection_time", None)

        if cat == "urine_output":
            interp = clinical_alert_engine.evaluate_urine_output(
                volume_ml=float(val) if isinstance(val, (int, float)) else 0.0,
                unit=unit or "mL",
                patient_weight_kg=weight_kg,
                observation_period=obs_period,
                start_time=sample_time,
                end_time=meas_time,
            )
            source = "CLINICAL_OBSERVATION"
        elif cat in ("blood_test", "other_lab"):
            interp = clinical_alert_engine.evaluate_lab_result(
                measurement_name=meas_name,
                value=val,
                unit=unit,
                reference_range_str=ref_range,
                patient_sex=patient_sex,
                patient_age=patient_age,
            )
            source = "LAB_RESULT"
        else:
            interp = clinical_alert_engine.evaluate_clinical_observation(
                measurement_name=meas_name,
                value=val,
                unit=unit,
            )
            source = "CLINICAL_OBSERVATION"

        trig_iso = meas_time.isoformat() if hasattr(meas_time, "isoformat") else str(meas_time or "")
        alert = alert_manager.process_clinical_interpretation(
            interpretation=interp,
            patient_id=patient_id,
            stay_id=stay_id,
            patientunitstayid=patientunitstayid,
            source=source,
            record_id=record_dict.get("record_id"),
            trigger_time=trig_iso or None,
        )
        return interp, alert
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception("Error evaluating clinical record for alerts: %s", exc)
        return None, None


@app.post("/api/clinical-records", status_code=201)
def create_clinical_record(record: ClinicalRecordCreate):
    if record.stay_id is not None and record.patientunitstayid is not None:
        raise HTTPException(
            status_code=400,
            detail="Cannot provide both historical patientunitstayid and registered stay_id.",
        )

    if record.stay_id is not None:
        stay = patient_store.get_stay(record.stay_id)
        if not stay:
            raise HTTPException(status_code=404, detail=f"ICU stay '{record.stay_id}' not found.")
        if stay["discharge_status"] != "active":
            raise HTTPException(status_code=400, detail="Cannot add clinical records to a discharged ICU stay.")
        if record.patient_id is not None:
            pat = patient_store.get_patient(record.patient_id)
            if not pat:
                raise HTTPException(status_code=404, detail=f"Patient '{record.patient_id}' not found.")
            if stay["patient_id"] != record.patient_id:
                raise HTTPException(status_code=400, detail="ICU stay does not belong to the specified patient.")
        else:
            record.patient_id = stay["patient_id"]
        # Strictly ensure historical identifier is None for registered admission records
        record.patientunitstayid = None
    elif record.patientunitstayid is not None:
        if record.patient_id is not None or record.stay_id is not None:
            raise HTTPException(
                status_code=400,
                detail="Historical clinical records cannot specify registered patient_id or stay_id.",
            )
        require_existing_stay(record.patientunitstayid)
        record.stay_id = None
        record.patient_id = None
    else:
        raise HTTPException(status_code=422, detail="Either stay_id or patientunitstayid is required.")

    try:
        created = clinical_record_store.create(record)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    interp, alert = evaluate_record_for_alerts(
        record_dict=created,
        record_obj=record,
        patient_id=record.patient_id,
        stay_id=record.stay_id,
        patientunitstayid=record.patientunitstayid,
    )
    if interp:
        created["interpretation"] = interp.model_dump()
    if alert:
        created["alert"] = alert

    if record.patient_id is not None:
        try:
            clinical_intelligence.evaluate_patient(record.patient_id)
        except Exception:
            pass

    return created


@app.get("/api/patients/{patient_ref}/clinical-records")
def get_patient_clinical_records(
    patient_ref: int | str,
    source: Literal["registered", "historical"] | None = Query(default=None),
    category: ClinicalCategory | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    include_voided: bool = False,
    stay_id: int | None = None,
):
    validate_clinical_date_range(start_time, end_time)
    ref_str = str(patient_ref).strip()

    if source == "registered":
        registered_patient = patient_store.get_patient(patient_ref)
        if not registered_patient:
            raise HTTPException(
                status_code=404,
                detail=f"Registered patient '{patient_ref}' not found.",
            )
        return {
            "patient_id": registered_patient["id"],
            "patient_code": registered_patient["patient_code"],
            "records": clinical_record_store.list_for_patient_id(
                registered_patient["id"],
                category=category,
                start_time=start_time,
                end_time=end_time,
                include_voided=include_voided,
                stay_id=stay_id,
            ),
        }

    if source == "historical":
        try:
            stay_id_int = int(patient_ref)
        except (ValueError, TypeError) as error:
            raise HTTPException(
                status_code=404,
                detail=f"Historical ICU stay '{patient_ref}' not found.",
            ) from error
        if df is None or not (df["patientunitstayid"] == stay_id_int).any():
            raise HTTPException(
                status_code=404,
                detail=f"Historical ICU stay '{patient_ref}' not found.",
            )
        return {
            "patientunitstayid": stay_id_int,
            "records": clinical_record_store.list_for_patient(
                stay_id_int,
                category=category,
                start_time=start_time,
                end_time=end_time,
                include_voided=include_voided,
            ),
        }

    # If source is None:
    if not ref_str.isdigit() and not (ref_str.startswith("-") and ref_str[1:].isdigit()):
        # Alphanumeric code (e.g. "P001") -> unambiguously registered patient
        registered_patient = patient_store.get_patient(patient_ref)
        if registered_patient:
            return {
                "patient_id": registered_patient["id"],
                "patient_code": registered_patient["patient_code"],
                "records": clinical_record_store.list_for_patient_id(
                    registered_patient["id"],
                    category=category,
                    start_time=start_time,
                    end_time=end_time,
                    include_voided=include_voided,
                    stay_id=stay_id,
                ),
            }
        raise HTTPException(status_code=404, detail=f"Registered patient '{patient_ref}' not found.")

    # Numeric identifier without explicit source: check ambiguity
    num_id = int(ref_str)
    reg_patient = patient_store.get_patient(num_id)
    has_historical = df is not None and (df["patientunitstayid"] == num_id).any()

    if reg_patient is not None and has_historical:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Identifier '{patient_ref}' is ambiguous: exists as both registered patient "
                f"'{reg_patient['patient_code']}' (ID {reg_patient['id']}) and historical eICU stay. "
                "Specify query parameter '?source=registered' or '?source=historical'."
            ),
        )
    if reg_patient is not None:
        return {
            "patient_id": reg_patient["id"],
            "patient_code": reg_patient["patient_code"],
            "records": clinical_record_store.list_for_patient_id(
                reg_patient["id"],
                category=category,
                start_time=start_time,
                end_time=end_time,
                include_voided=include_voided,
                stay_id=stay_id,
            ),
        }
    if has_historical:
        return {
            "patientunitstayid": num_id,
            "records": clinical_record_store.list_for_patient(
                num_id,
                category=category,
                start_time=start_time,
                end_time=end_time,
                include_voided=include_voided,
            ),
        }

    raise HTTPException(status_code=404, detail=f"Patient or historical stay '{patient_ref}' not found.")


@app.get("/api/stays/{stay_ref}/clinical-records")
def get_stay_clinical_records(
    stay_ref: str,
    source: Literal["registered", "historical"] | None = Query(default=None),
    category: ClinicalCategory | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    include_voided: bool = False,
):
    validate_clinical_date_range(start_time, end_time)
    ref_str = str(stay_ref).strip()

    if source == "registered":
        stay = patient_store.get_stay(stay_ref)
        if not stay:
            raise HTTPException(status_code=404, detail=f"ICU stay '{stay_ref}' not found.")
        records = clinical_record_store.list_for_stay_id(
            stay["id"],
            category=category,
            start_time=start_time,
            end_time=end_time,
            include_voided=include_voided,
        )
        return {
            "stay_id": stay["id"],
            "stay_code": stay["stay_code"],
            "patient_id": stay["patient_id"],
            "records": records,
        }

    if source == "historical":
        try:
            stay_id_int = int(stay_ref)
        except ValueError as error:
            raise HTTPException(
                status_code=404,
                detail=f"Historical ICU stay '{stay_ref}' not found.",
            ) from error
        if df is None or not (df["patientunitstayid"] == stay_id_int).any():
            raise HTTPException(status_code=404, detail=f"Historical ICU stay '{stay_ref}' not found.")
        return {
            "patientunitstayid": stay_id_int,
            "records": clinical_record_store.list_for_patient(
                stay_id_int,
                category=category,
                start_time=start_time,
                end_time=end_time,
                include_voided=include_voided,
            ),
        }

    # If source is None:
    if not ref_str.isdigit():
        stay = patient_store.get_stay(stay_ref)
        if not stay:
            raise HTTPException(status_code=404, detail=f"ICU stay '{stay_ref}' not found.")
        records = clinical_record_store.list_for_stay_id(
            stay["id"],
            category=category,
            start_time=start_time,
            end_time=end_time,
            include_voided=include_voided,
        )
        return {
            "stay_id": stay["id"],
            "stay_code": stay["stay_code"],
            "patient_id": stay["patient_id"],
            "records": records,
        }

    stay_id_int = int(ref_str)
    reg_stay = patient_store.get_stay(stay_id_int)
    has_historical = df is not None and (df["patientunitstayid"] == stay_id_int).any()

    if reg_stay is not None and has_historical:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Stay identifier '{stay_ref}' is ambiguous: exists as both registered stay "
                f"'{reg_stay['stay_code']}' (ID {reg_stay['id']}) and historical eICU stay. "
                "Specify query parameter '?source=registered' or '?source=historical'."
            ),
        )
    if reg_stay is not None:
        records = clinical_record_store.list_for_stay_id(
            reg_stay["id"],
            category=category,
            start_time=start_time,
            end_time=end_time,
            include_voided=include_voided,
        )
        return {
            "stay_id": reg_stay["id"],
            "stay_code": reg_stay["stay_code"],
            "patient_id": reg_stay["patient_id"],
            "records": records,
        }
    if has_historical:
        return {
            "patientunitstayid": stay_id_int,
            "records": clinical_record_store.list_for_patient(
                stay_id_int,
                category=category,
                start_time=start_time,
                end_time=end_time,
                include_voided=include_voided,
            ),
        }
    raise HTTPException(status_code=404, detail=f"ICU stay '{stay_ref}' not found.")


@app.get("/api/clinical-records/{record_id}/revisions")
def get_clinical_record_revisions(record_id: UUID):
    if clinical_record_store.get(str(record_id)) is None:
        raise HTTPException(status_code=404, detail="Clinical record not found.")
    return {
        "record_id": str(record_id),
        "revisions": clinical_record_store.revisions(str(record_id)),
    }


@app.get("/api/clinical-records/{record_id}")
def get_clinical_record(record_id: UUID):
    record = clinical_record_store.get(str(record_id))
    if record is None:
        raise HTTPException(status_code=404, detail="Clinical record not found.")
    return record


@app.put("/api/clinical-records/{record_id}")
def update_clinical_record(record_id: UUID, record: ClinicalRecordUpdate):
    try:
        updated = clinical_record_store.update(str(record_id), record)
    except ClinicalRecordConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if updated is None:
        raise HTTPException(
            status_code=404,
            detail="Clinical record not found or already voided.",
        )

    # Supersede previous active alerts linked to this record
    alert_manager.supersede_alerts_for_record(
        record_id=str(record_id),
        reason=f"Superseded by record revision: {record.change_reason}",
        voided_by=record.entered_by,
    )

    # Re-evaluate the updated record
    interp, alert = evaluate_record_for_alerts(
        record_dict=updated,
        record_obj=record,
        patient_id=updated.get("patient_id"),
        stay_id=updated.get("stay_id"),
        patientunitstayid=updated.get("patientunitstayid"),
    )
    if interp:
        updated["interpretation"] = interp.model_dump()
    if alert:
        updated["alert"] = alert

    if updated.get("patient_id") is not None:
        try:
            clinical_intelligence.evaluate_patient(updated["patient_id"])
        except Exception:
            pass

    return updated


@app.delete("/api/clinical-records/{record_id}")
def void_clinical_record(
    record_id: UUID,
    void: ClinicalRecordVoid = Body(...),
):
    try:
        voided = clinical_record_store.void(
            str(record_id),
            expected_version=void.expected_version,
            voided_by=void.voided_by,
            void_reason=void.void_reason,
        )
    except ClinicalRecordConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if voided is None:
        raise HTTPException(status_code=404, detail="Clinical record not found.")

    # Voiding a record marks its active alerts as superseded
    alert_manager.supersede_alerts_for_record(
        record_id=str(record_id),
        reason=f"Clinical record voided: {void.void_reason}",
        voided_by=void.voided_by,
    )

    if voided.get("patient_id") is not None:
        try:
            clinical_intelligence.evaluate_patient(voided["patient_id"])
        except Exception:
            pass

    return voided


# ============================================================
# SUMMARY
# ============================================================

@app.get("/api/summary")
def get_summary():

    if df is None:
        raise HTTPException(
            status_code=503,
            detail="Dataset not loaded."
        )

    priority_counts = (
        df[
            "final_alert_priority"
        ]
        .value_counts()
        .to_dict()
    )

    return {
        "total_rows": int(len(df)),

        "icu_stays": int(
            df[
                "patientunitstayid"
            ].nunique()
        ),

        "alert_priorities": {
            "normal": int(
                priority_counts.get(
                    "normal",
                    0
                )
            ),
            "low": int(
                priority_counts.get(
                    "low",
                    0
                )
            ),
            "moderate": int(
                priority_counts.get(
                    "moderate",
                    0
                )
            ),
            "high": int(
                priority_counts.get(
                    "high",
                    0
                )
            ),
            "critical": int(
                priority_counts.get(
                    "critical",
                    0
                )
            )
        },

        "ml_anomalies": int(
            df["ml_anomaly"].sum()
        ),

        "clinical_warnings": int(
            df["clinical_warning"].sum()
        ),

        "clinical_severe": int(
            df["clinical_severe"].sum()
        ),

        "risk_10_percent_or_higher": int(
            df["risk_high"].sum()
        ),

        "risk_25_percent_or_higher": int(
            df["risk_very_high"].sum()
        ),

        "risk_50_percent_or_higher": int(
            df["risk_extreme"].sum()
        ),

        "multi_signal_alerts": int(
            (
                df["alert_category"]
                == "multi-signal"
            ).sum()
        )
    }


# ============================================================
# PATIENTS & ADMISSIONS (REGISTERED)
# ============================================================

@app.post("/api/patients", status_code=201)
def create_patient(patient: PatientCreate):
    try:
        return patient_store.create_patient(patient)
    except ValueError as err:
        msg = str(err)
        raise HTTPException(
            status_code=409 if "already exists" in msg else 400,
            detail=msg,
        )


@app.get("/api/registered-patients")
def get_registered_patients(
    search: str | None = Query(default=None),
    status: str | None = Query(default="all"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    return patient_store.list_patients(
        search=search,
        status=status,
        limit=limit,
        offset=offset,
    )


@app.get("/api/registered-patients/{patient_ref}")
def get_registered_patient(patient_ref: str):
    patient = patient_store.get_patient(patient_ref)
    if not patient:
        raise HTTPException(
            status_code=404,
            detail=f"Registered patient '{patient_ref}' not found.",
        )
    return patient


@app.patch("/api/patients/{patient_ref}")
def update_patient(patient_ref: str, fields: PatientUpdate):
    try:
        updated = patient_store.update_patient(patient_ref, fields)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_ref}' not found.",
        )
    return updated


@app.post("/api/patients/{patient_ref}/archive")
def archive_patient(patient_ref: str):
    archived = patient_store.archive_patient(patient_ref)
    if not archived:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_ref}' not found.",
        )
    return archived


@app.delete("/api/patients/{patient_ref}")
def delete_patient(patient_ref: str):
    try:
        deleted = patient_store.delete_patient(patient_ref)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Patient '{patient_ref}' not found.",
        )
    return {"status": "deleted", "patient_id": patient_ref}


# ============================================================
# ICU STAYS / ADMISSIONS
# ============================================================

@app.post("/api/patients/{patient_ref}/stays", status_code=201)
def create_patient_stay(patient_ref: str, fields: IcuStayCreate):
    try:
        return patient_store.create_stay(patient_ref, fields)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err).strip("'"))
    except ValueError as err:
        msg = str(err)
        status_code = 409 if "already has an active ICU admission" in msg or "already exists" in msg else 400
        raise HTTPException(status_code=status_code, detail=msg)


@app.get("/api/patients/{patient_ref}/stays")
def list_patient_stays(patient_ref: str):
    try:
        return {
            "patient_ref": patient_ref,
            "stays": patient_store.list_stays_for_patient(patient_ref),
        }
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err).strip("'"))


@app.get("/api/stays/{stay_ref}")
def get_stay(stay_ref: str):
    stay = patient_store.get_stay(stay_ref)
    if not stay:
        raise HTTPException(
            status_code=404,
            detail=f"ICU stay '{stay_ref}' not found.",
        )
    return stay


@app.patch("/api/stays/{stay_ref}")
def update_stay(stay_ref: str, fields: IcuStayUpdate):
    updated = patient_store.update_stay(stay_ref, fields)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"ICU stay '{stay_ref}' not found.",
        )
    return updated


@app.post("/api/stays/{stay_ref}/discharge")
def discharge_stay(stay_ref: str, fields: IcuStayDischarge = Body(default_factory=IcuStayDischarge)):
    try:
        return patient_store.discharge_stay(
            stay_ref,
            discharge_datetime=fields.discharge_datetime,
            discharge_status=fields.discharge_status,
        )
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err).strip("'"))
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))


# ============================================================
# DEVICES & ASSIGNMENTS
# ============================================================

@app.post("/api/devices", status_code=201)
def register_device(device: DeviceCreate):
    try:
        return patient_store.register_device(device)
    except ValueError as err:
        raise HTTPException(status_code=409, detail=str(err))


@app.get("/api/devices")
def list_devices():
    return {"devices": patient_store.list_devices()}


@app.get("/api/devices/{device_id}")
def get_device(device_id: str):
    dev = patient_store.get_device(device_id)
    if not dev:
        raise HTTPException(
            status_code=404,
            detail=f"Device '{device_id}' not found.",
        )
    return dev


@app.post("/api/devices/{device_id}/assign")
def assign_device(device_id: str, payload: DeviceAssign):
    try:
        return patient_store.assign_device(
            device_id=device_id,
            patient_ref=payload.patient_id,
            stay_ref=payload.stay_id,
        )
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err).strip("'"))
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))


@app.post("/api/devices/{device_id}/unassign")
def unassign_device(device_id: str, payload: DeviceUnassign = Body(default_factory=DeviceUnassign)):
    try:
        return patient_store.unassign_device(device_id=device_id, reason=payload.reason)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err).strip("'"))


@app.get("/api/devices/{device_id}/assignments")
def list_device_assignments(device_id: str):
    dev = patient_store.get_device(device_id)
    if not dev:
        raise HTTPException(
            status_code=404,
            detail=f"Device '{device_id}' not found.",
        )
    return {
        "device_id": device_id,
        "assignments": patient_store.list_device_assignments(device_id),
    }


# ============================================================
# PATIENT TELEMETRY & TIMELINE
# ============================================================

@app.get("/api/patients/{patient_ref}/telemetry")
def get_patient_telemetry(
    patient_ref: str,
    limit: int = Query(default=100, ge=1, le=1000),
    start_time: datetime | None = None,
    end_time: datetime | None = None,
):
    try:
        return {
            "patient_ref": patient_ref,
            "telemetry": patient_store.get_patient_telemetry(
                patient_ref,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
            ),
        }
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err).strip("'"))


@app.get("/api/stays/{stay_ref}/telemetry")
def get_stay_telemetry(
    stay_ref: str,
    limit: int = Query(default=100, ge=1, le=1000),
):
    try:
        return {
            "stay_ref": stay_ref,
            "telemetry": patient_store.get_stay_telemetry(stay_ref, limit=limit),
        }
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err).strip("'"))


@app.post("/api/devices/{device_id}/telemetry", status_code=201)
def inject_device_telemetry(device_id: str, reading: TelemetryIngest):
    """
    Ingest or simulate incoming telemetry for a device.
    Associates reading with patient and stay strictly based on active backend assignment.
    """
    try:
        reading_dict = reading.model_dump()
        return patient_store.ingest_reading(device_id, reading_dict)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))



@app.get("/api/patients/{patient_ref}/timeline")
def get_patient_timeline(patient_ref: str):
    try:
        return patient_store.get_patient_timeline(patient_ref)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err).strip("'"))


# ============================================================
# HISTORICAL EICU PATIENT LIST & LEGACY LOOKUP
# ============================================================

@app.get("/api/patients")
def get_patients(
    priority: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    source: str | None = Query(default=None),
    search: str | None = Query(default=None),
    status: str | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
):
    if patient_store.database_path != clinical_record_store.database_path:
        patient_store.database_path = clinical_record_store.database_path

    # Route to registered patients if explicitly requested or searching
    is_registered = (
        (isinstance(source, str) and source == "registered")
        or (isinstance(search, str) and bool(search.strip()))
        or (isinstance(status, str) and status in ("active", "archived", "all"))
    )
    if is_registered:
        return patient_store.list_patients(
            search=search if isinstance(search, str) else None,
            status=status if isinstance(status, str) else "all",
            limit=limit if isinstance(limit, int) else 100,
            offset=offset if isinstance(offset, int) else 0,
        )

    # Otherwise return historical eICU dataset
    if df is None:
        raise HTTPException(
            status_code=503,
            detail="Dataset not loaded."
        )

    valid_priorities = {
        "normal",
        "low",
        "moderate",
        "high",
        "critical"
    }

    priority_val = priority if isinstance(priority, str) else None
    limit_val = limit if isinstance(limit, int) else 50

    if priority_val:
        priority_val = priority_val.lower()

        if priority_val not in valid_priorities:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid priority. "
                    "Use normal, low, moderate, "
                    "high, or critical."
                )
            )

    # Latest hour per patient
    latest = (
        df
        .sort_values(
            "hour"
        )
        .groupby(
            "patientunitstayid",
            as_index=False
        )
        .tail(1)
    )

    if priority_val:
        latest = latest[
            latest[
                "final_alert_priority"
            ].str.lower()
            == priority_val
        ]

    # Sort the patient list by alert severity
    severity_order = {
        "critical": 4,
        "high": 3,
        "moderate": 2,
        "low": 1,
        "normal": 0
    }

    latest = latest.copy()

    latest["severity_rank"] = (
        latest[
            "final_alert_priority"
        ]
        .map(severity_order)
        .fillna(0)
    )

    latest = latest.sort_values(
        [
            "severity_rank",
            "risk_probability"
        ],
        ascending=False
    )

    patients = []

    for _, row in latest.head(limit_val).iterrows():

        patients.append({
            "patientunitstayid":
                int(row["patientunitstayid"]),

            "hour":
                float(row["hour"]),

            "priority":
                row["final_alert_priority"],

            "alert_category":
                row["alert_category"],

            "risk_probability":
                clean_value(
                    row["risk_probability"]
                ),

            "risk_band":
                row["risk_band"],

            "anomaly_score":
                clean_value(
                    row["anomaly_score"]
                ),
	    "ml_anomaly":
    		bool(
        	    row["ml_anomaly"]
    		),

            "clinical_rule_severity":
                row["clinical_rule_severity_v2"],

            "clinical_explanation":
                row[
                    "clinical_rule_explanation_v2"
                ],

            "final_explanation":
                row[
                    "final_alert_explanation"
                ]
        })

    return {
        "count": len(patients),
        "patients": patients
    }


# ============================================================
# PATIENT DETAIL / HISTORY
# ============================================================

def get_patient_history(
    patientunitstayid: int,
    limit: int = Query(
        default=200,
        ge=1,
        le=2000
    )
):
    patient_df = require_patient(
        patientunitstayid
    )

    patient_df = (
        patient_df
        .sort_values("hour")
        .tail(limit)
    )

    records = []

    for _, row in patient_df.iterrows():
        records.append({
            "patientunitstayid": int(row["patientunitstayid"]),
            "hour": float(row["hour"]),
            "heart_rate": clean_value(row["heart_rate"]),
            "spo2": clean_value(row["spo2"]),
            "respiration": clean_value(row["respiration"]),
            "systolic_bp": clean_value(row["systolic_bp"]),
            "diastolic_bp": clean_value(row["diastolic_bp"]),
            "mean_bp": clean_value(row["mean_bp"]),
            "total_intake_ml": clean_value(row["total_intake_ml"]),
            "total_output_ml": clean_value(row["total_output_ml"]),
            "dialysis_ml": clean_value(row["dialysis_ml"]),
            "reported_net_ml": clean_value(row["reported_net_ml"]),
            "net_fluid_balance_ml": clean_value(row["net_fluid_balance_ml"]),
            "urine_output_ml": clean_value(row["urine_output_ml"]),
            "active_infusion_count": clean_value(row["active_infusion_count"]),
            "vasoactive_support": clean_value(row["vasoactive_support"]),
            "anomaly_score": clean_value(row["anomaly_score"]),
            "ml_anomaly": bool(row["ml_anomaly"]),
            "risk_probability": clean_value(row["risk_probability"]),
            "risk_band": row["risk_band"],
            "clinical_rule_severity": row["clinical_rule_severity_v2"],
            "clinical_explanation": row["clinical_rule_explanation_v2"],
            "priority": row["final_alert_priority"],
            "alert_category": row["alert_category"],
            "final_explanation": row["final_alert_explanation"]
        })

    return {
        "patientunitstayid": patientunitstayid,
        "rows": len(records),
        "history": records
    }


@app.get("/api/patients/{patient_ref}")
def get_patient_detail_or_history(
    patient_ref: str,
    source: Literal["registered", "historical"] | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
):
    ref_str = str(patient_ref).strip()

    if source == "registered":
        patient = patient_store.get_patient(patient_ref)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Registered patient '{patient_ref}' not found.")
        return patient

    if source == "historical":
        try:
            stay_id = int(patient_ref)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Historical eICU stay '{patient_ref}' not found.")
        return get_patient_history(stay_id, limit=limit)

    # If source is None:
    if not ref_str.isdigit() and not (ref_str.startswith("-") and ref_str[1:].isdigit()):
        patient = patient_store.get_patient(patient_ref)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Registered patient '{patient_ref}' not found.")
        return patient

    # Numeric string without explicit source
    num_id = int(ref_str)
    reg_patient = patient_store.get_patient(num_id)
    has_historical = df is not None and (df["patientunitstayid"] == num_id).any()

    if reg_patient is not None and has_historical:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Identifier '{patient_ref}' is ambiguous: exists as both registered patient "
                f"'{reg_patient['patient_code']}' (ID {reg_patient['id']}) and historical eICU stay. "
                "Specify query parameter '?source=registered' or '?source=historical'."
            ),
        )

    if reg_patient is not None:
        return reg_patient

    if has_historical:
        return get_patient_history(num_id, limit=limit)

    raise HTTPException(status_code=404, detail=f"Patient or historical stay '{patient_ref}' not found.")


# ============================================================
# PATIENT LATEST STATE
# ============================================================

@app.get(
    "/api/patients/{patientunitstayid}/latest"
)
def get_patient_latest(
    patientunitstayid: int
):

    patient_df = require_patient(
        patientunitstayid
    )

    latest = (
        patient_df
        .sort_values("hour")
        .iloc[-1]
    )

    return {
        "patientunitstayid":
            patientunitstayid,

        "hour":
            clean_value(
                latest["hour"]
            ),

        "vitals": {
            "heart_rate":
                clean_value(
                    latest["heart_rate"]
                ),

            "spo2":
                clean_value(
                    latest["spo2"]
                ),

            "respiration":
                clean_value(
                    latest["respiration"]
                ),

            "systolic_bp":
                clean_value(
                    latest["systolic_bp"]
                ),

            "diastolic_bp":
                clean_value(
                    latest["diastolic_bp"]
                ),

            "mean_bp":
                clean_value(
                    latest["mean_bp"]
                )
        },

        "fluid": {
            "net_balance_ml":
                clean_value(
                    latest[
                        "net_fluid_balance_ml"
                    ]
                ),

            "urine_output_ml":
                clean_value(
                    latest[
                        "urine_output_ml"
                    ]
                ),
        "total_intake_ml": clean_value(
                            latest["total_intake_ml"]),
        "total_output_ml": clean_value(
                            latest["total_output_ml"]),
        "dialysis_ml": clean_value(
                            latest["dialysis_ml"]),
        "reported_net_ml": clean_value(
                            latest["reported_net_ml"]),
        },
        "treatment": {
            "active_infusion_count":
                clean_value(
                    latest[
                        "active_infusion_count"
                    ]
                ),

            "vasoactive_support":
                clean_value(
                    latest[
                        "vasoactive_support"
                    ]
                )
        },

        "intelligence": {
            "priority":
                latest[
                    "final_alert_priority"
                ],

            "alert_category":
                latest[
                    "alert_category"
                ],

            "anomaly_score":
                clean_value(
                    latest[
                        "anomaly_score"
                    ]
                ),

            "ml_anomaly":
                bool(
                    latest[
                        "ml_anomaly"
                    ]
                ),

            "deterioration_risk":
                clean_value(
                    latest[
                        "risk_probability"
                    ]
                ),

            "risk_band":
                latest[
                    "risk_band"
                ],

            "clinical_severity":
                latest[
                    "clinical_rule_severity_v2"
                ],

            "explanation":
                latest[
                    "final_alert_explanation"
                ]
        }
    }


# ============================================================
# UNIFIED CLINICAL ALERTS API
# ============================================================

@app.get("/api/alerts")
def get_alerts(
    patient_id: int | None = Query(default=None),
    stay_id: int | None = Query(default=None),
    patientunitstayid: int | None = Query(default=None),
    severity: str | None = Query(default=None),
    source: str | None = Query(default=None),
    status: str | None = Query(default=None),
    category: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    alerts = alert_manager.list_alerts(
        patient_id=patient_id,
        stay_id=stay_id,
        patientunitstayid=patientunitstayid,
        severity=severity,
        source=source,
        status=status,
        category=category,
        limit=limit,
        offset=offset,
    )
    return {"alerts": alerts}


@app.get("/api/alerts/summary")
def get_alerts_summary():
    return alert_manager.get_summary()


@app.get("/api/alerts/{alert_id}")
def get_alert_by_id(alert_id: str):
    alert = alert_manager.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return alert


@app.get("/api/alerts/{alert_id}/history")
def get_alert_history(alert_id: str):
    alert = alert_manager.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return {"alert_id": alert_id, "history": alert_manager.get_alert_history(alert_id)}


@app.post("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str, req: AlertAcknowledgeRequest):
    try:
        updated = alert_manager.acknowledge_alert(
            alert_id=alert_id,
            acknowledged_by=req.acknowledged_by,
            note=req.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not updated:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return updated


@app.post("/api/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: str, req: AlertResolveRequest):
    try:
        updated = alert_manager.resolve_alert(
            alert_id=alert_id,
            resolved_by=req.resolved_by,
            note=req.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not updated:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return updated


@app.get("/api/patients/{patient_ref}/alerts")
def get_patient_alerts(
    patient_ref: int | str,
    status: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    pid = patient_store.resolve_patient_id(patient_ref)
    punit = None
    if pid is None:
        try:
            punit = int(patient_ref)
        except ValueError:
            pass
    if pid is None and punit is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_ref}' not found.")

    alerts = alert_manager.list_alerts(
        patient_id=pid,
        patientunitstayid=punit,
        status=status,
        severity=severity,
        limit=limit,
    )
    return {"alerts": alerts}


@app.get("/api/stays/{stay_ref}/alerts")
def get_stay_alerts(
    stay_ref: int | str,
    status: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    stay_id = patient_store.resolve_stay_id(stay_ref)
    if stay_id is None:
        raise HTTPException(status_code=404, detail=f"ICU stay '{stay_ref}' not found.")
    alerts = alert_manager.list_alerts(
        stay_id=stay_id,
        status=status,
        severity=severity,
        limit=limit,
    )
    return {"alerts": alerts}


class EvaluateRequest(BaseModel):
    category: str
    measurement_name: str
    value: float | str
    unit: str | None = None
    observation_period: str | None = None
    reference_range: str | None = None
    patient_id: int | None = None
    weight_kg: float | None = None


@app.post("/api/clinical-records/evaluate")
def preview_clinical_evaluation(req: EvaluateRequest):
    weight = req.weight_kg
    if weight is None and req.patient_id:
        pat = patient_store.get_patient(req.patient_id)
        if pat:
            weight = pat.get("weight_kg")

    if req.category == "urine_output":
        val_f = float(req.value) if isinstance(req.value, (int, float)) else 0.0
        interp = clinical_alert_engine.evaluate_urine_output(
            volume_ml=val_f,
            unit=req.unit or "mL",
            patient_weight_kg=weight,
            observation_period=req.observation_period,
        )
    elif req.category in ("blood_test", "other_lab"):
        interp = clinical_alert_engine.evaluate_lab_result(
            measurement_name=req.measurement_name,
            value=req.value,
            unit=req.unit,
            reference_range_str=req.reference_range,
        )
    else:
        interp = clinical_alert_engine.evaluate_clinical_observation(
            measurement_name=req.measurement_name,
            value=req.value,
            unit=req.unit,
        )
    return interp.model_dump()


# ============================================================
# ML INTELLIGENCE STATUS API
# ============================================================

@app.get("/api/ml/status")
def get_ml_status():
    return ml_service.get_status()


@app.get("/api/ml/predict/{stay_id}")
def predict_patient_ml(stay_id: int):
    """
    Run genuine retrospective rolling machine learning inference on an ICU stay.
    Evaluates unweighted Logistic Regression deterioration risk (target: ICU discharge marked expired within the next 24 hours from the prediction timestamp)
    and unsupervised IsolationForest anomaly detection.
    """
    patient_df = require_patient(stay_id)
    sorted_df = patient_df.sort_values("hour").reset_index(drop=True)

    det_result = ml_service.predict_deterioration(sorted_df)
    anom_result = ml_service.score_anomalies(sorted_df)

    if not det_result.get("available"):
        raise HTTPException(
            status_code=503 if det_result.get("status") == "UNAVAILABLE" else 422,
            detail=det_result.get("error", "ML prediction unavailable.")
        )

    latest_row = sorted_df.iloc[-1]
    latest_hour = int(latest_row["hour"])
    latest_pred = det_result.get("latest_prediction") or {}

    # Construct historical series for charts/timelines
    series = []
    probs = det_result.get("probabilities") or []
    bands = det_result.get("risk_bands") or []
    anom_scores = anom_result.get("scores") or [] if anom_result.get("available") else []
    anom_flags = anom_result.get("flags") or [] if anom_result.get("available") else []

    for i, row in sorted_df.iterrows():
        series.append({
            "hour": int(row["hour"]),
            "probability": probs[i] if i < len(probs) else None,
            "risk_band": bands[i] if i < len(bands) else "Unknown",
            "anomaly_score": anom_scores[i] if i < len(anom_scores) else None,
            "anomaly_flag": anom_flags[i] if i < len(anom_flags) else 0,
        })

    return {
        "patientunitstayid": stay_id,
        "total_hours": len(sorted_df),
        "latest_hour": latest_hour,
        "model_name": det_result.get("model_name"),
        "model_version": det_result.get("model_version"),
        "target": det_result.get("target"),
        "latest_prediction": latest_pred,
        "anomaly": {
            "available": anom_result.get("available", False),
            "latest_score": anom_scores[-1] if anom_scores else None,
            "latest_flag": anom_flags[-1] if anom_flags else 0,
        },
        "series": series,
        "disclaimer": det_result.get("disclaimer"),
    }


# ============================================================
# DATA CONSISTENCY & INTELLIGENCE ENGINE ENDPOINTS
# ============================================================

@app.get("/api/consistency/status")
def get_consistency_status():
    """
    Returns the operational status, reference distributions, and tolerance configuration
    of the Unified Clinical Data Consistency & Intelligence Engine.
    """
    return clinical_intelligence.get_status()


@app.get("/api/consistency/patient/{patient_id}")
def get_patient_consistency(patient_id: int):
    """
    Retrieves the latest multi-level clinical data consistency and anomaly detection
    evaluation for a registered patient's active ICU stay.
    """
    pat = patient_store.get_patient(patient_id)
    if not pat:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")
    try:
        return clinical_intelligence.evaluate_patient(patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Consistency evaluation failed: {exc}")


@app.get("/api/consistency/patient/{patient_id}/history")
def get_patient_consistency_history(
    patient_id: int,
    limit: int = Query(default=50, ge=1, le=500),
):
    """
    Retrieves historical consistency checks, patient trend anomalies, and historical
    population comparison records for a patient.
    """
    pat = patient_store.get_patient(patient_id)
    if not pat:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")
    return clinical_intelligence.get_patient_consistency_history(patient_id, limit=limit)


@app.get("/api/consistency/alerts")
def get_consistency_alerts(
    patient_id: int | None = Query(default=None),
    stay_id: int | None = Query(default=None),
    severity: str | None = Query(default=None),
    source: str | None = Query(default=None),
    status: str | None = Query(default="ACTIVE"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """
    Retrieves active or filtered alerts originating from the data consistency engine
    (CROSS_SOURCE_DISCREPANCY, PATIENT_TREND_ANOMALY, HISTORICAL_POPULATION_ANOMALY).
    """
    allowed_sources = {
        "CROSS_SOURCE_DISCREPANCY",
        "PATIENT_TREND_ANOMALY",
        "HISTORICAL_POPULATION_ANOMALY",
    }
    req_source = source if source in allowed_sources else None

    raw_alerts = alert_manager.list_alerts(
        patient_id=patient_id,
        stay_id=stay_id,
        severity=severity,
        source=req_source,
        status=status,
        limit=limit,
        offset=offset,
    )
    if not req_source:
        filtered = [a for a in raw_alerts if a.get("source") in allowed_sources]
    else:
        filtered = raw_alerts

    return {"alerts": filtered, "total": len(filtered)}


@app.post("/api/consistency/evaluate/{patient_id}")
def evaluate_patient_consistency(patient_id: int):
    """
    Explicitly triggers an on-demand multi-level consistency and anomaly re-evaluation
    for a patient's active ICU stay.
    """
    pat = patient_store.get_patient(patient_id)
    if not pat:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")
    try:
        return clinical_intelligence.evaluate_patient(patient_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")


@app.post("/api/consistency/alerts/{alert_id}/acknowledge")
def acknowledge_consistency_alert(alert_id: str, req: AlertAcknowledgeRequest):
    """
    Acknowledge a consistency or anomaly alert with clinician identification and notes.
    """
    try:
        updated = alert_manager.acknowledge_alert(
            alert_id=alert_id,
            acknowledged_by=req.acknowledged_by,
            note=req.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not updated:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return updated


@app.post("/api/consistency/alerts/{alert_id}/resolve")
def resolve_consistency_alert(alert_id: str, req: AlertResolveRequest):
    """
    Resolve a consistency or anomaly alert with clinician resolution notes.
    """
    try:
        updated = alert_manager.resolve_alert(
            alert_id=alert_id,
            resolved_by=req.resolved_by,
            note=req.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not updated:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return updated