from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
import os
from uuid import UUID

from services.serial_service import serial_monitor
from services.clinical_records import (
    ClinicalCategory,
    ClinicalRecordConflict,
    ClinicalRecordCreate,
    ClinicalRecordDatabaseBusy,
    ClinicalRecordStore,
    ClinicalRecordUpdate,
    ClinicalRecordVoid,
)

import pandas as pd
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


# ============================================================
# APP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global df
    print("Loading Smart-ICU historical dataset...")
    df = load_dataset()
    print(f"Loaded {len(df):,} patient-hour rows from {df['patientunitstayid'].nunique():,} ICU stays.")
    clinical_record_store.initialize()
    serial_monitor.start()
    try:
        yield
    finally:
        serial_monitor.stop()

app = FastAPI(
    title="Smart ICU Patient Safety System",
    description="Research prototype: historical eICU analytics and live ESP32 telemetry",
    version="0.3.0",
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


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
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
    """Latest parsed ESP32 reading; this is not linked to an eICU patient."""
    return serial_monitor.snapshot()


# ============================================================
# MANUALLY ENTERED CLINICAL RECORDS
# ============================================================

def require_existing_stay(patientunitstayid: int) -> None:
    if df is None:
        raise HTTPException(
            status_code=503,
            detail="Historical patient dataset is not loaded.",
        )
    if not (df["patientunitstayid"] == patientunitstayid).any():
        raise HTTPException(
            status_code=404,
            detail=f"ICU patient stay {patientunitstayid} not found.",
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


@app.post("/api/clinical-records", status_code=201)
def create_clinical_record(record: ClinicalRecordCreate):
    require_existing_stay(record.patientunitstayid)
    return clinical_record_store.create(record)


@app.get("/api/patients/{patientunitstayid}/clinical-records")
def get_patient_clinical_records(
    patientunitstayid: int,
    category: ClinicalCategory | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    include_voided: bool = False,
):
    require_existing_stay(patientunitstayid)
    validate_clinical_date_range(start_time, end_time)
    return {
        "patientunitstayid": patientunitstayid,
        "records": clinical_record_store.list_for_patient(
            patientunitstayid,
            category=category,
            start_time=start_time,
            end_time=end_time,
            include_voided=include_voided,
        ),
    }


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
# PATIENT LIST
# ============================================================

@app.get("/api/patients")
def get_patients(
    priority: str | None = Query(
        default=None
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500
    )
):

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

    if priority:
        priority = priority.lower()

        if priority not in valid_priorities:
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

    if priority:
        latest = latest[
            latest[
                "final_alert_priority"
            ].str.lower()
            == priority
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

    for _, row in latest.head(limit).iterrows():

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
# PATIENT HISTORY
# ============================================================

@app.get(
    "/api/patients/{patientunitstayid}"
)
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

            "patientunitstayid":
                int(
                    row["patientunitstayid"]
                ),

            "hour":
                float(row["hour"]),

            "heart_rate":
                clean_value(
                    row["heart_rate"]
                ),

            "spo2":
                clean_value(
                    row["spo2"]
                ),

            "respiration":
                clean_value(
                    row["respiration"]
                ),

            "systolic_bp":
                clean_value(
                    row["systolic_bp"]
                ),

            "diastolic_bp":
                clean_value(
                    row["diastolic_bp"]
                ),

            "mean_bp":
    clean_value(
        row["mean_bp"]
    ),

"total_intake_ml":
    clean_value(
        row["total_intake_ml"]
    ),

"total_output_ml":
    clean_value(
        row["total_output_ml"]
    ),

"dialysis_ml":
    clean_value(
        row["dialysis_ml"]
    ),

"reported_net_ml":
    clean_value(
        row["reported_net_ml"]
    ),

"net_fluid_balance_ml":
    clean_value(
        row["net_fluid_balance_ml"]
    ),

"urine_output_ml":
    clean_value(
        row["urine_output_ml"]
    ),

            "active_infusion_count":
                clean_value(
                    row["active_infusion_count"]
                ),

            "vasoactive_support":
                clean_value(
                    row["vasoactive_support"]
                ),

            "anomaly_score":
                clean_value(
                    row["anomaly_score"]
                ),

            "ml_anomaly":
                bool(
                    row["ml_anomaly"]
                ),

            "risk_probability":
                clean_value(
                    row["risk_probability"]
                ),

            "risk_band":
                row["risk_band"],

            "clinical_rule_severity":
                row[
                    "clinical_rule_severity_v2"
                ],

            "clinical_explanation":
                row[
                    "clinical_rule_explanation_v2"
                ],

            "priority":
                row[
                    "final_alert_priority"
                ],

            "alert_category":
                row[
                    "alert_category"
                ],

            "final_explanation":
                row[
                    "final_alert_explanation"
                ]
        })

    return {
        "patientunitstayid":
            patientunitstayid,

        "rows": len(records),

        "history": records
    }


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