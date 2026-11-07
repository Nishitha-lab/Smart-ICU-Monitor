from pathlib import Path
from contextlib import asynccontextmanager

from services.serial_service import serial_monitor

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "smart_icu_final_alerts.csv"


# ============================================================
# APP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global df
    print("Loading Smart-ICU historical dataset...")
    df = load_dataset()
    print(f"Loaded {len(df):,} patient-hour rows from {df['patientunitstayid'].nunique():,} ICU stays.")
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
        "version": "0.2.0"
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

    data = df.copy()

    if priority:

        priority = priority.lower()

        valid_priorities = {
            "normal",
            "low",
            "moderate",
            "high",
            "critical"
        }

        if priority not in valid_priorities:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid priority. "
                    "Use normal, low, moderate, "
                    "high, or critical."
                )
            )

        data = data[
            data[
                "final_alert_priority"
            ].str.lower()
            == priority
        ]

    # Latest hour per patient
    latest = (
        data
        .sort_values(
            "hour"
        )
        .groupby(
            "patientunitstayid",
            as_index=False
        )
        .tail(1)
    )

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