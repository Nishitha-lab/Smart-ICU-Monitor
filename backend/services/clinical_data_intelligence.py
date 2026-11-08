"""
Unified Clinical Data Consistency & Intelligence Coordinator.

Orchestrates:
- Level 1: Cross-Source Clinical Data Consistency (ESP32 Telemetry vs Manual Bedside Observations)
- Level 2: Patient-Specific Historical Trend Analysis (Stay Baseline, Rate of Change, Spikes/Drops)
- Level 3: Historical eICU Population-Level Anomaly Detection (Reference Distributions & Isolation Forest)

Stores evaluations in persistent SQLite tables (data_consistency_checks, patient_trend_anomalies, historical_population_anomalies)
and integrates seamlessly with AlertManager for fingerprint-based alert deduplication and lifecycle management.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field

from services.data_consistency_engine import (
    DataConsistencyEngine,
    DiscrepancyResult,
    data_consistency_engine,
)
from services.patient_trend_analyzer import (
    PatientTrendAnalyzer,
    TrendAnalysisResult,
    patient_trend_analyzer,
)
from services.historical_reference_engine import (
    HistoricalReferenceEngine,
    PopulationComparisonResult,
    IsolationForestEvaluationResult,
    historical_reference_engine,
)
from services.alert_manager import AlertManager


class ClinicalDataIntelligenceCoordinator:
    def __init__(
        self,
        database_path: Path,
        consistency_engine: DataConsistencyEngine | None = None,
        trend_analyzer: PatientTrendAnalyzer | None = None,
        reference_engine: HistoricalReferenceEngine | None = None,
        alert_manager: AlertManager | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.consistency_engine = consistency_engine or data_consistency_engine
        self.trend_analyzer = trend_analyzer or patient_trend_analyzer
        self.reference_engine = reference_engine or historical_reference_engine
        self.alert_manager = alert_manager or AlertManager(self.database_path)

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

    def initialize(self) -> None:
        """
        Create persistent tables for consistency checks and anomalies.
        """
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            # 1. Level 1 Cross-Source Consistency Checks
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS data_consistency_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL REFERENCES patients (id) ON DELETE RESTRICT,
                    stay_id INTEGER REFERENCES icu_stays (id) ON DELETE RESTRICT,
                    metric TEXT NOT NULL,
                    canonical_metric TEXT NOT NULL,
                    telemetry_value REAL,
                    telemetry_time TEXT,
                    clinical_value REAL,
                    clinical_time TEXT,
                    clinical_record_id TEXT,
                    discrepancy_level TEXT NOT NULL,
                    absolute_difference REAL,
                    relative_difference_pct REAL,
                    time_difference_seconds REAL,
                    explanation TEXT NOT NULL,
                    clinical_rationale TEXT,
                    recommended_action TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_consistency_patient ON data_consistency_checks (patient_id, created_at);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_consistency_stay ON data_consistency_checks (stay_id, created_at);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_consistency_metric ON data_consistency_checks (canonical_metric, discrepancy_level);"
            )

            # 2. Level 2 Patient Trend Anomalies
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_trend_anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL REFERENCES patients (id) ON DELETE RESTRICT,
                    stay_id INTEGER REFERENCES icu_stays (id) ON DELETE RESTRICT,
                    metric TEXT NOT NULL,
                    canonical_metric TEXT NOT NULL,
                    current_value REAL,
                    previous_value REAL,
                    baseline_mean REAL,
                    baseline_std REAL,
                    delta_from_baseline REAL,
                    rate_of_change REAL,
                    time_delta_seconds REAL,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    clinical_recommendation TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_trend_patient ON patient_trend_anomalies (patient_id, created_at);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_trend_stay ON patient_trend_anomalies (stay_id, created_at);"
            )

            # 3. Level 3 Historical Population Anomalies
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_population_anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL REFERENCES patients (id) ON DELETE RESTRICT,
                    stay_id INTEGER REFERENCES icu_stays (id) ON DELETE RESTRICT,
                    metric TEXT NOT NULL,
                    canonical_metric TEXT NOT NULL,
                    measured_value REAL NOT NULL,
                    reference_median REAL,
                    reference_iqr REAL,
                    eicu_percentile REAL,
                    robust_z_score REAL,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    clinical_recommendation TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_pop_patient ON historical_population_anomalies (patient_id, created_at);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_pop_stay ON historical_population_anomalies (stay_id, created_at);"
            )

    def evaluate_patient(self, patient_id: int) -> dict[str, Any]:
        """
        Execute full multi-level consistency and anomaly evaluation for a patient's active stay.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._connection() as connection:
            # 1. Resolve patient & active stay
            patient = connection.execute(
                "SELECT * FROM patients WHERE id = ?", (patient_id,)
            ).fetchone()
            if not patient:
                raise ValueError(f"Patient {patient_id} not found.")

            active_stay = connection.execute(
                "SELECT * FROM icu_stays WHERE patient_id = ? AND discharge_status = 'active' ORDER BY id DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
            stay_id = active_stay["id"] if active_stay else None

            # 2. Check active device assignment
            assignment = connection.execute(
                """
                SELECT a.device_id, d.status as device_status
                FROM device_assignments a
                JOIN devices d ON a.device_id = d.device_id
                WHERE a.patient_id = ? AND a.assignment_status = 'active'
                ORDER BY a.id DESC LIMIT 1
                """,
                (patient_id,),
            ).fetchone()
            active_device_id = assignment["device_id"] if assignment else None
            is_device_active = active_device_id is not None

            # 3. Retrieve latest telemetry reading
            latest_telemetry = None
            if stay_id:
                latest_telemetry = connection.execute(
                    """
                    SELECT * FROM telemetry_records
                    WHERE stay_id = ?
                    ORDER BY reading_time DESC LIMIT 1
                    """,
                    (stay_id,),
                ).fetchone()

            # 4. Retrieve recent non-voided clinical records for this stay (last 24 hours)
            clinical_records = []
            if stay_id:
                clinical_records = connection.execute(
                    """
                    SELECT * FROM clinical_records
                    WHERE stay_id = ? AND voided_at IS NULL
                    ORDER BY measurement_time DESC LIMIT 100
                    """,
                    (stay_id,),
                ).fetchall()

            # 5. Retrieve historical telemetry readings for trend analysis (up to 100 points)
            historical_telemetry = []
            if stay_id:
                historical_telemetry = connection.execute(
                    """
                    SELECT reading_time, bpm, spo2, fluid_ml, quality_metadata
                    FROM telemetry_records
                    WHERE stay_id = ?
                    ORDER BY reading_time ASC LIMIT 100
                    """,
                    (stay_id,),
                ).fetchall()

        # ============================================================
        # LEVEL 1: CROSS-SOURCE CONSISTENCY EVALUATION
        # ============================================================
        level1_results: list[DiscrepancyResult] = []

        telem_time = latest_telemetry["reading_time"] if latest_telemetry else None
        telem_bpm = latest_telemetry["bpm"] if latest_telemetry else None
        telem_spo2 = latest_telemetry["spo2"] if latest_telemetry else None
        telem_fluid = latest_telemetry["fluid_ml"] if latest_telemetry else None

        # Mask invalid telemetry values per quality_metadata
        if latest_telemetry and latest_telemetry["quality_metadata"]:
            try:
                qm = json.loads(latest_telemetry["quality_metadata"])
                if isinstance(qm, dict):
                    if not qm.get("bpm"):
                        telem_bpm = None
                    if not qm.get("spo2"):
                        telem_spo2 = None
                    if not qm.get("fluid"):
                        telem_fluid = None
            except Exception:
                pass

        # Helper to find latest clinical record matching candidate names
        def find_latest_record(names: list[str]) -> dict[str, Any] | None:
            names_lower = [n.lower() for n in names]
            for r in clinical_records:
                if r["measurement_name"].strip().lower() in names_lower:
                    try:
                        val = float(json.loads(r["value_json"]))
                        return {
                            "record_id": r["record_id"],
                            "measurement_name": r["measurement_name"],
                            "value": val,
                            "unit": r["unit"],
                            "time": r["measurement_time"],
                            "category": r["category"],
                        }
                    except Exception:
                        continue
            return None

        # (a) Heart Rate
        rec_hr = find_latest_record(["Heart rate", "Pulse", "HR"])
        res_hr = self.consistency_engine.evaluate_measurement_pair(
            canonical_metric="heart_rate",
            telemetry_value=telem_bpm,
            telemetry_time=telem_time,
            clinical_value=rec_hr["value"] if rec_hr else None,
            clinical_time=rec_hr["time"] if rec_hr else None,
            clinical_unit=rec_hr["unit"] if rec_hr else "bpm",
            clinical_record_id=rec_hr["record_id"] if rec_hr else None,
            clinical_measurement_name=rec_hr["measurement_name"] if rec_hr else "Heart rate",
        )
        level1_results.append(res_hr)

        # (b) SpO2
        rec_spo2 = find_latest_record(["Oxygen saturation", "SpO2", "SaO2"])
        res_spo2 = self.consistency_engine.evaluate_measurement_pair(
            canonical_metric="spo2",
            telemetry_value=telem_spo2,
            telemetry_time=telem_time,
            clinical_value=rec_spo2["value"] if rec_spo2 else None,
            clinical_time=rec_spo2["time"] if rec_spo2 else None,
            clinical_unit=rec_spo2["unit"] if rec_spo2 else "%",
            clinical_record_id=rec_spo2["record_id"] if rec_spo2 else None,
            clinical_measurement_name=rec_spo2["measurement_name"] if rec_spo2 else "Oxygen saturation",
        )
        level1_results.append(res_spo2)

        # (c) Fluid Volume (Intake vs Container Telemetry)
        rec_fluid = find_latest_record(["IV fluid infusion", "Intake volume", "IV Fluid"])
        res_fluid = self.consistency_engine.evaluate_measurement_pair(
            canonical_metric="fluid",
            telemetry_value=telem_fluid,
            telemetry_time=telem_time,
            clinical_value=rec_fluid["value"] if rec_fluid else None,
            clinical_time=rec_fluid["time"] if rec_fluid else None,
            clinical_unit=rec_fluid["unit"] if rec_fluid else "mL",
            clinical_record_id=rec_fluid["record_id"] if rec_fluid else None,
            clinical_measurement_name=rec_fluid["measurement_name"] if rec_fluid else "IV fluid infusion",
        )
        level1_results.append(res_fluid)

        # ============================================================
        # LEVEL 2: PATIENT-SPECIFIC HISTORICAL TREND ANALYSIS
        # ============================================================
        level2_results: list[TrendAnalysisResult] = []

        # 1. Stale telemetry check
        stale_res = self.trend_analyzer.check_stale_telemetry(
            latest_reading_time=telem_time,
            is_device_active=is_device_active,
        )
        if stale_res:
            level2_results.append(stale_res)

        # 2. Heart rate trajectory
        hr_series: list[tuple[datetime | str, float]] = []
        for row in historical_telemetry:
            if row["bpm"] is not None:
                hr_series.append((row["reading_time"], float(row["bpm"])))
        for r in clinical_records:
            if r["measurement_name"].strip().lower() in ["heart rate", "pulse"]:
                try:
                    hr_series.append((r["measurement_time"], float(json.loads(r["value_json"]))))
                except Exception:
                    pass
        res_trend_hr = self.trend_analyzer.analyze_trajectory("Heart Rate", hr_series)
        level2_results.append(res_trend_hr)

        # 3. SpO2 trajectory
        spo2_series: list[tuple[datetime | str, float]] = []
        for row in historical_telemetry:
            if row["spo2"] is not None:
                spo2_series.append((row["reading_time"], float(row["spo2"])))
        for r in clinical_records:
            if r["measurement_name"].strip().lower() in ["oxygen saturation", "spo2"]:
                try:
                    spo2_series.append((r["measurement_time"], float(json.loads(r["value_json"]))))
                except Exception:
                    pass
        res_trend_spo2 = self.trend_analyzer.analyze_trajectory("Oxygen Saturation", spo2_series)
        level2_results.append(res_trend_spo2)

        # 4. Fluid trajectory
        fluid_series: list[tuple[datetime | str, float]] = []
        for row in historical_telemetry:
            if row["fluid_ml"] is not None:
                fluid_series.append((row["reading_time"], float(row["fluid_ml"])))
        res_trend_fluid = self.trend_analyzer.analyze_trajectory("IV Fluid Volume", fluid_series)
        level2_results.append(res_trend_fluid)

        # ============================================================
        # LEVEL 3: HISTORICAL eICU POPULATION-LEVEL COMPARISON
        # ============================================================
        level3_results: list[PopulationComparisonResult] = []

        # Compare available vitals / latest records
        metrics_to_eval = [
            ("Heart rate", telem_bpm if telem_bpm is not None else (rec_hr["value"] if rec_hr else None)),
            ("Oxygen saturation", telem_spo2 if telem_spo2 is not None else (rec_spo2["value"] if rec_spo2 else None)),
        ]

        # Add clinical observation vitals if present
        rec_sbp = find_latest_record(["Systolic blood pressure", "SBP"])
        if rec_sbp:
            metrics_to_eval.append(("Systolic blood pressure", rec_sbp["value"]))
        rec_dbp = find_latest_record(["Diastolic blood pressure", "DBP"])
        if rec_dbp:
            metrics_to_eval.append(("Diastolic blood pressure", rec_dbp["value"]))
        rec_resp = find_latest_record(["Respiratory rate", "Respiration"])
        if rec_resp:
            metrics_to_eval.append(("Respiratory rate", rec_resp["value"]))

        # Add lab values if present
        for lab_name in ["Blood glucose", "Creatinine", "BUN", "Hemoglobin", "Potassium", "Sodium", "Urine output"]:
            rec_lab = find_latest_record([lab_name])
            if rec_lab:
                metrics_to_eval.append((lab_name, rec_lab["value"]))

        for m_name, m_val in metrics_to_eval:
            if m_val is not None:
                pop_res = self.reference_engine.evaluate_value(m_name, float(m_val))
                if pop_res:
                    level3_results.append(pop_res)

        # Evaluate Isolation Forest
        if_features: dict[str, float | None] = {
            "heart_rate": telem_bpm if telem_bpm is not None else (rec_hr["value"] if rec_hr else None),
            "spo2": telem_spo2 if telem_spo2 is not None else (rec_spo2["value"] if rec_spo2 else None),
            "respiration": rec_resp["value"] if rec_resp else None,
            "systolic_bp": rec_sbp["value"] if rec_sbp else None,
            "diastolic_bp": rec_dbp["value"] if rec_dbp else None,
            "mean_bp": (rec_sbp["value"] + 2 * rec_dbp["value"]) / 3 if (rec_sbp and rec_dbp) else None,
        }
        if_result = self.reference_engine.evaluate_isolation_forest(if_features)

        # ============================================================
        # PERSISTENCE & ALERT DISPATCH
        # ============================================================
        created_alerts: list[dict[str, Any]] = []

        with self._connection() as connection:
            # 1. Persist Level 1 Checks
            for res in level1_results:
                connection.execute(
                    """
                    INSERT INTO data_consistency_checks (
                        patient_id, stay_id, metric, canonical_metric, telemetry_value,
                        telemetry_time, clinical_value, clinical_time, clinical_record_id,
                        discrepancy_level, absolute_difference, relative_difference_pct,
                        time_difference_seconds, explanation, clinical_rationale,
                        recommended_action, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        patient_id,
                        stay_id,
                        res.metric,
                        res.canonical_metric,
                        res.telemetry_value,
                        res.telemetry_time,
                        res.clinical_value,
                        res.clinical_time,
                        res.clinical_record_id,
                        res.status,
                        res.absolute_difference,
                        res.relative_difference_pct,
                        res.time_difference_seconds,
                        res.explanation,
                        res.clinical_rationale,
                        res.recommended_action,
                        now_iso,
                    ),
                )

            # 2. Persist Level 2 Anomalies
            for tr in level2_results:
                if tr.is_anomaly:
                    connection.execute(
                        """
                        INSERT INTO patient_trend_anomalies (
                            patient_id, stay_id, metric, canonical_metric, current_value,
                            previous_value, baseline_mean, baseline_std, delta_from_baseline,
                            rate_of_change, time_delta_seconds, anomaly_type, severity,
                            explanation, clinical_recommendation, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            patient_id,
                            stay_id,
                            tr.metric,
                            tr.canonical_metric,
                            tr.current_value,
                            tr.previous_value,
                            tr.baseline_mean,
                            tr.baseline_std,
                            tr.delta_from_baseline,
                            tr.rate_of_change_per_minute,
                            tr.time_delta_seconds,
                            tr.status,
                            tr.severity,
                            tr.explanation,
                            tr.clinical_recommendation,
                            now_iso,
                        ),
                    )

            # 3. Persist Level 3 Anomalies
            for pop in level3_results:
                if pop.is_anomaly:
                    connection.execute(
                        """
                        INSERT INTO historical_population_anomalies (
                            patient_id, stay_id, metric, canonical_metric, measured_value,
                            reference_median, reference_iqr, eicu_percentile, robust_z_score,
                            anomaly_type, severity, explanation, clinical_recommendation, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            patient_id,
                            stay_id,
                            pop.metric,
                            pop.canonical_metric,
                            pop.measured_value,
                            pop.reference_median,
                            pop.reference_iqr,
                            pop.percentile_rank,
                            pop.robust_z_score,
                            pop.anomaly_type,
                            pop.severity,
                            pop.explanation,
                            pop.clinical_validation_recommendation,
                            now_iso,
                        ),
                    )

        # 4. Dispatch and resolve alerts outside the SQLite insert transaction
        for res in level1_results:
            if res.requires_alert:
                alt = self.alert_manager.process_consistency_alert(
                    patient_id=patient_id,
                    stay_id=stay_id,
                    source="CROSS_SOURCE_DISCREPANCY",
                    metric_name=res.metric,
                    measured_value=res.absolute_difference,
                    unit=self.consistency_engine.get_tolerance(res.canonical_metric).telemetry_unit if self.consistency_engine.get_tolerance(res.canonical_metric) else None,
                    severity=res.alert_severity,
                    category="consistency",
                    explanation=res.explanation,
                    recommendation=res.recommended_action,
                    trigger_time=res.telemetry_time or now_iso,
                    record_id=res.clinical_record_id,
                )
                if alt:
                    created_alerts.append(alt)
            elif res.status == "CONSISTENT":
                self.alert_manager.auto_resolve_consistency_alert(
                    patient_id=patient_id,
                    stay_id=stay_id,
                    source="CROSS_SOURCE_DISCREPANCY",
                    metric_name=res.metric,
                    resolution_reason=f"Subsequent cross-source measurement verified consistent: delta {res.absolute_difference:g}.",
                )

        for tr in level2_results:
            if tr.is_anomaly:
                alt = self.alert_manager.process_consistency_alert(
                    patient_id=patient_id,
                    stay_id=stay_id,
                    source="PATIENT_TREND_ANOMALY",
                    metric_name=tr.metric,
                    measured_value=tr.current_value,
                    unit="",
                    severity=tr.severity,
                    category="trend",
                    explanation=tr.explanation,
                    recommendation=tr.clinical_recommendation,
                    trigger_time=tr.current_timestamp or now_iso,
                )
                if alt:
                    created_alerts.append(alt)
            elif tr.status == "NORMAL_TREND":
                self.alert_manager.auto_resolve_consistency_alert(
                    patient_id=patient_id,
                    stay_id=stay_id,
                    source="PATIENT_TREND_ANOMALY",
                    metric_name=tr.metric,
                    resolution_reason=f"Subsequent readings returned to patient-specific baseline (mean: {tr.baseline_mean:g}).",
                )

        for pop in level3_results:
            if pop.is_anomaly:
                alt = self.alert_manager.process_consistency_alert(
                    patient_id=patient_id,
                    stay_id=stay_id,
                    source="HISTORICAL_POPULATION_ANOMALY",
                    metric_name=pop.metric,
                    measured_value=pop.measured_value,
                    unit="",
                    severity=pop.severity,
                    category="population_reference",
                    explanation=pop.explanation,
                    recommendation=pop.clinical_validation_recommendation,
                    trigger_time=now_iso,
                )
                if alt:
                    created_alerts.append(alt)
            elif pop.severity == "NORMAL":
                self.alert_manager.auto_resolve_consistency_alert(
                    patient_id=patient_id,
                    stay_id=stay_id,
                    source="HISTORICAL_POPULATION_ANOMALY",
                    metric_name=pop.metric,
                    resolution_reason=f"Measurement within eICU population normal range (z-score: {pop.robust_z_score:+.2f}).",
                )

        # Return full structured intelligence response
        return {
            "patient_id": patient_id,
            "stay_id": stay_id,
            "evaluated_at": now_iso,
            "level1_cross_source": [r.model_dump() for r in level1_results],
            "level2_patient_trends": [t.model_dump() for t in level2_results],
            "level3_population_comparisons": [p.model_dump() for p in level3_results],
            "isolation_forest_result": if_result.model_dump(),
            "active_alerts": created_alerts,
            "summary": {
                "level1_discrepant_count": sum(1 for r in level1_results if r.status in ("MINOR_DISCREPANCY", "SIGNIFICANT_DISCREPANCY", "CRITICAL_DISCREPANCY")),
                "level2_anomaly_count": sum(1 for t in level2_results if t.is_anomaly),
                "level3_anomaly_count": sum(1 for p in level3_results if p.is_anomaly),
                "stale_telemetry": any(t.status == "STALE_TELEMETRY" for t in level2_results),
            },
        }

    def get_patient_consistency_history(
        self,
        patient_id: int,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        Retrieve historical checks and anomaly logs for a patient.
        """
        with self._connection() as connection:
            checks = connection.execute(
                """
                SELECT * FROM data_consistency_checks
                WHERE patient_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (patient_id, limit),
            ).fetchall()

            trends = connection.execute(
                """
                SELECT * FROM patient_trend_anomalies
                WHERE patient_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (patient_id, limit),
            ).fetchall()

            population = connection.execute(
                """
                SELECT * FROM historical_population_anomalies
                WHERE patient_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (patient_id, limit),
            ).fetchall()

        return {
            "patient_id": patient_id,
            "consistency_checks": [dict(r) for r in checks],
            "trend_anomalies": [dict(r) for r in trends],
            "population_anomalies": [dict(r) for r in population],
        }

    def get_status(self) -> dict[str, Any]:
        """
        Global status of the consistency & intelligence engine.
        """
        ref_meta = self.reference_engine.get_metadata()
        with self._connection() as connection:
            total_checks = connection.execute("SELECT COUNT(*) FROM data_consistency_checks").fetchone()[0]
            total_trend_anomalies = connection.execute("SELECT COUNT(*) FROM patient_trend_anomalies").fetchone()[0]
            total_pop_anomalies = connection.execute("SELECT COUNT(*) FROM historical_population_anomalies").fetchone()[0]
            active_consistency_alerts = connection.execute(
                """
                SELECT COUNT(*) FROM clinical_alerts
                WHERE status IN ('ACTIVE', 'ACKNOWLEDGED')
                  AND source IN ('CROSS_SOURCE_DISCREPANCY', 'PATIENT_TREND_ANOMALY', 'HISTORICAL_POPULATION_ANOMALY')
                """
            ).fetchone()[0]

        return {
            "engine_status": "ONLINE",
            "version": "1.0.0",
            "reference_population": ref_meta,
            "tolerances": {k: v.model_dump() for k, v in self.consistency_engine.tolerances.items()},
            "statistics": {
                "total_consistency_checks": total_checks,
                "total_trend_anomalies": total_trend_anomalies,
                "total_population_anomalies": total_pop_anomalies,
                "active_consistency_alerts": active_consistency_alerts,
            },
        }
