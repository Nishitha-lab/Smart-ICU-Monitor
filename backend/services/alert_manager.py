"""
Unified Clinical Alert Manager for Smart ICU Patient Safety System.

Manages alert generation, lifecycle (ACTIVE, ACKNOWLEDGED, RESOLVED, SUPERSEDED),
deduplication via fingerprinting, escalation, and comprehensive audit history.
Integrates with clinical records, laboratory interpretation, and ESP32 telemetry.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


AlertStatus = Literal["ACTIVE", "ACKNOWLEDGED", "RESOLVED", "SUPERSEDED"]
AlertSeverity = Literal["INFO", "LOW", "MODERATE", "HIGH", "CRITICAL"]
AlertSource = Literal[
    "ESP32_TELEMETRY",
    "CLINICAL_OBSERVATION",
    "LAB_RESULT",
    "HISTORICAL_EICU",
    "ML_ANOMALY",
    "ML_RISK",
    "CROSS_SOURCE_DISCREPANCY",
    "PATIENT_TREND_ANOMALY",
    "HISTORICAL_POPULATION_ANOMALY",
]

SEVERITY_WEIGHT: dict[str, int] = {
    "INFO": 0,
    "LOW": 1,
    "MODERATE": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


class AlertAcknowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    acknowledged_by: str | None = Field(default=None, max_length=120)
    clinician_name: str | None = Field(default=None, max_length=120)
    note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def normalize_actor(self) -> AlertAcknowledgeRequest:
        actor = (self.acknowledged_by or self.clinician_name or "").strip()
        if not actor:
            raise ValueError("Either 'acknowledged_by' or 'clinician_name' must be provided.")
        self.acknowledged_by = actor
        return self


class AlertResolveRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    resolved_by: str | None = Field(default=None, max_length=120)
    clinician_name: str | None = Field(default=None, max_length=120)
    note: str | None = Field(default=None, max_length=500)
    resolution_note: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def normalize_actor_and_note(self) -> AlertResolveRequest:
        actor = (self.resolved_by or self.clinician_name or "").strip()
        if not actor:
            raise ValueError("Either 'resolved_by' or 'clinician_name' must be provided.")
        self.resolved_by = actor
        note = (self.note or self.resolution_note or "").strip()
        self.note = note or None
        return self


class AlertManager:
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

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS clinical_alerts (
                    alert_id TEXT PRIMARY KEY,
                    patient_id INTEGER,
                    stay_id INTEGER,
                    patientunitstayid INTEGER,
                    source TEXT NOT NULL,
                    measurement_name TEXT NOT NULL,
                    measured_value TEXT NOT NULL,
                    unit TEXT,
                    reference_range TEXT,
                    critical_threshold TEXT,
                    severity TEXT NOT NULL,
                    category TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    trigger_time TEXT NOT NULL,
                    record_id TEXT,
                    rule_version TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    acknowledged_at TEXT,
                    acknowledged_by TEXT,
                    acknowledgement_note TEXT,
                    resolved_at TEXT,
                    resolved_by TEXT,
                    resolution_note TEXT,
                    superseded_by TEXT,
                    fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_patient ON clinical_alerts (patient_id);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_stay ON clinical_alerts (stay_id);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_status ON clinical_alerts (status);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON clinical_alerts (severity);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_fingerprint ON clinical_alerts (fingerprint);"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_record ON clinical_alerts (record_id);"
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL REFERENCES clinical_alerts(alert_id) ON DELETE CASCADE,
                    action TEXT NOT NULL,
                    performed_by TEXT,
                    note TEXT,
                    timestamp TEXT NOT NULL,
                    details_json TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_alert_id ON alert_audit_logs (alert_id);"
            )

    def compute_fingerprint(
        self,
        patient_ref: str | int,
        source: str,
        canonical_test_or_metric: str,
    ) -> str:
        raw = f"{patient_ref}:{source.upper()}:{canonical_test_or_metric.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def process_clinical_interpretation(
        self,
        interpretation: Any,
        patient_id: int | None,
        stay_id: int | None,
        patientunitstayid: int | None,
        source: AlertSource,
        record_id: str | None = None,
        trigger_time: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Evaluate interpretation and create, escalate, or auto-resolve active alerts.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        trig_iso = trigger_time or now_iso

        patient_ref = str(stay_id or patientunitstayid or patient_id or "global")
        fingerprint = self.compute_fingerprint(
            patient_ref, source, interpretation.canonical_test_id or interpretation.measurement_name
        )

        with self._connection() as connection:
            # If result does NOT require an alert (i.e. NORMAL or UNINTERPRETABLE without alert),
            # check if an existing ACTIVE or ACKNOWLEDGED alert should be auto-resolved.
            if not interpretation.requires_alert or interpretation.status == "NORMAL":
                active = connection.execute(
                    """
                    SELECT alert_id, severity FROM clinical_alerts
                    WHERE fingerprint = ? AND status IN ('ACTIVE', 'ACKNOWLEDGED')
                    """,
                    (fingerprint,),
                ).fetchall()
                for row in active:
                    aid = row["alert_id"]
                    connection.execute(
                        """
                        UPDATE clinical_alerts
                        SET status = 'RESOLVED', resolved_at = ?, resolved_by = 'SYSTEM_AUTO_RESOLVE',
                            resolution_note = ?, updated_at = ?
                        WHERE alert_id = ?
                        """,
                        (
                            now_iso,
                            f"Auto-resolved: subsequent measurement was normal ({interpretation.measured_value} {interpretation.unit or ''}).",
                            now_iso,
                            aid,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO alert_audit_logs (alert_id, action, performed_by, note, timestamp, details_json)
                        VALUES (?, 'RESOLVED', 'SYSTEM_AUTO_RESOLVE', ?, ?, ?)
                        """,
                        (
                            aid,
                            "Auto-resolved due to subsequent normal measurement.",
                            now_iso,
                            json.dumps({"subsequent_value": interpretation.measured_value}),
                        ),
                    )
                return None

            # Result REQUIRES an alert
            severity = interpretation.severity or "MODERATE"
            new_weight = SEVERITY_WEIGHT.get(severity, 2)

            existing = connection.execute(
                """
                SELECT * FROM clinical_alerts
                WHERE fingerprint = ? AND status IN ('ACTIVE', 'ACKNOWLEDGED')
                ORDER BY created_at DESC LIMIT 1
                """,
                (fingerprint,),
            ).fetchone()

            if existing:
                existing_weight = SEVERITY_WEIGHT.get(existing["severity"], 2)
                alert_id = existing["alert_id"]

                # Escalation
                if new_weight > existing_weight:
                    connection.execute(
                        """
                        UPDATE clinical_alerts
                        SET severity = ?, measured_value = ?, unit = ?, reference_range = ?,
                            critical_threshold = ?, explanation = ?, trigger_time = ?,
                            record_id = ?, status = 'ACTIVE', updated_at = ?
                        WHERE alert_id = ?
                        """,
                        (
                            severity,
                            str(interpretation.measured_value),
                            interpretation.unit,
                            interpretation.reference_range,
                            interpretation.critical_threshold,
                            interpretation.explanation,
                            trig_iso,
                            record_id,
                            now_iso,
                            alert_id,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO alert_audit_logs (alert_id, action, performed_by, note, timestamp, details_json)
                        VALUES (?, 'ESCALATED', 'SYSTEM_RULES_ENGINE', ?, ?, ?)
                        """,
                        (
                            alert_id,
                            f"Severity escalated from {existing['severity']} to {severity}.",
                            now_iso,
                            json.dumps(
                                {
                                    "previous_severity": existing["severity"],
                                    "new_severity": severity,
                                    "measured_value": interpretation.measured_value,
                                }
                            ),
                        ),
                    )
                else:
                    # Update without duplicate spamming
                    connection.execute(
                        """
                        UPDATE clinical_alerts
                        SET measured_value = ?, explanation = ?, trigger_time = ?,
                            record_id = ?, updated_at = ?
                        WHERE alert_id = ?
                        """,
                        (
                            str(interpretation.measured_value),
                            interpretation.explanation,
                            trig_iso,
                            record_id,
                            now_iso,
                            alert_id,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO alert_audit_logs (alert_id, action, performed_by, note, timestamp, details_json)
                        VALUES (?, 'RE_EVALUATED', 'SYSTEM_RULES_ENGINE', ?, ?, ?)
                        """,
                        (
                            alert_id,
                            "Repeated abnormal measurement recorded; alert updated.",
                            now_iso,
                            json.dumps({"measured_value": interpretation.measured_value}),
                        ),
                    )

                updated = connection.execute(
                    "SELECT * FROM clinical_alerts WHERE alert_id = ?", (alert_id,)
                ).fetchone()
                return dict(updated)

            # Insert new Alert
            new_alert_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO clinical_alerts (
                    alert_id, patient_id, stay_id, patientunitstayid, source,
                    measurement_name, measured_value, unit, reference_range,
                    critical_threshold, severity, category, explanation,
                    trigger_time, record_id, rule_version, status, fingerprint,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?)
                """,
                (
                    new_alert_id,
                    patient_id,
                    stay_id,
                    patientunitstayid,
                    source,
                    interpretation.measurement_name,
                    str(interpretation.measured_value),
                    interpretation.unit,
                    interpretation.reference_range,
                    interpretation.critical_threshold,
                    severity,
                    interpretation.category,
                    interpretation.explanation,
                    trig_iso,
                    record_id,
                    interpretation.evaluation_version,
                    fingerprint,
                    now_iso,
                    now_iso,
                ),
            )
            connection.execute(
                """
                INSERT INTO alert_audit_logs (alert_id, action, performed_by, note, timestamp, details_json)
                VALUES (?, 'CREATED', 'SYSTEM_RULES_ENGINE', 'Alert generated by clinical rules engine.', ?, ?)
                """,
                (
                    new_alert_id,
                    now_iso,
                    json.dumps(
                        {
                            "source": source,
                            "severity": severity,
                            "measurement": interpretation.measurement_name,
                            "value": interpretation.measured_value,
                        }
                    ),
                ),
            )
            row = connection.execute(
                "SELECT * FROM clinical_alerts WHERE alert_id = ?", (new_alert_id,)
            ).fetchone()
            return dict(row)

    def process_consistency_alert(
        self,
        patient_id: int,
        stay_id: int | None,
        source: AlertSource,
        metric_name: str,
        measured_value: Any,
        unit: str | None,
        severity: AlertSeverity,
        category: str,
        explanation: str,
        recommendation: str | None = None,
        trigger_time: str | None = None,
        record_id: str | None = None,
        reference_range: str | None = None,
        critical_threshold: str | None = None,
        rule_version: str = "1.0.0",
    ) -> dict[str, Any] | None:
        """
        Create, escalate, or deduplicate consistency and anomaly alerts.
        Deduplicates via fingerprinting without creating duplicate active alerts on repeated polling.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        trig_iso = trigger_time or now_iso
        patient_ref = str(stay_id or patient_id)
        fingerprint = self.compute_fingerprint(patient_ref, source, metric_name)

        new_weight = SEVERITY_WEIGHT.get(severity, 2)

        with self._connection() as connection:
            existing = connection.execute(
                """
                SELECT * FROM clinical_alerts
                WHERE fingerprint = ? AND status IN ('ACTIVE', 'ACKNOWLEDGED')
                ORDER BY created_at DESC LIMIT 1
                """,
                (fingerprint,),
            ).fetchone()

            if existing:
                existing_weight = SEVERITY_WEIGHT.get(existing["severity"], 2)
                alert_id = existing["alert_id"]

                # Escalation if severity strictly increased
                if new_weight > existing_weight:
                    connection.execute(
                        """
                        UPDATE clinical_alerts
                        SET severity = ?, measured_value = ?, unit = ?, reference_range = ?,
                            critical_threshold = ?, explanation = ?, trigger_time = ?,
                            record_id = ?, status = 'ACTIVE', updated_at = ?
                        WHERE alert_id = ?
                        """,
                        (
                            severity,
                            str(measured_value),
                            unit,
                            reference_range,
                            critical_threshold,
                            explanation,
                            trig_iso,
                            record_id,
                            now_iso,
                            alert_id,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO alert_audit_logs (alert_id, action, performed_by, note, timestamp, details_json)
                        VALUES (?, 'ESCALATED', 'SYSTEM_CONSISTENCY_ENGINE', ?, ?, ?)
                        """,
                        (
                            alert_id,
                            f"Alert escalated from {existing['severity']} to {severity}.",
                            now_iso,
                            json.dumps(
                                {
                                    "old_severity": existing["severity"],
                                    "new_severity": severity,
                                    "measured_value": measured_value,
                                    "reason": "Discrepancy or anomaly severity worsened.",
                                }
                            ),
                        ),
                    )
                    row = connection.execute(
                        "SELECT * FROM clinical_alerts WHERE alert_id = ?", (alert_id,)
                    ).fetchone()
                    return dict(row)

                # Deduplicated: active alert already exists with >= severity
                return dict(existing)

            # Create brand-new alert
            new_alert_id = f"ALT-{uuid.uuid4().hex[:12].upper()}"
            connection.execute(
                """
                INSERT INTO clinical_alerts (
                    alert_id, patient_id, stay_id, patientunitstayid, source,
                    measurement_name, measured_value, unit, reference_range,
                    critical_threshold, severity, category, explanation,
                    trigger_time, record_id, rule_version, status, fingerprint,
                    created_at, updated_at
                ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?)
                """,
                (
                    new_alert_id,
                    patient_id,
                    stay_id,
                    source,
                    metric_name,
                    str(measured_value),
                    unit,
                    reference_range,
                    critical_threshold,
                    severity,
                    category,
                    explanation,
                    trig_iso,
                    record_id,
                    rule_version,
                    fingerprint,
                    now_iso,
                    now_iso,
                ),
            )
            connection.execute(
                """
                INSERT INTO alert_audit_logs (alert_id, action, performed_by, note, timestamp, details_json)
                VALUES (?, 'CREATED', 'SYSTEM_CONSISTENCY_ENGINE', 'Alert generated by data consistency engine.', ?, ?)
                """,
                (
                    new_alert_id,
                    now_iso,
                    json.dumps(
                        {
                            "source": source,
                            "severity": severity,
                            "metric": metric_name,
                            "value": measured_value,
                            "recommendation": recommendation,
                        }
                    ),
                ),
            )
            row = connection.execute(
                "SELECT * FROM clinical_alerts WHERE alert_id = ?", (new_alert_id,)
            ).fetchone()
            return dict(row)

    def auto_resolve_consistency_alert(
        self,
        patient_id: int,
        stay_id: int | None,
        source: AlertSource,
        metric_name: str,
        resolution_reason: str,
    ) -> None:
        """
        Auto-resolves any active consistency or anomaly alert when values become normal or consistent.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        patient_ref = str(stay_id or patient_id)
        fingerprint = self.compute_fingerprint(patient_ref, source, metric_name)

        with self._connection() as connection:
            active = connection.execute(
                """
                SELECT alert_id FROM clinical_alerts
                WHERE fingerprint = ? AND status IN ('ACTIVE', 'ACKNOWLEDGED')
                """,
                (fingerprint,),
            ).fetchall()

            for row in active:
                aid = row["alert_id"]
                connection.execute(
                    """
                    UPDATE clinical_alerts
                    SET status = 'RESOLVED', resolved_at = ?, resolved_by = 'SYSTEM_AUTO_RESOLVE',
                        resolution_note = ?, updated_at = ?
                    WHERE alert_id = ?
                    """,
                    (now_iso, resolution_reason, now_iso, aid),
                )
                connection.execute(
                    """
                    INSERT INTO alert_audit_logs (alert_id, action, performed_by, note, timestamp, details_json)
                    VALUES (?, 'RESOLVED', 'SYSTEM_AUTO_RESOLVE', ?, ?, ?)
                    """,
                    (aid, resolution_reason, now_iso, json.dumps({"reason": resolution_reason})),
                )

    def supersede_alerts_for_record(self, record_id: str, reason: str, voided_by: str = "SYSTEM") -> int:
        """
        Mark all active/acknowledged alerts linked to this record_id as SUPERSEDED.
        Preserves history and audit trails.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT alert_id FROM clinical_alerts WHERE record_id = ? AND status IN ('ACTIVE', 'ACKNOWLEDGED')",
                (record_id,),
            ).fetchall()
            count = 0
            for row in rows:
                aid = row["alert_id"]
                connection.execute(
                    """
                    UPDATE clinical_alerts
                    SET status = 'SUPERSEDED', updated_at = ?
                    WHERE alert_id = ?
                    """,
                    (now_iso, aid),
                )
                connection.execute(
                    """
                    INSERT INTO alert_audit_logs (alert_id, action, performed_by, note, timestamp, details_json)
                    VALUES (?, 'SUPERSEDED', ?, ?, ?, ?)
                    """,
                    (aid, voided_by, reason, now_iso, json.dumps({"record_id": record_id})),
                )
                count += 1
            return count

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        note: str | None = None,
    ) -> dict[str, Any] | None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            alert = connection.execute(
                "SELECT * FROM clinical_alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()
            if not alert:
                return None
            if alert["status"] not in ("ACTIVE", "ACKNOWLEDGED"):
                raise ValueError(f"Cannot acknowledge alert with status '{alert['status']}'.")

            connection.execute(
                """
                UPDATE clinical_alerts
                SET status = 'ACKNOWLEDGED', acknowledged_at = ?, acknowledged_by = ?,
                    acknowledgement_note = ?, updated_at = ?
                WHERE alert_id = ?
                """,
                (now_iso, acknowledged_by.strip(), (note or "").strip() or None, now_iso, alert_id),
            )
            connection.execute(
                """
                INSERT INTO alert_audit_logs (alert_id, action, performed_by, note, timestamp, details_json)
                VALUES (?, 'ACKNOWLEDGED', ?, ?, ?, ?)
                """,
                (
                    alert_id,
                    acknowledged_by.strip(),
                    note or "Alert acknowledged by clinician.",
                    now_iso,
                    json.dumps({"acknowledged_by": acknowledged_by}),
                ),
            )
            updated = connection.execute(
                "SELECT * FROM clinical_alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()
            return dict(updated)

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        note: str | None = None,
    ) -> dict[str, Any] | None:
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            alert = connection.execute(
                "SELECT * FROM clinical_alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()
            if not alert:
                return None
            if alert["status"] in ("RESOLVED", "SUPERSEDED"):
                return dict(alert)

            connection.execute(
                """
                UPDATE clinical_alerts
                SET status = 'RESOLVED', resolved_at = ?, resolved_by = ?,
                    resolution_note = ?, updated_at = ?
                WHERE alert_id = ?
                """,
                (now_iso, resolved_by.strip(), (note or "").strip() or None, now_iso, alert_id),
            )
            connection.execute(
                """
                INSERT INTO alert_audit_logs (alert_id, action, performed_by, note, timestamp, details_json)
                VALUES (?, 'RESOLVED', ?, ?, ?, ?)
                """,
                (
                    alert_id,
                    resolved_by.strip(),
                    note or "Alert resolved by clinician.",
                    now_iso,
                    json.dumps({"resolved_by": resolved_by}),
                ),
            )
            updated = connection.execute(
                "SELECT * FROM clinical_alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()
            return dict(updated)

    def get_alert(self, alert_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM clinical_alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_alert_history(self, alert_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM alert_audit_logs WHERE alert_id = ? ORDER BY id ASC",
                (alert_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def list_alerts(
        self,
        patient_id: int | None = None,
        stay_id: int | None = None,
        patientunitstayid: int | None = None,
        severity: str | None = None,
        source: str | None = None,
        status: str | None = None,
        category: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        with self._connection() as connection:
            clauses: list[str] = []
            params: list[Any] = []

            if patient_id is not None:
                clauses.append("patient_id = ?")
                params.append(patient_id)
            if stay_id is not None:
                clauses.append("stay_id = ?")
                params.append(stay_id)
            if patientunitstayid is not None:
                clauses.append("patientunitstayid = ?")
                params.append(patientunitstayid)
            if severity:
                clauses.append("LOWER(severity) = LOWER(?)")
                params.append(severity.strip())
            if source:
                clauses.append("LOWER(source) = LOWER(?)")
                params.append(source.strip())
            if status:
                if status.lower() != "all":
                    clauses.append("LOWER(status) = LOWER(?)")
                    params.append(status.strip())
            else:
                clauses.append("status IN ('ACTIVE', 'ACKNOWLEDGED')")

            if category:
                clauses.append("LOWER(category) = LOWER(?)")
                params.append(category.strip())

            where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""

            # Custom sort: critical first, then high, then moderate, then low, then info
            order_by = """
                ORDER BY
                    CASE status
                        WHEN 'ACTIVE' THEN 1
                        WHEN 'ACKNOWLEDGED' THEN 2
                        WHEN 'RESOLVED' THEN 3
                        WHEN 'SUPERSEDED' THEN 4
                        ELSE 5
                    END ASC,
                    CASE severity
                        WHEN 'CRITICAL' THEN 1
                        WHEN 'HIGH' THEN 2
                        WHEN 'MODERATE' THEN 3
                        WHEN 'LOW' THEN 4
                        WHEN 'INFO' THEN 5
                        ELSE 6
                    END ASC,
                    trigger_time DESC
            """
            sql = f"SELECT * FROM clinical_alerts {where_str} {order_by} LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = connection.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def get_summary(self) -> dict[str, Any]:
        with self._connection() as connection:
            active_count = connection.execute(
                "SELECT COUNT(*) FROM clinical_alerts WHERE status = 'ACTIVE'"
            ).fetchone()[0]
            critical_count = connection.execute(
                "SELECT COUNT(*) FROM clinical_alerts WHERE status = 'ACTIVE' AND severity = 'CRITICAL'"
            ).fetchone()[0]
            high_count = connection.execute(
                "SELECT COUNT(*) FROM clinical_alerts WHERE status = 'ACTIVE' AND severity = 'HIGH'"
            ).fetchone()[0]
            unack_count = connection.execute(
                "SELECT COUNT(*) FROM clinical_alerts WHERE status = 'ACTIVE' AND acknowledged_at IS NULL"
            ).fetchone()[0]
            resolved_count = connection.execute(
                "SELECT COUNT(*) FROM clinical_alerts WHERE status = 'RESOLVED'"
            ).fetchone()[0]
            total_alerts = connection.execute(
                "SELECT COUNT(*) FROM clinical_alerts"
            ).fetchone()[0]

            by_source_rows = connection.execute(
                "SELECT source, COUNT(*) as cnt FROM clinical_alerts WHERE status IN ('ACTIVE', 'ACKNOWLEDGED') GROUP BY source"
            ).fetchall()
            by_source = {r["source"]: r["cnt"] for r in by_source_rows}

            by_severity_rows = connection.execute(
                "SELECT severity, COUNT(*) as cnt FROM clinical_alerts WHERE status IN ('ACTIVE', 'ACKNOWLEDGED') GROUP BY severity"
            ).fetchall()
            by_severity = {r["severity"].lower(): r["cnt"] for r in by_severity_rows}

            return {
                "active_alerts": active_count,
                "critical_alerts": critical_count,
                "high_priority_alerts": high_count,
                "unacknowledged_alerts": unack_count,
                "resolved_alerts": resolved_count,
                "total_alerts": total_alerts,
                "by_source": by_source,
                "by_severity": by_severity,
            }
