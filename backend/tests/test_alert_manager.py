"""Tests for Alert Manager lifecycle, deduplication, escalation, audit logs, and record sync."""

import pytest
from pathlib import Path

from services.alert_manager import (
    AlertManager,
    AlertAcknowledgeRequest,
    AlertResolveRequest,
)
from services.clinical_alert_engine import ClinicalInterpretation


@pytest.fixture
def alert_mgr(tmp_path: Path) -> AlertManager:
    db = tmp_path / "test_alerts.sqlite3"
    mgr = AlertManager(db)
    mgr.initialize()
    return mgr


def test_alert_creation_and_retrieval(alert_mgr: AlertManager):
    interp = ClinicalInterpretation(
        status="CRITICAL_HIGH",
        severity="CRITICAL",
        category="electrolyte",
        measurement_name="Potassium",
        canonical_test_id="POTASSIUM",
        measured_value=6.5,
        unit="mmol/L",
        reference_range="3.5 - 5.0 mmol/L",
        critical_threshold=">= 6.0 mmol/L",
        reference_source="Tietz",
        explanation="CRITICAL HIGH Potassium 6.5 mmol/L",
        requires_alert=True,
    )
    alert = alert_mgr.process_clinical_interpretation(
        interpretation=interp,
        patient_id=1,
        stay_id=1,
        patientunitstayid=None,
        source="LAB_RESULT",
        record_id="rec-1",
    )
    assert alert is not None
    assert alert["status"] == "ACTIVE"
    assert alert["severity"] == "CRITICAL"
    assert alert["measurement_name"] == "Potassium"

    # Fetch alert by ID
    fetched = alert_mgr.get_alert(alert["alert_id"])
    assert fetched is not None
    assert fetched["alert_id"] == alert["alert_id"]

    # History contains CREATED action
    hist = alert_mgr.get_alert_history(alert["alert_id"])
    assert len(hist) == 1
    assert hist[0]["action"] == "CREATED"


def test_alert_deduplication_without_spamming(alert_mgr: AlertManager):
    interp = ClinicalInterpretation(
        status="HIGH",
        severity="MODERATE",
        category="renal",
        measurement_name="Creatinine",
        canonical_test_id="CREATININE",
        measured_value=2.0,
        unit="mg/dL",
        reference_range="0.6 - 1.2 mg/dL",
        reference_source="KDIGO",
        explanation="HIGH Creatinine 2.0 mg/dL",
        requires_alert=True,
    )
    # First alert
    a1 = alert_mgr.process_clinical_interpretation(
        interp, patient_id=1, stay_id=1, patientunitstayid=None, source="LAB_RESULT", record_id="rec-1"
    )
    # Second measurement with same severity
    interp2 = ClinicalInterpretation(
        status="HIGH",
        severity="MODERATE",
        category="renal",
        measurement_name="Creatinine",
        canonical_test_id="CREATININE",
        measured_value=2.1,
        unit="mg/dL",
        reference_range="0.6 - 1.2 mg/dL",
        reference_source="KDIGO",
        explanation="HIGH Creatinine 2.1 mg/dL",
        requires_alert=True,
    )
    a2 = alert_mgr.process_clinical_interpretation(
        interp2, patient_id=1, stay_id=1, patientunitstayid=None, source="LAB_RESULT", record_id="rec-2"
    )
    # Same alert ID reused (deduplicated)
    assert a1["alert_id"] == a2["alert_id"]
    assert a2["measured_value"] == "2.1"

    # Only 1 active alert exists
    active = alert_mgr.list_alerts(stay_id=1, status="ACTIVE")
    assert len(active) == 1


def test_alert_escalation(alert_mgr: AlertManager):
    # First measurement: MODERATE severity
    interp1 = ClinicalInterpretation(
        status="HIGH",
        severity="MODERATE",
        category="electrolyte",
        measurement_name="Potassium",
        canonical_test_id="POTASSIUM",
        measured_value=5.3,
        unit="mmol/L",
        reference_range="3.5 - 5.0 mmol/L",
        reference_source="Tietz",
        explanation="HIGH Potassium 5.3",
        requires_alert=True,
    )
    a1 = alert_mgr.process_clinical_interpretation(
        interp1, patient_id=1, stay_id=1, patientunitstayid=None, source="LAB_RESULT"
    )
    assert a1["severity"] == "MODERATE"

    # Second measurement escalates to CRITICAL
    interp2 = ClinicalInterpretation(
        status="CRITICAL_HIGH",
        severity="CRITICAL",
        category="electrolyte",
        measurement_name="Potassium",
        canonical_test_id="POTASSIUM",
        measured_value=6.4,
        unit="mmol/L",
        reference_range="3.5 - 5.0 mmol/L",
        critical_threshold=">= 6.0 mmol/L",
        reference_source="Tietz",
        explanation="CRITICAL HIGH Potassium 6.4",
        requires_alert=True,
    )
    a2 = alert_mgr.process_clinical_interpretation(
        interp2, patient_id=1, stay_id=1, patientunitstayid=None, source="LAB_RESULT"
    )
    assert a2["alert_id"] == a1["alert_id"]
    assert a2["severity"] == "CRITICAL"

    # Audit log contains ESCALATED
    hist = alert_mgr.get_alert_history(a1["alert_id"])
    actions = [h["action"] for h in hist]
    assert "CREATED" in actions
    assert "ESCALATED" in actions


def test_alert_auto_resolution_on_normal(alert_mgr: AlertManager):
    # Setup active alert
    interp_abnormal = ClinicalInterpretation(
        status="HIGH",
        severity="MODERATE",
        category="hemodynamic",
        measurement_name="Heart Rate",
        canonical_test_id="HEART_RATE",
        measured_value=125,
        unit="BPM",
        reference_range="50 - 120 BPM",
        reference_source="AHA",
        explanation="Tachycardia 125 BPM",
        requires_alert=True,
    )
    a = alert_mgr.process_clinical_interpretation(
        interp_abnormal, patient_id=1, stay_id=1, patientunitstayid=None, source="ESP32_TELEMETRY"
    )
    assert a["status"] == "ACTIVE"

    # Subsequent reading is NORMAL
    interp_normal = ClinicalInterpretation(
        status="NORMAL",
        category="hemodynamic",
        measurement_name="Heart Rate",
        canonical_test_id="HEART_RATE",
        measured_value=78,
        unit="BPM",
        reference_range="50 - 120 BPM",
        reference_source="AHA",
        explanation="Normal HR 78 BPM",
        requires_alert=False,
    )
    res = alert_mgr.process_clinical_interpretation(
        interp_normal, patient_id=1, stay_id=1, patientunitstayid=None, source="ESP32_TELEMETRY"
    )
    assert res is None

    # Alert should now be RESOLVED
    updated = alert_mgr.get_alert(a["alert_id"])
    assert updated["status"] == "RESOLVED"
    assert "Auto-resolved" in updated["resolution_note"]


def test_manual_acknowledgement_and_resolution(alert_mgr: AlertManager):
    interp = ClinicalInterpretation(
        status="LOW",
        severity="HIGH",
        category="renal",
        measurement_name="Urine output",
        canonical_test_id="URINE_OUTPUT",
        measured_value=0.35,
        unit="mL/kg/h",
        reference_range="0.5 - 2.0 mL/kg/h",
        reference_source="KDIGO",
        explanation="KDIGO Stage 1",
        requires_alert=True,
    )
    alert = alert_mgr.process_clinical_interpretation(
        interp, patient_id=2, stay_id=2, patientunitstayid=None, source="CLINICAL_OBSERVATION"
    )
    aid = alert["alert_id"]

    # Acknowledge
    ack = alert_mgr.acknowledge_alert(aid, acknowledged_by="Nurse Smith", note="Notified attending physician.")
    assert ack["status"] == "ACKNOWLEDGED"
    assert ack["acknowledged_by"] == "Nurse Smith"
    assert ack["acknowledgement_note"] == "Notified attending physician."

    # Resolve
    resolved = alert_mgr.resolve_alert(aid, resolved_by="Dr. Jones", note="Fluid challenge administered; output normalized.")
    assert resolved["status"] == "RESOLVED"
    assert resolved["resolved_by"] == "Dr. Jones"
    assert resolved["resolution_note"] == "Fluid challenge administered; output normalized."


def test_supersede_alerts_for_voided_record(alert_mgr: AlertManager):
    interp = ClinicalInterpretation(
        status="CRITICAL_HIGH",
        severity="CRITICAL",
        category="electrolyte",
        measurement_name="Sodium",
        canonical_test_id="SODIUM",
        measured_value=165,
        unit="mmol/L",
        reference_range="135 - 145 mmol/L",
        critical_threshold=">= 160 mmol/L",
        reference_source="Tietz",
        explanation="Severe hypernatremia",
        requires_alert=True,
    )
    alert = alert_mgr.process_clinical_interpretation(
        interp, patient_id=3, stay_id=3, patientunitstayid=None, source="LAB_RESULT", record_id="rec-sodium-99"
    )
    aid = alert["alert_id"]
    assert alert["status"] == "ACTIVE"

    # Voiding record supersedes alert
    count = alert_mgr.supersede_alerts_for_record("rec-sodium-99", "Lab specimen was mislabeled.", voided_by="Lab Director")
    assert count == 1

    updated = alert_mgr.get_alert(aid)
    assert updated["status"] == "SUPERSEDED"

    hist = alert_mgr.get_alert_history(aid)
    actions = [h["action"] for h in hist]
    assert "SUPERSEDED" in actions


def test_alert_summary_metrics(alert_mgr: AlertManager):
    # Insert 1 critical and 1 high alert
    interp_crit = ClinicalInterpretation(
        status="CRITICAL_HIGH",
        severity="CRITICAL",
        category="electrolyte",
        measurement_name="Potassium",
        canonical_test_id="POTASSIUM",
        measured_value=6.5,
        unit="mmol/L",
        reference_range="3.5 - 5.0",
        reference_source="Tietz",
        explanation="Crit",
        requires_alert=True,
    )
    interp_high = ClinicalInterpretation(
        status="LOW",
        severity="HIGH",
        category="respiratory",
        measurement_name="SpO2",
        canonical_test_id="SPO2",
        measured_value=90,
        unit="%",
        reference_range="92 - 100%",
        reference_source="Hypoxemia",
        explanation="Hypoxemia",
        requires_alert=True,
    )
    alert_mgr.process_clinical_interpretation(interp_crit, 1, 1, None, "LAB_RESULT")
    alert_mgr.process_clinical_interpretation(interp_high, 2, 2, None, "ESP32_TELEMETRY")

    summary = alert_mgr.get_summary()
    assert summary["active_alerts"] == 2
    assert summary["critical_alerts"] == 1
    assert summary["high_priority_alerts"] == 1
    assert summary["by_source"]["LAB_RESULT"] == 1
    assert summary["by_source"]["ESP32_TELEMETRY"] == 1


def test_alert_request_schema_supports_aliases():
    # Test AlertAcknowledgeRequest with clinician_name
    ack_alias = AlertAcknowledgeRequest.model_validate({"clinician_name": "Dr. Sarah Chen, MD"})
    assert ack_alias.acknowledged_by == "Dr. Sarah Chen, MD"

    ack_canon = AlertAcknowledgeRequest.model_validate({"acknowledged_by": "Dr. Kelly", "note": "Reviewed"})
    assert ack_canon.acknowledged_by == "Dr. Kelly"
    assert ack_canon.note == "Reviewed"

    # Test AlertResolveRequest with clinician_name and resolution_note
    res_alias = AlertResolveRequest.model_validate({
        "clinician_name": "Dr. Sarah Chen, MD",
        "resolution_note": "Potassium normalized following insulin-glucose infusion.",
    })
    assert res_alias.resolved_by == "Dr. Sarah Chen, MD"
    assert res_alias.note == "Potassium normalized following insulin-glucose infusion."

    res_canon = AlertResolveRequest.model_validate({
        "resolved_by": "Dr. Jones",
        "note": "Resolved following lab redraw.",
    })
    assert res_canon.resolved_by == "Dr. Jones"
    assert res_canon.note == "Resolved following lab redraw."
