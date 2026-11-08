import { useCallback, useEffect, useMemo, useState } from "react";
import { API_BASE } from "./api";
import "./AlertPanel.css";

export type AlertSeverity = "CRITICAL" | "HIGH" | "MODERATE" | "LOW" | "INFO";
export type AlertStatus = "ACTIVE" | "ACKNOWLEDGED" | "RESOLVED" | "SUPERSEDED";
export type AlertSource =
  | "ESP32_TELEMETRY"
  | "LAB_RESULT"
  | "CLINICAL_OBSERVATION"
  | "URINE_OUTPUT"
  | "NURSE_OBSERVATION"
  | "HISTORICAL_EICU"
  | "ML_ANOMALY"
  | "ML_RISK"
  | string;

export type AlertItem = {
  alert_id: string;
  patient_id: number | null;
  patient_code?: string | null;
  patient_name?: string | null;
  stay_id: number | null;
  patientunitstayid: number | null;
  source: AlertSource;
  record_id: string | null;
  fingerprint: string;
  severity: AlertSeverity;
  category: string;
  measurement_name: string;
  measured_value: number | string | null;
  unit: string | null;
  reference_range: string | null;
  critical_threshold: string | null;
  reference_source?: string | null;
  rule_version?: string | null;
  explanation: string;
  status: AlertStatus;
  trigger_time: string;
  acknowledged_at: string | null;
  acknowledged_by: string | null;
  acknowledgement_note?: string | null;
  resolved_at: string | null;
  resolved_by: string | null;
  resolution_note: string | null;
  superseded_by?: string | null;
  superseded_by_alert_id?: string | null;
  escalation_count?: number;
  created_at: string;
  updated_at: string;
};

export type AlertAuditLog = {
  log_id: number;
  alert_id: string;
  action: string;
  performed_by: string;
  note: string | null;
  timestamp: string;
  details_json: string | null;
};

export type AlertSummary = {
  total: number;
  active: number;
  acknowledged: number;
  resolved: number;
  superseded?: number;
  critical: number;
  high: number;
  moderate?: number;
  low?: number;
  active_alerts?: number;
  critical_alerts?: number;
  high_priority_alerts?: number;
  unacknowledged_alerts?: number;
  resolved_alerts?: number;
  total_alerts?: number;
  by_source?: Record<string, number>;
  by_severity?: Record<string, number>;
};

const SOURCE_LABELS: Record<string, string> = {
  ESP32_TELEMETRY: "ESP32 Sensor",
  LAB_RESULT: "Laboratory Result",
  CLINICAL_OBSERVATION: "Clinical Observation",
  URINE_OUTPUT: "Urine Output (KDIGO)",
  NURSE_OBSERVATION: "Nurse Observation",
  HISTORICAL_EICU: "Historical eICU",
  ML_ANOMALY: "ML Anomaly",
  ML_RISK: "ML Risk Model",
};

export default function AlertPanel() {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [summary, setSummary] = useState<AlertSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState<string>("ACTIVE");
  const [severityFilter, setSeverityFilter] = useState<string>("ALL");
  const [sourceFilter, setSourceFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState("");
  const [groupBy, setGroupBy] = useState<"none" | "patient" | "source">("none");
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Expanded details & history
  const [expandedAlertId, setExpandedAlertId] = useState<string | null>(null);
  const [auditLogs, setAuditLogs] = useState<Record<string, AlertAuditLog[]>>({});
  const [auditLoading, setAuditLoading] = useState<string | null>(null);

  // Modal / action state
  const [actionAlert, setActionAlert] = useState<AlertItem | null>(null);
  const [actionType, setActionType] = useState<"ack" | "resolve" | null>(null);
  const [clinicianName, setClinicianName] = useState("Dr. ICU Attending");
  const [resolutionNote, setResolutionNote] = useState("");
  const [actionSubmitting, setActionSubmitting] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const fetchAlerts = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const params = new URLSearchParams();

      // When statusFilter is default "ACTIVE", do not send status param so backend returns all active & in-progress alerts
      if (statusFilter === "UNACKNOWLEDGED") {
        params.append("status", "ACTIVE");
      } else if (statusFilter === "ALL") {
        params.append("status", "ALL");
      } else if (statusFilter !== "ACTIVE") {
        params.append("status", statusFilter);
      }

      if (severityFilter !== "ALL") params.append("severity", severityFilter);
      if (sourceFilter !== "ALL") params.append("source", sourceFilter);

      const qs = params.toString();
      const url = `${API_BASE}/api/alerts${qs ? `?${qs}` : ""}`;
      const alertsRes = await fetch(url);

      if (!alertsRes.ok) throw new Error("Failed to load clinical alerts.");

      const rawAlerts = await alertsRes.json();
      const alertsData: AlertItem[] = Array.isArray(rawAlerts)
        ? rawAlerts
        : Array.isArray(rawAlerts?.alerts)
        ? rawAlerts.alerts
        : [];

      setAlerts(alertsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error connecting to alert service");
    } finally {
      setLoading(false);
    }
  }, [statusFilter, severityFilter, sourceFilter]);

  const fetchSummary = useCallback(async () => {
    try {
      const summaryRes = await fetch(`${API_BASE}/api/alerts/summary`);
      if (!summaryRes.ok) throw new Error("Failed to load alert summary.");

      const rawSummary = await summaryRes.json();
      const normalizedSummary: AlertSummary = {
        total: rawSummary.total_alerts ?? rawSummary.total ?? 0,
        active: rawSummary.active_alerts ?? rawSummary.active ?? 0,
        acknowledged:
          rawSummary.acknowledged ??
          (rawSummary.total_alerts != null && rawSummary.unacknowledged_alerts != null
            ? Math.max(0, rawSummary.total_alerts - rawSummary.unacknowledged_alerts)
            : 0),
        resolved: rawSummary.resolved_alerts ?? rawSummary.resolved ?? 0,
        superseded: rawSummary.superseded ?? 0,
        critical:
          rawSummary.critical_alerts ??
          rawSummary.critical ??
          (rawSummary.by_severity?.critical ?? 0),
        high:
          rawSummary.high_priority_alerts ??
          rawSummary.high ??
          (rawSummary.by_severity?.high ?? 0),
        moderate: rawSummary.moderate ?? (rawSummary.by_severity?.moderate ?? 0),
        low: rawSummary.low ?? (rawSummary.by_severity?.low ?? 0),
      };

      setSummary(normalizedSummary);
    } catch {
      // Summary failure does not block alerts list
    }
  }, []);

  // Fetch alerts and summary independently upon mount
  useEffect(() => {
    const timer = setTimeout(() => {
      void fetchAlerts();
      void fetchSummary();
    }, 0);
    return () => clearTimeout(timer);
  }, [fetchAlerts, fetchSummary]);

  // Auto-refresh timer for continuous surveillance
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      void fetchAlerts();
      void fetchSummary();
    }, 10000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchAlerts, fetchSummary]);

  const toggleExpand = async (alertId: string) => {
    if (expandedAlertId === alertId) {
      setExpandedAlertId(null);
      return;
    }
    setExpandedAlertId(alertId);
    if (!auditLogs[alertId]) {
      try {
        setAuditLoading(alertId);
        const res = await fetch(`${API_BASE}/api/alerts/${alertId}/history`);
        if (res.ok) {
          const rawHistory = await res.json();
          const logs: AlertAuditLog[] = Array.isArray(rawHistory)
            ? rawHistory
            : Array.isArray(rawHistory?.history)
            ? rawHistory.history
            : [];
          setAuditLogs((prev) => ({ ...prev, [alertId]: logs }));
        }
      } catch {
        // silent fail for history
      } finally {
        setAuditLoading(null);
      }
    }
  };

  const handleOpenAction = (alert: AlertItem, type: "ack" | "resolve") => {
    setActionAlert(alert);
    setActionType(type);
    setActionError(null);
    setResolutionNote(type === "resolve" ? "Patient condition evaluated and stabilized." : "");
  };

  const handleCloseAction = () => {
    setActionAlert(null);
    setActionType(null);
    setActionError(null);
  };

  const handlePerformAction = async () => {
    if (!actionAlert || !actionType) return;
    if (actionType === "resolve" && !resolutionNote.trim()) {
      setActionError("A clinical resolution note is required to resolve an alert.");
      return;
    }

    try {
      setActionSubmitting(true);
      setActionError(null);

      const endpoint =
        actionType === "ack"
          ? `${API_BASE}/api/alerts/${actionAlert.alert_id}/acknowledge`
          : `${API_BASE}/api/alerts/${actionAlert.alert_id}/resolve`;

      const body =
        actionType === "ack"
          ? {
              acknowledged_by: clinicianName,
              clinician_name: clinicianName,
            }
          : {
              resolved_by: clinicianName,
              clinician_name: clinicianName,
              note: resolutionNote,
              resolution_note: resolutionNote,
            };

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to ${actionType} alert.`);
      }

      handleCloseAction();
      await Promise.all([fetchAlerts(), fetchSummary()]);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Action failed");
    } finally {
      setActionSubmitting(false);
    }
  };

  const filteredAlerts = useMemo(() => {
    if (!searchQuery.trim()) return alerts;
    const q = searchQuery.toLowerCase();
    return alerts.filter(
      (a) =>
        a.measurement_name.toLowerCase().includes(q) ||
        a.explanation.toLowerCase().includes(q) ||
        (a.patient_code && a.patient_code.toLowerCase().includes(q)) ||
        (a.stay_id && String(a.stay_id).includes(q)) ||
        (a.patientunitstayid && String(a.patientunitstayid).includes(q))
    );
  }, [alerts, searchQuery]);

  const groupedAlerts = useMemo(() => {
    if (groupBy === "none") {
      return [{ groupTitle: "All Alerts", items: filteredAlerts }];
    }
    if (groupBy === "patient") {
      const map = new Map<string, AlertItem[]>();
      for (const a of filteredAlerts) {
        const key = a.stay_id
          ? `Registered Stay #${a.stay_id} (Patient #${a.patient_id ?? "Unknown"})`
          : a.patientunitstayid
          ? `Historical ICU Stay #${a.patientunitstayid}`
          : "Unattributed / Global";
        if (!map.has(key)) map.set(key, []);
        map.get(key)!.push(a);
      }
      return Array.from(map.entries()).map(([groupTitle, items]) => ({ groupTitle, items }));
    }
    if (groupBy === "source") {
      const map = new Map<string, AlertItem[]>();
      for (const a of filteredAlerts) {
        const key = SOURCE_LABELS[a.source] || a.source;
        if (!map.has(key)) map.set(key, []);
        map.get(key)!.push(a);
      }
      return Array.from(map.entries()).map(([groupTitle, items]) => ({ groupTitle, items }));
    }
    return [{ groupTitle: "All Alerts", items: filteredAlerts }];
  }, [filteredAlerts, groupBy]);

  return (
    <div className="alert-panel-container">
      {/* Header & Stat Cards */}
      <div className="alert-panel-header">
        <div>
          <h2>Unified Clinical Alert &amp; Interpretation Engine</h2>
          <p className="alert-panel-subhead">
            Continuous surveillance synthesizing ESP32 sensor telemetry, KDIGO urine output, laboratory panels, and nursing observations.
          </p>
        </div>
        <div className="alert-panel-actions">
          <label className="auto-refresh-toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh (10s)
          </label>
          <button
            className="refresh-btn"
            onClick={() => {
              void fetchAlerts();
              void fetchSummary();
            }}
            disabled={loading}
          >
            {loading ? "Refreshing..." : "↻ Refresh Alerts"}
          </button>
        </div>
      </div>

      {summary && (
        <div className="alert-summary-cards">
          <div className="stat-card stat-active">
            <span className="stat-value">{summary.active}</span>
            <span className="stat-label">Active Alerts</span>
          </div>
          <div className="stat-card stat-critical">
            <span className="stat-value">{summary.critical}</span>
            <span className="stat-label">Critical Alerts</span>
          </div>
          <div className="stat-card stat-high">
            <span className="stat-value">{summary.high}</span>
            <span className="stat-label">High Severity</span>
          </div>
          <div className="stat-card stat-ack">
            <span className="stat-value">{summary.acknowledged}</span>
            <span className="stat-label">Acknowledged</span>
          </div>
          <div className="stat-card stat-resolved">
            <span className="stat-value">{summary.resolved}</span>
            <span className="stat-label">Resolved</span>
          </div>
        </div>
      )}

      {/* Filter and Group Bar */}
      <div className="alert-filter-bar">
        <div className="filter-group">
          <label htmlFor="status-filter">Status:</label>
          <select
            id="status-filter"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="ACTIVE">Active</option>
            <option value="UNACKNOWLEDGED">Unacknowledged Only</option>
            <option value="ACKNOWLEDGED">Acknowledged Only</option>
            <option value="RESOLVED">Resolved</option>
            <option value="SUPERSEDED">Superseded</option>
            <option value="ALL">All Statuses</option>
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="severity-filter">Severity:</label>
          <select
            id="severity-filter"
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
          >
            <option value="ALL">All Severities</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="MODERATE">Moderate</option>
            <option value="LOW">Low</option>
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="source-filter">Source:</label>
          <select
            id="source-filter"
            value={sourceFilter}
            onChange={(e) => setSourceFilter(e.target.value)}
          >
            <option value="ALL">All Sources</option>
            <option value="ESP32_TELEMETRY">ESP32 Sensor Telemetry</option>
            <option value="LAB_RESULT">Laboratory Result</option>
            <option value="CLINICAL_OBSERVATION">Clinical Observation</option>
            <option value="URINE_OUTPUT">Urine Output (KDIGO)</option>
            <option value="NURSE_OBSERVATION">Nurse Observation</option>
          </select>
        </div>

        <div className="filter-group">
          <label htmlFor="groupby-filter">Group By:</label>
          <select
            id="groupby-filter"
            value={groupBy}
            onChange={(e) => setGroupBy(e.target.value as "none" | "patient" | "source")}
          >
            <option value="none">Flat (Chronological)</option>
            <option value="patient">Patient / ICU Stay</option>
            <option value="source">Alert Source</option>
          </select>
        </div>

        <div className="filter-group search-group">
          <label htmlFor="alert-search">Search:</label>
          <input
            id="alert-search"
            type="search"
            placeholder="Search measurement, stay, explanation..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {error && <div className="alert-error-banner" role="alert">{error}</div>}

      {/* Alert List */}
      <div className="alert-list-container">
        {loading && alerts.length === 0 && (
          <div className="alert-empty-state alert-loading-state" role="status">
            Loading active alerts...
          </div>
        )}

        {error && alerts.length === 0 && (
          <div className="alert-empty-state alert-error-state" role="alert">
            <p><strong>Failed to load clinical alerts</strong></p>
            <span className="empty-subtitle">{error}</span>
            <button
              className="refresh-btn"
              onClick={() => {
                void fetchAlerts();
                void fetchSummary();
              }}
              style={{ marginTop: "12px" }}
            >
              Retry
            </button>
          </div>
        )}

        {!loading && !error && filteredAlerts.length === 0 && (
          <div className="alert-empty-state">
            <p>No alerts match the selected criteria.</p>
            <span className="empty-subtitle">ICU patient vitals and telemetry currently in safe physiological bands.</span>
          </div>
        )}

        {groupedAlerts.map(({ groupTitle, items }) => (
          <div key={groupTitle} className="alert-group-section">
            {groupBy !== "none" && <h3 className="alert-group-title">{groupTitle} ({items.length})</h3>}
            {items.map((alert) => {
              const isExpanded = expandedAlertId === alert.alert_id;
              const hasLogs = auditLogs[alert.alert_id];

              return (
                <div
                  key={alert.alert_id}
                  className={`alert-card alert-severity-${alert.severity.toLowerCase()} alert-status-${alert.status.toLowerCase()}`}
                >
                  <div className="alert-card-header">
                    <div className="alert-badges">
                      <span className={`badge-severity badge-${alert.severity.toLowerCase()}`}>
                        {alert.severity}
                      </span>
                      <span className={`badge-status badge-st-${alert.status.toLowerCase()}`}>
                        {alert.status}
                      </span>
                      <span className="badge-source">
                        {SOURCE_LABELS[alert.source] || alert.source}
                      </span>
                      {(alert.escalation_count ?? 0) > 0 && (
                        <span className="badge-escalated">
                          Escalated x{alert.escalation_count ?? 0}
                        </span>
                      )}
                    </div>

                    <div className="alert-card-time">
                      <span>Triggered: {new Date(alert.trigger_time).toLocaleTimeString()} ({new Date(alert.trigger_time).toLocaleDateString()})</span>
                    </div>
                  </div>

                  <div className="alert-card-body">
                    <div className="alert-main-info">
                      <h4 className="alert-title">{alert.measurement_name}</h4>
                      <div className="alert-value-line">
                        <strong className="alert-value">
                          {alert.measured_value !== null ? alert.measured_value : "—"} {alert.unit || ""}
                        </strong>
                        <span className="alert-explanation">{alert.explanation}</span>
                      </div>
                    </div>

                    <div className="alert-patient-context">
                      {alert.stay_id ? (
                        <span>Stay #{alert.stay_id} (Patient #{alert.patient_id})</span>
                      ) : alert.patientunitstayid ? (
                        <span>Historical Stay #{alert.patientunitstayid}</span>
                      ) : (
                        <span>Unassigned Telemetry</span>
                      )}
                    </div>
                  </div>

                  {/* Quick Action Footer */}
                  <div className="alert-card-footer">
                    <div className="alert-action-buttons">
                      {alert.status === "ACTIVE" && (
                        <button
                          className="btn-ack"
                          onClick={() => handleOpenAction(alert, "ack")}
                        >
                          ✓ Acknowledge
                        </button>
                      )}
                      {(alert.status === "ACTIVE" || alert.status === "ACKNOWLEDGED") && (
                        <button
                          className="btn-resolve"
                          onClick={() => handleOpenAction(alert, "resolve")}
                        >
                          Resolve Alert
                        </button>
                      )}
                    </div>

                    <button
                      className="btn-expand-details"
                      onClick={() => toggleExpand(alert.alert_id)}
                      aria-expanded={isExpanded}
                    >
                      {isExpanded ? "▲ Hide Diagnostic Details" : "▼ Diagnostic Details (7 Questions)"}
                    </button>
                  </div>

                  {/* 7 Diagnostic Questions Deep-Dive Panel */}
                  {isExpanded && (
                    <div className="alert-diagnostic-panel">
                      <div className="diagnostic-header">
                        <h5>Clinical Diagnostic Inspection (7-Question Analysis)</h5>
                      </div>

                      <div className="diagnostic-grid">
                        <div className="diagnostic-item">
                          <span className="diagnostic-q">1. What triggered this alert?</span>
                          <span className="diagnostic-a">
                            <strong>{alert.measurement_name}</strong> under category <em>{alert.category}</em> from <strong>{SOURCE_LABELS[alert.source] || alert.source}</strong> at {alert.trigger_time}.
                          </span>
                        </div>

                        <div className="diagnostic-item">
                          <span className="diagnostic-q">2. What was the exact measured value and unit?</span>
                          <span className="diagnostic-a">
                            Value: <strong>{String(alert.measured_value)}</strong> {alert.unit || "N/A"}.
                          </span>
                        </div>

                        <div className="diagnostic-item">
                          <span className="diagnostic-q">3. What is the reference range and critical threshold?</span>
                          <span className="diagnostic-a">
                            Reference Range: <strong>{alert.reference_range || "N/A"}</strong> | Critical Threshold: <strong>{alert.critical_threshold || "None"}</strong>.
                          </span>
                        </div>

                        <div className="diagnostic-item">
                          <span className="diagnostic-q">4. Which clinical guideline or evidence basis was applied?</span>
                          <span className="diagnostic-a">
                            {alert.reference_source || "Hospital Standard ICU Protocol"}
                          </span>
                        </div>

                        <div className="diagnostic-item">
                          <span className="diagnostic-q">5. Which patient / ICU stay does this belong to?</span>
                          <span className="diagnostic-a">
                            {alert.stay_id ? `ICU Stay #${alert.stay_id} (Patient ID: ${alert.patient_id})` : alert.patientunitstayid ? `Historical Stay #${alert.patientunitstayid}` : "Unassigned telemetry"}.
                            {alert.record_id && ` Record Ref: ${alert.record_id}`}
                          </span>
                        </div>

                        <div className="diagnostic-item">
                          <span className="diagnostic-q">6. Is this reading confirmed, escalating, or superseded?</span>
                          <span className="diagnostic-a">
                            Current Status: <strong>{alert.status}</strong>. Escalation count: <strong>{alert.escalation_count ?? 0}</strong>.
                            {(alert.superseded_by || alert.superseded_by_alert_id) && ` Superseded by Alert ${alert.superseded_by || alert.superseded_by_alert_id}.`}
                            {alert.resolution_note && ` Resolution Note: ${alert.resolution_note}`}
                          </span>
                        </div>

                        <div className="diagnostic-item diagnostic-full-width">
                          <span className="diagnostic-q">7. What audit history or clinician interventions have occurred?</span>
                          <div className="diagnostic-audit-table-wrapper">
                            {auditLoading === alert.alert_id ? (
                              <span>Loading audit log...</span>
                            ) : hasLogs && hasLogs.length > 0 ? (
                              <table className="diagnostic-audit-table">
                                <thead>
                                  <tr>
                                    <th>Timestamp</th>
                                    <th>Action</th>
                                    <th>Performed By</th>
                                    <th>Clinical Note</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {hasLogs.map((log) => (
                                    <tr key={log.log_id}>
                                      <td>{new Date(log.timestamp).toLocaleString()}</td>
                                      <td><span className={`log-action log-${log.action.toLowerCase()}`}>{log.action}</span></td>
                                      <td>{log.performed_by}</td>
                                      <td>{log.note || "—"}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            ) : (
                              <span className="no-audit">No audit entries recorded yet.</span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Action Dialog / Modal */}
      {actionAlert && actionType && (
        <div className="alert-modal-backdrop" role="dialog" aria-modal="true">
          <div className="alert-modal">
            <h3>{actionType === "ack" ? "Acknowledge Clinical Alert" : "Resolve Clinical Alert"}</h3>
            <p className="modal-alert-summary">
              <strong>{actionAlert.measurement_name}</strong> ({actionAlert.severity}): {actionAlert.explanation}
            </p>
            <p className="modal-disclaimer" style={{ fontSize: "0.8rem", color: "#6b7280", margin: "4px 0 12px" }}>
              Notice: Prototype environment. Clinician identity is self-reported and not verified by an authentication service.
            </p>

            {actionError && <div className="modal-error">{actionError}</div>}

            <div className="modal-field">
              <label htmlFor="clinician-name">Clinician Name / Title:</label>
              <input
                id="clinician-name"
                type="text"
                value={clinicianName}
                onChange={(e) => setClinicianName(e.target.value)}
                placeholder="Dr. John Doe, MD / Nurse Specialist"
              />
            </div>

            {actionType === "resolve" && (
              <div className="modal-field">
                <label htmlFor="resolution-note">Resolution Note (Required):</label>
                <textarea
                  id="resolution-note"
                  rows={3}
                  value={resolutionNote}
                  onChange={(e) => setResolutionNote(e.target.value)}
                  placeholder="State clinical intervention taken or normalization verified..."
                />
              </div>
            )}

            <div className="modal-actions">
              <button
                className="btn-modal-cancel"
                onClick={handleCloseAction}
                disabled={actionSubmitting}
              >
                Cancel
              </button>
              <button
                className={actionType === "ack" ? "btn-modal-ack" : "btn-modal-resolve"}
                onClick={handlePerformAction}
                disabled={actionSubmitting}
              >
                {actionSubmitting ? "Submitting..." : actionType === "ack" ? "Confirm Acknowledgment" : "Confirm Resolution"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
