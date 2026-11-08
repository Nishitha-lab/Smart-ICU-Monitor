import React, { useState, useEffect, useCallback } from "react";
import { API_BASE } from "./api";
import "./DataConsistencyPanel.css";

export interface DiscrepancyResult {
  metric: string;
  canonical_metric: string;
  telemetry_value: number | null;
  telemetry_time: string | null;
  clinical_value: number | null;
  clinical_time: string | null;
  clinical_record_id: string | null;
  time_difference_seconds: number | null;
  status: string;
  absolute_difference: number | null;
  relative_difference_pct: number | null;
  explanation: string;
  clinical_rationale: string;
  recommended_action: string;
  requires_alert: boolean;
  alert_severity: "INFO" | "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
}

export interface TrendAnalysisResult {
  metric: string;
  canonical_metric: string;
  current_value: number | null;
  current_time: string | null;
  previous_value: number | null;
  previous_time: string | null;
  baseline_mean: number | null;
  baseline_std: number | null;
  observation_count: number;
  delta_from_baseline: number | null;
  delta_from_previous: number | null;
  rate_of_change_per_minute: number | null;
  time_delta_seconds: number | null;
  status: string;
  is_anomaly: boolean;
  severity: "INFO" | "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
  explanation: string;
  clinical_recommendation: string;
}

export interface PopulationComparisonResult {
  metric: string;
  canonical_metric: string;
  measured_value: number;
  reference_median: number;
  reference_iqr: number;
  percentile_rank: number;
  robust_z_score: number;
  is_anomaly: boolean;
  anomaly_type: string;
  severity: "INFO" | "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
  explanation: string;
  clinical_validation_recommendation: string;
}

export interface IsolationForestEvaluationResult {
  is_evaluated: boolean;
  status: string;
  anomaly_score: number | null;
  is_anomaly: boolean;
  severity: "INFO" | "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
  evaluated_features: string[];
  missing_features: string[];
  explanation: string;
  recommendation: string;
}

export interface ConsistencyEvaluationReport {
  patient_id: number;
  stay_id: number | null;
  evaluated_at: string;
  level1_cross_source: DiscrepancyResult[];
  level2_patient_trends: TrendAnalysisResult[];
  level3_population_comparison: PopulationComparisonResult[];
  isolation_forest: IsolationForestEvaluationResult;
  overall_status: string;
  active_alerts_count: number;
}

export interface DataConsistencyPanelProps {
  patientId: number;
  stayId?: number | null;
}

export const DataConsistencyPanel: React.FC<DataConsistencyPanelProps> = ({
  patientId,
  stayId,
}) => {
  const [activeTab, setActiveTab] = useState<"level1" | "level2" | "level3" | "history">("level1");
  const [report, setReport] = useState<ConsistencyEvaluationReport | null>(null);
  const [history, setHistory] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [evaluating, setEvaluating] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [engineStatus, setEngineStatus] = useState<string>("ONLINE");
  const [expandedMetric, setExpandedMetric] = useState<string | null>(null);

  const fetchEvaluation = useCallback(async () => {
    if (!patientId) return;
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`${API_BASE}/api/consistency/patient/${patientId}`);
      if (!res.ok) {
        if (res.status === 404) {
          setReport(null);
          return;
        }
        throw new Error(`Failed to load consistency data (HTTP ${res.status})`);
      }
      const data = await res.json();
      if (data && Array.isArray(data.level1_cross_source)) {
        setReport(data);
      } else {
        setReport(null);
      }
    } catch (err: any) {
      setError(err.message || "Failed to load consistency report");
    } finally {
      setLoading(false);
    }
  }, [patientId]);

  const fetchHistory = useCallback(async () => {
    if (!patientId) return;
    try {
      const res = await fetch(`${API_BASE}/api/consistency/patient/${patientId}/history?limit=30`);
      if (res.ok) {
        const histData = await res.json();
        setHistory(histData);
      }
    } catch {
      // non-critical
    }
  }, [patientId]);

  const handleReevaluate = async () => {
    if (!patientId || evaluating) return;
    try {
      setEvaluating(true);
      setError(null);
      const res = await fetch(`${API_BASE}/api/consistency/evaluate/${patientId}`, {
        method: "POST",
      });
      if (!res.ok) {
        throw new Error(`Re-evaluation failed (HTTP ${res.status})`);
      }
      const updated = await res.json();
      if (updated && Array.isArray(updated.level1_cross_source)) {
        setReport(updated);
      }
      fetchHistory();
    } catch (err: any) {
      setError(err.message || "Re-evaluation request failed");
    } finally {
      setEvaluating(false);
    }
  };

  useEffect(() => {
    fetchEvaluation();
    fetchHistory();
  }, [fetchEvaluation, fetchHistory]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      fetchEvaluation();
    }, 15000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchEvaluation]);

  const formatTimeDelta = (seconds: number | null) => {
    if (seconds === null || seconds === undefined) return "—";
    const mins = Math.round(seconds / 60);
    if (mins === 0) return "< 1 min";
    return `${mins} min`;
  };

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case "CONSISTENT":
      case "NORMAL_TREND":
      case "NORMAL":
        return "badge-status consistent";
      case "MINOR_DISCREPANCY":
        return "badge-status minor-discrepancy";
      case "SIGNIFICANT_DISCREPANCY":
      case "POPULATION_EXTREME_LOW":
        return "badge-status significant-discrepancy";
      case "CRITICAL_DISCREPANCY":
      case "POPULATION_EXTREME_HIGH":
      case "SUDDEN_SPIKE":
      case "SUDDEN_DROP":
        return "badge-status critical-discrepancy";
      case "INSUFFICIENT_PATIENT_HISTORY":
        return "badge-status insufficient-history";
      case "INCOMPATIBLE_METRIC":
      case "INCOMPATIBLE_UNITS":
        return "badge-status incompatible-metric";
      default:
        return "badge-status insufficient-data";
    }
  };

  return (
    <div className="consistency-panel-container" data-testid="data-consistency-panel">
      <div className="consistency-card">
        {/* Panel Header */}
        <div className="consistency-header">
          <div>
            <h2>
              <span>Clinical Data Consistency & Intelligence Engine</span>
              <span className="engine-status-badge">● {engineStatus}</span>
            </h2>
            <small style={{ color: "#64748b" }}>
              Multi-source cross-validation, stay trend intelligence, and 139,432 eICU reference population benchmarking.
            </small>
          </div>

          <div className="consistency-actions">
            <label style={{ fontSize: "13px", display: "flex", alignItems: "center", gap: "6px", color: "#475569" }}>
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              Live Sync (15s)
            </label>
            <button
              type="button"
              className="btn-reevaluate"
              onClick={handleReevaluate}
              disabled={evaluating}
            >
              {evaluating ? "Evaluating..." : "🔄 Re-evaluate Now"}
            </button>
          </div>
        </div>

        {error && (
          <div style={{ background: "#fef2f2", color: "#b91c1c", padding: "10px 14px", borderRadius: "8px", marginBottom: "16px", fontSize: "13px" }}>
            <strong>Notice:</strong> {error}
          </div>
        )}

        {/* Navigation Tabs */}
        <div className="tab-nav">
          <button
            type="button"
            className={`tab-btn ${activeTab === "level1" ? "active" : ""}`}
            onClick={() => setActiveTab("level1")}
          >
            Level 1: Source Consistency ({report?.level1_cross_source?.length || 0})
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === "level2" ? "active" : ""}`}
            onClick={() => setActiveTab("level2")}
          >
            Level 2: Patient Trend Intelligence ({report?.level2_patient_trends?.length || 0})
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === "level3" ? "active" : ""}`}
            onClick={() => setActiveTab("level3")}
          >
            Level 3: Historical eICU Comparison ({report?.level3_population_comparison?.length || 0})
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === "history" ? "active" : ""}`}
            onClick={() => {
              setActiveTab("history");
              fetchHistory();
            }}
          >
            History & Audits
          </button>
        </div>

        {loading && !report ? (
          <div style={{ padding: "30px", textAlign: "center", color: "#64748b" }}>
            Loading multi-source consistency evaluation...
          </div>
        ) : null}

        {/* ============================================================ */}
        {/* SUB-PANEL 1: LEVEL 1 CROSS-SOURCE CONSISTENCY */}
        {/* ============================================================ */}
        {activeTab === "level1" && report && (
          <div>
            <p style={{ fontSize: "13px", color: "#64748b", margin: "0 0 12px 0" }}>
              Compares streaming ESP32 hardware telemetry with manual bedside records and lab observations within a configurable &plusmn;30-minute matching window.
            </p>

            <table className="discrepancy-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>ESP32 Telemetry</th>
                  <th>Clinical Record</th>
                  <th>&Delta; Time</th>
                  <th>Absolute &Delta;</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {report?.level1_cross_source?.map((item, idx) => (
                  <React.Fragment key={idx}>
                    <tr>
                      <td>
                        <strong>{item.metric}</strong>
                        {item.canonical_metric === "fluid" && (
                          <div style={{ fontSize: "11px", color: "#6b7280" }}>
                            IV reservoir volume (isolated from urine output)
                          </div>
                        )}
                      </td>
                      <td>
                        {item.telemetry_value !== null ? (
                          <span>
                            <strong>{item.telemetry_value}</strong>
                            <small style={{ display: "block", color: "#64748b", fontSize: "11px" }}>
                              {item.telemetry_time ? new Date(item.telemetry_time).toLocaleTimeString() : ""}
                            </small>
                          </span>
                        ) : (
                          <span style={{ color: "#94a3b8" }}>No telemetry</span>
                        )}
                      </td>
                      <td>
                        {item.clinical_value !== null ? (
                          <span>
                            <strong>{item.clinical_value}</strong>
                            <small style={{ display: "block", color: "#64748b", fontSize: "11px" }}>
                              {item.clinical_time ? new Date(item.clinical_time).toLocaleTimeString() : ""}
                            </small>
                          </span>
                        ) : (
                          <span style={{ color: "#94a3b8" }}>No bedside entry</span>
                        )}
                      </td>
                      <td>{formatTimeDelta(item.time_difference_seconds)}</td>
                      <td>
                        {item.absolute_difference !== null ? (
                          <span>
                            {item.absolute_difference}
                            {item.relative_difference_pct !== null ? ` (${Math.round(item.relative_difference_pct)}%)` : ""}
                          </span>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td>
                        <span className={getStatusBadgeClass(item.status)}>
                          {item.status.replace(/_/g, " ")}
                        </span>
                      </td>
                      <td>
                        <button
                          type="button"
                          style={{
                            background: "none",
                            border: "1px solid #cbd5e1",
                            borderRadius: "6px",
                            padding: "4px 8px",
                            fontSize: "12px",
                            cursor: "pointer",
                            color: "#1e293b",
                          }}
                          onClick={() =>
                            setExpandedMetric(expandedMetric === item.metric ? null : item.metric)
                          }
                        >
                          {expandedMetric === item.metric ? "Hide Details" : "Explainability"}
                        </button>
                      </td>
                    </tr>
                    {expandedMetric === item.metric && (
                      <tr>
                        <td colSpan={7} style={{ background: "#f8fafc", padding: "14px" }}>
                          <div
                            className={`explainability-box ${
                              item.status === "CRITICAL_DISCREPANCY"
                                ? "critical"
                                : item.status === "SIGNIFICANT_DISCREPANCY"
                                ? "significant"
                                : ""
                            }`}
                          >
                            <div className="clinical-rationale-title">Physiological & Sensor Rationale:</div>
                            <div>{item.clinical_rationale}</div>
                            <div className="validation-action-item">
                              Recommended Validation: {item.recommended_action}
                            </div>
                            <small style={{ display: "block", marginTop: "6px", color: "#64748b" }}>
                              System Policy: Automated consistency checks do not override clinical judgment. Both sources remain documented in the record.
                            </small>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* ============================================================ */}
        {/* SUB-PANEL 2: LEVEL 2 PATIENT TREND INTELLIGENCE */}
        {/* ============================================================ */}
        {activeTab === "level2" && report && (
          <div>
            <p style={{ fontSize: "13px", color: "#64748b", margin: "0 0 12px 0" }}>
              Monitors patient-specific moving baseline trajectory within the active stay. Enforces strict minimum observation guard (&ge;3 required) before establishing baseline.
            </p>

            <div className="trend-grid">
              {report?.level2_patient_trends?.map((trend, idx) => (
                <div
                  key={idx}
                  className={`trend-card ${trend.is_anomaly ? "anomaly" : ""}`}
                >
                  <div className="trend-card-header">
                    <span className="trend-metric-name">{trend.metric}</span>
                    <span className={getStatusBadgeClass(trend.status)}>
                      {trend.status.replace(/_/g, " ")}
                    </span>
                  </div>

                  {trend.status === "INSUFFICIENT_PATIENT_HISTORY" ? (
                    <div className="insufficient-history-banner">
                      <strong>Insufficient Patient History</strong>
                      <div style={{ marginTop: "4px" }}>
                        Minimum 3 observations required for patient-specific moving baseline (currently {trend.observation_count}).
                      </div>
                    </div>
                  ) : trend.status === "STALE_DATA" ? (
                    <div className="stale-telemetry-banner">
                      <span>⚠️ Stale Telemetry:</span> {trend.explanation}
                    </div>
                  ) : (
                    <>
                      <div className="trend-values-row">
                        <div className="trend-current-value">
                          {trend.current_value !== null ? trend.current_value : "—"}
                        </div>
                        <div className="trend-baseline-val">
                          Baseline &mu;: {trend.baseline_mean !== null ? trend.baseline_mean.toFixed(1) : "—"} &plusmn;{" "}
                          {trend.baseline_std !== null ? trend.baseline_std.toFixed(1) : "0"} (N={trend.observation_count})
                        </div>
                      </div>

                      {trend.rate_of_change_per_minute !== null && (
                        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                          <span className="rate-of-change-tag">
                            &Delta; Rate: {trend.rate_of_change_per_minute > 0 ? "+" : ""}
                            {trend.rate_of_change_per_minute.toFixed(2)}/min
                          </span>
                          {trend.delta_from_baseline !== null && (
                            <span style={{ fontSize: "12px", color: "#64748b" }}>
                              &Delta; Baseline: {trend.delta_from_baseline > 0 ? "+" : ""}
                              {trend.delta_from_baseline.toFixed(1)}
                            </span>
                          )}
                        </div>
                      )}

                      <div style={{ fontSize: "12px", color: "#334155", background: "#f8fafc", padding: "8px 10px", borderRadius: "6px" }}>
                        <div>{trend.explanation}</div>
                        <div style={{ color: "#0369a1", fontWeight: 600, marginTop: "4px" }}>
                          Recommendation: {trend.clinical_recommendation}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* SUB-PANEL 3: LEVEL 3 HISTORICAL eICU POPULATION COMPARISON */}
        {/* ============================================================ */}
        {activeTab === "level3" && report && (
          <div>
            <p style={{ fontSize: "13px", color: "#64748b", margin: "0 0 12px 0" }}>
              Benchmarking patient vital signs and labs against 139,432 patient-hour reference distributions from 2,419 ICU stays.
            </p>

            <div className="population-grid">
              {report?.level3_population_comparison?.map((pop, idx) => (
                <div key={idx} className="pop-metric-row">
                  <div className="pop-metric-header">
                    <div>
                      <strong>{pop.metric}</strong>
                      <span style={{ marginLeft: "8px", fontSize: "13px", color: "#0f172a", fontWeight: 700 }}>
                        {pop.measured_value}
                      </span>
                    </div>
                    <span className={getStatusBadgeClass(pop.anomaly_type)}>
                      {pop.anomaly_type === "NORMAL" ? "Normal in eICU" : pop.anomaly_type.replace(/_/g, " ")}
                    </span>
                  </div>

                  <div className="percentile-bar-container">
                    <div
                      className={`percentile-bar-fill ${
                        pop.percentile_rank < 5 || pop.percentile_rank > 95
                          ? "extreme"
                          : pop.percentile_rank < 15 || pop.percentile_rank > 85
                          ? "warning"
                          : "normal"
                      }`}
                      style={{ width: `${Math.min(100, Math.max(2, pop.percentile_rank))}%` }}
                    />
                  </div>

                  <div className="pop-stats-row">
                    <span>eICU Percentile: <strong>{pop.percentile_rank.toFixed(1)}%</strong></span>
                    <span>Robust Z-Score: <strong>{pop.robust_z_score.toFixed(2)}</strong></span>
                    <span>Population Median [IQR]: <strong>{pop.reference_median} [{pop.reference_iqr}]</strong></span>
                  </div>

                  {pop.is_anomaly && (
                    <div style={{ marginTop: "10px", fontSize: "12px", color: "#991b1b", background: "#fef2f2", padding: "8px 10px", borderRadius: "6px" }}>
                      <div>{pop.explanation}</div>
                      <div style={{ fontWeight: 600, marginTop: "2px" }}>Action: {pop.clinical_validation_recommendation}</div>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Isolation Forest Multi-Feature Anomaly Card */}
            {report.isolation_forest && (
              <div
                className={`isolation-forest-card ${
                  report.isolation_forest.is_anomaly ? "outlier" : ""
                }`}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <strong>Multi-Feature Isolation Forest Evaluation</strong>
                  <span
                    className={
                      report.isolation_forest.is_anomaly
                        ? "badge-status critical-discrepancy"
                        : "badge-status consistent"
                    }
                  >
                    {report.isolation_forest.status}
                  </span>
                </div>
                <div style={{ fontSize: "13px", color: "#334155", marginTop: "8px" }}>
                  {report.isolation_forest.explanation}
                </div>
                {report.isolation_forest.anomaly_score !== null && (
                  <div style={{ fontSize: "12px", color: "#64748b", marginTop: "4px" }}>
                    Anomaly Score: {report.isolation_forest.anomaly_score.toFixed(4)} | Evaluated Features:{" "}
                    {report.isolation_forest.evaluated_features.join(", ") || "None"}
                  </div>
                )}
              </div>
            )}

            {/* Academic Research Disclaimer */}
            <div className="research-disclaimer">
              <strong>Academic Research Prototype Notice:</strong> Population reference distributions are computed retrospectively from the eICU Collaborative Research Database (139,432 patient hours, 2,419 ICU stays). This comparison is designed for anomaly screening and exploratory quality verification; bedside clinical assessment and localized hospital clinical protocols always supersede automated benchmarks.
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* SUB-PANEL 4: HISTORY & AUDITS */}
        {/* ============================================================ */}
        {activeTab === "history" && (
          <div>
            <h3>Consistency & Anomaly Audit Log</h3>
            {history && history.consistency_checks && history.consistency_checks.length > 0 ? (
              <div style={{ maxHeight: "350px", overflowY: "auto" }}>
                <table className="discrepancy-table">
                  <thead>
                    <tr>
                      <th>Timestamp</th>
                      <th>Metric</th>
                      <th>Telemetry</th>
                      <th>Clinical Entry</th>
                      <th>Status</th>
                      <th>Recommended Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.consistency_checks.map((c: any) => (
                      <tr key={c.check_id}>
                        <td>{new Date(c.created_at).toLocaleString()}</td>
                        <td>{c.metric}</td>
                        <td>{c.telemetry_value !== null ? c.telemetry_value : "—"}</td>
                        <td>{c.clinical_value !== null ? c.clinical_value : "—"}</td>
                        <td>
                          <span className={getStatusBadgeClass(c.discrepancy_level)}>
                            {c.discrepancy_level.replace(/_/g, " ")}
                          </span>
                        </td>
                        <td style={{ fontSize: "12px" }}>{c.recommended_action || "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p style={{ color: "#64748b" }}>No historical consistency checks found for this patient.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DataConsistencyPanel;
