import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

type Summary = {
  total_rows: number;
  icu_stays: number;
  alert_priorities: {
    normal: number;
    low: number;
    moderate: number;
    high: number;
    critical: number;
  };
  ml_anomalies: number;
  clinical_warnings: number;
  clinical_severe: number;
  risk_10_percent_or_higher: number;
  risk_25_percent_or_higher: number;
  risk_50_percent_or_higher: number;
  multi_signal_alerts: number;
};

type Patient = {
  patientunitstayid: number;
  hour: number;
  priority: string;
  alert_category: string;
  risk_probability: number | null;
  risk_band: string;
  anomaly_score: number | null;
  ml_anomaly: boolean;
  clinical_rule_severity: string;
  clinical_explanation: string;
  final_explanation: string;
};

type PatientHistoryRow = {
  patientunitstayid: number;
  hour: number;

  heart_rate: number | null;
  spo2: number | null;
  respiration: number | null;

  systolic_bp: number | null;
  diastolic_bp: number | null;
  mean_bp: number | null;

  total_intake_ml: number | null;
  total_output_ml: number | null;
  dialysis_ml: number | null;
  reported_net_ml: number | null;
  net_fluid_balance_ml: number | null;
  urine_output_ml: number | null;

  active_infusion_count: number | null;
  vasoactive_support: number | null;

  anomaly_score: number | null;
  ml_anomaly: boolean;

  risk_probability: number | null;
  risk_band: string;

  clinical_rule_severity: string;
  clinical_explanation: string;

  priority: string;
  alert_category: string;
  final_explanation: string;
};
type IvTelemetry = {
  connected: boolean;
  bag_weight_g: number;
  initial_weight_g: number;
  remaining_volume_ml: number;
  flow_rate_ml_per_hr: number;
  level_percent: number;
  status: "normal" | "low" | "critical";
  last_update: string;
};
type TrendChartProps = {
  title: string;
  unit: string;
  data: PatientHistoryRow[];
  field: keyof PatientHistoryRow;
  height?: number;
};

function formatNumber(
  value: number | null | undefined,
  digits = 1
) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(value)
  ) {
    return "—";
  }

  return value.toFixed(digits);
}

function formatRisk(
  value: number | null | undefined
) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(value)
  ) {
    return "—";
  }

  return `${(value * 100).toFixed(1)}%`;
}
function getTrendDirection(
  data: PatientHistoryRow[],
  field: keyof PatientHistoryRow
) {
  const values = data
    .map((row) => trendValue(row, field))
    .filter(
      (value): value is number =>
        value !== null && Number.isFinite(value)
    );

  if (values.length < 2) {
    return "stable";
  }

  const first = values[0];
  const last = values[values.length - 1];

  const difference = last - first;

  if (Math.abs(difference) < 0.05) {
    return "stable";
  }

  return difference > 0 ? "up" : "down";
}

function trendSymbol(direction: string) {
  if (direction === "up") return "↑";
  if (direction === "down") return "↓";
  return "→";
}

function trendClass(direction: string) {
  return `trend-indicator trend-${direction}`;
}

function alertSource(row: PatientHistoryRow) {
  const hasClinical =
    row.clinical_rule_severity !== "normal";

  if (row.ml_anomaly && hasClinical) {
    return "ML + clinical";
  }

  if (row.ml_anomaly) {
    return "ML anomaly";
  }

  if (hasClinical) {
    return "Clinical";
  }

  return "None";
}

function priorityClass(priority: string) {
  return `badge badge-${priority}`;
}

function riskClass(riskBand: string) {
  return `risk-band risk-${riskBand}`;
}

function trendValue(
  row: PatientHistoryRow,
  field: keyof PatientHistoryRow
): number | null {
  const value = row[field];

  if (typeof value !== "number" || Number.isNaN(value)) {
    return null;
  }

  return value;
}

function TrendChart({
  title,
  unit,
  data,
  field,
  height = 220,
}: TrendChartProps) {
  const points = data
    .map((row) => ({
      hour: row.hour,
      value: trendValue(row, field),
    }))
    .filter(
      (point): point is { hour: number; value: number } =>
        point.value !== null
    );

  if (points.length === 0) {
    return (
      <div className="chart-card">
        <div className="chart-header">
          <div>
            <h3>{title}</h3>
            <span>{unit}</span>
          </div>
        </div>

        <div className="chart-empty">
          No recorded values available.
        </div>
      </div>
    );
  }

  const width = 760;
  const paddingLeft = 46;
  const paddingRight = 20;
  const paddingTop = 18;
  const paddingBottom = 32;

  const plotWidth = width - paddingLeft - paddingRight;
  const plotHeight = height - paddingTop - paddingBottom;

  const values = points.map((point) => point.value);

  let minValue = Math.min(...values);
  let maxValue = Math.max(...values);

  if (minValue === maxValue) {
    minValue -= 1;
    maxValue += 1;
  }

  const range = maxValue - minValue;

  const x = (index: number) => {
    if (points.length === 1) {
      return paddingLeft + plotWidth / 2;
    }

    return (
      paddingLeft +
      (index / (points.length - 1)) * plotWidth
    );
  };

  const y = (value: number) =>
    paddingTop +
    ((maxValue - value) / range) * plotHeight;

  const linePoints = points
    .map((point, index) => `${x(index)},${y(point.value)}`)
    .join(" ");

  const firstHour = points[0].hour;
  const lastHour = points[points.length - 1].hour;
  const midValue = (minValue + maxValue) / 2;

  return (
    <div className="chart-card">
      <div className="chart-header">
        <div>
          <h3>{title}</h3>
          <span>{unit}</span>
        </div>

        <div className="chart-range">
          H{firstHour} → H{lastHour}
        </div>
      </div>

      <div className="chart-wrapper">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="trend-chart"
          role="img"
          aria-label={`${title} trend`}
          preserveAspectRatio="none"
        >
          {[0, 0.5, 1].map((fraction) => {
            const yPosition =
              paddingTop + fraction * plotHeight;

            const value =
              maxValue - fraction * range;

            return (
              <g key={fraction}>
                <line
                  x1={paddingLeft}
                  y1={yPosition}
                  x2={width - paddingRight}
                  y2={yPosition}
                  className="chart-grid"
                />

                <text
                  x={paddingLeft - 8}
                  y={yPosition + 4}
                  textAnchor="end"
                  className="chart-label"
                >
                  {value.toFixed(0)}
                </text>
              </g>
            );
          })}

          <polyline
            points={linePoints}
            className="chart-line"
            fill="none"
          />

          {points.map((point, index) => (
            <circle
              key={`${point.hour}-${index}`}
              cx={x(index)}
              cy={y(point.value)}
              r="2.7"
              className="chart-point"
            />
          ))}

          <text
            x={paddingLeft}
            y={height - 10}
            className="chart-label"
          >
            H{firstHour}
          </text>

          <text
            x={width - paddingRight}
            y={height - 10}
            textAnchor="end"
            className="chart-label"
          >
            H{lastHour}
          </text>

          <text
            x={width - paddingRight}
            y={paddingTop + 12}
            textAnchor="end"
            className="chart-value"
          >
            Max {maxValue.toFixed(1)}
          </text>

          <text
            x={width - paddingRight}
            y={paddingTop + plotHeight - 2}
            textAnchor="end"
            className="chart-value"
          >
            Min {minValue.toFixed(1)}
          </text>
        </svg>
      </div>

      <div className="chart-footer">
        <span>Min {minValue.toFixed(1)}</span>
        <span>Avg {midValue.toFixed(1)}</span>
        <span>Max {maxValue.toFixed(1)}</span>
      </div>
    </div>
  );
}

function App() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [priorityFilter, setPriorityFilter] = useState("all");

  const [selectedPatientId, setSelectedPatientId] =
    useState<number | null>(null);

  const [patientHistory, setPatientHistory] = useState<
    PatientHistoryRow[]
  >([]);

  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);

  const [error, setError] = useState("");
  const [detailError, setDetailError] = useState("");
  const [ivTelemetry, setIvTelemetry] =
  useState<IvTelemetry | null>(null);
  async function loadDashboard() {
    try {
      setLoading(true);
      setError("");

      const summaryResponse = await fetch(
        `${API_BASE}/api/summary`
      );

      if (!summaryResponse.ok) {
        throw new Error("Failed to load summary");
      }

      const summaryData: Summary =
        await summaryResponse.json();

      const patientUrl =
        priorityFilter === "all"
          ? `${API_BASE}/api/patients?limit=100`
          : `${API_BASE}/api/patients?priority=${priorityFilter}&limit=100`;

      const patientResponse =
        await fetch(patientUrl);

      if (!patientResponse.ok) {
        throw new Error("Failed to load patients");
      }

      const patientData =
        await patientResponse.json();

      setSummary(summaryData);
      setPatients(patientData.patients ?? []);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to Smart-ICU backend."
      );
    } finally {
      setLoading(false);
    }
  }

  async function openPatient(patientId: number) {
  try {
    setSelectedPatientId(patientId);
    setPatientHistory([]);
    setIvTelemetry(null);
    setDetailLoading(true);
    setDetailError("");

    const response = await fetch(
      `${API_BASE}/api/patients/${patientId}`
    );

    if (!response.ok) {
      throw new Error(
        `Failed to load patient #${patientId}`
      );
    }

    const data = await response.json();

    const patientHistoryData: PatientHistoryRow[] =
  Array.isArray(data)
    ? data
    : Array.isArray(data.history)
      ? data.history
      : [];

    setPatientHistory(patientHistoryData);

    // Live ESP32 telemetry is polled independently from historical eICU data.
    // No simulated IV values are assigned to a historical patient record.

  } catch (err) {
    setDetailError(
      err instanceof Error
        ? err.message
        : "Unable to load patient history."
    );
  } finally {
    setDetailLoading(false);
  }
}

  function closePatient() {
    setSelectedPatientId(null);
    setPatientHistory([]);
    setDetailError("");
  }

  useEffect(() => {
    loadDashboard();
  }, [priorityFilter]);

  useEffect(() => {
    let active = true;
    const loadHardware = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/hardware/telemetry`);
        if (!response.ok) throw new Error("Hardware telemetry unavailable");
        const data = await response.json();
        if (!active) return;
        const connected = Boolean(data.connected);
        const fluid = typeof data.fluid_ml === "number" ? data.fluid_ml : 0;
        setIvTelemetry({
          connected,
          bag_weight_g: fluid,
          initial_weight_g: 1000,
          remaining_volume_ml: fluid,
          flow_rate_ml_per_hr: 0,
          level_percent: Math.max(0, Math.min(100, fluid / 10)),
          status: data.device_status === "CRITICAL" ? "critical" : data.device_status === "WARNING" ? "low" : "normal",
          last_update: data.last_update ?? data.connection_status ?? "Waiting for ESP32",
        });
      } catch {
        if (active) setIvTelemetry(null);
      }
    };
    loadHardware();
    const timer = window.setInterval(loadHardware, 1000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  const visiblePatients = useMemo(
    () => patients,
    [patients]
  );
  const latestDetailRow =
  patientHistory.length > 0
    ? patientHistory[patientHistory.length - 1]
    : null;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <div className="brand">SMART-ICU</div>

          <div className="subtitle">
            AI-Based ICU Patient Safety & Alert System
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot" />
          SYSTEM ONLINE
        </div>
      </header>

      <main className="dashboard">
        {loading && (
          <div className="loading-card">
            Loading ICU intelligence...
          </div>
        )}

        {error && (
          <div className="error-card">
            <strong>Backend connection error</strong>
            <span>{error}</span>
            <span>
              Make sure FastAPI is running on port 8000.
            </span>
          </div>
        )}

        {summary && !loading && !error && (
          <>
            <section className="hero-section">
              <div>
                <p className="eyebrow">
                  REAL-TIME ICU OVERVIEW
                </p>

                <h1>
                  Patient Safety Command Center
                </h1>

                <p className="hero-text">
                  Clinical rules, anomaly detection,
                  deterioration risk, treatment context,
                  and fluid status are combined into
                  one intelligent alert layer.
                </p>
              </div>

              <div className="hero-stat">
                <span>ICU STAYS</span>

                <strong>
                  {summary.icu_stays.toLocaleString()}
                </strong>
              </div>
            </section>

            <section className="stats-grid">
              <div className="stat-card">
                <span>Total patient-hours</span>

                <strong>
                  {summary.total_rows.toLocaleString()}
                </strong>
              </div>

              <div className="stat-card critical-card">
                <span>Critical</span>

                <strong>
                  {summary.alert_priorities.critical.toLocaleString()}
                </strong>
              </div>

              <div className="stat-card high-card">
                <span>High</span>

                <strong>
                  {summary.alert_priorities.high.toLocaleString()}
                </strong>
              </div>

              <div className="stat-card">
                <span>ML anomalies</span>

                <strong>
                  {summary.ml_anomalies.toLocaleString()}
                </strong>
              </div>

              <div className="stat-card">
                <span>Clinical severe</span>

                <strong>
                  {summary.clinical_severe.toLocaleString()}
                </strong>
              </div>

              <div className="stat-card multi-card">
                <span>Multi-signal</span>

                <strong>
                  {summary.multi_signal_alerts.toLocaleString()}
                </strong>
              </div>
            </section>

            <section className="risk-strip">
              <div>
                <span>Risk ≥ 10%</span>

                <strong>
                  {summary.risk_10_percent_or_higher.toLocaleString()}
                </strong>
              </div>

              <div>
                <span>Risk ≥ 25%</span>

                <strong>
                  {summary.risk_25_percent_or_higher.toLocaleString()}
                </strong>
              </div>

              <div>
                <span>Risk ≥ 50%</span>

                <strong>
                  {summary.risk_50_percent_or_higher.toLocaleString()}
                </strong>
              </div>

              <div>
                <span>Clinical warnings</span>

                <strong>
                  {summary.clinical_warnings.toLocaleString()}
                </strong>
              </div>
            </section>

            <section className="patients-section">
              <div className="section-header">
                <div>
                  <p className="eyebrow">
                    PATIENT MONITORING
                  </p>

                  <h2>ICU Patient Queue</h2>
                </div>

                <div className="filters">
                  {[
                    "all",
                    "critical",
                    "high",
                    "moderate",
                    "low",
                    "normal",
                  ].map((filter) => (
                    <button
                      key={filter}
                      className={
                        priorityFilter === filter
                          ? "filter active"
                          : "filter"
                      }
                      onClick={() =>
                        setPriorityFilter(filter)
                      }
                    >
                      {filter}
                    </button>
                  ))}
                </div>
              </div>

              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>Patient</th>
                      <th>Hour</th>
                      <th>Priority</th>
                      <th>Risk</th>
                      <th>ML anomaly</th>
                      <th>Clinical state</th>
                      <th>Explanation</th>
                    </tr>
                  </thead>

                  <tbody>
                    {visiblePatients.map(
                      (patient) => (
                        <tr
                          key={
                            patient.patientunitstayid
                          }
                          className="patient-row"
                          onClick={() =>
                            openPatient(
                              patient.patientunitstayid
                            )
                          }
                        >
                          <td className="patient-id">
                            <button
                              className="patient-link"
                              onClick={(event) => {
                                event.stopPropagation();

                                openPatient(
                                  patient.patientunitstayid
                                );
                              }}
                            >
                              #{patient.patientunitstayid}
                            </button>
                          </td>

                          <td>{patient.hour}</td>

                          <td>
                            <span
                              className={priorityClass(
                                patient.priority
                              )}
                            >
                              {patient.priority}
                            </span>
                          </td>

                          <td className="risk-value">
                            {formatRisk(
                              patient.risk_probability
                            )}
                          </td>

                          <td>
                            {patient.ml_anomaly ? (
                              <span className="yes">
                                YES
                              </span>
                            ) : (
                              <span className="no">
                                NO
                              </span>
                            )}
                          </td>

                          <td>
                            {patient.clinical_rule_severity}
                          </td>

                          <td className="explanation">
                            {patient.final_explanation}
                          </td>
                        </tr>
                      )
                    )}

                    {visiblePatients.length === 0 && (
                      <tr>
                        <td
                          colSpan={7}
                          className="empty"
                        >
                          No patients found.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </main>

      <footer className="footer">
        Smart-ICU research prototype · Model outputs are
        decision-support signals, not clinical diagnoses.
      </footer>

      {selectedPatientId !== null && (
        <div className="patient-overlay">
          <div className="patient-detail-panel">
            <div className="detail-topbar">
              <div>
                <p className="eyebrow">
                  PATIENT DETAIL / ICU HISTORY
                </p>

                <h2>
                  Patient #{selectedPatientId}
                </h2>
              </div>

              <button
                className="close-button"
                onClick={closePatient}
              >
                ×
              </button>
            </div>

            {detailLoading && (
              <div className="detail-loading">
                Loading patient history...
              </div>
            )}

            {detailError && (
              <div className="detail-error">
                {detailError}
              </div>
            )}

            {!detailLoading &&
              !detailError &&
              latestDetailRow && (
                <>
                  <section className="detail-summary">
                    <div className="detail-main-status">
                      <span
                        className={priorityClass(
                          latestDetailRow.priority
                        )}
                      >
                        {latestDetailRow.priority}
                      </span>

                      <div>
                        <span className="detail-label">
                          Current hour
                        </span>

                        <strong>
                          H{latestDetailRow.hour}
                        </strong>
                      </div>
                    </div>

                    <div className="detail-metric">
                      <span>24h deterioration risk</span>

                      <strong>
                        {formatRisk(
                          latestDetailRow.risk_probability
                        )}
                      </strong>

                      <span
                        className={riskClass(
                          latestDetailRow.risk_band
                        )}
                      >
                        {latestDetailRow.risk_band}
                      </span>
                    </div>

                    <div className="detail-metric">
                      <span>ML anomaly</span>

                      <strong
                        className={
                          latestDetailRow.ml_anomaly
                            ? "signal-danger"
                            : "signal-normal"
                        }
                      >
                        {latestDetailRow.ml_anomaly
                          ? "DETECTED"
                          : "NONE"}
                      </strong>
                    </div>

                    <div className="detail-metric">
                      <span>Anomaly score</span>

                      <strong>
                        {formatNumber(
                          latestDetailRow.anomaly_score,
                          3
                        )}
                      </strong>
                    </div>
                  </section>

                  <section className="detail-grid">
                    <div className="detail-card">
                      <span className="detail-card-label">
                        HEART RATE
                      </span>

                      <strong>
                        {formatNumber(
                          latestDetailRow.heart_rate
                        )}
                      </strong>

                      <small>bpm</small>
                    </div>

                    <div className="detail-card">
                      <span className="detail-card-label">
                        SpO₂
                      </span>

                      <strong>
                        {formatNumber(
                          latestDetailRow.spo2
                        )}
                      </strong>

                      <small>%</small>
                    </div>

                    <div className="detail-card">
                      <span className="detail-card-label">
                        RESPIRATION
                      </span>

                      <strong>
                        {formatNumber(
                          latestDetailRow.respiration
                        )}
                      </strong>

                      <small>/min</small>
                    </div>

                    <div className="detail-card">
                      <span className="detail-card-label">
                        MEAN BP
                      </span>

                      <strong>
                        {formatNumber(
                          latestDetailRow.mean_bp
                        )}
                      </strong>

                      <small>mmHg</small>
                    </div>

                    <div className="detail-card">
                      <span className="detail-card-label">
                        SYSTOLIC BP
                      </span>

                      <strong>
                        {formatNumber(
                          latestDetailRow.systolic_bp
                        )}
                      </strong>

                      <small>mmHg</small>
                    </div>

                    <div className="detail-card">
                      <span className="detail-card-label">
                        DIASTOLIC BP
                      </span>

                      <strong>
                        {formatNumber(
                          latestDetailRow.diastolic_bp
                        )}
                      </strong>

                      <small>mmHg</small>
                    </div>
                  </section>
                  <section className="recent-context-section">
  <div className="section-header detail-section-header">
    <div>
      <p className="eyebrow">
        RECENT CONTEXT
      </p>

      <h2>Last 6 hours</h2>
    </div>

    <span className="history-count">
      H{Math.max(0, latestDetailRow.hour - 5)}
      {" → "}
      H{latestDetailRow.hour}
    </span>
  </div>

  <div className="recent-context-grid">
    <div className="context-card">
      <span>HR TREND</span>

      <strong
        className={trendClass(
          getTrendDirection(
            patientHistory.slice(-6),
            "heart_rate"
          )
        )}
      >
        {trendSymbol(
          getTrendDirection(
            patientHistory.slice(-6),
            "heart_rate"
          )
        )}
      </strong>

      <small>Heart rate</small>
    </div>

    <div className="context-card">
      <span>SpO₂ TREND</span>

      <strong
        className={trendClass(
          getTrendDirection(
            patientHistory.slice(-6),
            "spo2"
          )
        )}
      >
        {trendSymbol(
          getTrendDirection(
            patientHistory.slice(-6),
            "spo2"
          )
        )}
      </strong>

      <small>Oxygen saturation</small>
    </div>

    <div className="context-card">
      <span>MAP TREND</span>

      <strong
        className={trendClass(
          getTrendDirection(
            patientHistory.slice(-6),
            "mean_bp"
          )
        )}
      >
        {trendSymbol(
          getTrendDirection(
            patientHistory.slice(-6),
            "mean_bp"
          )
        )}
      </strong>

      <small>Mean arterial pressure</small>
    </div>

    <div className="context-card">
      <span>RISK TREND</span>

      <strong
        className={trendClass(
          getTrendDirection(
            patientHistory.slice(-6),
            "risk_probability"
          )
        )}
      >
        {trendSymbol(
          getTrendDirection(
            patientHistory.slice(-6),
            "risk_probability"
          )
        )}
      </strong>

      <small>
        {formatRisk(
          patientHistory.slice(-6)[0]?.risk_probability ?? null
        )}
        {" → "}
        {formatRisk(latestDetailRow.risk_probability)}
      </small>
    </div>
  </div>
</section>

                  <section className="charts-section">
                    <div className="section-header detail-section-header">
                      <div>
                        <p className="eyebrow">
                          PHYSIOLOGICAL TRENDS
                        </p>

                        <h2>Patient trajectory</h2>
                      </div>

                      <span className="history-count">
                        {patientHistory.length} hourly records
                      </span>
                    </div>

                    <div className="charts-grid">
                      <TrendChart
                        title="Heart Rate"
                        unit="beats / min"
                        data={patientHistory}
                        field="heart_rate"
                      />

                      <TrendChart
                        title="SpO₂"
                        unit="oxygen saturation %"
                        data={patientHistory}
                        field="spo2"
                      />

                      <TrendChart
                        title="Mean Arterial Pressure"
                        unit="mmHg"
                        data={patientHistory}
                        field="mean_bp"
                      />

                      <TrendChart
                        title="Respiration"
                        unit="breaths / min"
                        data={patientHistory}
                        field="respiration"
                      />

                      <TrendChart
                        title="Net Fluid Balance"
                        unit="mL / hour"
                        data={patientHistory}
                        field="net_fluid_balance_ml"
                      />

                      <TrendChart
                        title="Urine Output"
                        unit="mL / hour"
                        data={patientHistory}
                        field="urine_output_ml"
                      />
                    </div>
                  </section>
                  <section className="iv-monitor-section">
  <div className="section-header detail-section-header">
    <div>
      <p className="eyebrow">
        IV FLUID MONITORING
      </p>

      <h2>Infusion status</h2>
    </div>

    <div className="iv-connection-status">
      <span
        className={
          ivTelemetry?.connected
            ? "iv-dot connected"
            : "iv-dot demo"
        }
      />

      {ivTelemetry?.connected
        ? "ESP32 CONNECTED"
        : "WAITING FOR ESP32"}
    </div>
  </div>

  <div className="iv-grid">
    <div className="iv-level-card">
      <div className="iv-card-header">
        <div>
          <span className="iv-label">
            ESTIMATED IV LEVEL
          </span>

          <strong>
            {formatNumber(
              ivTelemetry?.remaining_volume_ml ?? null,
              0
            )}{" "}
            mL
          </strong>
        </div>

        <span
          className={`iv-status iv-${ivTelemetry?.status ?? "normal"}`}
        >
          {ivTelemetry?.status ?? "normal"}
        </span>
      </div>

      <div className="iv-tank">
        <div
          className="iv-liquid"
          style={{
            height: `${ivTelemetry?.level_percent ?? 0}%`,
          }}
        />
      </div>

      <div className="iv-scale">
        <span>0%</span>
        <strong>
          {formatNumber(
            ivTelemetry?.level_percent ?? null,
            0
          )}
          %
        </strong>
        <span>100%</span>
      </div>
    </div>

    <div className="iv-metrics">
      <div className="iv-metric">
        <span>Bag weight</span>

        <strong>
          {formatNumber(
            ivTelemetry?.bag_weight_g ?? null,
            0
          )}{" "}
          g
        </strong>
      </div>

      <div className="iv-metric">
        <span>Initial weight</span>

        <strong>
          {formatNumber(
            ivTelemetry?.initial_weight_g ?? null,
            0
          )}{" "}
          g
        </strong>
      </div>

      <div className="iv-metric">
        <span>Estimated flow</span>

        <strong>
          {formatNumber(
            ivTelemetry?.flow_rate_ml_per_hr ?? null,
            0
          )}{" "}
          mL/hr
        </strong>
      </div>

      <div className="iv-metric">
        <span>Telemetry</span>

        <strong>
          {ivTelemetry?.last_update ?? "—"}
        </strong>
      </div>
    </div>

    <div className="iv-clinical-card">
      <p className="eyebrow">
        CLINICAL FLUID CONTEXT
      </p>

      <div className="iv-clinical-row">
        <span>Total intake</span>

        <strong>
          {formatNumber(
            latestDetailRow.total_intake_ml
          )}{" "}
          mL
        </strong>
      </div>

      <div className="iv-clinical-row">
        <span>Total output</span>

        <strong>
          {formatNumber(
            latestDetailRow.total_output_ml
          )}{" "}
          mL
        </strong>
      </div>

      <div className="iv-clinical-row">
        <span>Net balance</span>

        <strong>
          {formatNumber(
            latestDetailRow.net_fluid_balance_ml
          )}{" "}
          mL
        </strong>
      </div>

      <div className="iv-clinical-row">
        <span>Urine output</span>

        <strong>
          {formatNumber(
            latestDetailRow.urine_output_ml
          )}{" "}
          mL
        </strong>
      </div>
    </div>
  </div>

  <div className="iv-notice">
    <strong>Live hardware feed:</strong>{" "}
    ESP32 readings are displayed independently from historical eICU patient records. A disconnected or stale feed is not treated as a normal reading.
  </div>
</section>

                  <section className="detail-lower-grid">
                    <div className="intelligence-card">
  <p className="eyebrow">
    SIGNAL CONVERGENCE
  </p>

  <h3>
    Why this patient is being flagged
  </h3>

  <div className="convergence-grid">
    <div className="convergence-item">
      <span>Clinical rules</span>

      <strong
        className={
          latestDetailRow.clinical_rule_severity ===
          "critical"
            ? "convergence-danger"
            : ""
        }
      >
        {latestDetailRow.clinical_rule_severity.toUpperCase()}
      </strong>
    </div>

    <div className="convergence-item">
      <span>ML anomaly</span>

      <strong
        className={
          latestDetailRow.ml_anomaly
            ? "convergence-danger"
            : "convergence-safe"
        }
      >
        {latestDetailRow.ml_anomaly
          ? "DETECTED"
          : "NONE"}
      </strong>
    </div>

    <div className="convergence-item">
      <span>24h risk</span>

      <strong>
        {formatRisk(
          latestDetailRow.risk_probability
        )}
      </strong>
    </div>

    <div className="convergence-item">
      <span>Physiological trend</span>

      <strong className="convergence-danger">
        {getTrendDirection(
          patientHistory.slice(-6),
          "mean_bp"
        ) === "down"
          ? "DETERIORATING"
          : "STABLE / RISING"}
      </strong>
    </div>

    <div className="convergence-item">
      <span>Treatment context</span>

      <strong>
        {latestDetailRow.vasoactive_support
          ? "VASOACTIVE"
          : latestDetailRow.active_infusion_count &&
              latestDetailRow.active_infusion_count > 0
            ? "INFUSIONS"
            : "NONE DETECTED"}
      </strong>
    </div>

    <div className="convergence-item">
      <span>Fluid context</span>

      <strong>
        {formatNumber(
          latestDetailRow.net_fluid_balance_ml
        )}{" "}
        mL
      </strong>
    </div>
  </div>

                      <div className="signal-list">
                        <div className="signal-row">
                          <span>
                            Clinical state
                          </span>

                          <strong>
                            {
                              latestDetailRow.clinical_rule_severity
                            }
                          </strong>
                        </div>

                        <div className="signal-row">
                          <span>ML anomaly</span>

                          <strong>
                            {latestDetailRow.ml_anomaly
                              ? "Detected"
                              : "Not detected"}
                          </strong>
                        </div>

                        <div className="signal-row">
                          <span>Active infusions</span>

                          <strong>
                            {formatNumber(
                              latestDetailRow.active_infusion_count,
                              0
                            )}
                          </strong>
                        </div>

                        <div className="signal-row">
                          <span>Vasoactive support</span>

                          <strong>
                            {latestDetailRow.vasoactive_support
                              ? "Yes"
                              : "No"}
                          </strong>
                        </div>

                        <div className="signal-row">
                          <span>Fluid balance</span>

                          <strong>
                            {formatNumber(
                              latestDetailRow.net_fluid_balance_ml
                            )}{" "}
                            mL
                          </strong>
                        </div>

                        <div className="signal-row">
                          <span>Urine output</span>

                          <strong>
                            {formatNumber(
                              latestDetailRow.urine_output_ml
                            )}{" "}
                            mL
                          </strong>
                        </div>
                      </div>

                      <div className="explanation-box">
                        {latestDetailRow.final_explanation ||
                          "No active explanation recorded."}
                      </div>
                    </div>

                    <div className="alert-history-card">
                      <div className="alert-history-header">
                        <div>
                          <p className="eyebrow">
                            ALERT HISTORY
                          </p>

                          <h3>
                            Recent patient states
                          </h3>
                        </div>
                      </div>

                      <div className="history-list">
                        {patientHistory
                          .slice(-10)
                          .reverse()
                          .map((row) => (
                            <div
                              className="history-row"
                              key={`${row.patientunitstayid}-${row.hour}`}
                            >
                              <div className="history-hour">
                                H{row.hour}
                              </div>

                              <span
                                className={priorityClass(
                                  row.priority
                                )}
                              >
                                {row.priority}
                              </span>

                              <div className="history-risk">
  {formatRisk(
    row.risk_probability
  )}
</div>

<div className="history-source">
  {alertSource(row)}
</div>

<div className="history-text">
                                {row.final_explanation ||
                                  row.clinical_explanation ||
                                  "No active alert."}
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  </section>

                  <div className="detail-disclaimer">
                    Recorded measurements and model outputs are
                    shown for research and demonstration.
                    They are not a medical diagnosis or a
                    replacement for clinician assessment.
                  </div>
                </>
              )}

            {!detailLoading &&
              !detailError &&
              !latestDetailRow && (
                <div className="detail-empty">
                  No history data was returned for this patient.
                </div>
              )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;