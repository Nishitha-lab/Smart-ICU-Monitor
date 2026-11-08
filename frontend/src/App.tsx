import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import "./App.css";
import ClinicalRecords from "./ClinicalRecords";
import PatientManagement from "./PatientManagement";
import DeviceManagement from "./DeviceManagement";
import AlertPanel from "./AlertPanel";
import { API_BASE } from "./api";

type MLStatus = {
  status: "ONLINE" | "UNAVAILABLE" | "DEGRADED";
  academic_disclaimer?: string;
  deterioration_model: {
    artifact?: string;
    artifact_path?: string;
    model_name?: string;
    model_type?: string;
    version?: string;
    target?: string;
    target_definition?: string;
    status: string;
    loaded?: boolean;
    available?: boolean;
    missing_path?: string | null;
    decision_threshold?: number;
    validation_metrics?: {
      roc_auc: number;
      pr_auc: number;
      brier_score: number;
      decision_threshold: number;
      precision: number;
      recall_sensitivity: number;
      specificity: number;
      f1_score: number;
    } | null;
    training_samples?: number;
    positive_samples?: number;
    feature_count?: number;
    top_features?: Array<{
      feature: string;
      weight?: number;
      abs_weight?: number;
    }>;
    notes?: string;
  };
  anomaly_model: {
    artifact?: string;
    artifact_path?: string;
    model_name?: string;
    model_type?: string;
    version?: string;
    status: string;
    loaded?: boolean;
    available?: boolean;
    missing_path?: string | null;
    statistical_interpretation?: string;
    notes?: string;
  };
  winning_model?: string;
  diagnostic_notes?: string[];
};

type MLPredictionResult = {
  patientunitstayid: number;
  total_hours: number;
  latest_hour: number;
  model_name: string;
  model_version: string;
  target: string;
  latest_prediction?: {
    probability: number;
    risk_band: string;
    decision_threshold: number;
    threshold_exceeded: boolean;
    top_contributing_features: Array<{
      feature: string;
      value: number | null;
      weight: number;
      contribution: number;
      abs_contribution: number;
      direction: string;
    }>;
  };
  anomaly?: {
    available: boolean;
    latest_score: number | null;
    latest_flag: number;
  };
  disclaimer?: string;
};

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
  fluidMl: number | null;
  deviceStatus: string | null;
  connectionStatus: string;
  lastUpdate: string | null;
};
type HardwareTelemetryResponse = {
  connected: boolean;
  connection_status: string;
  fluid_ml: number | null;
  device_status: string | null;
  last_update: string | null;
  validity: {
    fluid: boolean;
  };
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

function toIvTelemetry(
  data: HardwareTelemetryResponse
): IvTelemetry {
  const fluidMl =
    data.connected &&
    data.validity.fluid &&
    typeof data.fluid_ml === "number" &&
    Number.isFinite(data.fluid_ml)
      ? data.fluid_ml
      : null;

  return {
    fluidMl,
    deviceStatus:
      fluidMl !== null && typeof data.device_status === "string"
        ? data.device_status
        : null,
    connectionStatus: data.connection_status,
    lastUpdate: data.last_update,
  };
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
  const [activeSection, setActiveSection] = useState<
    "dashboard" | "patients" | "clinical" | "devices" | "alerts"
  >(() => {
    if (typeof window !== "undefined") {
      const hash = window.location.hash.replace("#", "").toLowerCase();
      if (hash === "alerts") return "alerts";
      const params = new URLSearchParams(window.location.search);
      const section = params.get("section") || params.get("tab");
      if (section && ["dashboard", "patients", "clinical", "devices", "alerts"].includes(section.toLowerCase())) {
        return section.toLowerCase() as "dashboard" | "patients" | "clinical" | "devices" | "alerts";
      }
    }
    return "dashboard";
  });
  const [mlStatus, setMlStatus] = useState<MLStatus | null>(null);
  const [priorityFilter, setPriorityFilter] = useState("all");

  const [selectedPatientId, setSelectedPatientId] =
    useState<number | null>(null);

  const [patientHistory, setPatientHistory] = useState<
    PatientHistoryRow[]
  >([]);
  const [patientML, setPatientML] = useState<MLPredictionResult | null>(null);

  useEffect(() => {
    async function loadMLStatus() {
      try {
        const res = await fetch(`${API_BASE}/api/ml/status`);
        if (res.ok) {
          const data: MLStatus = await res.json();
          setMlStatus(data);
        }
      } catch {
        // ML status offline
      }
    }
    void loadMLStatus();
  }, []);

  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);

  const [error, setError] = useState("");
  const [detailError, setDetailError] = useState("");
  const [ivTelemetry, setIvTelemetry] =
    useState<IvTelemetry | null>(null);
  const [telemetryError, setTelemetryError] = useState("");
  const detailRequestRef = useRef<AbortController | null>(null);
  const detailPanelRef = useRef<HTMLDivElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    async function loadDashboard() {
      try {
        const summaryResponse = await fetch(
          `${API_BASE}/api/summary`,
          { signal: controller.signal }
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

        const patientResponse = await fetch(patientUrl, {
          signal: controller.signal,
        });

        if (!patientResponse.ok) {
          throw new Error("Failed to load patients");
        }

        const patientData = await patientResponse.json();

        if (controller.signal.aborted) {
          return;
        }

        setSummary(summaryData);
        setPatients(patientData.patients ?? []);
        setError("");
      } catch (err) {
        if (controller.signal.aborted) {
          return;
        }

        setError(
          err instanceof Error
            ? err.message
            : "Unable to connect to Smart-ICU backend."
        );
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void loadDashboard();
    return () => controller.abort();
  }, [priorityFilter]);

  async function openPatient(patientId: number) {
    detailRequestRef.current?.abort();
    const controller = new AbortController();
    detailRequestRef.current = controller;

    if (selectedPatientId === null) {
      returnFocusRef.current =
        document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null;
    }

    setSelectedPatientId(patientId);
    setPatientHistory([]);
    setPatientML(null);
    setDetailLoading(true);
    setDetailError("");

    try {
      const response = await fetch(
        `${API_BASE}/api/patients/${patientId}`,
        { signal: controller.signal }
      );

      if (!response.ok) {
        throw new Error(
          `Failed to load patient #${patientId}`
        );
      }

      const data = await response.json();

      if (controller.signal.aborted) {
        return;
      }

      const patientHistoryData: PatientHistoryRow[] =
        Array.isArray(data)
          ? data
          : Array.isArray(data.history)
            ? data.history
            : [];

      setPatientHistory(patientHistoryData);

      // Attempt to load genuine retrospective ML prediction & explainability
      try {
        const mlResponse = await fetch(
          `${API_BASE}/api/ml/predict/${patientId}`,
          { signal: controller.signal }
        );
        if (mlResponse.ok && !controller.signal.aborted) {
          const mlData: MLPredictionResult = await mlResponse.json();
          setPatientML(mlData);
        }
      } catch {
        // Retrospective ML prediction not available for non-eICU patient
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        setDetailError(
          err instanceof Error
            ? err.message
            : "Unable to load patient history."
        );
      }
    } finally {
      if (detailRequestRef.current === controller) {
        detailRequestRef.current = null;
        setDetailLoading(false);
      }
    }
  }

  const closePatient = useCallback(() => {
    detailRequestRef.current?.abort();
    detailRequestRef.current = null;
    setSelectedPatientId(null);
    setPatientHistory([]);
    setPatientML(null);
    setDetailError("");
  }, []);

  useEffect(() => {
    if (selectedPatientId === null) {
      return;
    }

    closeButtonRef.current?.focus();

    function handleDialogKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        closePatient();
        return;
      }

      if (event.key !== "Tab" || !detailPanelRef.current) {
        return;
      }

      const focusableElements =
        detailPanelRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
      const first = focusableElements[0];
      const last = focusableElements[focusableElements.length - 1];

      if (!first || !last) {
        event.preventDefault();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", handleDialogKeyDown);
    return () => {
      document.removeEventListener("keydown", handleDialogKeyDown);
      returnFocusRef.current?.focus();
      returnFocusRef.current = null;
    };
  }, [closePatient, selectedPatientId]);

  useEffect(() => () => {
    detailRequestRef.current?.abort();
  }, []);

  useEffect(() => {
    let active = true;
    const loadHardware = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/hardware/telemetry`);
        if (!response.ok) throw new Error("Hardware telemetry unavailable");
        const data: HardwareTelemetryResponse =
          await response.json();
        if (!active) return;
        setIvTelemetry(toIvTelemetry(data));
        setTelemetryError("");
      } catch (err) {
        if (active) {
          setIvTelemetry(null);
          setTelemetryError(
            err instanceof Error
              ? err.message
              : "Hardware telemetry unavailable."
          );
        }
      }
    };
    loadHardware();
    const timer = window.setInterval(loadHardware, 1000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  const latestDetailRow =
  patientHistory.length > 0
    ? patientHistory[patientHistory.length - 1]
    : null;
  const deviceStatus = ivTelemetry?.deviceStatus?.toUpperCase() ?? null;
  const deviceStatusClass =
    deviceStatus === "NORMAL"
      ? "normal"
      : deviceStatus === "WARNING"
        ? "low"
        : deviceStatus === "CRITICAL"
          ? "critical"
          : "unknown";

  function selectPriorityFilter(filter: string) {
    if (filter !== priorityFilter) {
      setLoading(true);
      setPriorityFilter(filter);
    }
  }

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
          <span
            className={`status-dot ${error ? "status-dot-offline" : ""}`}
          />
          {loading
            ? "HISTORICAL DATA CONNECTING"
            : error
              ? "HISTORICAL DATA UNAVAILABLE"
              : "HISTORICAL DATA LOADED"}
        </div>
      </header>

      {mlStatus && mlStatus.status === "UNAVAILABLE" && (
        <div className="ml-status-banner" role="status">
          <div className="ml-status-badge">ML OFFLINE</div>
          <div className="ml-status-text">
            <strong>Model Artifacts Missing:</strong> Neither <code>deterioration_logistic_v1.joblib</code> nor <code>anomaly_isolation_forest_v2.joblib</code> are deployed.
            Predictive deterioration target remains strictly <em>ICU discharge marked expired within the next 24 hours from the prediction timestamp</em>. Deterministic clinical rule guidelines (KDIGO 2012, Tietz, AHA) and hardware safeguards active.
          </div>
        </div>
      )}

      {mlStatus && (mlStatus.status === "ONLINE" || mlStatus.deterioration_model?.available) && (
        <div className="ml-status-banner ml-status-banner-online" role="status">
          <div className="ml-status-badge ml-status-badge-online">ML ONLINE</div>
          <div className="ml-status-text">
            <strong>Active Model:</strong> <code>{mlStatus.deterioration_model.model_name || "deterioration_logistic_v1"}</code> (v{mlStatus.deterioration_model.version || "1.0.0"}) &bull; Unweighted Logistic Regression (natural training prevalence) &bull; Target: <em>{mlStatus.deterioration_model.target || "ICU discharge marked expired within the next 24 hours from the prediction timestamp"}</em> &bull;
            <strong> Metrics:</strong> ROC-AUC: <strong>{mlStatus.deterioration_model.validation_metrics?.roc_auc?.toFixed(4) ?? "0.6875"}</strong> &bull; PR-AUC: <strong>{mlStatus.deterioration_model.validation_metrics?.pr_auc?.toFixed(4) ?? "0.1257"}</strong> &bull; Brier Score: <strong>{mlStatus.deterioration_model.validation_metrics?.brier_score?.toFixed(4) ?? "0.0154"}</strong> &bull; Decision Threshold: <strong>{mlStatus.deterioration_model.decision_threshold?.toFixed(4) ?? "0.0823"}</strong> &bull; Anomaly: <code>{mlStatus.anomaly_model.model_name || "anomaly_isolation_forest_v2"}</code>
            <span className="ml-research-pill">Academic Research Prototype</span>
          </div>
        </div>
      )}

      <nav className="module-navigation" aria-label="Application sections">
        <button
          type="button"
          aria-current={activeSection === "dashboard" ? "page" : undefined}
          onClick={() => setActiveSection("dashboard")}
        >
          Dashboard
        </button>
        <button
          type="button"
          aria-current={activeSection === "patients" ? "page" : undefined}
          onClick={() => setActiveSection("patients")}
        >
          Patients
        </button>
        <button
          type="button"
          aria-current={activeSection === "clinical" ? "page" : undefined}
          onClick={() => setActiveSection("clinical")}
        >
          Clinical Observations &amp; Lab Results
        </button>
        <button
          type="button"
          aria-current={activeSection === "devices" ? "page" : undefined}
          onClick={() => setActiveSection("devices")}
        >
          Device Management
        </button>
        <button
          type="button"
          aria-current={activeSection === "alerts" ? "page" : undefined}
          onClick={() => setActiveSection("alerts")}
        >
          Clinical Alerts
        </button>
      </nav>

      <main className="dashboard">
        {activeSection === "dashboard" && loading && (
          <div className="loading-card">
            Loading ICU intelligence...
          </div>
        )}

        {activeSection === "dashboard" && error && (
          <div className="error-card">
            <strong>Backend connection error</strong>
            <span>{error}</span>
            <span>
              Make sure FastAPI is running on port 8000.
            </span>
          </div>
        )}

        {activeSection === "patients" && (
          <PatientManagement />
        )}

        {activeSection === "clinical" && (
          <ClinicalRecords />
        )}

        {activeSection === "devices" && (
          <DeviceManagement />
        )}

        {activeSection === "alerts" && (
          <AlertPanel />
        )}

        {summary && !loading && !error && activeSection === "dashboard" && (
          <>
            <section className="hero-section">
              <div>
                <p className="eyebrow">
                  HISTORICAL ICU OVERVIEW
                </p>

                <h1>
                  Patient Safety Command Center
                </h1>

                <p className="hero-text">
                  Historical eICU records and precomputed research
                  alerts are shown separately from live ESP32
                  telemetry, when available.
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
                        selectPriorityFilter(filter)
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
                    {patients.map(
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

                    {patients.length === 0 && (
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

            <p className="research-scope">
              Historical eICU demo data. The model target is expired ICU
              discharge within 24 hours, not general clinical deterioration.
              Not validated for patient-care use.
            </p>
          </>
        )}
      </main>

      <footer className="footer">
        Smart-ICU research prototype · Retrospective demo outputs are not
        clinical diagnoses and are not for patient-care use.
      </footer>

      {selectedPatientId !== null && (
        <div className="patient-overlay">
          <div
            className="patient-detail-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="patient-detail-title"
            ref={detailPanelRef}
            tabIndex={-1}
          >
            <div className="detail-topbar">
              <div>
                <p className="eyebrow">
                  PATIENT DETAIL / ICU HISTORY
                </p>

                <h2>
                  <span id="patient-detail-title">
                  Patient #{selectedPatientId}
                  </span>
                </h2>
              </div>

              <button
                className="close-button"
                aria-label="Close patient detail"
                ref={closeButtonRef}
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
                          Latest recorded hour
                        </span>

                        <strong>
                          H{latestDetailRow.hour}
                        </strong>
                      </div>
                    </div>

                    <div className="detail-metric">
                      <span>
                        Probability of expired ICU discharge within 24 hours
                      </span>

                      <strong>
                        {patientML?.latest_prediction
                          ? `${(patientML.latest_prediction.probability * 100).toFixed(1)}%`
                          : formatRisk(latestDetailRow.risk_probability)}
                      </strong>

                      <span
                        className={riskClass(
                          patientML?.latest_prediction?.risk_band || latestDetailRow.risk_band
                        )}
                      >
                        {patientML?.latest_prediction?.risk_band || latestDetailRow.risk_band}
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
          ivTelemetry?.connectionStatus === "CONNECTED"
            ? "iv-dot connected"
            : "iv-dot demo"
        }
      />

      {telemetryError
        ? "TELEMETRY SERVICE UNAVAILABLE"
        : ivTelemetry?.connectionStatus.replaceAll("_", " ")
          ?? "WAITING FOR TELEMETRY"}
    </div>
  </div>

  <div className="iv-grid">
    <div className="iv-level-card">
      <div className="iv-card-header">
        <div>
          <span className="iv-label">
            ESP32 FLUID SENSOR READING
          </span>

          <strong>
            {formatNumber(
              ivTelemetry?.fluidMl ?? null
            )}{" "}
            mL
          </strong>
        </div>

        <span
          className={`iv-status iv-${deviceStatusClass}`}
        >
          {deviceStatus ?? "UNAVAILABLE"}
        </span>
      </div>

      <div className="iv-reading-note">
        {ivTelemetry?.fluidMl === null || !ivTelemetry
          ? "No fresh, valid fluid reading is available."
          : "Raw device value; remaining volume and flow are not inferred."}
      </div>
    </div>

    <div className="iv-metrics">
      <div className="iv-metric">
        <span>Connection state</span>

        <strong>
          {telemetryError
            ? "UNAVAILABLE"
            : ivTelemetry?.connectionStatus.replaceAll("_", " ")
              ?? "WAITING"}
        </strong>
      </div>

      <div className="iv-metric">
        <span>Last valid update</span>

        <strong>
          {ivTelemetry?.lastUpdate ?? "—"}
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
    {telemetryError ? (
      <>
        <strong>Telemetry unavailable:</strong> {telemetryError}
      </>
    ) : (
      <>
        <strong>Live hardware feed:</strong>{" "}
        ESP32 telemetry is separate from historical eICU records. Only fresh,
        valid fluid readings are shown; remaining volume and flow are not
        inferred.
      </>
    )}
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
        {patientML?.latest_prediction
          ? `${(patientML.latest_prediction.probability * 100).toFixed(1)}%`
          : formatRisk(latestDetailRow.risk_probability)}
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

                    {patientML?.latest_prediction && (
                      <div className="ml-model-card" aria-label="Machine Learning Explainability">
                        <p className="eyebrow">PREDICTIVE ML INTELLIGENCE</p>
                        <h3>Risk Drivers &amp; Attribution</h3>
                        <div className="ml-card-model-meta">
                          <span>Model: <strong>{patientML.model_name}</strong> (v{patientML.model_version})</span>
                          <span>Target: <em>{patientML.target}</em></span>
                          <span>Estimated 24h Risk: <strong>{(patientML.latest_prediction.probability * 100).toFixed(2)}%</strong></span>
                          <span>Experimental Band: <strong className={riskClass(patientML.latest_prediction.risk_band)}>{patientML.latest_prediction.risk_band}</strong></span>
                          <span>Threshold: <strong>{patientML.latest_prediction.decision_threshold.toFixed(4)}</strong> (<span className={patientML.latest_prediction.threshold_exceeded ? "badge-exceeded" : "badge-normal"}>{patientML.latest_prediction.threshold_exceeded ? "Exceeded" : "Nominal"}</span>)</span>
                        </div>

                        {patientML.latest_prediction.top_contributing_features.length > 0 && (
                          <div className="ml-feature-attribution-table">
                            <div className="ml-feature-table-title">Top Contributing Features (Logistic Regression Log-Odds Impact)</div>
                            <table className="ml-feature-table">
                              <thead>
                                <tr>
                                  <th>Feature</th>
                                  <th>Observed Value</th>
                                  <th>Model Weight</th>
                                  <th>Impact</th>
                                  <th>Direction</th>
                                </tr>
                              </thead>
                              <tbody>
                                {patientML.latest_prediction.top_contributing_features.map((feat) => (
                                  <tr key={feat.feature}>
                                    <td><code>{feat.feature}</code></td>
                                    <td>{feat.value !== null ? feat.value : "—"}</td>
                                    <td>{feat.weight > 0 ? `+${feat.weight.toFixed(4)}` : feat.weight.toFixed(4)}</td>
                                    <td>{feat.contribution > 0 ? `+${feat.contribution.toFixed(4)}` : feat.contribution.toFixed(4)}</td>
                                    <td>
                                      <span className={feat.direction === "INCREASING_RISK" ? "impact-badge impact-up" : "impact-badge impact-down"}>
                                        {feat.direction === "INCREASING_RISK" ? "▲ Escalates Risk" : "▼ Mitigates Risk"}
                                      </span>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                        <p className="ml-card-disclaimer">
                          <strong>Academic Prototype Notice:</strong> Retrospective rolling 24-hour prediction model for investigational research decision-support only. Risk bands are experimental presentation categories, not validated clinical severity classifications. Not an FDA/CE-cleared medical device.
                        </p>
                      </div>
                    )}
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