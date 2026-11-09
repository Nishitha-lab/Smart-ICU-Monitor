import "@testing-library/jest-dom/vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import DataConsistencyPanel, {
  type ConsistencyEvaluationReport,
} from "./DataConsistencyPanel";

const mockReport: ConsistencyEvaluationReport = {
  patient_id: 101,
  stay_id: 202,
  evaluated_at: "2026-11-09T01:30:00Z",
  level1_cross_source: [
    {
      metric: "Heart Rate",
      canonical_metric: "heart_rate",
      telemetry_value: 80.0,
      telemetry_time: "2026-11-09T01:29:50Z",
      clinical_value: 125.0,
      clinical_time: "2026-11-09T01:29:55Z",
      clinical_record_id: "rec-hr-01",
      time_difference_seconds: 5.0,
      status: "CRITICAL_DISCREPANCY",
      absolute_difference: 45.0,
      relative_difference_pct: 56.25,
      explanation: "Significant divergence between sensor and bedside pulse reading (45 bpm).",
      clinical_rationale: "Rapid changes in cardiac rhythm or loose sensor clip can cause rate mismatch.",
      recommended_action: "Perform manual radial pulse check and inspect ESP32 PPG sensor clip.",
      requires_alert: true,
      alert_severity: "CRITICAL",
    },
    {
      metric: "Oxygen Saturation",
      canonical_metric: "spo2",
      telemetry_value: 98.0,
      telemetry_time: "2026-11-09T01:29:50Z",
      clinical_value: 98.0,
      clinical_time: "2026-11-09T01:29:55Z",
      clinical_record_id: "rec-spo2-01",
      time_difference_seconds: 5.0,
      status: "CONSISTENT",
      absolute_difference: 0.0,
      relative_difference_pct: 0.0,
      explanation: "Telemetry and bedside record are concordant (0% delta).",
      clinical_rationale: "Measurement concordant within physiological variance.",
      recommended_action: "Continue standard ICU monitoring protocol.",
      requires_alert: false,
      alert_severity: "INFO",
    },
  ],
  level2_patient_trends: [
    {
      metric: "Heart Rate",
      canonical_metric: "heart_rate",
      current_value: 80.0,
      current_time: "2026-11-09T01:29:50Z",
      previous_value: null,
      previous_time: null,
      baseline_mean: null,
      baseline_std: null,
      observation_count: 1,
      delta_from_baseline: null,
      delta_from_previous: null,
      rate_of_change_per_minute: null,
      time_delta_seconds: null,
      status: "INSUFFICIENT_PATIENT_HISTORY",
      is_anomaly: false,
      severity: "INFO",
      explanation: "Minimum 3 observations required for patient-specific moving baseline (currently 1).",
      clinical_recommendation: "Allow additional telemetry or bedside data collection before establishing patient-specific moving baseline.",
    },
  ],
  level3_population_comparison: [
    {
      metric: "Heart rate",
      canonical_metric: "heart_rate",
      measured_value: 80.0,
      reference_median: 84.0,
      reference_iqr: 22.0,
      percentile_rank: 41.5,
      robust_z_score: -0.12,
      is_anomaly: false,
      anomaly_type: "NORMAL",
      severity: "INFO",
      explanation: "Within expected eICU ICU reference distribution (41.5th percentile).",
      clinical_validation_recommendation: "Value consistent with typical ICU population distribution.",
    },
  ],
  isolation_forest: {
    is_evaluated: true,
    status: "INLIER",
    anomaly_score: 0.1245,
    is_anomaly: false,
    severity: "INFO",
    evaluated_features: ["heart_rate", "spo2"],
    missing_features: ["respiration", "systolic_bp"],
    explanation: "Multi-feature pattern evaluated as typical relative to historical ICU population (score: 0.1245).",
    recommendation: "Maintain standard observation protocol.",
  },
  overall_status: "ANOMALIES_DETECTED",
  active_alerts_count: 1,
};

const mockHistory = {
  patient_id: 101,
  stay_id: 202,
  consistency_checks: [
    {
      check_id: 1,
      metric: "Heart Rate",
      telemetry_value: 80.0,
      clinical_value: 125.0,
      discrepancy_level: "CRITICAL_DISCREPANCY",
      recommended_action: "Perform manual radial pulse check",
      created_at: "2026-11-09T01:30:00Z",
    },
  ],
  patient_trend_anomalies: [],
  historical_population_anomalies: [],
};

describe("DataConsistencyPanel", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        if (url.includes("/api/consistency/patient/101/history")) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () => Promise.resolve(mockHistory),
          });
        }
        if (url.includes("/api/consistency/patient/101")) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () => Promise.resolve(mockReport),
          });
        }
        if (url.includes("/api/consistency/evaluate/101")) {
          return Promise.resolve({
            ok: true,
            status: 200,
            json: () => Promise.resolve(mockReport),
          });
        }
        return Promise.reject(new Error(`Unhandled URL: ${url}`));
      })
    );
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders panel header and Level 1 cross-source discrepancy table", async () => {
    render(<DataConsistencyPanel patientId={101} stayId={202} />);

    expect(
      screen.getByText(/Clinical Data Consistency & Intelligence Engine/i)
    ).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText(/Level 1: Source Consistency/i)).toBeInTheDocument();
    });

    expect(screen.getByText("Heart Rate")).toBeInTheDocument();
    expect(screen.getByText(/CRITICAL DISCREPANCY/i)).toBeInTheDocument();
    expect(screen.getByText(/CONSISTENT/i)).toBeInTheDocument();
  });

  it("displays explainability rationale when button clicked", async () => {
    render(<DataConsistencyPanel patientId={101} stayId={202} />);

    await waitFor(() => {
      expect(screen.getAllByText(/Explainability/i)[0]).toBeInTheDocument();
    });

    fireEvent.click(screen.getAllByText(/Explainability/i)[0]);

    await waitFor(() => {
      expect(
        screen.getByText(/Physiological & Sensor Rationale:/i)
      ).toBeInTheDocument();
      expect(
        screen.getByText(/Perform manual radial pulse check/i)
      ).toBeInTheDocument();
    });
  });

  it("renders Level 2 patient trend with minimum history guard banner", async () => {
    render(<DataConsistencyPanel patientId={101} stayId={202} />);

    await waitFor(() => {
      expect(
        screen.getByText(/Level 2: Patient Trend Intelligence/i)
      ).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText(/Level 2: Patient Trend Intelligence/i));

    await waitFor(() => {
      expect(screen.getAllByText(/Insufficient Patient History/i).length).toBeGreaterThanOrEqual(1);
      expect(
        screen.getByText(/Minimum 3 observations required for patient-specific moving baseline/i)
      ).toBeInTheDocument();
    });
  });

  it("renders Level 3 historical eICU population benchmarking and research disclaimer", async () => {
    render(<DataConsistencyPanel patientId={101} stayId={202} />);

    await waitFor(() => {
      expect(
        screen.getByText(/Level 3: Historical eICU Comparison/i)
      ).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText(/Level 3: Historical eICU Comparison/i));

    await waitFor(() => {
      expect(screen.getByText(/41.5%/i)).toBeInTheDocument();
      expect(screen.getByText(/Academic Research Prototype Notice:/i)).toBeInTheDocument();
      expect(
        screen.getByText(/Multi-Feature Isolation Forest Evaluation/i)
      ).toBeInTheDocument();
    });
  });

  it("triggers re-evaluation when button is clicked", async () => {
    render(<DataConsistencyPanel patientId={101} stayId={202} />);

    await waitFor(() => {
      expect(screen.getByText(/Re-evaluate Now/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText(/Re-evaluate Now/i));

    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledWith(
        expect.stringContaining("/api/consistency/evaluate/101"),
        expect.objectContaining({ method: "POST" })
      );
    });
  });
});
