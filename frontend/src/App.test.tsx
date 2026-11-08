import "@testing-library/jest-dom/vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const summary = {
  total_rows: 2,
  icu_stays: 2,
  alert_priorities: {
    normal: 0,
    low: 1,
    moderate: 0,
    high: 0,
    critical: 1,
  },
  ml_anomalies: 0,
  clinical_warnings: 0,
  clinical_severe: 0,
  risk_10_percent_or_higher: 1,
  risk_25_percent_or_higher: 0,
  risk_50_percent_or_higher: 0,
  multi_signal_alerts: 0,
};

function patient(id: number, priority: string) {
  return {
    patientunitstayid: id,
    hour: 12,
    priority,
    alert_category: "single-signal",
    risk_probability: 0.2,
    risk_band: "low",
    anomaly_score: null,
    ml_anomaly: false,
    clinical_rule_severity: "normal",
    clinical_explanation: "",
    final_explanation: "",
  };
}

function historyRow(id: number, hour: number) {
  return {
    patientunitstayid: id,
    hour,
    heart_rate: 80,
    spo2: 96,
    respiration: 18,
    systolic_bp: 120,
    diastolic_bp: 70,
    mean_bp: 86,
    total_intake_ml: 50,
    total_output_ml: 20,
    dialysis_ml: 0,
    reported_net_ml: 30,
    net_fluid_balance_ml: 30,
    urine_output_ml: 20,
    active_infusion_count: 0,
    vasoactive_support: 0,
    anomaly_score: null,
    ml_anomaly: false,
    risk_probability: 0.2,
    risk_band: "low",
    clinical_rule_severity: "normal",
    clinical_explanation: "",
    priority: "low",
    alert_category: "single-signal",
    final_explanation: "",
  };
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}

describe("Smart ICU dashboard", () => {
  let historyResponses: Map<number, Promise<Response>>;
  let hardwarePayload: Record<string, unknown>;
  let requestUrls: string[];
  let failSummaryRequest: boolean;
  let failClinicalPatientSearch: boolean;
  let mlStatusPayload: Record<string, unknown> | null;
  let mlPredictPayload: Record<string, unknown> | null;

  beforeEach(() => {
    requestUrls = [];
    failSummaryRequest = false;
    failClinicalPatientSearch = false;
    mlStatusPayload = null;
    mlPredictPayload = null;
    hardwarePayload = {
      connected: false,
      connection_status: "DISCONNECTED",
      fluid_ml: null,
      device_status: null,
      last_update: null,
      validity: { fluid: false },
    };
    historyResponses = new Map([
      [123, Promise.resolve(jsonResponse([historyRow(123, 10)]))],
      [456, Promise.resolve(jsonResponse([historyRow(456, 12)]))],
    ]);

    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        requestUrls.push(url);

        if (url.endsWith("/api/ml/status")) {
          if (mlStatusPayload !== null) {
            return Promise.resolve(jsonResponse(mlStatusPayload));
          }
          return Promise.resolve(new Response("Not found", { status: 404 }));
        }
        if (url.includes("/api/ml/predict/")) {
          if (mlPredictPayload !== null) {
            return Promise.resolve(jsonResponse(mlPredictPayload));
          }
          return Promise.resolve(new Response("Not found", { status: 404 }));
        }
        if (url.endsWith("/api/summary")) {
          if (failSummaryRequest) {
            return Promise.resolve(
              new Response("Unavailable", { status: 503 })
            );
          }
          return Promise.resolve(jsonResponse(summary));
        }
        if (url.includes("/api/patients?")) {
          return Promise.resolve(
            jsonResponse({
              count: 2,
              patients: [patient(123, "critical"), patient(456, "low")],
            })
          );
        }
        if (url.endsWith("/api/hardware/telemetry")) {
          return Promise.resolve(jsonResponse(hardwarePayload));
        }
        if (url.includes("/api/clinical-records/patient-stays")) {
          if (failClinicalPatientSearch) {
            return Promise.resolve(
              new Response(
                JSON.stringify({ detail: "Clinical API unavailable" }),
                { status: 503, headers: { "Content-Type": "application/json" } },
              ),
            );
          }
          return Promise.resolve(jsonResponse({ patient_ids: [123, 456] }));
        }
        if (url.includes("/api/patients/123/clinical-records")) {
          return Promise.resolve(jsonResponse({ records: [] }));
        }
        if (url.includes("/api/alerts/summary")) {
          return Promise.resolve(
            jsonResponse({
              total_alerts: 2,
              active_alerts: 2,
              unacknowledged_alerts: 2,
              critical_alerts: 0,
              high_priority_alerts: 0,
              resolved_alerts: 0,
            })
          );
        }
        if (url.includes("/api/alerts")) {
          return Promise.resolve(
            jsonResponse({
              alerts: [
                {
                  alert_id: "test-alert-1",
                  patient_id: 1,
                  stay_id: 1,
                  patientunitstayid: null,
                  source: "ESP32_TELEMETRY",
                  record_id: "esp-1",
                  fingerprint: "pat-1:ESP32:IV",
                  severity: "MODERATE",
                  category: "infusion",
                  measurement_name: "IV Fluid Level",
                  measured_value: 450,
                  unit: "mL",
                  reference_range: "> 500 mL",
                  critical_threshold: "< 200 mL",
                  reference_source: "Hospital Standard Infusion Protocol",
                  explanation: "Low IV fluid level (450 mL)",
                  status: "ACTIVE",
                  trigger_time: "2026-11-08T10:00:00Z",
                  acknowledged_at: null,
                  acknowledged_by: null,
                  resolved_at: null,
                  resolved_by: null,
                  resolution_note: null,
                  superseded_by_alert_id: null,
                  escalation_count: 0,
                  created_at: "2026-11-08T10:00:00Z",
                  updated_at: "2026-11-08T10:00:00Z",
                },
                {
                  alert_id: "test-alert-2",
                  patient_id: 1,
                  stay_id: 1,
                  patientunitstayid: null,
                  source: "LAB_RESULT",
                  record_id: "lab-1",
                  fingerprint: "pat-1:LAB:HGB",
                  severity: "MODERATE",
                  category: "hematology",
                  measurement_name: "Hemoglobin",
                  measured_value: 9.0,
                  unit: "g/dL",
                  reference_range: "12.0 - 16.0 g/dL",
                  critical_threshold: "< 7.0 g/dL",
                  reference_source: "Tietz Textbook of Clinical Chemistry",
                  explanation: "Moderate Anemia 9.0 g/dL",
                  status: "ACTIVE",
                  trigger_time: "2026-11-08T10:05:00Z",
                  acknowledged_at: null,
                  acknowledged_by: null,
                  resolved_at: null,
                  resolved_by: null,
                  resolution_note: null,
                  superseded_by_alert_id: null,
                  escalation_count: 0,
                  created_at: "2026-11-08T10:05:00Z",
                  updated_at: "2026-11-08T10:05:00Z",
                },
              ],
            })
          );
        }

        const patientId = Number(url.split("/").pop());
        return historyResponses.get(patientId) ??
          Promise.resolve(jsonResponse([]));
      })
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("labels the historical dataset and unavailable telemetry accurately", async () => {
    render(<App />);

    expect(
      await screen.findByText("HISTORICAL ICU OVERVIEW")
    ).toBeInTheDocument();
    expect(
      screen.getByText(/expired ICU discharge within 24 hours/)
    ).toBeInTheDocument();

    const patientButton = await screen.findByRole("button", {
      name: "#123",
    });
    patientButton.focus();
    fireEvent.click(patientButton);

    const dialog = await screen.findByRole("dialog");
    const closeButton = within(dialog).getByRole("button", {
      name: "Close patient detail",
    });
    expect(closeButton).toHaveFocus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(closeButton).toHaveFocus();

    expect(
      within(dialog).getByText("Latest recorded hour")
    ).toBeInTheDocument();
    expect(
      within(dialog).getByText(
        "Probability of expired ICU discharge within 24 hours"
      )
    ).toBeInTheDocument();

    const telemetry = dialog.querySelector<HTMLElement>(
      ".iv-monitor-section"
    );
    if (!telemetry) {
      throw new Error("IV telemetry section was not rendered.");
    }
    const telemetrySection = within(telemetry);
    expect(
      telemetrySection.getByText("No fresh, valid fluid reading is available.")
    ).toBeInTheDocument();
    expect(telemetrySection.getByText("UNAVAILABLE")).toBeInTheDocument();
    expect(telemetrySection.queryByText("1000 g")).not.toBeInTheDocument();
    expect(telemetrySection.queryByText("0 mL/hr")).not.toBeInTheDocument();

    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
    expect(patientButton).toHaveFocus();
  });

  it("requests the selected priority filter", async () => {
    render(<App />);
    await screen.findByText("HISTORICAL ICU OVERVIEW");

    fireEvent.click(screen.getByRole("button", { name: "critical" }));

    await waitFor(() => {
      expect(requestUrls).toContain(
        "http://127.0.0.1:8000/api/patients?priority=critical&limit=100"
      );
    });
  });

  it("opens the independent clinical records section", async () => {
    render(<App />);
    await screen.findByText("HISTORICAL ICU OVERVIEW");
    fireEvent.click(
      screen.getByRole("button", {
        name: "Clinical Observations & Lab Results",
      })
    );

    expect(
      await screen.findByRole("heading", {
        name: "Clinical Observations & Lab Results",
      })
    ).toBeInTheDocument();
    expect(
      screen.getByText(/separate from historical eICU data, ESP32 telemetry/)
    ).toBeInTheDocument();
  });

  it("scopes the unavailable status to historical dashboard data", async () => {
    failSummaryRequest = true;
    render(<App />);

    expect(
      await screen.findByText("HISTORICAL DATA UNAVAILABLE"),
    ).toBeInTheDocument();
    expect(screen.getByText("Failed to load summary")).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", {
        name: "Clinical Observations & Lab Results",
      }),
    );
    expect(
      await screen.findByRole("heading", {
        name: "Clinical Observations & Lab Results",
      }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("option", { name: "123" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Clinical records API: available")).toBeInTheDocument();
  });

  it("keeps historical dashboard status loaded when the clinical API fails", async () => {
    failClinicalPatientSearch = true;
    render(<App />);
    await screen.findByText("HISTORICAL ICU OVERVIEW");
    fireEvent.click(
      screen.getByRole("button", {
        name: "Clinical Observations & Lab Results",
      }),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Clinical API unavailable",
    );
    expect(
      screen.getByText("Clinical records API: unavailable"),
    ).toBeInTheDocument();
    expect(screen.getByText("HISTORICAL DATA LOADED")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Dashboard" }));
    expect(await screen.findByText("HISTORICAL ICU OVERVIEW")).toBeInTheDocument();
    expect(screen.getByText("HISTORICAL DATA LOADED")).toBeInTheDocument();
  });

  it("ignores an older patient response after a newer selection", async () => {
    const delayedFirstHistory = deferred<Response>();
    historyResponses.set(123, delayedFirstHistory.promise);

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "#123" }));
    fireEvent.click(screen.getByRole("button", { name: "#456" }));

    expect(
      await screen.findByRole("heading", { name: "Patient #456" })
    ).toBeInTheDocument();
    expect(
      within(screen.getByRole("dialog")).getAllByText("H12").length
    ).toBeGreaterThan(0);

    delayedFirstHistory.resolve(
      jsonResponse([historyRow(123, 10)])
    );

    await waitFor(() => {
      expect(
        screen.getByRole("heading", { name: "Patient #456" })
      ).toBeInTheDocument();
      expect(
        within(screen.getByRole("dialog")).getAllByText("H12").length
      ).toBeGreaterThan(0);
      expect(screen.queryByText("H10")).not.toBeInTheDocument();
    });
  });

  it("keeps a closed detail view closed when its request completes", async () => {
    const delayedHistory = deferred<Response>();
    historyResponses.set(123, delayedHistory.promise);

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "#123" }));

    const dialog = await screen.findByRole("dialog");
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Close patient detail" })
    );
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    delayedHistory.resolve(
      jsonResponse([historyRow(123, 10)])
    );

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
  });

  it("shows a fresh validated sensor value without estimating volume or flow", async () => {
    hardwarePayload = {
      connected: true,
      connection_status: "CONNECTED",
      fluid_ml: 42.5,
      device_status: "NORMAL",
      last_update: "2026-03-19T10:00:00+00:00",
      validity: { fluid: true },
    };

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "#123" }));

    const dialog = await screen.findByRole("dialog");
    const reading = dialog.querySelector(".iv-card-header strong");
    const status = dialog.querySelector(".iv-status");
    expect(reading?.textContent).toContain("42.5");
    expect(status?.textContent).toBe("NORMAL");
    expect(within(dialog).queryByText("Estimated flow")).not.toBeInTheDocument();
    expect(within(dialog).queryByText("Estimated IV level")).not.toBeInTheDocument();
  });

  it("does not display invalid zero telemetry as a normal reading", async () => {
    hardwarePayload = {
      connected: true,
      connection_status: "CONNECTED",
      fluid_ml: 0,
      device_status: "NORMAL",
      last_update: "2026-03-19T10:00:00+00:00",
      validity: { fluid: false },
    };

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "#123" }));

    const dialog = await screen.findByRole("dialog");
    const telemetry = dialog.querySelector<HTMLElement>(
      ".iv-monitor-section"
    );
    if (!telemetry) {
      throw new Error("IV telemetry section was not rendered.");
    }
    const telemetrySection = within(telemetry);
    expect(telemetrySection.getByText("UNAVAILABLE")).toBeInTheDocument();
    expect(telemetrySection.getByText("No fresh, valid fluid reading is available."))
      .toBeInTheDocument();
  });

  it("navigates to Clinical Alerts section and renders active clinical alerts", async () => {
    render(<App />);

    const alertsNavButton = await screen.findByRole("button", {
      name: "Clinical Alerts",
    });
    fireEvent.click(alertsNavButton);

    expect(
      await screen.findByText("Unified Clinical Alert & Interpretation Engine")
    ).toBeInTheDocument();
    expect(await screen.findByText("IV Fluid Level")).toBeInTheDocument();
    expect(await screen.findByText("Hemoglobin")).toBeInTheDocument();
  });

  it("renders ML Online status banner with real validation metrics when ML is active", async () => {
    mlStatusPayload = {
      status: "ONLINE",
      deterioration_model: {
        artifact: "deterioration_logistic_v1.joblib",
        model_name: "deterioration_logistic_v1",
        version: "1.0.0",
        target: "ICU discharge marked expired within the next 24 hours from the prediction timestamp",
        available: true,
        loaded: true,
        status: "AVAILABLE",
        decision_threshold: 0.0823,
        validation_metrics: {
          roc_auc: 0.6875,
          pr_auc: 0.1257,
          brier_score: 0.0154,
          decision_threshold: 0.0823,
          precision: 0.1758,
          recall_sensitivity: 0.1709,
          specificity: 0.9864,
          f1_score: 0.1733,
        },
      },
      anomaly_model: {
        artifact: "anomaly_isolation_forest_v2.joblib",
        model_name: "anomaly_isolation_forest_v2",
        version: "2.0.0",
        available: true,
        loaded: true,
        status: "AVAILABLE",
      },
    };

    render(<App />);

    expect(await screen.findByText("ML ONLINE")).toBeInTheDocument();
    expect(screen.getByText("deterioration_logistic_v1")).toBeInTheDocument();
    expect(screen.getByText("0.6875")).toBeInTheDocument();
    expect(screen.getByText("0.1257")).toBeInTheDocument();
    expect(screen.getByText("0.0154")).toBeInTheDocument();
    expect(screen.getByText("Academic Research Prototype")).toBeInTheDocument();
  });

  it("displays genuine ML predictions and top contributing features in patient detail", async () => {
    mlPredictPayload = {
      patientunitstayid: 123,
      total_hours: 1,
      latest_hour: 10,
      model_name: "deterioration_logistic_v1",
      model_version: "1.0.0",
      target: "ICU discharge marked expired within the next 24 hours from the prediction timestamp",
      latest_prediction: {
        probability: 0.235,
        risk_band: "High",
        decision_threshold: 0.0823,
        threshold_exceeded: true,
        top_contributing_features: [
          {
            feature: "mean_bp_rolling_mean_3h",
            value: 65.0,
            weight: -0.2987,
            contribution: 0.4521,
            abs_contribution: 0.4521,
            direction: "INCREASING_RISK",
          },
          {
            feature: "clinical_severe_count",
            value: 2.0,
            weight: 0.256,
            contribution: 0.3812,
            abs_contribution: 0.3812,
            direction: "INCREASING_RISK",
          },
        ],
      },
      anomaly: {
        available: true,
        latest_score: 0.123,
        latest_flag: 0,
      },
      disclaimer: "Academic prototype disclaimer",
    };

    render(<App />);

    const patientButton = await screen.findByRole("button", { name: "#123" });
    fireEvent.click(patientButton);

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getAllByText("23.5%").length).toBeGreaterThanOrEqual(1);
    expect(within(dialog).getAllByText("High").length).toBeGreaterThanOrEqual(1);
    expect(within(dialog).getByText("PREDICTIVE ML INTELLIGENCE")).toBeInTheDocument();
    expect(within(dialog).getByText("mean_bp_rolling_mean_3h")).toBeInTheDocument();
    expect(within(dialog).getByText("clinical_severe_count")).toBeInTheDocument();
    expect(within(dialog).getByText("+0.4521")).toBeInTheDocument();
  });
});
