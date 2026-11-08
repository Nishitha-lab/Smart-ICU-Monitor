import "@testing-library/jest-dom/vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import AlertPanel, { type AlertItem } from "./AlertPanel";

const mockBackendSummary = {
  total_alerts: 4,
  active_alerts: 4,
  unacknowledged_alerts: 4,
  critical_alerts: 1,
  high_priority_alerts: 2,
  resolved_alerts: 0,
  by_severity: { critical: 1, high: 2, moderate: 1, low: 0 },
  by_source: {
    LAB_RESULT: 1,
    URINE_OUTPUT: 1,
    ESP32_TELEMETRY: 1,
    CLINICAL_OBSERVATION: 1,
  },
};

const mockAlerts: AlertItem[] = [
  {
    alert_id: "alert-1111",
    patient_id: 1,
    stay_id: 1,
    patientunitstayid: null,
    source: "LAB_RESULT",
    record_id: "rec-1",
    fingerprint: "pat-1:LAB_RESULT:POTASSIUM",
    severity: "CRITICAL",
    category: "electrolyte",
    measurement_name: "Potassium",
    measured_value: 6.8,
    unit: "mmol/L",
    reference_range: "3.5 - 5.0 mmol/L",
    critical_threshold: ">= 6.0 mmol/L",
    reference_source: "Tietz Textbook of Clinical Chemistry",
    explanation: "Severe Hyperkalemia 6.8 mmol/L (cardiac arrest risk)",
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
    alert_id: "alert-2222",
    patient_id: 1,
    stay_id: 1,
    patientunitstayid: null,
    source: "URINE_OUTPUT",
    record_id: "rec-2",
    fingerprint: "pat-1:URINE_OUTPUT:URINE_OUTPUT",
    severity: "HIGH",
    category: "renal",
    measurement_name: "Urine Output",
    measured_value: 15.0,
    unit: "mL/h",
    reference_range: ">= 0.5 mL/kg/h",
    critical_threshold: "< 0.3 mL/kg/h",
    reference_source: "KDIGO 2012 Acute Kidney Injury Guidelines",
    explanation: "KDIGO Stage 2 Oliguria (0.21 mL/kg/h over 6 hours)",
    status: "ACTIVE",
    trigger_time: "2026-11-08T09:30:00Z",
    acknowledged_at: null,
    acknowledged_by: null,
    resolved_at: null,
    resolved_by: null,
    resolution_note: null,
    superseded_by_alert_id: null,
    escalation_count: 1,
    created_at: "2026-11-08T09:30:00Z",
    updated_at: "2026-11-08T09:30:00Z",
  },
  {
    alert_id: "alert-3333",
    patient_id: 2,
    stay_id: 2,
    patientunitstayid: null,
    source: "ESP32_TELEMETRY",
    record_id: "esp-1",
    fingerprint: "pat-2:ESP32:IV_FLUID",
    severity: "MODERATE",
    category: "infusion",
    measurement_name: "IV Fluid Level",
    measured_value: 450,
    unit: "mL",
    reference_range: "> 500 mL",
    critical_threshold: "< 200 mL",
    reference_source: "Hospital Standard Infusion Protocol",
    explanation: "Low IV fluid volume remaining (450 mL)",
    status: "ACTIVE",
    trigger_time: "2026-11-08T09:00:00Z",
    acknowledged_at: null,
    acknowledged_by: null,
    resolved_at: null,
    resolved_by: null,
    resolution_note: null,
    superseded_by_alert_id: null,
    escalation_count: 0,
    created_at: "2026-11-08T09:00:00Z",
    updated_at: "2026-11-08T09:00:00Z",
  },
  {
    alert_id: "alert-4444",
    patient_id: 2,
    stay_id: 2,
    patientunitstayid: null,
    source: "CLINICAL_OBSERVATION",
    record_id: "obs-1",
    fingerprint: "pat-2:OBS:TEMP",
    severity: "HIGH",
    category: "clinical_observation",
    measurement_name: "Body temperature",
    measured_value: 39.5,
    unit: "°C",
    reference_range: "36.5 - 37.5 °C",
    critical_threshold: ">= 39.0 °C",
    reference_source: "Hospital Standard ICU Protocol",
    explanation: "High Fever (39.5 °C)",
    status: "ACTIVE",
    trigger_time: "2026-11-08T08:45:00Z",
    acknowledged_at: null,
    acknowledged_by: null,
    resolved_at: null,
    resolved_by: null,
    resolution_note: null,
    superseded_by_alert_id: null,
    escalation_count: 0,
    created_at: "2026-11-08T08:45:00Z",
    updated_at: "2026-11-08T08:45:00Z",
  },
];

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("AlertPanel Component", () => {
  let requests: Array<{ url: string; init?: RequestInit }>;

  beforeEach(() => {
    requests = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        requests.push({ url, init });

        if (url.includes("/api/alerts/summary")) {
          return jsonResponse(mockBackendSummary);
        }
        if (url.includes("/api/alerts/alert-1111/history")) {
          return jsonResponse({
            alert_id: "alert-1111",
            history: [
              {
                log_id: 1,
                alert_id: "alert-1111",
                action: "CREATED",
                performed_by: "SYSTEM_ALERT_ENGINE",
                note: "Initial trigger",
                timestamp: "2026-11-08T10:00:00Z",
                details_json: null,
              },
            ],
          });
        }
        if (url.includes("/acknowledge")) {
          return jsonResponse({
            ...mockAlerts[0],
            status: "ACKNOWLEDGED",
            acknowledged_by: "Dr. ICU Attending",
          });
        }
        if (url.includes("/resolve")) {
          return jsonResponse({
            ...mockAlerts[0],
            status: "RESOLVED",
            resolved_by: "Dr. ICU Attending",
            resolution_note: "Patient treated with insulin-dextrose.",
          });
        }
        if (url.includes("/api/alerts")) {
          // FastAPI backend returns { alerts: [...] }
          return jsonResponse({ alerts: mockAlerts });
        }
        return jsonResponse({});
      })
    );
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders summary statistics and active alert cards", async () => {
    render(<AlertPanel />);

    // Check header
    expect(
      screen.getByText("Unified Clinical Alert & Interpretation Engine")
    ).toBeInTheDocument();

    // Check summary statistics
    await waitFor(() => {
      expect(screen.getByText("Active Alerts")).toBeInTheDocument();
      expect(screen.getByText("Critical Alerts")).toBeInTheDocument();
    });

    // Check alert cards
    expect(screen.getByText("Potassium")).toBeInTheDocument();
    expect(screen.getByText("Urine Output")).toBeInTheDocument();
    expect(screen.getByText("6.8 mmol/L")).toBeInTheDocument();
  });

  it("filters alerts by search query", async () => {
    render(<AlertPanel />);

    await waitFor(() => {
      expect(screen.getByText("Potassium")).toBeInTheDocument();
    });

    const searchInput = screen.getByLabelText("Search:");
    fireEvent.change(searchInput, { target: { value: "Oliguria" } });

    expect(screen.queryByText("Potassium")).not.toBeInTheDocument();
    expect(screen.getByText("Urine Output")).toBeInTheDocument();
  });

  it("expands 7 diagnostic questions and fetches audit history", async () => {
    render(<AlertPanel />);

    await waitFor(() => {
      expect(screen.getAllByText("▼ Diagnostic Details (7 Questions)")[0]).toBeInTheDocument();
    });

    const toggleBtn = screen.getAllByText("▼ Diagnostic Details (7 Questions)")[0];
    fireEvent.click(toggleBtn);

    // Verify 7 diagnostic questions rendered
    expect(screen.getByText(/1\. What triggered this alert\?/)).toBeInTheDocument();
    expect(screen.getByText(/2\. What was the exact measured value and unit\?/)).toBeInTheDocument();
    expect(screen.getByText(/3\. What is the reference range and critical threshold\?/)).toBeInTheDocument();
    expect(screen.getByText(/4\. Which clinical guideline or evidence basis was applied\?/)).toBeInTheDocument();
    expect(screen.getByText(/5\. Which patient \/ ICU stay does this belong to\?/)).toBeInTheDocument();
    expect(screen.getByText(/6\. Is this reading confirmed, escalating, or superseded\?/)).toBeInTheDocument();
    expect(screen.getByText(/7\. What audit history or clinician interventions have occurred\?/)).toBeInTheDocument();

    // Verify evidence basis and guideline
    expect(screen.getByText("Tietz Textbook of Clinical Chemistry")).toBeInTheDocument();

    // Verify audit log fetched
    await waitFor(() => {
      expect(screen.getByText("SYSTEM_ALERT_ENGINE")).toBeInTheDocument();
      expect(screen.getByText("Initial trigger")).toBeInTheDocument();
    });
  });

  it("allows acknowledging an active alert", async () => {
    render(<AlertPanel />);

    await waitFor(() => {
      expect(screen.getAllByText("✓ Acknowledge")[0]).toBeInTheDocument();
    });

    fireEvent.click(screen.getAllByText("✓ Acknowledge")[0]);

    // Modal opens
    expect(screen.getByText("Acknowledge Clinical Alert")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Dr. ICU Attending")).toBeInTheDocument();

    const confirmBtn = screen.getByText("Confirm Acknowledgment");
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      const ackReq = requests.find((r) => r.url.includes("/acknowledge"));
      expect(ackReq).toBeDefined();
    });
  });

  it("allows resolving an alert with required resolution note", async () => {
    render(<AlertPanel />);

    await waitFor(() => {
      expect(screen.getAllByText("Resolve Alert")[0]).toBeInTheDocument();
    });

    fireEvent.click(screen.getAllByText("Resolve Alert")[0]);

    // Modal opens
    expect(screen.getByText("Resolve Clinical Alert")).toBeInTheDocument();

    // Set resolution note
    const noteInput = screen.getByLabelText("Resolution Note (Required):");
    fireEvent.change(noteInput, {
      target: { value: "Calcium gluconate and insulin given. Potassium re-checked normal." },
    });

    const confirmBtn = screen.getByText("Confirm Resolution");
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      const resReq = requests.find((r) => r.url.includes("/resolve"));
      expect(resReq).toBeDefined();
      const body = JSON.parse(String(resReq?.init?.body));
      expect(body.resolution_note).toContain("Calcium gluconate");
    });

    // Verify alerts and summary were both refreshed
    await waitFor(() => {
      const summaryReqs = requests.filter((r) => r.url.includes("/api/alerts/summary"));
      expect(summaryReqs.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("supports alerts from ESP32_TELEMETRY, LAB_RESULT, and CLINICAL_OBSERVATION", async () => {
    render(<AlertPanel />);

    await waitFor(() => {
      expect(screen.getByText("IV Fluid Level")).toBeInTheDocument();
    });

    // Badges for all sources
    expect(screen.getAllByText("ESP32 Sensor").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Laboratory Result").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Clinical Observation").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Urine Output (KDIGO)").length).toBeGreaterThanOrEqual(1);
  });

  it("fetches /api/alerts cleanly without status param by default to prevent hiding active alerts", async () => {
    render(<AlertPanel />);

    await waitFor(() => {
      expect(screen.getByText("Potassium")).toBeInTheDocument();
    });

    const alertReq = requests.find((r) => r.url.includes("/api/alerts") && !r.url.includes("summary"));
    expect(alertReq).toBeDefined();
    // Default filter should not restrict status=ACTIVE so both unacknowledged and acknowledged are returned
    expect(alertReq!.url).not.toContain("status=");
  });

  it("displays distinct empty state when no alerts exist", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/alerts/summary")) {
          return jsonResponse({ ...mockBackendSummary, active_alerts: 0, total_alerts: 0 });
        }
        if (url.includes("/api/alerts")) {
          return jsonResponse({ alerts: [] });
        }
        return jsonResponse({});
      })
    );

    render(<AlertPanel />);

    await waitFor(() => {
      expect(screen.getByText("No alerts match the selected criteria.")).toBeInTheDocument();
    });
    expect(
      screen.getByText("ICU patient vitals and telemetry currently in safe physiological bands.")
    ).toBeInTheDocument();
  });

  it("displays distinct error state when alert service fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/alerts/summary")) {
          return jsonResponse(mockBackendSummary);
        }
        if (url.includes("/api/alerts")) {
          return new Response("Service Unavailable", { status: 503 });
        }
        return jsonResponse({});
      })
    );

    render(<AlertPanel />);

    await waitFor(() => {
      expect(screen.getByText("Failed to load clinical alerts")).toBeInTheDocument();
    });
    expect(screen.getAllByRole("alert")[0]).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});
