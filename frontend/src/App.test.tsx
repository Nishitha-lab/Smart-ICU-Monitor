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

  beforeEach(() => {
    requestUrls = [];
    failSummaryRequest = false;
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

  it("shows a clear unavailable state when the API fails", async () => {
    failSummaryRequest = true;
    render(<App />);

    expect(await screen.findByText("API UNAVAILABLE")).toBeInTheDocument();
    expect(screen.getByText("Failed to load summary")).toBeInTheDocument();
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
});
