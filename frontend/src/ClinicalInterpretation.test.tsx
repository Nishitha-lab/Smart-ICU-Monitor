import "@testing-library/jest-dom/vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import ClinicalRecords from "./ClinicalRecords";

const baseRecord = {
  record_id: "rec-1111-2222",
  patientunitstayid: 123,
  category: "blood_test",
  measurement_name: "Potassium",
  value: 6.8,
  unit: "mmol/L",
  measurement_time: "2026-11-08T12:00:00Z",
  entry_time: "2026-11-08T12:01:00Z",
  version: 1,
  entered_by: "Dr. Testing",
  source: "MANUAL_ENTRY",
  notes: null,
  observation_period: null,
  reference_range: "3.5 - 5.0 mmol/L",
  sample_collection_time: "2026-11-08T11:45:00Z",
  result_time: "2026-11-08T12:00:00Z",
  updated_at: null,
  voided: false,
  voided_at: null,
  voided_by: null,
  void_reason: null,
  interpretation: {
    status: "CRITICAL_HIGH",
    severity: "CRITICAL",
    category: "electrolyte",
    measurement_name: "Potassium",
    measured_value: 6.8,
    unit: "mmol/L",
    reference_range: "3.5 - 5.0 mmol/L",
    critical_threshold: ">= 6.0 mmol/L",
    reference_source: "Tietz Textbook of Clinical Chemistry",
    explanation: "Severe Hyperkalemia 6.8 mmol/L (cardiac arrest risk)",
    requires_alert: true,
  },
};

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("Clinical Interpretation & Previews", () => {
  let records: Record<string, unknown>[];

  beforeEach(() => {
    records = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);

        if (url.includes("/api/clinical-records/patient-stays")) {
          return jsonResponse({ patient_ids: [123, 456] });
        }
        if (url.includes("/clinical-records") && (!init?.method || init.method === "GET")) {
          return jsonResponse({ records });
        }
        if (url.includes("/api/clinical-records/evaluate")) {
          return jsonResponse({
            status: "CRITICAL_HIGH",
            severity: "CRITICAL",
            category: "electrolyte",
            measurement_name: "Potassium",
            measured_value: 6.8,
            unit: "mmol/L",
            reference_range: "3.5 - 5.0 mmol/L",
            critical_threshold: ">= 6.0 mmol/L",
            reference_source: "Tietz Textbook of Clinical Chemistry",
            explanation: "Severe Hyperkalemia 6.8 mmol/L",
            requires_alert: true,
          });
        }
        if (url.endsWith("/api/clinical-records") && init?.method === "POST") {
          const body = JSON.parse(String(init.body));
          const created = {
            ...baseRecord,
            ...body,
            interpretation: {
              status: "CRITICAL_HIGH",
              severity: "CRITICAL",
              category: "electrolyte",
              measurement_name: "Potassium",
              measured_value: Number(body.value),
              unit: "mmol/L",
              reference_range: "3.5 - 5.0 mmol/L",
              critical_threshold: ">= 6.0 mmol/L",
              reference_source: "Tietz Textbook of Clinical Chemistry",
              explanation: "Severe Hyperkalemia 6.8 mmol/L",
              requires_alert: true,
            },
            alert: {
              alert_id: "alt-9999",
              severity: "CRITICAL",
              status: "ACTIVE",
            },
          };
          records = [created];
          return jsonResponse(created, 201);
        }
        return jsonResponse({});
      })
    );
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("displays live preview while typing abnormal lab value", async () => {
    render(<ClinicalRecords initialStayId={123} />);

    // Select blood test
    const categorySelect = screen.getByLabelText("Category");
    fireEvent.change(categorySelect, { target: { value: "blood_test" } });

    // Enter potassium 6.8
    const valInput = screen.getByLabelText("Result / volume");
    fireEvent.change(valInput, { target: { value: "6.8" } });

    // Check live preview appears
    await waitFor(() => {
      expect(screen.getByText("LIVE PREVIEW")).toBeInTheDocument();
      expect(screen.getByText("CRITICAL HIGH")).toBeInTheDocument();
      expect(screen.getByText("Ref Range:")).toBeInTheDocument();
      expect(screen.getByText("3.5 - 5.0 mmol/L")).toBeInTheDocument();
    });
  });

  it("displays evaluated interpretation banner with evidence upon saving", async () => {
    render(<ClinicalRecords initialStayId={123} />);

    // Select blood test
    const categorySelect = screen.getByLabelText("Category");
    fireEvent.change(categorySelect, { target: { value: "blood_test" } });

    const valInput = screen.getByLabelText("Result / volume");
    fireEvent.change(valInput, { target: { value: "6.8" } });

    const sampleInput = screen.getByLabelText("Sample collection time");
    fireEvent.change(sampleInput, { target: { value: "2026-11-08T11:45" } });

    const opInput = screen.getByLabelText("Operator identifier or display name");
    fireEvent.change(opInput, { target: { value: "Nurse Specialist" } });

    const saveBtn = screen.getByRole("button", { name: "Save clinical record" });
    fireEvent.click(saveBtn);

    // Verify evaluated interpretation banner appears
    await waitFor(() => {
      expect(screen.getByRole("region", { name: "Clinical Interpretation Result" })).toBeInTheDocument();
      expect(screen.getByText("EVALUATED")).toBeInTheDocument();
      expect(screen.getByText("🚨 Alert Triggered")).toBeInTheDocument();
      expect(screen.getByText("Tietz Textbook of Clinical Chemistry")).toBeInTheDocument();
    });

    // Verify record appears in history with interpretation badge
    await waitFor(() => {
      expect(screen.getByText("Record ID: rec-1111-2222")).toBeInTheDocument();
    });
  });
});
