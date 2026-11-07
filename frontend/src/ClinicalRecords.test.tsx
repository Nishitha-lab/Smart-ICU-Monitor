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
import ClinicalRecords from "./ClinicalRecords";

const record = {
  record_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
  patientunitstayid: 123,
  category: "urine_output",
  measurement_name: "Urine output",
  value: 125,
  unit: "mL",
  measurement_time: "2026-11-07T12:00:00Z",
  entry_time: "2026-11-07T12:01:00Z",
  version: 1,
  entered_by: "Operator One",
  source: "MANUAL_ENTRY",
  notes: null,
  observation_period: "1 hour",
  reference_range: null,
  sample_collection_time: null as string | null,
  result_time: null as string | null,
  updated_at: null,
  voided: false,
  voided_at: null,
  voided_by: null,
  void_reason: null,
};

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("Clinical observations and laboratory records", () => {
  let requests: Array<{ url: string; init?: RequestInit }>;
  let records: typeof record[];
  let failPost: boolean;
  let failHistory: boolean;
  let failUpdate: boolean;
  let failVoid: boolean;

  beforeEach(() => {
    requests = [];
    records = [];
    failPost = false;
    failHistory = false;
    failUpdate = false;
    failVoid = false;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        requests.push({ url, init });
        if (url.includes("/api/clinical-records/patient-stays")) {
          return jsonResponse({ patient_ids: [123, 456] });
        }
        if (url.includes("/api/patients/123/clinical-records")) {
          if (failHistory) return jsonResponse({ detail: "History unavailable" }, 503);
          return jsonResponse({ records: records.filter((item) => !item.voided) });
        }
        if (
          url.includes("/api/clinical-records/") &&
          (!init?.method || init.method === "GET")
        ) {
          return jsonResponse(records[0] ?? record);
        }
        if (url.includes("/api/clinical-records/") && init?.method === "PUT") {
          if (failUpdate) {
            failUpdate = false;
            records = [{ ...records[0], value: 140, version: 2 }];
            return jsonResponse(
              {
                detail:
                  "This record has changed since it was loaded. Reload the latest record before correcting it.",
              },
              409,
            );
          }
          const payload = JSON.parse(String(init.body));
          const updated = {
            ...records[0],
            ...payload,
            version: (records[0]?.version ?? 1) + 1,
          };
          delete updated.expected_version;
          delete updated.change_reason;
          records = [updated as typeof record];
          return jsonResponse(updated);
        }
        if (url.includes("/api/clinical-records/") && init?.method === "DELETE") {
          if (failVoid) {
            failVoid = false;
            records = [{ ...records[0], version: records[0].version + 1 }];
            return jsonResponse(
              { detail: "This record has changed. Reload the latest version before voiding." },
              409,
            );
          }
          const payload = JSON.parse(String(init.body));
          const voided = {
            ...records[0],
            version: records[0].version + 1,
            voided: true,
            voided_by: payload.voided_by,
            void_reason: payload.void_reason,
          };
          records = [voided as typeof record];
          return jsonResponse(voided);
        }
        if (url.endsWith("/api/clinical-records") && init?.method === "POST") {
          if (failPost) {
            return jsonResponse({ detail: "Record rejected" }, 422);
          }
          const created = {
            ...record,
            ...JSON.parse(String(init.body)),
          };
          records = [created as typeof record];
          return jsonResponse(created, 201);
        }
        return jsonResponse({ detail: "Unexpected request" }, 404);
      }),
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows an empty state and saves a unit-labelled patient record", async () => {
    render(<ClinicalRecords />);
    const patientSelect = await screen.findByRole("combobox", {
      name: "Select ICU stay ID",
    });
    await screen.findByRole("option", { name: "123" });
    fireEvent.change(patientSelect, { target: { value: "123" } });

    expect(
      await screen.findByText("No manual clinical records for this stay and filter."),
    ).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Result / volume"), {
      target: { value: "125" },
    });
    fireEvent.change(screen.getByLabelText("Operator identifier or display name"), {
      target: { value: "Operator One" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save clinical record" }));

    expect(await screen.findByText("Clinical record saved.")).toBeInTheDocument();
    expect(await screen.findByText("125 mL")).toBeInTheDocument();
    expect(screen.getByText("MANUAL_ENTRY")).toBeInTheDocument();
    expect(screen.getByText("Operator One")).toBeInTheDocument();

    const post = requests.find(
      (request) =>
        request.url.endsWith("/api/clinical-records") &&
        request.init?.method === "POST",
    );
    expect(post).toBeDefined();
    const payload = JSON.parse(String(post?.init?.body));
    expect(payload.patientunitstayid).toBe(123);
    expect(payload.value).toBe(125);
    expect(payload.unit).toBe("mL");
    expect(payload.measurement_time).toMatch(/Z$/);
  });

  it("updates dynamic blood-test and other-lab fields and filters history", async () => {
    render(<ClinicalRecords />);
    const patientSelect = await screen.findByRole("combobox", {
      name: "Select ICU stay ID",
    });
    await screen.findByRole("option", { name: "123" });
    fireEvent.change(patientSelect, { target: { value: "123" } });

    fireEvent.change(screen.getByLabelText("Category"), {
      target: { value: "blood_test" },
    });
    fireEvent.change(screen.getByLabelText("Blood test"), {
      target: { value: "Sodium" },
    });
    expect(screen.getByText("Unit: mmol/L (fixed; no conversion)")).toBeInTheDocument();
    expect(screen.getByLabelText("Sample collection time")).toBeRequired();
    expect(screen.getByLabelText("Measurement time")).toBeRequired();
    expect(screen.getByLabelText("Laboratory result time (optional)")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Category"), {
      target: { value: "other_lab" },
    });
    expect(screen.getByLabelText("Test name")).toBeInTheDocument();
    expect(screen.getByLabelText("Unit")).toBeRequired();
    expect(screen.getByLabelText("Reference range (optional)")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Category filter"), {
      target: { value: "other_lab" },
    });
    await waitFor(() => {
      expect(
        requests.some((request) =>
          request.url.includes("category=other_lab"),
        ),
      ).toBe(true);
    });
  });

  it("shows clear errors when saving or loading records fails", async () => {
    failHistory = true;
    failPost = true;
    render(<ClinicalRecords />);
    const patientSelect = await screen.findByRole("combobox", {
      name: "Select ICU stay ID",
    });
    await screen.findByRole("option", { name: "123" });
    fireEvent.change(patientSelect, { target: { value: "123" } });

    expect(await screen.findByRole("alert")).toHaveTextContent("History unavailable");
    fireEvent.change(screen.getByLabelText("Result / volume"), {
      target: { value: "10" },
    });
    fireEvent.change(screen.getByLabelText("Operator identifier or display name"), {
      target: { value: "Operator One" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save clinical record" }));
    expect(await screen.findByText("Record rejected")).toBeInTheDocument();
  });

  it("clears category-specific fields when changing categories", async () => {
    render(<ClinicalRecords />);
    const patientSelect = await screen.findByRole("combobox", {
      name: "Select ICU stay ID",
    });
    await screen.findByRole("option", { name: "123" });
    fireEvent.change(patientSelect, { target: { value: "123" } });

    fireEvent.change(screen.getByLabelText("Observation period (optional)"), {
      target: { value: "1 hour" },
    });
    fireEvent.change(screen.getByLabelText("Category"), {
      target: { value: "blood_test" },
    });
    fireEvent.change(screen.getByLabelText("Category"), {
      target: { value: "urine_output" },
    });

    expect(screen.getByLabelText("Observation period (optional)")).toHaveValue("");
    fireEvent.change(screen.getByLabelText("Result / volume"), {
      target: { value: "20" },
    });
    fireEvent.change(screen.getByLabelText("Operator identifier or display name"), {
      target: { value: "Operator One" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save clinical record" }));

    await screen.findByText("Clinical record saved.");
    const post = requests.find(
      (request) =>
        request.url.endsWith("/api/clinical-records") &&
        request.init?.method === "POST",
    );
    expect(JSON.parse(String(post?.init?.body)).observation_period).toBeNull();
  });

  it("prevents duplicate submissions while a save is in progress", async () => {
    render(<ClinicalRecords />);
    const patientSelect = await screen.findByRole("combobox", {
      name: "Select ICU stay ID",
    });
    await screen.findByRole("option", { name: "123" });
    fireEvent.change(patientSelect, { target: { value: "123" } });
    fireEvent.change(screen.getByLabelText("Result / volume"), {
      target: { value: "25" },
    });
    fireEvent.change(screen.getByLabelText("Operator identifier or display name"), {
      target: { value: "Operator One" },
    });
    const form = screen
      .getByRole("button", { name: "Save clinical record" })
      .closest("form");
    if (!form) throw new Error("Clinical record form was not rendered.");

    fireEvent.submit(form);
    fireEvent.submit(form);

    await screen.findByText("Clinical record saved.");
    expect(
      requests.filter(
        (request) =>
          request.url.endsWith("/api/clinical-records") &&
          request.init?.method === "POST",
      ),
    ).toHaveLength(1);
  });

  it("sends the expected version and offers to reload after a correction conflict", async () => {
    records = [{ ...record, version: 1 }];
    failUpdate = true;
    render(<ClinicalRecords />);
    const patientSelect = await screen.findByRole("combobox", {
      name: "Select ICU stay ID",
    });
    await screen.findByRole("option", { name: "123" });
    fireEvent.change(patientSelect, { target: { value: "123" } });
    fireEvent.click(await screen.findByRole("button", { name: "Correct record" }));

    fireEvent.change(screen.getByLabelText("Result / volume"), {
      target: { value: "130" },
    });
    fireEvent.change(screen.getByLabelText("Correction reason"), {
      target: { value: "Corrected transcription" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save correction" }));

    expect(
      await screen.findByRole("alert"),
    ).toHaveTextContent("Reload the latest record before correcting it.");
    const firstUpdate = requests.find((request) => request.init?.method === "PUT");
    expect(JSON.parse(String(firstUpdate?.init?.body)).expected_version).toBe(1);

    fireEvent.click(screen.getByRole("button", { name: "Reload latest record" }));
    await screen.findByText("Latest version loaded. Review it before correcting.");
    expect(screen.getByLabelText("Result / volume")).toHaveValue(140);

    fireEvent.change(screen.getByLabelText("Result / volume"), {
      target: { value: "145" },
    });
    fireEvent.change(screen.getByLabelText("Correction reason"), {
      target: { value: "Applied after reload" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save correction" }));
    await screen.findByText("Clinical record correction saved.");
    const updateRequests = requests.filter(
      (request) => request.init?.method === "PUT",
    );
    expect(JSON.parse(String(updateRequests[1].init?.body)).expected_version).toBe(2);
  });

  it("displays all laboratory timestamps separately and labels local time", async () => {
    records = [
      {
        ...record,
        category: "other_lab",
        measurement_time: "2026-11-07T10:00:00Z",
        sample_collection_time: "2026-11-07T11:00:00Z",
        result_time: "2026-11-07T12:00:00Z",
      },
    ];
    render(<ClinicalRecords />);
    const patientSelect = await screen.findByRole("combobox", {
      name: "Select ICU stay ID",
    });
    await screen.findByRole("option", { name: "123" });
    fireEvent.change(patientSelect, { target: { value: "123" } });

    expect(await screen.findByText("Measurement time")).toBeInTheDocument();
    expect(screen.getByText("Sample collection")).toBeInTheDocument();
    expect(screen.getByText("Result time")).toBeInTheDocument();
    expect(screen.getByText("Entered")).toBeInTheDocument();
    expect(screen.getAllByText(/\(local time\)/).length).toBe(4);
  });

  it("requires a reason and voids the active record without deleting its audit record", async () => {
    records = [{ ...record }];
    render(<ClinicalRecords />);
    const patientSelect = await screen.findByRole("combobox", {
      name: "Select ICU stay ID",
    });
    await screen.findByRole("option", { name: "123" });
    fireEvent.change(patientSelect, { target: { value: "123" } });

    const voidButton = await screen.findByRole("button", { name: "Void record" });
    voidButton.focus();
    fireEvent.click(voidButton);

    const dialog = screen.getByRole("dialog", { name: "Void clinical record?" });
    const reason = within(dialog).getByRole("textbox", { name: "Reason for voiding" });
    expect(reason).toHaveFocus();
    expect(reason).toBeRequired();
    expect(within(dialog).getByRole("button", { name: "Confirm void" }).closest("form"))
      .not.toBeNull();

    fireEvent.change(reason, { target: { value: "Duplicate synthetic entry" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Confirm void" }));

    expect(
      await screen.findByText(/Record voided and removed from active measurements/),
    ).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByText("125 mL")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Patient-specific record history" }))
      .toHaveFocus();

    const request = requests.find((item) => item.init?.method === "DELETE");
    expect(request).toBeDefined();
    expect(JSON.parse(String(request?.init?.body))).toMatchObject({
      expected_version: 1,
      voided_by: "Operator One",
      void_reason: "Duplicate synthetic entry",
    });
    expect(records[0]).toMatchObject({
      voided: true,
      void_reason: "Duplicate synthetic entry",
    });
  });

  it("restores focus on cancel and closes the void dialog with Escape", async () => {
    records = [{ ...record }];
    render(<ClinicalRecords />);
    const patientSelect = await screen.findByRole("combobox", {
      name: "Select ICU stay ID",
    });
    await screen.findByRole("option", { name: "123" });
    fireEvent.change(patientSelect, { target: { value: "123" } });

    const voidButton = await screen.findByRole("button", { name: "Void record" });
    voidButton.focus();
    fireEvent.click(voidButton);
    const dialog = screen.getByRole("dialog");
    expect(screen.getByRole("textbox", { name: "Reason for voiding" })).toHaveFocus();
    fireEvent.keyDown(dialog, { key: "Escape" });

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(voidButton).toHaveFocus();
    expect(requests.some((item) => item.init?.method === "DELETE")).toBe(false);
  });

  it("recovers from a stale void version and retries with the reloaded version", async () => {
    records = [{ ...record, version: 1 }];
    failVoid = true;
    render(<ClinicalRecords />);
    const patientSelect = await screen.findByRole("combobox", {
      name: "Select ICU stay ID",
    });
    await screen.findByRole("option", { name: "123" });
    fireEvent.change(patientSelect, { target: { value: "123" } });
    fireEvent.click(await screen.findByRole("button", { name: "Void record" }));

    const dialog = screen.getByRole("dialog");
    fireEvent.change(screen.getByLabelText("Reason for voiding"), {
      target: { value: "Duplicate synthetic entry" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Confirm void" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Reload the latest version before voiding.",
    );

    fireEvent.click(within(dialog).getByRole("button", { name: "Reload latest record" }));
    expect(
      await screen.findByText("Latest version loaded. Review it and enter the void reason again."),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Reason for voiding")).toHaveValue("");
    fireEvent.change(screen.getByLabelText("Reason for voiding"), {
      target: { value: "Reviewed after concurrent update" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Confirm void" }));

    await screen.findByText(/Record voided and removed from active measurements/);
    const voidRequests = requests.filter((item) => item.init?.method === "DELETE");
    expect(JSON.parse(String(voidRequests[0].init?.body)).expected_version).toBe(1);
    expect(JSON.parse(String(voidRequests[1].init?.body)).expected_version).toBe(2);
  });

  it("keeps keyboard focus within the void dialog", async () => {
    records = [{ ...record }];
    render(<ClinicalRecords />);
    const patientSelect = await screen.findByRole("combobox", {
      name: "Select ICU stay ID",
    });
    await screen.findByRole("option", { name: "123" });
    fireEvent.change(patientSelect, { target: { value: "123" } });
    fireEvent.click(await screen.findByRole("button", { name: "Void record" }));

    const dialog = screen.getByRole("dialog");
    const operator = screen.getByLabelText("Operator identifier or display name", {
      selector: "#clinical-voided-by",
    });
    const confirm = screen.getByRole("button", { name: "Confirm void" });
    confirm.focus();
    fireEvent.keyDown(dialog, { key: "Tab" });
    expect(operator).toHaveFocus();
    fireEvent.keyDown(dialog, { key: "Tab", shiftKey: true });
    expect(confirm).toHaveFocus();
  });
});
