import "@testing-library/jest-dom/vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PatientManagement from "./PatientManagement";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const mockPatient1 = {
  id: 1,
  patient_code: "P001",
  name: "Alice Walker",
  age: 58,
  sex: "F",
  date_of_birth: "1968-04-12",
  contact_information: "+1-555-0101",
  status: "active",
  created_at: "2026-11-08T00:00:00Z",
  updated_at: "2026-11-08T00:00:00Z",
  stays_count: 1,
  active_stay: {
    id: 101,
    stay_code: "S001",
    patient_id: 1,
    admission_datetime: "2026-11-08T00:10:00Z",
    discharge_datetime: null,
    bed_number: "ICU-Bed-04",
    ward_or_unit: "Cardiothoracic ICU",
    admission_reason: "Post-op coronary bypass observation",
    discharge_status: "active",
    created_at: "2026-11-08T00:10:00Z",
    updated_at: "2026-11-08T00:10:00Z",
  },
  active_device: {
    device_id: "ESP32-001",
    device_name: "ICU Bed 4 Telemetry Unit",
    status: "connected",
    assigned_at: "2026-11-08T00:15:00Z",
    assignment_status: "active",
    last_telemetry: {
      fluid_ml: 350,
      bpm: 78,
      spo2: 99,
      device_status: "NORMAL",
      reading_time: "2026-11-08T01:00:00Z",
    },
  },
};

const mockPatientDetail = {
  ...mockPatient1,
  stays: [mockPatient1.active_stay],
  device_assignments: [
    {
      id: 1,
      device_id: "ESP32-001",
      device_name: "ICU Bed 4 Telemetry Unit",
      patient_id: 1,
      stay_id: 101,
      assigned_at: "2026-11-08T00:15:00Z",
      unassigned_at: null,
      assignment_status: "active",
    },
  ],
  latest_telemetry: {
    reading_time: "2026-11-08T01:00:00Z",
    fluid_ml: 350,
    bpm: 78,
    spo2: 99,
    device_status: "NORMAL",
    connection_status: "CONNECTED",
  },
};

const mockTimeline = {
  patient_id: 1,
  events: [
    {
      event_type: "admission",
      timestamp: "2026-11-08T00:10:00Z",
      summary: "ICU Admission S001 created",
      source: "ICU Admission",
      details: {},
    },
    {
      event_type: "device_assigned",
      timestamp: "2026-11-08T00:15:00Z",
      summary: "ESP32-001 assigned to stay S001",
      source: "Device Assignment",
      details: {},
    },
  ],
};

describe("PatientManagement Component", () => {
  let requests: Array<{ url: string; init?: RequestInit }>;

  beforeEach(() => {
    requests = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        requests.push({ url, init });

        if (url.includes("/api/patients?") || url.endsWith("/api/patients")) {
          if (init?.method === "POST") {
            const body = JSON.parse(String(init.body));
            return jsonResponse({
              id: 2,
              patient_code: body.patient_code || "P002",
              name: body.name,
              age: body.age || null,
              sex: body.sex || null,
              date_of_birth: body.date_of_birth || null,
              contact_information: body.contact_information || null,
              status: "active",
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
              stays_count: 0,
              active_stay: null,
              active_device: null,
            });
          }
          return jsonResponse({ patients: [mockPatient1] });
        }

        if (url.includes("/api/registered-patients/1")) {
          return jsonResponse(mockPatientDetail);
        }

        if (url.includes("/api/registered-patients/2")) {
          return jsonResponse({
            id: 2,
            patient_code: "P002",
            name: "Bob Smith",
            age: 62,
            sex: null,
            date_of_birth: null,
            contact_information: null,
            status: "active",
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            stays: [],
            active_stay: null,
            active_device: null,
            device_assignments: [],
            latest_telemetry: null,
          });
        }

        if (url.includes("/timeline")) {
          return jsonResponse(mockTimeline);
        }

        if (url.includes("/stays") && init?.method === "POST") {
          const body = JSON.parse(String(init.body));
          return jsonResponse({
            id: 102,
            stay_code: "S002",
            patient_id: 1,
            admission_datetime: new Date().toISOString(),
            discharge_datetime: null,
            bed_number: body.bed_number || "ICU-Bed-05",
            ward_or_unit: body.ward_or_unit || "General ICU",
            admission_reason: body.admission_reason || "Observation",
            discharge_status: "active",
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          });
        }

        if (url.includes("/discharge")) {
          return jsonResponse({
            ...mockPatient1.active_stay,
            discharge_datetime: new Date().toISOString(),
            discharge_status: "discharged",
          });
        }

        if (url.includes("/api/devices") && !url.includes("assign")) {
          return jsonResponse({
            devices: [
              {
                id: 1,
                device_id: "ESP32-001",
                device_name: "ICU Bed 4 Telemetry Unit",
                connection_type: "serial",
                port: "COM9",
                status: "connected",
                active_assignment: {
                  patient_id: 1,
                  stay_id: 101,
                  patient_name: "Alice Walker",
                  stay_code: "S001",
                },
              },
            ],
          });
        }

        if (url.includes("/clinical-records")) {
          return jsonResponse({ records: [] });
        }

        return jsonResponse({});
      }),
    );
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders registered patients list with patient information and badges", async () => {
    render(<PatientManagement />);

    await waitFor(() => {
      expect(screen.getByText("Alice Walker")).toBeInTheDocument();
    });

    expect(screen.getByText("P001")).toBeInTheDocument();
    expect(screen.getByText("ICU-Bed-04")).toBeInTheDocument();
    expect(screen.getByText(/ESP32-001/)).toBeInTheDocument();
  });

  it("opens patient detail view and displays stays and timeline", async () => {
    render(<PatientManagement />);

    await waitFor(() => {
      expect(screen.getByText("Alice Walker")).toBeInTheDocument();
    });

    const viewButton = screen.getByRole("button", { name: "View Details →" });
    fireEvent.click(viewButton);

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /Alice Walker/ })).toBeInTheDocument();
    });

    expect(screen.getByText(/P001/)).toBeInTheDocument();
    expect(screen.getAllByText(/Cardiothoracic ICU/).length).toBeGreaterThan(0);
    expect(screen.getByText("350.0")).toBeInTheDocument();
    expect(screen.getByText("78")).toBeInTheDocument();
    expect(screen.getByText("99%")).toBeInTheDocument();

    // Timeline section
    expect(screen.getByText("ICU Admission S001 created")).toBeInTheDocument();
    expect(screen.getByText("ESP32-001 assigned to stay S001")).toBeInTheDocument();

    // Back button returns to list
    const backButton = screen.getByRole("button", { name: "← Back to Patient List" });
    fireEvent.click(backButton);

    await waitFor(() => {
      expect(screen.getByText("Alice Walker")).toBeInTheDocument();
    });
  });

  it("does not let a stale patient response overwrite the current selection", async () => {
    let resolveAlice!: (response: Response) => void;
    let resolveBob!: (response: Response) => void;
    const aliceResponse = new Promise<Response>((resolve) => {
      resolveAlice = resolve;
    });
    const bobResponse = new Promise<Response>((resolve) => {
      resolveBob = resolve;
    });
    const bobListItem = {
      ...mockPatient1,
      id: 2,
      patient_code: "P002",
      name: "Bob Smith",
      active_stay: null,
      active_device: null,
    };

    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("/api/patients?")) {
          return Promise.resolve(jsonResponse({ patients: [mockPatient1, bobListItem] }));
        }
        if (url.endsWith("/api/registered-patients/1")) return aliceResponse;
        if (url.endsWith("/api/registered-patients/2")) return bobResponse;
        if (url.includes("/timeline")) return Promise.resolve(jsonResponse({ events: [] }));
        return Promise.resolve(jsonResponse({}));
      }),
    );

    render(<PatientManagement />);

    await waitFor(() => {
      expect(screen.getAllByRole("button", { name: "View Details →" })).toHaveLength(2);
    });

    fireEvent.click(screen.getAllByRole("button", { name: "View Details →" })[0]);
    fireEvent.click(screen.getByRole("button", { name: "← Back to Patient List" }));
    await waitFor(() => {
      expect(screen.getAllByRole("button", { name: "View Details →" })).toHaveLength(2);
    });
    fireEvent.click(screen.getAllByRole("button", { name: "View Details →" })[1]);

    resolveBob(
      jsonResponse({
        ...mockPatientDetail,
        id: 2,
        patient_code: "P002",
        name: "Bob Smith",
        active_stay: null,
        active_device: null,
        stays: [],
        latest_telemetry: null,
      }),
    );
    await screen.findByRole("heading", { name: /Bob Smith/ });

    resolveAlice(jsonResponse(mockPatientDetail));
    await waitFor(() => {
      expect(screen.getByRole("heading", { name: /Bob Smith/ })).toBeInTheDocument();
      expect(screen.queryByRole("heading", { name: /Alice Walker/ })).not.toBeInTheDocument();
    });
  });

  it("allows registering a new patient via modal form", async () => {
    render(<PatientManagement />);

    await waitFor(() => {
      expect(screen.getByText("Alice Walker")).toBeInTheDocument();
    });

    const registerBtn = screen.getByRole("button", { name: "+ Register Patient" });
    fireEvent.click(registerBtn);

    expect(screen.getByRole("heading", { name: "Register New ICU Patient" })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/Full Patient Name \*/i), {
      target: { value: "Bob Smith" },
    });
    fireEvent.change(screen.getByLabelText(/Patient Code/i), {
      target: { value: "P002" },
    });
    fireEvent.change(screen.getByLabelText(/Age/i), {
      target: { value: "62" },
    });

    fireEvent.click(screen.getByRole("button", { name: "Register Patient" }));

    await waitFor(() => {
      const postReq = requests.find(
        (r) => r.url.endsWith("/api/patients") && r.init?.method === "POST",
      );
      expect(postReq).toBeDefined();
      expect(JSON.parse(String(postReq?.init?.body))).toMatchObject({
        name: "Bob Smith",
        patient_code: "P002",
        age: 62,
      });
    });
  });

  it("supports discharging an active ICU stay", async () => {
    render(<PatientManagement />);

    await waitFor(() => {
      expect(screen.getByText("Alice Walker")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "View Details →" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Discharge Stay" })).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: "Discharge Stay" }));

    expect(screen.getByRole("heading", { name: "Discharge ICU Stay" })).toBeInTheDocument();

    const confirmBtn = screen.getByRole("button", { name: "Confirm Discharge" });
    fireEvent.click(confirmBtn);

    await waitFor(() => {
      const dischargeReq = requests.find(
        (r) => r.url.includes("/discharge") && r.init?.method === "POST",
      );
      expect(dischargeReq).toBeDefined();
    });
  });
});
