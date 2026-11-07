import "@testing-library/jest-dom/vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import DeviceManagement from "./DeviceManagement";

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const mockDeviceAssigned = {
  id: 1,
  device_id: "ESP32-001",
  device_name: "ICU Bed 4 Telemetry Unit",
  connection_type: "serial",
  port: "COM9",
  status: "connected",
  created_at: "2026-11-08T00:00:00Z",
  updated_at: "2026-11-08T00:00:00Z",
  active_assignment: {
    id: 1,
    patient_id: 1,
    stay_id: 101,
    assigned_at: "2026-11-08T00:15:00Z",
    patient_name: "Alice Walker",
    patient_code: "P001",
    stay_code: "S001",
    bed_number: "ICU-Bed-04",
  },
  last_reading: {
    fluid_ml: 350,
    heart_rate_bpm: 78,
    spo2_percent: 99,
    device_status: "NORMAL",
    reading_timestamp: "2026-11-08T01:00:00Z",
  },
};

const mockDeviceUnassigned = {
  id: 2,
  device_id: "ESP32-002",
  device_name: "Portable ICU Monitor B",
  connection_type: "serial",
  port: "COM10",
  status: "connected",
  created_at: "2026-11-08T00:00:00Z",
  updated_at: "2026-11-08T00:00:00Z",
  active_assignment: null,
  last_reading: null,
};

const mockActivePatients = [
  {
    id: 2,
    patient_code: "P002",
    name: "Bob Smith",
    status: "active",
    active_stay: {
      id: 102,
      stay_code: "S002",
      bed_number: "ICU-Bed-08",
      ward_or_unit: "General ICU",
      admission_reason: "Post-op",
    },
  },
];

const mockHistory = [
  {
    id: 1,
    device_id: "ESP32-001",
    patient_id: 1,
    stay_id: 101,
    patient_name: "Alice Walker",
    patient_code: "P001",
    stay_code: "S001",
    assigned_at: "2026-11-08T00:15:00Z",
    unassigned_at: null,
    assignment_status: "active",
  },
];

describe("DeviceManagement Component", () => {
  let requests: Array<{ url: string; init?: RequestInit }>;

  beforeEach(() => {
    requests = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        requests.push({ url, init });

        if (url.endsWith("/api/devices") && (!init || init.method === "GET")) {
          return jsonResponse({
            devices: [mockDeviceAssigned, mockDeviceUnassigned],
          });
        }

        if (url.endsWith("/api/devices") && init?.method === "POST") {
          const body = JSON.parse(String(init.body));
          return jsonResponse({
            id: 3,
            device_id: body.device_id,
            device_name: body.device_name,
            connection_type: body.connection_type || "serial",
            port: body.port || "COM9",
            status: "disconnected",
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            active_assignment: null,
            last_reading: null,
          });
        }

        if (url.includes("/api/patients?source=registered")) {
          return jsonResponse({ patients: mockActivePatients });
        }

        if (url.includes("/assign") && init?.method === "POST") {
          return jsonResponse({
            id: 2,
            device_id: "ESP32-002",
            patient_id: 2,
            stay_id: 102,
            assigned_at: new Date().toISOString(),
            unassigned_at: null,
            assignment_status: "active",
          });
        }

        if (url.includes("/unassign") && init?.method === "POST") {
          return jsonResponse({
            id: 1,
            device_id: "ESP32-001",
            patient_id: 1,
            stay_id: 101,
            assignment_status: "unassigned",
            unassigned_at: new Date().toISOString(),
          });
        }

        if (url.includes("/assignments")) {
          return jsonResponse({ assignments: mockHistory });
        }

        if (url.includes("/telemetry") && init?.method === "POST") {
          const body = JSON.parse(String(init.body));
          return jsonResponse({
            id: 1,
            device_id: "ESP32-001",
            reading_timestamp: new Date().toISOString(),
            fluid_ml: body.fluid_ml,
            bpm: body.bpm,
            spo2: body.spo2,
            device_status: body.device_status,
            patient_id: 1,
            stay_id: 101,
          });
        }

        return jsonResponse({});
      }),
    );
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders registered devices with assignment status and unassigned warning banner", async () => {
    render(<DeviceManagement />);

    expect(screen.getByText("Device Management & Assignment")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getAllByText(/ESP32-001/).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/ESP32-002/).length).toBeGreaterThan(0);
    });

    // Check assigned device details
    expect(screen.getByText(/Alice Walker/)).toBeInTheDocument();
    expect(screen.getAllByText(/S001/).length).toBeGreaterThan(0);

    // Check unassigned device shows warning
    expect(screen.getByText(/Device Not Assigned/)).toBeInTheDocument();
  });

  it("allows registering a new ESP32 device", async () => {
    render(<DeviceManagement />);

    const registerBtn = screen.getByRole("button", { name: "+ Register New ESP32" });
    fireEvent.click(registerBtn);

    expect(screen.getByRole("heading", { name: "Register ESP32 Device" })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/Device Identifier \*/i), {
      target: { value: "ESP32-003" },
    });
    fireEvent.change(screen.getByLabelText(/Device Name \*/i), {
      target: { value: "ICU Bed 8 Telemetry" },
    });
    fireEvent.change(screen.getByLabelText(/Serial Port \(Optional\)/i), {
      target: { value: "COM11" },
    });

    fireEvent.click(screen.getByRole("button", { name: "Register Device" }));

    await waitFor(() => {
      const postReq = requests.find(
        (r) => r.url.endsWith("/api/devices") && r.init?.method === "POST",
      );
      expect(postReq).toBeDefined();
      expect(JSON.parse(String(postReq?.init?.body))).toMatchObject({
        device_id: "ESP32-003",
        device_name: "ICU Bed 8 Telemetry",
        port: "COM11",
      });
    });
  });

  it("allows assigning an unassigned device to an active patient and stay", async () => {
    render(<DeviceManagement />);

    await waitFor(() => {
      expect(screen.getAllByText(/ESP32-002/).length).toBeGreaterThan(0);
    });

    const assignBtns = screen.getAllByRole("button", { name: "Assign to Patient" });
    fireEvent.click(assignBtns[0]);

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Assign ESP32-002 to Patient" })).toBeInTheDocument();
    });

    // Select Bob Smith
    const selectEl = screen.getByLabelText(/Select Active ICU Patient/i);
    fireEvent.change(selectEl, { target: { value: "2" } });

    fireEvent.click(screen.getByRole("button", { name: "Confirm Assignment" }));

    await waitFor(() => {
      const assignReq = requests.find(
        (r) => r.url.includes("/api/devices/ESP32-002/assign") && r.init?.method === "POST",
      );
      expect(assignReq).toBeDefined();
      expect(JSON.parse(String(assignReq?.init?.body))).toMatchObject({
        patient_id: 2,
        stay_id: 102,
      });
    });
  });

  it("allows unassigning an active device", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<DeviceManagement />);

    await waitFor(() => {
      expect(screen.getAllByText(/ESP32-001/).length).toBeGreaterThan(0);
    });

    const unassignBtn = screen.getByRole("button", { name: "Unassign Patient" });
    fireEvent.click(unassignBtn);

    await waitFor(() => {
      const unassignReq = requests.find(
        (r) => r.url.includes("/api/devices/ESP32-001/unassign") && r.init?.method === "POST",
      );
      expect(unassignReq).toBeDefined();
    });
  });

  it("supports hardware-free telemetry simulation test injection", async () => {
    render(<DeviceManagement />);

    await waitFor(() => {
      expect(screen.getAllByText(/ESP32-001/).length).toBeGreaterThan(0);
    });

    const simBtns = screen.getAllByRole("button", { name: "⚡ Send Synthetic Reading" });
    fireEvent.click(simBtns[0]);

    await waitFor(() => {
      const simReq = requests.find(
        (r) => r.url.includes("/telemetry") && r.init?.method === "POST",
      );
      expect(simReq).toBeDefined();
      expect(screen.getAllByText(/Reading accepted and attributed to Patient #1/).length).toBeGreaterThan(0);
    });
  });
});
