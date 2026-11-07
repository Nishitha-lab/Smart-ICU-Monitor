import {
  useCallback,
  useEffect,
  useState,
  type FormEvent,
} from "react";
import { API_BASE } from "./api";
import "./DeviceManagement.css";

export type DeviceInfo = {
  device_id: string;
  device_name: string;
  connection_type: string;
  port: string | null;
  status: string;
  created_at: string;
  updated_at: string;
  active_assignment: {
    id: number;
    patient_id: number;
    stay_id: number;
    assigned_at: string;
    patient_code: string;
    patient_name: string;
    stay_code: string;
    bed_number: string | null;
    stay_status: string;
  } | null;
  last_telemetry: {
    reading_time: string;
    fluid_ml: number | null;
    bpm: number | null;
    spo2: number | null;
    device_status: string | null;
    connection_status: string;
  } | null;
};

export type DeviceAssignmentHistory = {
  id: number;
  device_id: string;
  patient_id: number;
  stay_id: number;
  assigned_at: string;
  unassigned_at: string | null;
  assignment_status: string;
  patient_code: string;
  patient_name: string;
  stay_code: string;
  bed_number: string | null;
};

export type ActivePatientOption = {
  id: number;
  patient_code: string;
  name: string;
  active_stay: {
    id: number;
    stay_code: string;
    bed_number: string | null;
  } | null;
};

export default function DeviceManagement() {
  const [devices, setDevices] = useState<DeviceInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [statusMessage, setStatusMessage] = useState("");

  // Register Device Modal
  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [regDeviceId, setRegDeviceId] = useState("");
  const [regDeviceName, setRegDeviceName] = useState("");
  const [regPort, setRegPort] = useState("COM9");
  const [regLoading, setRegLoading] = useState(false);
  const [regError, setRegError] = useState("");

  // Assign Modal
  const [assigningDevice, setAssigningDevice] = useState<DeviceInfo | null>(null);
  const [activePatients, setActivePatients] = useState<ActivePatientOption[]>([]);
  const [selectedPatientRef, setSelectedPatientRef] = useState<string>("");
  const [assignLoading, setAssignLoading] = useState(false);
  const [assignError, setAssignError] = useState("");

  // History inspection
  const [viewHistoryDeviceId, setViewHistoryDeviceId] = useState<string | null>(null);
  const [assignmentHistory, setAssignmentHistory] = useState<DeviceAssignmentHistory[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Telemetry Simulation
  const [simFluid, setSimFluid] = useState("450");
  const [simBpm, setSimBpm] = useState("75");
  const [simSpo2, setSimSpo2] = useState("98");
  const [simStatus, setSimStatus] = useState("NORMAL");
  const [simFeedback, setSimFeedback] = useState<string>("");
  const [simLoading, setSimLoading] = useState(false);

  const loadDevices = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/devices`);
      if (!response.ok) {
        throw new Error("Failed to load registered devices.");
      }
      const data = await response.json();
      setDevices(data.devices || []);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load devices.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let ignore = false;
    async function fetchInitialDevices() {
      try {
        const response = await fetch(`${API_BASE}/api/devices`);
        if (!response.ok) {
          throw new Error("Failed to load registered devices.");
        }
        const data = await response.json();
        if (!ignore) {
          setDevices(data.devices || []);
          setError("");
        }
      } catch (err) {
        if (!ignore) {
          setError(err instanceof Error ? err.message : "Failed to load devices.");
        }
      } finally {
        if (!ignore) {
          setLoading(false);
        }
      }
    }

    fetchInitialDevices();
    const interval = setInterval(loadDevices, 5000);
    return () => {
      ignore = true;
      clearInterval(interval);
    };
  }, [loadDevices]);

  // Load Active Patients for assignment
  async function openAssignModal(device: DeviceInfo) {
    setAssigningDevice(device);
    setAssignError("");
    try {
      const response = await fetch(`${API_BASE}/api/patients?source=registered&status=active`);
      if (response.ok) {
        const data = await response.json();
        const available = (data.patients || []).filter(
          (p: ActivePatientOption) => p.active_stay !== null
        );
        setActivePatients(available);
        if (available.length > 0) {
          setSelectedPatientRef(String(available[0].id));
        }
      }
    } catch {
      // ignore
    }
  }

  // Handle Assign
  async function handleAssignSubmit(e: FormEvent) {
    e.preventDefault();
    if (!assigningDevice || !selectedPatientRef) return;
    setAssignLoading(true);
    setAssignError("");

    const chosenPatient = activePatients.find(
      (p) => String(p.id) === selectedPatientRef
    );
    if (!chosenPatient || !chosenPatient.active_stay) {
      setAssignError("Selected patient does not have an active ICU admission.");
      setAssignLoading(false);
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/api/devices/${assigningDevice.device_id}/assign`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: chosenPatient.id,
          stay_id: chosenPatient.active_stay.id,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Failed to assign device.");
      }

      setAssigningDevice(null);
      setStatusMessage(`Device ${assigningDevice.device_id} assigned to ${chosenPatient.name}.`);
      loadDevices();
    } catch (err) {
      setAssignError(err instanceof Error ? err.message : "Failed to assign device.");
    } finally {
      setAssignLoading(false);
    }
  }

  // Handle Unassign
  async function handleUnassign(deviceId: string) {
    if (!window.confirm(`Unassign device ${deviceId}? Subsequent readings will not be attributed to any patient.`)) {
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/api/devices/${deviceId}/unassign`, {
        method: "POST",
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Failed to unassign device.");
      }
      setStatusMessage(`Device ${deviceId} unassigned.`);
      loadDevices();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to unassign device.");
    }
  }

  // Handle Register Device
  async function handleRegisterDevice(e: FormEvent) {
    e.preventDefault();
    setRegLoading(true);
    setRegError("");
    try {
      const response = await fetch(`${API_BASE}/api/devices`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          device_id: regDeviceId.trim(),
          device_name: regDeviceName.trim(),
          connection_type: "serial",
          port: regPort.trim() || undefined,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Failed to register device.");
      }

      setShowRegisterModal(false);
      setRegDeviceId("");
      setRegDeviceName("");
      setStatusMessage(`Device registered successfully.`);
      loadDevices();
    } catch (err) {
      setRegError(err instanceof Error ? err.message : "Failed to register device.");
    } finally {
      setRegLoading(false);
    }
  }

  // View Device Assignment History
  async function toggleHistory(deviceId: string) {
    if (viewHistoryDeviceId === deviceId) {
      setViewHistoryDeviceId(null);
      return;
    }
    setViewHistoryDeviceId(deviceId);
    setHistoryLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/devices/${deviceId}/assignments`);
      if (response.ok) {
        const data = await response.json();
        setAssignmentHistory(data.assignments || []);
      }
    } catch {
      // ignore
    } finally {
      setHistoryLoading(false);
    }
  }

  // Simulate Telemetry Reading
  async function handleSimulateReading(deviceId: string) {
    setSimLoading(true);
    setSimFeedback("");
    try {
      const payload = {
        fluid_ml: parseFloat(simFluid),
        bpm: parseFloat(simBpm),
        spo2: parseFloat(simSpo2),
        device_status: simStatus,
      };

      const response = await fetch(`${API_BASE}/api/devices/${deviceId}/telemetry`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error("Simulation request failed.");
      }

      const result = await response.json();
      if (result.patient_id) {
        setSimFeedback(
          `✓ Reading accepted and attributed to Patient #${result.patient_id} (Stay #${result.stay_id}).`
        );
      } else {
        setSimFeedback(
          `⚠️ Device is unassigned: Reading accepted by hardware log, but NOT attributed to any patient.`
        );
      }
      loadDevices();
    } catch (err) {
      setSimFeedback(err instanceof Error ? err.message : "Error sending synthetic reading.");
    } finally {
      setSimLoading(false);
    }
  }

  return (
    <section className="device-management-section">
      <div className="dm-header">
        <div>
          <h1>Device Management &amp; Assignment</h1>
          <p className="dm-subtitle">
            Register ESP32 monitoring devices, establish strict patient assignments, and simulate hardware feeds.
          </p>
        </div>
        <div>
          <button
            type="button"
            className="pm-btn-primary"
            onClick={() => setShowRegisterModal(true)}
          >
            + Register New ESP32
          </button>
        </div>
      </div>

      {statusMessage && (
        <div className="pm-success-banner" role="status">
          {statusMessage}
          <button
            type="button"
            style={{ float: "right", background: "none", border: "none", cursor: "pointer", fontWeight: "bold" }}
            onClick={() => setStatusMessage("")}
          >
            ✕
          </button>
        </div>
      )}

      {loading && devices.length === 0 && (
        <div className="dm-card" style={{ textAlign: "center", padding: "40px", color: "#64748b" }}>
          Loading registered devices...
        </div>
      )}

      {error && (
        <div className="pm-error-banner" role="alert">
          {error}
        </div>
      )}

      <div className="dm-grid">
        {devices.map((dev) => (
          <div key={dev.device_id} className="dm-card">
            <h2>
              <span>{dev.device_name} ({dev.device_id})</span>
              <span className={`dm-status-badge dm-status-${dev.status.toLowerCase()}`}>
                {dev.status}
              </span>
            </h2>

            <div className="pm-info-list" style={{ marginBottom: "12px" }}>
              <div className="pm-info-item">
                <label>Connection</label>
                <span>{dev.connection_type.toUpperCase()} ({dev.port || "COM9"})</span>
              </div>
              <div className="pm-info-item">
                <label>Baud Rate</label>
                <span>115200</span>
              </div>
            </div>

            {/* Assignment Status Box */}
            {dev.active_assignment ? (
              <div className="dm-assignment-box assigned">
                <div style={{ color: "#1d4ed8", fontWeight: 700, fontSize: "14px", marginBottom: "6px" }}>
                  ● Active Patient Assignment
                </div>
                <div className="pm-info-list">
                  <div className="pm-info-item">
                    <label>Patient</label>
                    <span>{dev.active_assignment.patient_name} ({dev.active_assignment.patient_code})</span>
                  </div>
                  <div className="pm-info-item">
                    <label>Stay / Bed</label>
                    <span>{dev.active_assignment.stay_code} — {dev.active_assignment.bed_number || "Bed N/A"}</span>
                  </div>
                  <div className="pm-info-item">
                    <label>Assigned At</label>
                    <span>{new Date(dev.active_assignment.assigned_at).toLocaleTimeString()}</span>
                  </div>
                </div>

                <div style={{ marginTop: "12px" }}>
                  <button
                    type="button"
                    className="pm-btn-secondary"
                    onClick={() => handleUnassign(dev.device_id)}
                  >
                    Unassign Patient
                  </button>
                </div>
              </div>
            ) : (
              <div className="dm-assignment-box unassigned">
                <div className="dm-unassigned-warning">
                  ⚠️ Device Not Assigned
                </div>
                <p style={{ color: "#9a3412", fontSize: "12px", margin: "6px 0 10px 0" }}>
                  Readings from this device will be monitored but NOT attributed to any patient or displayed in clinical patient records.
                </p>
                <button
                  type="button"
                  className="pm-btn-primary"
                  onClick={() => openAssignModal(dev)}
                >
                  Assign to Patient
                </button>
              </div>
            )}

            {/* Latest Telemetry Preview */}
            <div style={{ background: "#f8fafc", padding: "12px", borderRadius: "8px", marginTop: "12px" }}>
              <div style={{ fontSize: "12px", fontWeight: 700, color: "#475569", marginBottom: "6px" }}>
                Latest Sensor Packet:
              </div>
              {dev.last_telemetry ? (
                <div style={{ display: "flex", gap: "14px", flexWrap: "wrap", fontSize: "13px" }}>
                  <span><strong>Fluid:</strong> {dev.last_telemetry.fluid_ml ?? "—"} mL</span>
                  <span><strong>BPM:</strong> {dev.last_telemetry.bpm ?? "—"}</span>
                  <span><strong>SpO2:</strong> {dev.last_telemetry.spo2 ?? "—"}%</span>
                  <span><strong>Status:</strong> {dev.last_telemetry.device_status || "NORMAL"}</span>
                </div>
              ) : (
                <span style={{ color: "#94a3b8", fontSize: "12px" }}>No readings received yet.</span>
              )}
            </div>

            {/* Simulation Feed Panel for Demonstration */}
            <div className="dm-sim-panel">
              <h3>Interactive Telemetry Simulator ({dev.device_id})</h3>
              <p style={{ fontSize: "12px", color: "#6b21a8", margin: "0 0 8px 0" }}>
                Inject synthetic serial data without needing physical hardware to verify attribution behavior.
              </p>
              <div className="dm-sim-form">
                <div>
                  <label htmlFor={`sim-fluid-${dev.device_id}`}>Fluid (mL)</label>
                  <input
                    id={`sim-fluid-${dev.device_id}`}
                    type="number"
                    value={simFluid}
                    onChange={(e) => setSimFluid(e.target.value)}
                  />
                </div>
                <div>
                  <label htmlFor={`sim-bpm-${dev.device_id}`}>BPM</label>
                  <input
                    id={`sim-bpm-${dev.device_id}`}
                    type="number"
                    value={simBpm}
                    onChange={(e) => setSimBpm(e.target.value)}
                  />
                </div>
                <div>
                  <label htmlFor={`sim-spo2-${dev.device_id}`}>SpO2 (%)</label>
                  <input
                    id={`sim-spo2-${dev.device_id}`}
                    type="number"
                    value={simSpo2}
                    onChange={(e) => setSimSpo2(e.target.value)}
                  />
                </div>
                <div>
                  <label htmlFor={`sim-status-${dev.device_id}`}>Status</label>
                  <select
                    id={`sim-status-${dev.device_id}`}
                    value={simStatus}
                    onChange={(e) => setSimStatus(e.target.value)}
                  >
                    <option value="NORMAL">NORMAL</option>
                    <option value="WARNING">WARNING</option>
                    <option value="CRITICAL">CRITICAL</option>
                  </select>
                </div>
              </div>

              <div style={{ marginTop: "10px", display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                <button
                  type="button"
                  className="pm-btn-primary"
                  style={{ background: "#7e22ce" }}
                  disabled={simLoading}
                  onClick={() => handleSimulateReading(dev.device_id)}
                >
                  {simLoading ? "Sending..." : "⚡ Send Synthetic Reading"}
                </button>
                {simFeedback && (
                  <span style={{ fontSize: "12px", fontWeight: 600 }}>{simFeedback}</span>
                )}
              </div>
            </div>

            {/* Toggle History Button */}
            <div style={{ marginTop: "14px", borderTop: "1px solid #f1f5f9", paddingTop: "10px" }}>
              <button
                type="button"
                className="pm-btn-secondary"
                style={{ fontSize: "12px", width: "100%" }}
                onClick={() => toggleHistory(dev.device_id)}
              >
                {viewHistoryDeviceId === dev.device_id ? "Hide Assignment History" : "View Assignment History"}
              </button>

              {viewHistoryDeviceId === dev.device_id && (
                <div style={{ marginTop: "10px" }}>
                  {historyLoading ? (
                    <small>Loading history...</small>
                  ) : assignmentHistory.length === 0 ? (
                    <small style={{ color: "#64748b" }}>No past assignments for this device.</small>
                  ) : (
                    <div className="dm-table-wrap">
                      <table className="dm-history-table">
                        <thead>
                          <tr>
                            <th>Patient</th>
                            <th>Stay</th>
                            <th>Assigned</th>
                            <th>Unassigned</th>
                            <th>Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {assignmentHistory.map((h) => (
                            <tr key={h.id}>
                              <td>{h.patient_name} ({h.patient_code})</td>
                              <td>{h.stay_code}</td>
                              <td>{new Date(h.assigned_at).toLocaleTimeString()}</td>
                              <td>{h.unassigned_at ? new Date(h.unassigned_at).toLocaleTimeString() : "—"}</td>
                              <td>
                                <span className={h.assignment_status === "active" ? "badge-status-active" : "badge-status-archived"}>
                                  {h.assignment_status}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* ============================================================
          ASSIGN DEVICE MODAL
          ============================================================ */}
      {assigningDevice && (
        <div className="pm-modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="modal-assign-title">
          <div className="pm-modal">
            <h2 id="modal-assign-title">Assign {assigningDevice.device_id} to Patient</h2>
            {assignError && <div className="pm-error-banner" style={{ marginBottom: "12px" }}>{assignError}</div>}
            <form onSubmit={handleAssignSubmit}>
              {activePatients.length === 0 ? (
                <div style={{ color: "#b91c1c", fontSize: "13px" }}>
                  No active ICU patients found. Admit a patient to ICU before assigning a monitoring device.
                </div>
              ) : (
                <div>
                  <label htmlFor="select-patient-assign">Select Active ICU Patient</label>
                  <select
                    id="select-patient-assign"
                    className="pm-select"
                    style={{ width: "100%" }}
                    value={selectedPatientRef}
                    onChange={(e) => setSelectedPatientRef(e.target.value)}
                  >
                    {activePatients.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name} ({p.patient_code}) — {p.active_stay?.stay_code} ({p.active_stay?.bed_number || "No bed"})
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <p style={{ color: "#64748b", fontSize: "12px", marginTop: "10px" }}>
                The backend will route all incoming serial telemetry packets from {assigningDevice.device_id} to this patient's active ICU stay.
              </p>

              <div className="pm-modal-actions">
                <button
                  type="button"
                  className="pm-btn-secondary"
                  disabled={assignLoading}
                  onClick={() => setAssigningDevice(null)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="pm-btn-primary"
                  disabled={assignLoading || activePatients.length === 0}
                >
                  {assignLoading ? "Assigning..." : "Confirm Assignment"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ============================================================
          REGISTER DEVICE MODAL
          ============================================================ */}
      {showRegisterModal && (
        <div className="pm-modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="modal-reg-device">
          <div className="pm-modal">
            <h2 id="modal-reg-device">Register ESP32 Device</h2>
            {regError && <div className="pm-error-banner" style={{ marginBottom: "12px" }}>{regError}</div>}
            <form onSubmit={handleRegisterDevice}>
              <div>
                <label htmlFor="reg-device-id">Device Identifier *</label>
                <input
                  id="reg-device-id"
                  type="text"
                  required
                  placeholder="e.g. ESP32-002"
                  value={regDeviceId}
                  onChange={(e) => setRegDeviceId(e.target.value)}
                />
              </div>

              <div>
                <label htmlFor="reg-device-name">Device Name *</label>
                <input
                  id="reg-device-name"
                  type="text"
                  required
                  placeholder="e.g. ICU Bedside Monitor 2"
                  value={regDeviceName}
                  onChange={(e) => setRegDeviceName(e.target.value)}
                />
              </div>

              <div>
                <label htmlFor="reg-port">Serial Port (Optional)</label>
                <input
                  id="reg-port"
                  type="text"
                  placeholder="e.g. COM9"
                  value={regPort}
                  onChange={(e) => setRegPort(e.target.value)}
                />
              </div>

              <div className="pm-modal-actions">
                <button
                  type="button"
                  className="pm-btn-secondary"
                  disabled={regLoading}
                  onClick={() => setShowRegisterModal(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="pm-btn-primary"
                  disabled={regLoading}
                >
                  {regLoading ? "Registering..." : "Register Device"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </section>
  );
}
