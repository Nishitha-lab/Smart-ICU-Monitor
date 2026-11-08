import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";
import { API_BASE } from "./api";
import ClinicalRecords from "./ClinicalRecords";
import DataConsistencyPanel from "./DataConsistencyPanel";
import "./PatientManagement.css";

export type PatientListItem = {
  id: number;
  patient_code: string;
  name: string;
  age: number | null;
  sex: string | null;
  date_of_birth: string | null;
  status: "active" | "archived";
  created_at: string;
  updated_at: string;
  active_stay: {
    id: number;
    stay_code: string;
    bed_number: string | null;
    ward_or_unit: string | null;
    admission_datetime: string;
    discharge_status: string;
  } | null;
  active_device: {
    device_id: string;
    device_name: string;
    status: string;
  } | null;
};

export type PatientDetail = PatientListItem & {
  contact_information: string | null;
  stays: Array<{
    id: number;
    stay_code: string;
    patient_id: number;
    admission_datetime: string;
    discharge_datetime: string | null;
    bed_number: string | null;
    ward_or_unit: string | null;
    admission_reason: string | null;
    discharge_status: string;
    created_at: string;
    updated_at: string;
  }>;
  latest_telemetry: {
    reading_time: string;
    fluid_ml: number | null;
    bpm: number | null;
    spo2: number | null;
    device_status: string | null;
    connection_status: string;
  } | null;
};

export type TimelineEvent = {
  event_type: string;
  timestamp: string;
  summary: string;
  source: string;
  details: Record<string, unknown>;
};

export type DeviceOption = {
  device_id: string;
  device_name: string;
  status: string;
  active_assignment: {
    patient_name: string;
    stay_code: string;
  } | null;
};

function formatDateTime(isoString: string | null | undefined): string {
  if (!isoString) return "—";
  const date = new Date(isoString);
  return Number.isNaN(date.getTime())
    ? isoString
    : `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
}

export default function PatientManagement() {
  // List state
  const [patients, setPatients] = useState<PatientListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "active" | "archived">("all");

  // Selection / Detail state
  const [selectedPatientId, setSelectedPatientId] = useState<number | string | null>(null);
  const [patientDetail, setPatientDetail] = useState<PatientDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");

  // Timeline state
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [timelineLoading, setTimelineLoading] = useState(false);
  const detailRequestRef = useRef(0);
  const timelineRequestRef = useRef(0);

  // Modals state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [createName, setCreateName] = useState("");
  const [createCode, setCreateCode] = useState("");
  const [createAge, setCreateAge] = useState("");
  const [createSex, setCreateSex] = useState<string>("unknown");
  const [createDob, setCreateDob] = useState("");
  const [createContact, setCreateContact] = useState("");
  const [createLoading, setCreateLoading] = useState(false);
  const [createError, setCreateError] = useState("");

  // Admission Modal
  const [showAdmissionModal, setShowAdmissionModal] = useState(false);
  const [admissionBed, setAdmissionBed] = useState("");
  const [admissionWard, setAdmissionWard] = useState("ICU-Main");
  const [admissionReason, setAdmissionReason] = useState("");
  const [admissionLoading, setAdmissionLoading] = useState(false);
  const [admissionError, setAdmissionError] = useState("");

  // Discharge Modal
  const [showDischargeModal, setShowDischargeModal] = useState(false);
  const [dischargeLoading, setDischargeLoading] = useState(false);
  const [dischargeError, setDischargeError] = useState("");

  // Device Assign Modal
  const [showAssignModal, setShowAssignModal] = useState(false);
  const [availableDevices, setAvailableDevices] = useState<DeviceOption[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState("");
  const [assignLoading, setAssignLoading] = useState(false);
  const [assignError, setAssignError] = useState("");

  // Device Unassign Modal
  const [showUnassignModal, setShowUnassignModal] = useState(false);
  const [unassignLoading, setUnassignLoading] = useState(false);
  const [unassignError, setUnassignError] = useState("");

  // Embedded Clinical Records toggle
  const [showEmbeddedClinicalForm, setShowEmbeddedClinicalForm] = useState(false);
  const [bannerMessage, setBannerMessage] = useState("");

  // Load Patient List
  const loadPatients = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams();
      params.set("source", "registered");
      if (searchQuery.trim()) params.set("search", searchQuery.trim());
      if (statusFilter !== "all") params.set("status", statusFilter);

      const response = await fetch(`${API_BASE}/api/patients?${params.toString()}`);
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Failed to load registered patients.");
      }
      const data = await response.json();
      setPatients(data.patients || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load registered patients.");
    } finally {
      setLoading(false);
    }
  }, [searchQuery, statusFilter]);

  useEffect(() => {
    const timer = setTimeout(() => {
      loadPatients();
    }, 200);
    return () => clearTimeout(timer);
  }, [loadPatients]);

  // Load Patient Detail
  const loadPatientDetail = useCallback(async (patientId: number | string) => {
    const requestId = ++detailRequestRef.current;
    const timelineRequestId = ++timelineRequestRef.current;
    setDetailLoading(true);
    setDetailError("");
    try {
      const response = await fetch(`${API_BASE}/api/registered-patients/${patientId}`);
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Failed to load patient details.");
      }
      const data: PatientDetail = await response.json();
      if (requestId !== detailRequestRef.current) return;
      setPatientDetail(data);

      // Load timeline in parallel
      setTimelineLoading(true);
      setTimeline([]);
      try {
        const timelineResponse = await fetch(`${API_BASE}/api/patients/${patientId}/timeline`);
        if (!timelineResponse.ok) return;
        const timelineData = await timelineResponse.json();
        if (timelineRequestId === timelineRequestRef.current) {
          setTimeline(timelineData.events || []);
        }
      } catch {
        if (timelineRequestId === timelineRequestRef.current) {
          setTimeline([]);
        }
      } finally {
        if (timelineRequestId === timelineRequestRef.current) {
          setTimelineLoading(false);
        }
      }
    } catch (err) {
      if (requestId === detailRequestRef.current) {
        setPatientDetail(null);
        setDetailError(err instanceof Error ? err.message : "Failed to load patient details.");
      }
    } finally {
      if (requestId === detailRequestRef.current) {
        setDetailLoading(false);
      }
    }
  }, []);

  const handleSelectPatient = useCallback(
    (patientId: number | string) => {
      setSelectedPatientId(patientId);
      setPatientDetail(null);
      setTimeline([]);
      setDetailError("");
      loadPatientDetail(patientId);
    },
    [loadPatientDetail],
  );

  const handleBackToList = useCallback(() => {
    detailRequestRef.current += 1;
    timelineRequestRef.current += 1;
    setSelectedPatientId(null);
    setPatientDetail(null);
    setTimeline([]);
    setDetailLoading(false);
    setTimelineLoading(false);
    loadPatients();
  }, [loadPatients]);

  // Handle Create Patient
  async function handleCreatePatient(e: FormEvent) {
    e.preventDefault();
    setCreateLoading(true);
    setCreateError("");
    try {
      const payload = {
        name: createName.trim(),
        patient_code: createCode.trim() || undefined,
        age: createAge ? parseInt(createAge, 10) : undefined,
        sex: createSex || undefined,
        date_of_birth: createDob || undefined,
        contact_information: createContact.trim() || undefined,
      };

      const response = await fetch(`${API_BASE}/api/patients`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Failed to register patient.");
      }

      const created = await response.json();
      setShowCreateModal(false);
      setCreateName("");
      setCreateCode("");
      setCreateAge("");
      setCreateDob("");
      setCreateContact("");
      setBannerMessage(`Patient ${created.name} (${created.patient_code}) successfully registered.`);
      loadPatients();
      handleSelectPatient(created.id);
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : "Failed to register patient.");
    } finally {
      setCreateLoading(false);
    }
  }

  // Handle Admit to ICU
  async function handleAdmitPatient(e: FormEvent) {
    e.preventDefault();
    if (!patientDetail) return;
    setAdmissionLoading(true);
    setAdmissionError("");
    try {
      const payload = {
        bed_number: admissionBed.trim() || undefined,
        ward_or_unit: admissionWard.trim() || undefined,
        admission_reason: admissionReason.trim() || undefined,
      };

      const response = await fetch(`${API_BASE}/api/patients/${patientDetail.id}/stays`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Failed to admit patient to ICU.");
      }

      const stay = await response.json();
      setShowAdmissionModal(false);
      setAdmissionBed("");
      setAdmissionReason("");
      setBannerMessage(`ICU admission ${stay.stay_code} created for ${patientDetail.name}.`);
      loadPatientDetail(patientDetail.id);
      loadPatients();
    } catch (err) {
      setAdmissionError(err instanceof Error ? err.message : "Failed to admit patient.");
    } finally {
      setAdmissionLoading(false);
    }
  }

  // Handle Discharge Stay
  async function handleDischargeStay() {
    if (!patientDetail?.active_stay) return;
    setDischargeLoading(true);
    setDischargeError("");
    try {
      const response = await fetch(
        `${API_BASE}/api/stays/${patientDetail.active_stay.id}/discharge`,
        { method: "POST" }
      );

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Failed to discharge ICU stay.");
      }

      setShowDischargeModal(false);
      setBannerMessage(`Stay ${patientDetail.active_stay.stay_code} discharged successfully.`);
      loadPatientDetail(patientDetail.id);
      loadPatients();
    } catch (err) {
      setDischargeError(err instanceof Error ? err.message : "Failed to discharge stay.");
    } finally {
      setDischargeLoading(false);
    }
  }

  // Open Assign Modal: Load available devices
  async function openAssignModal() {
    setShowAssignModal(true);
    setAssignError("");
    try {
      const response = await fetch(`${API_BASE}/api/devices`);
      if (response.ok) {
        const data = await response.json();
        setAvailableDevices(data.devices || []);
        if (data.devices?.length > 0) {
          setSelectedDeviceId(data.devices[0].device_id);
        }
      }
    } catch {
      // ignore
    }
  }

  // Handle Assign Device
  async function handleAssignDevice(e: FormEvent) {
    e.preventDefault();
    if (!patientDetail?.active_stay || !selectedDeviceId) return;
    setAssignLoading(true);
    setAssignError("");
    try {
      const response = await fetch(`${API_BASE}/api/devices/${selectedDeviceId}/assign`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patient_id: patientDetail.id,
          stay_id: patientDetail.active_stay.id,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Failed to assign device.");
      }

      setShowAssignModal(false);
      setBannerMessage(`Device ${selectedDeviceId} assigned to ${patientDetail.name}.`);
      loadPatientDetail(patientDetail.id);
      loadPatients();
    } catch (err) {
      setAssignError(err instanceof Error ? err.message : "Failed to assign device.");
    } finally {
      setAssignLoading(false);
    }
  }

  // Handle Unassign Device
  async function handleUnassignDevice() {
    if (!patientDetail?.active_device) return;
    setUnassignLoading(true);
    setUnassignError("");
    try {
      const response = await fetch(
        `${API_BASE}/api/devices/${patientDetail.active_device.device_id}/unassign`,
        { method: "POST" }
      );

      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Failed to unassign device.");
      }

      setShowUnassignModal(false);
      setBannerMessage(`Device ${patientDetail.active_device.device_id} unassigned.`);
      loadPatientDetail(patientDetail.id);
      loadPatients();
    } catch (err) {
      setUnassignError(err instanceof Error ? err.message : "Failed to unassign device.");
    } finally {
      setUnassignLoading(false);
    }
  }

  // Handle Archive Patient
  async function handleArchivePatient() {
    if (!patientDetail) return;
    if (!window.confirm(`Archive patient ${patientDetail.name}? Active admissions and device assignments will be closed.`)) {
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/api/patients/${patientDetail.id}/archive`, {
        method: "POST",
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.detail || "Failed to archive patient.");
      }
      setBannerMessage(`Patient ${patientDetail.name} archived.`);
      loadPatientDetail(patientDetail.id);
      loadPatients();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Failed to archive patient.");
    }
  }

  return (
    <section className="patient-management-section">
      {bannerMessage && (
        <div className="pm-success-banner" role="status">
          {bannerMessage}
          <button
            type="button"
            style={{ float: "right", background: "none", border: "none", cursor: "pointer", fontWeight: "bold" }}
            onClick={() => setBannerMessage("")}
          >
            ✕
          </button>
        </div>
      )}

      {/* ============================================================
          PATIENT LIST VIEW
          ============================================================ */}
      {selectedPatientId === null && (
        <>
          <div className="pm-header">
            <div>
              <h1>Patient Management</h1>
              <p className="pm-subtitle">
                Registered ICU patients, admissions, device assignments, and clinical history.
              </p>
            </div>
            <div className="pm-controls">
              <input
                type="search"
                className="pm-search-input"
                placeholder="Search patient name or code..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                aria-label="Search patients"
              />
              <select
                className="pm-select"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value as "all" | "active" | "archived")}
                aria-label="Filter status"
              >
                <option value="all">All Statuses</option>
                <option value="active">Active Only</option>
                <option value="archived">Archived Only</option>
              </select>
              <button
                type="button"
                className="pm-btn-primary"
                onClick={() => setShowCreateModal(true)}
              >
                + Register Patient
              </button>
            </div>
          </div>

          {loading && (
            <div className="pm-card" style={{ textAlign: "center", padding: "40px", color: "#64748b" }}>
              Loading registered patients...
            </div>
          )}

          {error && (
            <div className="pm-error-banner" role="alert">
              <strong>Error:</strong> {error}
            </div>
          )}

          {!loading && !error && patients.length === 0 && (
            <div className="pm-card pm-empty-state">
              <h3>No Registered Patients Found</h3>
              <p>
                {searchQuery || statusFilter !== "all"
                  ? "No patients match the specified search or filter criteria."
                  : "No registered patients yet. Register a patient to start monitoring and ICU management."}
              </p>
              <button
                type="button"
                className="pm-btn-primary"
                onClick={() => setShowCreateModal(true)}
              >
                Register First Patient
              </button>
            </div>
          )}

          {!loading && !error && patients.length > 0 && (
            <div className="pm-table-wrapper">
              <table className="pm-table" aria-label="Registered patients table">
                <thead>
                  <tr>
                    <th>Patient ID</th>
                    <th>Name</th>
                    <th>Age / Sex</th>
                    <th>Active Stay</th>
                    <th>Bed Number</th>
                    <th>Admission Date</th>
                    <th>Assigned Device</th>
                    <th>Status</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map((p) => (
                    <tr key={p.id}>
                      <td>
                        <strong>{p.patient_code}</strong>
                      </td>
                      <td>
                        <span style={{ fontWeight: 600 }}>{p.name}</span>
                      </td>
                      <td>
                        {p.age !== null ? `${p.age} yrs` : "—"}{" "}
                        {p.sex ? `(${p.sex.charAt(0).toUpperCase()})` : ""}
                      </td>
                      <td>
                        {p.active_stay ? (
                          <span style={{ color: "#047857", fontWeight: 600 }}>
                            {p.active_stay.stay_code}
                          </span>
                        ) : (
                          <span style={{ color: "#94a3b8" }}>None</span>
                        )}
                      </td>
                      <td>
                        {p.active_stay?.bed_number ? (
                          <span>{p.active_stay.bed_number}</span>
                        ) : (
                          <span style={{ color: "#94a3b8" }}>—</span>
                        )}
                      </td>
                      <td>{formatDateTime(p.active_stay?.admission_datetime)}</td>
                      <td>
                        {p.active_device ? (
                          <span className="badge-device-assigned">
                            ● {p.active_device.device_id}
                          </span>
                        ) : (
                          <span className="badge-device-none">Unassigned</span>
                        )}
                      </td>
                      <td>
                        <span className={`badge-status-${p.status}`}>
                          {p.status}
                        </span>
                      </td>
                      <td>
                        <button
                          type="button"
                          className="pm-btn-secondary"
                          onClick={() => handleSelectPatient(p.id)}
                        >
                          View Details →
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      {/* ============================================================
          PATIENT DETAIL VIEW
          ============================================================ */}
      {selectedPatientId !== null && (
        <div className="pm-detail-grid">
          <div className="pm-detail-top-nav">
            <button
              type="button"
              className="pm-btn-secondary"
              onClick={handleBackToList}
            >
              ← Back to Patient List
            </button>
            {patientDetail && (
              <div style={{ display: "flex", gap: "10px" }}>
                {patientDetail.status === "active" && (
                  <button
                    type="button"
                    className="pm-btn-danger"
                    onClick={handleArchivePatient}
                  >
                    Archive Patient
                  </button>
                )}
              </div>
            )}
          </div>

          {detailLoading && (
            <div className="pm-card" style={{ textAlign: "center", padding: "40px", color: "#64748b" }}>
              Loading patient profile and admissions...
            </div>
          )}

          {detailError && (
            <div className="pm-error-banner" role="alert">
              <strong>Error loading patient:</strong> {detailError}
            </div>
          )}

          {!detailLoading && patientDetail && (
            <>
              {/* Profile Overview Card */}
              <div className="pm-card">
                <h2>
                  <span>
                    {patientDetail.name}{" "}
                    <span style={{ color: "#155eef", fontWeight: 700 }}>
                      ({patientDetail.patient_code})
                    </span>
                  </span>
                  <span className={`badge-status-${patientDetail.status}`}>
                    {patientDetail.status}
                  </span>
                </h2>
                <div className="pm-info-list">
                  <div className="pm-info-item">
                    <label>Age</label>
                    <span>{patientDetail.age !== null ? `${patientDetail.age} years` : "Not recorded"}</span>
                  </div>
                  <div className="pm-info-item">
                    <label>Sex</label>
                    <span>{patientDetail.sex || "Not recorded"}</span>
                  </div>
                  <div className="pm-info-item">
                    <label>Date of Birth</label>
                    <span>{patientDetail.date_of_birth || "Not recorded"}</span>
                  </div>
                  <div className="pm-info-item">
                    <label>Contact Info</label>
                    <span>{patientDetail.contact_information || "None"}</span>
                  </div>
                  <div className="pm-info-item">
                    <label>Registered At</label>
                    <span>{formatDateTime(patientDetail.created_at)}</span>
                  </div>
                </div>
              </div>

              {/* Grid: Active Admission & Device Monitoring */}
              <div className="pm-detail-cards">
                {/* Active ICU Stay Card */}
                <div className="pm-card">
                  <h2>
                    <span>Active ICU Stay</span>
                    {patientDetail.active_stay ? (
                      <span className="badge-status-active">Admitted</span>
                    ) : (
                      <span className="badge-device-none">Not Admitted</span>
                    )}
                  </h2>

                  {patientDetail.active_stay ? (
                    <div>
                      <div className="pm-info-list">
                        <div className="pm-info-item">
                          <label>Stay Code</label>
                          <span>{patientDetail.active_stay.stay_code}</span>
                        </div>
                        <div className="pm-info-item">
                          <label>Bed Number</label>
                          <span>{patientDetail.active_stay.bed_number || "Unassigned"}</span>
                        </div>
                        <div className="pm-info-item">
                          <label>Ward / Unit</label>
                          <span>{patientDetail.active_stay.ward_or_unit || "ICU"}</span>
                        </div>
                        <div className="pm-info-item">
                          <label>Admitted Since</label>
                          <span>{formatDateTime(patientDetail.active_stay.admission_datetime)}</span>
                        </div>
                      </div>

                      <div style={{ marginTop: "18px", display: "flex", gap: "10px", flexWrap: "wrap" }}>
                        <button
                          type="button"
                          className="pm-btn-primary"
                          onClick={() => setShowEmbeddedClinicalForm((v) => !v)}
                        >
                          {showEmbeddedClinicalForm ? "Hide Clinical Entry" : "+ Add Clinical Record"}
                        </button>
                        <button
                          type="button"
                          className="pm-btn-danger"
                          onClick={() => setShowDischargeModal(true)}
                        >
                          Discharge Stay
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div style={{ textAlign: "center", padding: "20px 0" }}>
                      <p style={{ color: "#64748b", margin: "0 0 14px 0" }}>
                        This patient has no active ICU admission. A stay is required for live monitoring and clinical records.
                      </p>
                      {patientDetail.status === "active" && (
                        <button
                          type="button"
                          className="pm-btn-primary"
                          onClick={() => setShowAdmissionModal(true)}
                        >
                          + Admit to ICU
                        </button>
                      )}
                    </div>
                  )}
                </div>

                {/* Device Assignment Card */}
                <div className="pm-card">
                  <h2>
                    <span>Assigned ESP32 Hardware</span>
                    {patientDetail.active_device ? (
                      <span className="badge-device-assigned">● Active</span>
                    ) : (
                      <span className="badge-device-none">Not Assigned</span>
                    )}
                  </h2>

                  {patientDetail.active_device ? (
                    <div>
                      <div className="pm-info-list">
                        <div className="pm-info-item">
                          <label>Device ID</label>
                          <span>{patientDetail.active_device.device_id}</span>
                        </div>
                        <div className="pm-info-item">
                          <label>Device Name</label>
                          <span>{patientDetail.active_device.device_name}</span>
                        </div>
                        <div className="pm-info-item">
                          <label>Connection Status</label>
                          <span style={{ textTransform: "uppercase" }}>
                            {patientDetail.active_device.status}
                          </span>
                        </div>
                      </div>

                      {/* Live Telemetry Banner for this patient */}
                      <div className="pm-vitals-banner">
                        <div className="pm-vital-box">
                          <div className="pm-vital-val">
                            {patientDetail.latest_telemetry?.fluid_ml !== null &&
                            patientDetail.latest_telemetry?.fluid_ml !== undefined
                              ? `${patientDetail.latest_telemetry.fluid_ml.toFixed(1)}`
                              : "—"}
                          </div>
                          <div className="pm-vital-lbl">Fluid (mL)</div>
                        </div>
                        <div className="pm-vital-box">
                          <div className="pm-vital-val">
                            {patientDetail.latest_telemetry?.bpm !== null &&
                            patientDetail.latest_telemetry?.bpm !== undefined
                              ? `${Math.round(patientDetail.latest_telemetry.bpm)}`
                              : "—"}
                          </div>
                          <div className="pm-vital-lbl">Heart Rate (BPM)</div>
                        </div>
                        <div className="pm-vital-box">
                          <div className="pm-vital-val">
                            {patientDetail.latest_telemetry?.spo2 !== null &&
                            patientDetail.latest_telemetry?.spo2 !== undefined
                              ? `${Math.round(patientDetail.latest_telemetry.spo2)}%`
                              : "—"}
                          </div>
                          <div className="pm-vital-lbl">SpO2</div>
                        </div>
                        <div className="pm-vital-box">
                          <div className="pm-vital-val" style={{ fontSize: "14px", marginTop: "4px" }}>
                            {patientDetail.latest_telemetry?.device_status || "NORMAL"}
                          </div>
                          <div className="pm-vital-lbl">Status</div>
                        </div>
                      </div>
                      <small style={{ color: "#64748b", display: "block", marginTop: "6px" }}>
                        Last Reading: {formatDateTime(patientDetail.latest_telemetry?.reading_time)}
                      </small>

                      <div style={{ marginTop: "14px" }}>
                        <button
                          type="button"
                          className="pm-btn-secondary"
                          onClick={() => setShowUnassignModal(true)}
                        >
                          Unassign Device
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div style={{ textAlign: "center", padding: "20px 0" }}>
                      <p style={{ color: "#64748b", margin: "0 0 14px 0" }}>
                        No ESP32 monitoring device is assigned to this patient. Unassigned device readings are not attributed to any patient.
                      </p>
                      {patientDetail.active_stay && (
                        <button
                          type="button"
                          className="pm-btn-primary"
                          onClick={openAssignModal}
                        >
                          Assign Monitoring Device
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Embedded Clinical Record Form */}
              {showEmbeddedClinicalForm && patientDetail.active_stay && (
                <div className="pm-card" style={{ border: "2px solid #155eef" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "14px" }}>
                    <h3 style={{ margin: 0, color: "#155eef" }}>
                      Add Clinical Record for {patientDetail.name} (Stay {patientDetail.active_stay.stay_code})
                    </h3>
                    <button
                      type="button"
                      className="pm-btn-secondary"
                      onClick={() => setShowEmbeddedClinicalForm(false)}
                    >
                      Close Entry Form
                    </button>
                  </div>
                  <ClinicalRecords
                    initialPatientId={patientDetail.id}
                    initialStayId={patientDetail.active_stay.id}
                    patientName={patientDetail.name}
                    onRecordSaved={() => {
                      loadPatientDetail(patientDetail.id);
                    }}
                  />
                </div>
              )}

              {/* Clinical Data Consistency & Intelligence Engine */}
              {patientDetail.active_stay && (
                <DataConsistencyPanel
                  patientId={patientDetail.id}
                  stayId={patientDetail.active_stay.id}
                />
              )}

              {/* Stay History Table */}
              <div className="pm-card">
                <h2>ICU Admission History ({patientDetail.stays?.length || 0})</h2>
                {patientDetail.stays && patientDetail.stays.length > 0 ? (
                  <div className="pm-table-wrapper" style={{ boxShadow: "none" }}>
                    <table className="pm-table">
                      <thead>
                        <tr>
                          <th>Stay Code</th>
                          <th>Admitted</th>
                          <th>Discharged</th>
                          <th>Bed</th>
                          <th>Ward</th>
                          <th>Reason</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {patientDetail.stays.map((s) => (
                          <tr key={s.id}>
                            <td><strong>{s.stay_code}</strong></td>
                            <td>{formatDateTime(s.admission_datetime)}</td>
                            <td>{formatDateTime(s.discharge_datetime)}</td>
                            <td>{s.bed_number || "—"}</td>
                            <td>{s.ward_or_unit || "—"}</td>
                            <td>{s.admission_reason || "—"}</td>
                            <td>
                              <span className={s.discharge_status === "active" ? "badge-status-active" : "badge-status-archived"}>
                                {s.discharge_status}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p style={{ color: "#64748b" }}>No ICU admissions recorded for this patient.</p>
                )}
              </div>

              {/* Chronological Patient Timeline */}
              <div className="pm-card">
                <h2>
                  <span>Patient Chronological Timeline</span>
                  <span style={{ fontSize: "13px", color: "#64748b", fontWeight: 400 }}>
                    Auditable events &amp; measurements
                  </span>
                </h2>

                {timelineLoading && <p style={{ color: "#64748b" }}>Loading timeline events...</p>}

                {!timelineLoading && timeline.length === 0 && (
                  <p style={{ color: "#64748b" }}>No events recorded for this patient yet.</p>
                )}

                {!timelineLoading && timeline.length > 0 && (
                  <div className="pm-timeline">
                    {timeline.map((evt, idx) => (
                      <div key={idx} className="pm-timeline-item">
                        <div className={`pm-timeline-dot ${evt.event_type}`} />
                        <div className="pm-timeline-header">
                          <span className="pm-timeline-time">{formatDateTime(evt.timestamp)}</span>
                          <span className="pm-timeline-source">{evt.source}</span>
                        </div>
                        <div className="pm-timeline-summary">{evt.summary}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {/* ============================================================
          REGISTER PATIENT MODAL
          ============================================================ */}
      {showCreateModal && (
        <div className="pm-modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="create-patient-title">
          <div className="pm-modal">
            <h2 id="create-patient-title">Register New ICU Patient</h2>
            {createError && <div className="pm-error-banner" style={{ marginBottom: "12px" }}>{createError}</div>}
            <form onSubmit={handleCreatePatient}>
              <div>
                <label htmlFor="create-name">Full Patient Name *</label>
                <input
                  id="create-name"
                  type="text"
                  required
                  placeholder="e.g. John Doe"
                  value={createName}
                  onChange={(e) => setCreateName(e.target.value)}
                />
              </div>

              <div>
                <label htmlFor="create-code">Patient Code (Optional)</label>
                <input
                  id="create-code"
                  type="text"
                  placeholder="Auto-generated (e.g. P001) if blank"
                  value={createCode}
                  onChange={(e) => setCreateCode(e.target.value)}
                />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
                <div>
                  <label htmlFor="create-age">Age (Optional)</label>
                  <input
                    id="create-age"
                    type="number"
                    min="0"
                    max="130"
                    placeholder="e.g. 55"
                    value={createAge}
                    onChange={(e) => setCreateAge(e.target.value)}
                  />
                </div>
                <div>
                  <label htmlFor="create-sex">Sex (Optional)</label>
                  <select
                    id="create-sex"
                    value={createSex}
                    onChange={(e) => setCreateSex(e.target.value)}
                  >
                    <option value="unknown">Unknown</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>

              <div>
                <label htmlFor="create-dob">Date of Birth (Optional)</label>
                <input
                  id="create-dob"
                  type="date"
                  value={createDob}
                  onChange={(e) => setCreateDob(e.target.value)}
                />
              </div>

              <div>
                <label htmlFor="create-contact">Contact Info (Optional)</label>
                <input
                  id="create-contact"
                  type="text"
                  placeholder="Phone, emergency contact, or unit notes"
                  value={createContact}
                  onChange={(e) => setCreateContact(e.target.value)}
                />
              </div>

              <div className="pm-modal-actions">
                <button
                  type="button"
                  className="pm-btn-secondary"
                  disabled={createLoading}
                  onClick={() => setShowCreateModal(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="pm-btn-primary"
                  disabled={createLoading}
                >
                  {createLoading ? "Registering..." : "Register Patient"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ============================================================
          ADMIT TO ICU MODAL
          ============================================================ */}
      {showAdmissionModal && patientDetail && (
        <div className="pm-modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="admit-patient-title">
          <div className="pm-modal">
            <h2 id="admit-patient-title">Admit {patientDetail.name} to ICU</h2>
            {admissionError && <div className="pm-error-banner" style={{ marginBottom: "12px" }}>{admissionError}</div>}
            <form onSubmit={handleAdmitPatient}>
              <div>
                <label htmlFor="admit-bed">Bed Number</label>
                <input
                  id="admit-bed"
                  type="text"
                  placeholder="e.g. Bed 04"
                  value={admissionBed}
                  onChange={(e) => setAdmissionBed(e.target.value)}
                />
              </div>

              <div>
                <label htmlFor="admit-ward">Ward or Unit</label>
                <input
                  id="admit-ward"
                  type="text"
                  placeholder="e.g. Medical ICU"
                  value={admissionWard}
                  onChange={(e) => setAdmissionWard(e.target.value)}
                />
              </div>

              <div>
                <label htmlFor="admit-reason">Admission Reason / Context</label>
                <textarea
                  id="admit-reason"
                  placeholder="e.g. Post-operative monitoring, acute respiratory observation"
                  value={admissionReason}
                  onChange={(e) => setAdmissionReason(e.target.value)}
                />
              </div>

              <div className="pm-modal-actions">
                <button
                  type="button"
                  className="pm-btn-secondary"
                  disabled={admissionLoading}
                  onClick={() => setShowAdmissionModal(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="pm-btn-primary"
                  disabled={admissionLoading}
                >
                  {admissionLoading ? "Admitting..." : "Confirm Admission"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ============================================================
          DISCHARGE MODAL
          ============================================================ */}
      {showDischargeModal && patientDetail?.active_stay && (
        <div className="pm-modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="discharge-stay-title">
          <div className="pm-modal">
            <h2 id="discharge-stay-title">Discharge ICU Stay</h2>
            {dischargeError && <div className="pm-error-banner" style={{ marginBottom: "12px" }}>{dischargeError}</div>}
            <p>
              Are you sure you want to discharge <strong>{patientDetail.name}</strong> from ICU Stay{" "}
              <strong>{patientDetail.active_stay.stay_code}</strong>?
            </p>
            <p style={{ color: "#64748b", fontSize: "13px" }}>
              This will record the discharge timestamp and automatically unassign any active monitoring device.
              Historical telemetry and clinical records will remain preserved.
            </p>
            <div className="pm-modal-actions">
              <button
                type="button"
                className="pm-btn-secondary"
                disabled={dischargeLoading}
                onClick={() => setShowDischargeModal(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="pm-btn-danger"
                disabled={dischargeLoading}
                onClick={handleDischargeStay}
              >
                {dischargeLoading ? "Discharging..." : "Confirm Discharge"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ============================================================
          ASSIGN DEVICE MODAL
          ============================================================ */}
      {showAssignModal && patientDetail?.active_stay && (
        <div className="pm-modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="assign-device-title">
          <div className="pm-modal">
            <h2 id="assign-device-title">Assign ESP32 Device</h2>
            {assignError && <div className="pm-error-banner" style={{ marginBottom: "12px" }}>{assignError}</div>}
            <form onSubmit={handleAssignDevice}>
              <p style={{ color: "#334155", fontSize: "14px", margin: "0 0 10px 0" }}>
                Assigning device to <strong>{patientDetail.name}</strong> (Stay {patientDetail.active_stay.stay_code})
              </p>
              <div>
                <label htmlFor="assign-device-select">Select Hardware Device</label>
                <select
                  id="assign-device-select"
                  className="pm-select"
                  style={{ width: "100%" }}
                  value={selectedDeviceId}
                  onChange={(e) => setSelectedDeviceId(e.target.value)}
                >
                  {availableDevices.map((d) => (
                    <option key={d.device_id} value={d.device_id}>
                      {d.device_id} — {d.device_name} ({d.status})
                      {d.active_assignment ? ` [Currently: ${d.active_assignment.patient_name}]` : " [Unassigned]"}
                    </option>
                  ))}
                </select>
              </div>

              <p style={{ fontSize: "12px", color: "#64748b", marginTop: "8px" }}>
                Reassigning will safely end any prior patient assignment and immediately route future readings to this patient.
              </p>

              <div className="pm-modal-actions">
                <button
                  type="button"
                  className="pm-btn-secondary"
                  disabled={assignLoading}
                  onClick={() => setShowAssignModal(false)}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="pm-btn-primary"
                  disabled={assignLoading || !selectedDeviceId}
                >
                  {assignLoading ? "Assigning..." : "Confirm Assignment"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ============================================================
          UNASSIGN DEVICE MODAL
          ============================================================ */}
      {showUnassignModal && patientDetail?.active_device && (
        <div className="pm-modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="unassign-device-title">
          <div className="pm-modal">
            <h2 id="unassign-device-title">Unassign Hardware Device</h2>
            {unassignError && <div className="pm-error-banner" style={{ marginBottom: "12px" }}>{unassignError}</div>}
            <p>
              Unassign device <strong>{patientDetail.active_device.device_id}</strong> from{" "}
              <strong>{patientDetail.name}</strong>?
            </p>
            <p style={{ color: "#64748b", fontSize: "13px" }}>
              Future incoming readings from this ESP32 will be treated as unassigned and will no longer be attributed to this patient.
            </p>
            <div className="pm-modal-actions">
              <button
                type="button"
                className="pm-btn-secondary"
                disabled={unassignLoading}
                onClick={() => setShowUnassignModal(false)}
              >
                Cancel
              </button>
              <button
                type="button"
                className="pm-btn-danger"
                disabled={unassignLoading}
                onClick={handleUnassignDevice}
              >
                {unassignLoading ? "Unassigning..." : "Confirm Unassign"}
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
