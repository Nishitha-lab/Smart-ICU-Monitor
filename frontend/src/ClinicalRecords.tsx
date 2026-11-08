import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import { API_BASE } from "./api";
import "./ClinicalRecords.css";

type Category =
  | "urine_output"
  | "blood_test"
  | "other_lab"
  | "clinical_observation";

export type ClinicalInterpretation = {
  status:
    | "CRITICAL_HIGH"
    | "CRITICAL_LOW"
    | "ABNORMAL_HIGH"
    | "ABNORMAL_LOW"
    | "NORMAL"
    | "UNINTERPRETABLE";
  severity?: "CRITICAL" | "HIGH" | "MODERATE" | "LOW" | null;
  category: string;
  measurement_name: string;
  canonical_test_id?: string | null;
  measured_value: number | string;
  unit?: string | null;
  reference_range?: string | null;
  critical_threshold?: string | null;
  reference_source?: string | null;
  explanation: string;
  requires_alert: boolean;
  evaluation_version?: string;
  evaluated_at?: string;
  details?: Record<string, unknown>;
};

type ClinicalRecord = {
  record_id: string;
  patientunitstayid: number;
  category: Category;
  measurement_name: string;
  value: number | string;
  unit: string | null;
  measurement_time: string;
  entry_time: string;
  version: number;
  entered_by: string;
  source: "MANUAL_ENTRY";
  notes: string | null;
  observation_period: string | null;
  reference_range: string | null;
  sample_collection_time: string | null;
  result_time: string | null;
  updated_at: string | null;
  voided: boolean;
  voided_at: string | null;
  voided_by: string | null;
  void_reason: string | null;
  interpretation?: ClinicalInterpretation | null;
  alert?: Record<string, unknown> | null;
};

const CATEGORY_LABELS: Record<Category, string> = {
  urine_output: "Urine Output",
  blood_test: "Blood Tests",
  other_lab: "Other Laboratory Results",
  clinical_observation: "Clinical Observations",
};

const BLOOD_TEST_UNITS: Record<string, string> = {
  Hemoglobin: "g/dL",
  Creatinine: "mg/dL",
  BUN: "mg/dL",
  Sodium: "mmol/L",
  Potassium: "mmol/L",
  "Blood glucose": "mg/dL",
};

const OBSERVATION_UNITS: Record<string, string | null> = {
  "Body temperature": "°C",
  "Respiratory rate": "breaths/min",
  "Systolic blood pressure": "mmHg",
  "Diastolic blood pressure": "mmHg",
  "Heart rate": "bpm",
  Pulse: "bpm",
  "Oxygen saturation": "%",
  SpO2: "%",
  "IV fluid infusion": "mL",
  "Intake volume": "mL",
  "Consciousness / general observation": null,
  "Clinical note": null,
};

function localDateTimeValue(date = new Date()) {
  const offset = date.getTimezoneOffset() * 60_000;
  return new Date(date.getTime() - offset).toISOString().slice(0, 16);
}

function asIsoTimestamp(value: string) {
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) {
    throw new Error("Enter a valid date and time.");
  }
  return date.toISOString();
}

function asDateFilter(value: string, endOfDay = false) {
  if (!value) return "";
  const date = new Date(`${value}T${endOfDay ? "23:59:59.999" : "00:00:00"}`);
  return date.toISOString();
}

function displayTimestamp(value: string | null | undefined) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : `${date.toLocaleString()} (local time)`;
}

async function responseError(response: Response, fallback: string) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch {
    return fallback;
  }
  return fallback;
}

function localDateTimeInput(value: string | null | undefined) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : localDateTimeValue(date);
}

export type ClinicalRecordsProps = {
  initialPatientId?: number | string;
  initialStayId?: number | string;
  patientName?: string;
  onRecordSaved?: () => void;
};

export default function ClinicalRecords(props: ClinicalRecordsProps = {}) {
  const [patientSearch, setPatientSearch] = useState("");
  const [patientIds, setPatientIds] = useState<number[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>(() =>
    props.initialStayId
      ? String(props.initialStayId)
      : props.initialPatientId
        ? String(props.initialPatientId)
        : "",
  );
  const [prevInitialStayId, setPrevInitialStayId] = useState(props.initialStayId);
  const [prevInitialPatientId, setPrevInitialPatientId] = useState(props.initialPatientId);

  if (props.initialStayId !== prevInitialStayId || props.initialPatientId !== prevInitialPatientId) {
    setPrevInitialStayId(props.initialStayId);
    setPrevInitialPatientId(props.initialPatientId);
    if (props.initialStayId) {
      setSelectedPatientId(String(props.initialStayId));
    } else if (props.initialPatientId) {
      setSelectedPatientId(String(props.initialPatientId));
    }
  }
  const [patientLoading, setPatientLoading] = useState(false);
  const [patientError, setPatientError] = useState("");
  const [clinicalApiStatus, setClinicalApiStatus] = useState<
    "checking" | "available" | "unavailable"
  >("checking");

  const [category, setCategory] = useState<Category>("urine_output");
  const [measurementName, setMeasurementName] = useState("Urine output");
  const [value, setValue] = useState("");
  const [unit, setUnit] = useState("");
  const [measurementTime, setMeasurementTime] = useState(
    localDateTimeValue,
  );
  const [sampleCollectionTime, setSampleCollectionTime] = useState("");
  const [resultTime, setResultTime] = useState("");
  const [enteredBy, setEnteredBy] = useState("");
  const [observationPeriod, setObservationPeriod] = useState("");
  const [referenceRange, setReferenceRange] = useState("");
  const [notes, setNotes] = useState("");

  const [historyCategory, setHistoryCategory] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [records, setRecords] = useState<ClinicalRecord[]>([]);
  const [recordsLoading, setRecordsLoading] = useState(false);
  const [recordsError, setRecordsError] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [latestInterpretation, setLatestInterpretation] =
    useState<ClinicalInterpretation | null>(null);
  const [latestAlert, setLatestAlert] = useState<Record<string, unknown> | null>(null);
  const [previewInterpretation, setPreviewInterpretation] =
    useState<ClinicalInterpretation | null>(null);
  const [editingRecord, setEditingRecord] = useState<ClinicalRecord | null>(null);
  const [changeReason, setChangeReason] = useState("");
  const [conflictRecordId, setConflictRecordId] = useState<string | null>(null);
  const submissionInProgress = useRef(false);
  const [voidingRecord, setVoidingRecord] = useState<ClinicalRecord | null>(null);
  const [voidedBy, setVoidedBy] = useState("");
  const [voidReason, setVoidReason] = useState("");
  const [voiding, setVoiding] = useState(false);
  const [voidError, setVoidError] = useState("");
  const [voidConflictRecordId, setVoidConflictRecordId] = useState<string | null>(null);
  const voidSubmissionInProgress = useRef(false);
  const voidDialogRef = useRef<HTMLDivElement | null>(null);
  const voidReasonInputRef = useRef<HTMLTextAreaElement | null>(null);
  const reloadVoidButtonRef = useRef<HTMLButtonElement | null>(null);
  const voidTriggerRef = useRef<HTMLButtonElement | null>(null);
  const voidDialogWasOpen = useRef(false);
  const historyHeadingRef = useRef<HTMLHeadingElement | null>(null);

  const selectedBloodTestUnit = BLOOD_TEST_UNITS[measurementName];
  const selectedObservationUnit = OBSERVATION_UNITS[measurementName];
  const isNarrativeObservation =
    category === "clinical_observation" &&
    selectedObservationUnit === null;

  const activePreview =
    !value || isNarrativeObservation ? null : previewInterpretation;

  useEffect(() => {
    if (!value || isNarrativeObservation) {
      return;
    }
    const timer = setTimeout(async () => {
      try {
        const resolvedUnit =
          category === "urine_output"
            ? "mL"
            : category === "blood_test"
              ? selectedBloodTestUnit
              : category === "clinical_observation"
                ? selectedObservationUnit
                : unit.trim();
        const payload = {
          category,
          measurement_name:
            category === "urine_output" ? "Urine output" : measurementName.trim(),
          value: Number(value),
          unit: resolvedUnit || null,
          observation_period:
            category === "urine_output" ? observationPeriod.trim() || null : null,
          reference_range:
            category === "other_lab" ? referenceRange.trim() || null : null,
          patient_id: props.initialPatientId ? Number(props.initialPatientId) : undefined,
        };
        const res = await fetch(`${API_BASE}/api/clinical-records/evaluate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (res.ok) {
          const data: ClinicalInterpretation = await res.json();
          setPreviewInterpretation(data);
        }
      } catch {
        // Ignore preview failure
      }
    }, 250);
    return () => clearTimeout(timer);
  }, [
    value,
    category,
    measurementName,
    observationPeriod,
    referenceRange,
    unit,
    selectedBloodTestUnit,
    selectedObservationUnit,
    isNarrativeObservation,
    props.initialPatientId,
  ]);

  const historyQuery = useMemo(() => {
    if (!selectedPatientId) return "";
    const query = new URLSearchParams();
    if (historyCategory) query.set("category", historyCategory);
    if (startDate) query.set("start_time", asDateFilter(startDate));
    if (endDate) query.set("end_time", asDateFilter(endDate, true));
    return query.toString();
  }, [endDate, historyCategory, selectedPatientId, startDate]);

  const historyEndpoint = props.initialStayId
    ? `${API_BASE}/api/stays/${props.initialStayId}/clinical-records`
    : `${API_BASE}/api/patients/${selectedPatientId}/clinical-records`;

  useEffect(() => {
    if (props.initialStayId) {
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setPatientLoading(true);
      setPatientError("");
      try {
        const query = new URLSearchParams({
          query: patientSearch.trim(),
          limit: "25",
        });
        const response = await fetch(
          `${API_BASE}/api/clinical-records/patient-stays?${query}`,
          { signal: controller.signal },
        );
        if (!response.ok) {
          throw new Error(
            await responseError(response, "Unable to find existing ICU stay IDs."),
          );
        }
        const data: { patient_ids: number[] } = await response.json();
        if (!controller.signal.aborted) {
          setPatientIds(data.patient_ids);
          setClinicalApiStatus("available");
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setPatientIds([]);
          setClinicalApiStatus("unavailable");
          setPatientError(
            error instanceof Error ? error.message : "Patient search failed.",
          );
        }
      } finally {
        if (!controller.signal.aborted) setPatientLoading(false);
      }
    }, 150);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [patientSearch, props.initialStayId]);

  useEffect(() => {
    if (!selectedPatientId) {
      return;
    }
    const controller = new AbortController();
    async function loadRecords() {
      setRecordsLoading(true);
      setRecordsError("");
      try {
        const suffix = historyQuery ? `?${historyQuery}` : "";
        const response = await fetch(
          `${historyEndpoint}${suffix}`,
          { signal: controller.signal },
        );
        if (!response.ok) {
          throw new Error(
            await responseError(response, "Unable to load clinical records."),
          );
        }
        const data: { records: ClinicalRecord[] } = await response.json();
        if (!controller.signal.aborted) {
          setRecords(data.records ?? []);
          setClinicalApiStatus("available");
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setClinicalApiStatus("unavailable");
          setRecordsError(
            error instanceof Error ? error.message : "Clinical records failed to load.",
          );
        }
      } finally {
        if (!controller.signal.aborted) setRecordsLoading(false);
      }
    }
    void loadRecords();
    return () => controller.abort();
  }, [historyEndpoint, historyQuery, selectedPatientId]);

  useEffect(() => {
    if (voidingRecord) {
      voidDialogWasOpen.current = true;
      voidReasonInputRef.current?.focus();
      return;
    }
    if (!voidDialogWasOpen.current) return;
    voidDialogWasOpen.current = false;
    if (voidTriggerRef.current?.isConnected) {
      voidTriggerRef.current.focus();
    } else {
      historyHeadingRef.current?.focus();
    }
  }, [voidingRecord]);

  useEffect(() => {
    if (voidConflictRecordId) reloadVoidButtonRef.current?.focus();
  }, [voidConflictRecordId]);

  function changeCategory(nextCategory: Category) {
    setCategory(nextCategory);
    setValue("");
    setUnit("");
    setReferenceRange("");
    setObservationPeriod("");
    setSampleCollectionTime("");
    setResultTime("");
    setChangeReason("");
    setConflictRecordId(null);
    setSaveError("");
    setSuccessMessage("");
    if (nextCategory === "urine_output") setMeasurementName("Urine output");
    if (nextCategory === "blood_test") setMeasurementName("Hemoglobin");
    if (nextCategory === "other_lab") setMeasurementName("");
    if (nextCategory === "clinical_observation") {
      setMeasurementName("Body temperature");
    }
  }

  function beginEditingRecord(record: ClinicalRecord) {
    setEditingRecord(record);
    setCategory(record.category);
    setMeasurementName(record.measurement_name);
    setValue(String(record.value));
    setUnit(record.unit ?? "");
    setMeasurementTime(localDateTimeInput(record.measurement_time));
    setSampleCollectionTime(localDateTimeInput(record.sample_collection_time));
    setResultTime(localDateTimeInput(record.result_time));
    setEnteredBy(record.entered_by);
    setObservationPeriod(record.observation_period ?? "");
    setReferenceRange(record.reference_range ?? "");
    setNotes(record.notes ?? "");
    setChangeReason("");
    setConflictRecordId(null);
    setSaveError("");
    setSuccessMessage("");
  }

  async function reloadConflictRecord() {
    if (!conflictRecordId) return;
    try {
      const response = await fetch(
        `${API_BASE}/api/clinical-records/${conflictRecordId}`,
      );
      if (!response.ok) throw new Error("Unable to reload the latest record.");
      const latestRecord: ClinicalRecord = await response.json();
      setRecords((current) =>
        current.map((item) =>
          item.record_id === latestRecord.record_id ? latestRecord : item,
        ),
      );
      beginEditingRecord(latestRecord);
      setSuccessMessage("Latest version loaded. Review it before correcting.");
    } catch (error) {
      setSaveError(
        error instanceof Error ? error.message : "Unable to reload the latest record.",
      );
    }
  }

  function openVoidDialog(record: ClinicalRecord, trigger: HTMLButtonElement) {
      voidTriggerRef.current = trigger;
      setVoidingRecord(record);
      setVoidedBy(enteredBy.trim() || record.entered_by);
      setVoidReason("");
      setVoidError("");
      setVoidConflictRecordId(null);
    }

  function closeVoidDialog() {
      if (voiding) return;
      setVoidingRecord(null);
      setVoidReason("");
      setVoidError("");
      setVoidConflictRecordId(null);
    }

  function handleVoidDialogKeyDown(event: KeyboardEvent<HTMLDivElement>) {
      if (event.key === "Escape") {
        event.preventDefault();
        closeVoidDialog();
        return;
      }
      if (event.key !== "Tab" || !voidDialogRef.current) return;

      const focusable = Array.from(
        voidDialogRef.current.querySelectorAll<HTMLElement>(
          "input:not(:disabled), textarea:not(:disabled), select:not(:disabled), button:not(:disabled)",
        ),
      );
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

  async function reloadLatestVoidRecord() {
      if (!voidConflictRecordId) return;
      try {
        const response = await fetch(
          `${API_BASE}/api/clinical-records/${voidConflictRecordId}`,
        );
        if (!response.ok) throw new Error("Unable to reload the latest record.");
        const latestRecord: ClinicalRecord = await response.json();
        if (latestRecord.voided) {
          setRecords((current) =>
            current.filter((item) => item.record_id !== latestRecord.record_id),
          );
          setVoidingRecord(null);
          setSuccessMessage("This record was already voided and is no longer active.");
          return;
        }
        setRecords((current) =>
          current.map((item) =>
            item.record_id === latestRecord.record_id ? latestRecord : item,
          ),
        );
        setVoidingRecord(latestRecord);
        setVoidedBy(latestRecord.entered_by);
        setVoidReason("");
        setVoidConflictRecordId(null);
        setVoidError("Latest version loaded. Review it and enter the void reason again.");
      } catch (error) {
        setVoidError(
          error instanceof Error ? error.message : "Unable to reload the latest record.",
        );
      }
    }

  async function submitVoid(event: FormEvent<HTMLFormElement>) {
      event.preventDefault();
      if (!voidingRecord || voidSubmissionInProgress.current) return;
      if (!voidedBy.trim() || !voidReason.trim()) {
        setVoidError("Enter an operator identifier and a reason for voiding.");
        return;
      }
      voidSubmissionInProgress.current = true;
      setVoiding(true);
      setVoidError("");
      setSuccessMessage("");
      let voidCompleted = false;
      try {
        const response = await fetch(
          `${API_BASE}/api/clinical-records/${voidingRecord.record_id}`,
          {
            method: "DELETE",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              expected_version: voidingRecord.version,
              voided_by: voidedBy.trim(),
              void_reason: voidReason.trim(),
            }),
          },
        );
        if (!response.ok) {
          const detail = await responseError(response, "Unable to void this record.");
          if (response.status === 409) {
            setVoidConflictRecordId(voidingRecord.record_id);
            throw new Error(
              detail || "This record has changed. Reload the latest version before voiding.",
            );
          }
          throw new Error(detail);
        }

        setRecords((current) =>
          current.filter((item) => item.record_id !== voidingRecord.record_id),
        );
        voidCompleted = true;
        setVoidingRecord(null);
        setVoidReason("");
        setVoidConflictRecordId(null);
        setSuccessMessage(
          "Record voided and removed from active measurements. Its audit history is retained.",
        );

        const suffix = historyQuery ? `?${historyQuery}` : "";
        const historyResponse = await fetch(
          `${historyEndpoint}${suffix}`,
        );
        if (!historyResponse.ok) {
          throw new Error("Record voided, but the active history refresh failed.");
        }
        const historyData: { records: ClinicalRecord[] } =
          await historyResponse.json();
        setRecords(historyData.records ?? []);
        setRecordsError("");
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unable to void this record.";
        if (voidCompleted) {
          setRecordsError(message);
        } else {
          setVoidError(message);
        }
      } finally {
        voidSubmissionInProgress.current = false;
        setVoiding(false);
    }
  }

  async function saveRecord(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submissionInProgress.current) return;
    submissionInProgress.current = true;
    setSaving(true);
    setSaveError("");
    setSuccessMessage("");
    try {
      if (!selectedPatientId) {
        throw new Error("Select an existing ICU stay before saving.");
      }
      const sampleCollectionIso =
        category === "blood_test" || category === "other_lab"
          ? asIsoTimestamp(sampleCollectionTime)
          : null;
      const resolvedUnit =
        category === "urine_output"
          ? "mL"
          : category === "blood_test"
            ? selectedBloodTestUnit
            : category === "clinical_observation"
              ? selectedObservationUnit
              : unit.trim();
      const body = {
        category,
        measurement_name:
          category === "urine_output" ? "Urine output" : measurementName.trim(),
        value: isNarrativeObservation ? value.trim() : Number(value),
        unit: resolvedUnit || null,
        measurement_time: asIsoTimestamp(measurementTime),
        sample_collection_time: sampleCollectionIso,
        result_time: resultTime ? asIsoTimestamp(resultTime) : null,
        entered_by: enteredBy.trim(),
        notes: notes.trim() || null,
        observation_period:
          category === "urine_output" ? observationPeriod.trim() || null : null,
        reference_range:
          category === "other_lab" ? referenceRange.trim() || null : null,
        ...(editingRecord
          ? {
              expected_version: editingRecord.version,
              change_reason: changeReason.trim(),
            }
          : props.initialStayId
            ? {
                stay_id: Number(props.initialStayId),
                patient_id: props.initialPatientId
                  ? Number(props.initialPatientId)
                  : undefined,
              }
            : {
                patientunitstayid: Number(selectedPatientId),
              }),
      };
      const response = await fetch(
        editingRecord
          ? `${API_BASE}/api/clinical-records/${editingRecord.record_id}`
          : `${API_BASE}/api/clinical-records`,
        {
          method: editingRecord ? "PUT" : "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
      if (!response.ok) {
        const errorBody = await response.json().catch(() => null);
        const detail = errorBody?.detail;
        if (response.status === 409 && editingRecord) {
          setConflictRecordId(editingRecord.record_id);
          throw new Error(
            typeof detail === "string"
              ? detail
              : "This record changed since it was loaded. Reload the latest record before correcting it.",
          );
        }
        throw new Error(
          typeof detail === "string" ? detail : "Unable to save this record.",
        );
      }
      const savedResult = await response.json().catch(() => null);
      if (savedResult?.interpretation) {
        setLatestInterpretation(savedResult.interpretation);
      }
      if (savedResult?.alert) {
        setLatestAlert(savedResult.alert);
      } else {
        setLatestAlert(null);
      }
      setSuccessMessage(
        editingRecord ? "Clinical record correction saved." : "Clinical record saved.",
      );
      props.onRecordSaved?.();
      setEditingRecord(null);
      setValue("");
      setNotes("");
      setObservationPeriod("");
      setReferenceRange("");
      setSampleCollectionTime("");
      setResultTime("");
      setChangeReason("");
      setConflictRecordId(null);
      const query = historyQuery ? `?${historyQuery}` : "";
      const historyResponse = await fetch(
        `${historyEndpoint}${query}`,
      );
      if (!historyResponse.ok) {
        throw new Error("Record saved, but the history refresh failed.");
      }
      const historyData: { records: ClinicalRecord[] } =
        await historyResponse.json();
      setRecords(historyData.records ?? []);
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : "Unable to save record.");
    } finally {
      submissionInProgress.current = false;
      setSaving(false);
    }
  }

  return (
    <section className="clinical-records-section" aria-labelledby="clinical-records-title">
      <header className="clinical-records-header">
        <div>
          <p className="eyebrow">SEPARATE MANUAL RECORDS</p>
          <h1 id="clinical-records-title">Clinical Observations &amp; Lab Results</h1>
          <p>
            Prototype data entry only. These records are separate from
            historical eICU data, ESP32 telemetry, and alert or ML pipelines.
            Entries are not independently verified.
          </p>
          <p
            className={`clinical-api-status${clinicalApiStatus === "unavailable" ? " clinical-error" : ""}`}
            role="status"
          >
            Clinical records API: {clinicalApiStatus}
          </p>
        </div>
      </header>

      <div className="clinical-records-grid">
        <section className="clinical-panel" aria-labelledby="new-record-title">
          <h2 id="new-record-title">
            {editingRecord ? "Correct a clinical record" : "Enter a clinical record"}
          </h2>
          <p className="clinical-history-scope">
            Numeric limits are technical input safeguards, not clinical
            reference ranges or interpretation.
          </p>
          {props.initialStayId ? (
            <div
              className="clinical-preselected-stay-banner"
              style={{
                background: "#f0fdf4",
                border: "1px solid #bbf7d0",
                padding: "12px 14px",
                borderRadius: "8px",
                marginBottom: "14px",
              }}
            >
              <div
                style={{
                  color: "#166534",
                  fontWeight: 700,
                  fontSize: "14px",
                  marginBottom: "4px",
                }}
              >
                ✓ Preselected ICU Admission
              </div>
              <div style={{ color: "#1e293b", fontSize: "13px" }}>
                <strong>Patient:</strong>{" "}
                {props.patientName || `Patient ${props.initialPatientId || ""}`}
              </div>
              <div style={{ color: "#1e293b", fontSize: "13px" }}>
                <strong>Stay ID:</strong> {props.initialStayId}
              </div>
            </div>
          ) : (
            <div className="clinical-patient-picker">
              <label htmlFor="clinical-patient-search">Search existing ICU stay IDs</label>
              <input
                id="clinical-patient-search"
                type="search"
                inputMode="numeric"
                value={patientSearch}
                onChange={(event) => {
                  setPatientSearch(event.target.value);
                  setSelectedPatientId("");
                  setRecords([]);
                  setEditingRecord(null);
                }}
                placeholder="Enter all or part of a stay ID"
              />
              <label htmlFor="clinical-patient-id">Select ICU stay ID</label>
              <select
                id="clinical-patient-id"
                required
                value={selectedPatientId}
                onChange={(event) => {
                  setSelectedPatientId(event.target.value);
                  setRecords([]);
                  setEditingRecord(null);
                }}
              >
                <option value="">Select an existing stay</option>
                {patientIds.map((patientId) => (
                  <option key={patientId} value={patientId}>
                    {patientId}
                  </option>
                ))}
              </select>
              {patientLoading && <small>Searching stays…</small>}
              {patientError && <p className="clinical-error" role="alert">{patientError}</p>}
            </div>
          )}

          <form onSubmit={saveRecord}>
            <label htmlFor="clinical-category">Category</label>
            <select
              id="clinical-category"
              value={category}
              onChange={(event) => changeCategory(event.target.value as Category)}
            >
              {Object.entries(CATEGORY_LABELS).map(([key, label]) => (
                <option key={key} value={key}>{label}</option>
              ))}
            </select>

            {category === "blood_test" && (
              <>
                <label htmlFor="clinical-measurement">Blood test</label>
                <select
                  id="clinical-measurement"
                  value={measurementName}
                  onChange={(event) => setMeasurementName(event.target.value)}
                >
                  {Object.entries(BLOOD_TEST_UNITS).map(([name, testUnit]) => (
                    <option key={name} value={name}>{name} ({testUnit})</option>
                  ))}
                </select>
              </>
            )}

            {category === "other_lab" && (
              <>
                <label htmlFor="clinical-measurement">Test name</label>
                <input
                  id="clinical-measurement"
                  required
                  maxLength={120}
                  value={measurementName}
                  onChange={(event) => setMeasurementName(event.target.value)}
                />
              </>
            )}

            {category === "clinical_observation" && (
              <>
                <label htmlFor="clinical-measurement">Observation type</label>
                <select
                  id="clinical-measurement"
                  value={measurementName}
                  onChange={(event) => setMeasurementName(event.target.value)}
                >
                  {Object.entries(OBSERVATION_UNITS).map(([name, observationUnit]) => (
                    <option key={name} value={name}>
                      {name}{observationUnit ? ` (${observationUnit})` : ""}
                    </option>
                  ))}
                </select>
              </>
            )}

            {category === "other_lab" && (
              <>
                <label htmlFor="clinical-unit">Unit</label>
                <input
                  id="clinical-unit"
                  required
                  maxLength={40}
                  value={unit}
                  onChange={(event) => setUnit(event.target.value)}
                  placeholder="Enter the reported unit exactly"
                />
              </>
            )}

            {category === "urine_output" && (
              <p className="clinical-unit-note">Unit: mL (fixed; no conversion)</p>
            )}
            {category === "blood_test" && (
              <p className="clinical-unit-note">
                Unit: {selectedBloodTestUnit} (fixed; no conversion)
              </p>
            )}
            {category === "clinical_observation" && (
              <p className="clinical-unit-note">
                Unit: {selectedObservationUnit ?? "not applicable"}
              </p>
            )}

            <label htmlFor="clinical-value">
              {isNarrativeObservation ? "Observation text" : "Result / volume"}
            </label>
            {isNarrativeObservation ? (
              <textarea
                id="clinical-value"
                required
                maxLength={5000}
                value={value}
                onChange={(event) => setValue(event.target.value)}
              />
            ) : (
              <input
                id="clinical-value"
                type="number"
                required
                step="any"
                min={category === "other_lab" ? undefined : 0}
                max={1_000_000_000_000}
                value={value}
                onChange={(event) => setValue(event.target.value)}
              />
            )}

            <label htmlFor="clinical-measurement-time">
              Measurement time
            </label>
            <input
              id="clinical-measurement-time"
              type="datetime-local"
              required
              value={measurementTime}
              onChange={(event) => setMeasurementTime(event.target.value)}
            />

            {(category === "blood_test" || category === "other_lab") && (
              <>
                <label htmlFor="clinical-sample-collection-time">
                  Sample collection time
                </label>
                <input
                  id="clinical-sample-collection-time"
                  type="datetime-local"
                  required
                  value={sampleCollectionTime}
                  onChange={(event) => setSampleCollectionTime(event.target.value)}
                />
                <label htmlFor="clinical-result-time">Laboratory result time (optional)</label>
                <input
                  id="clinical-result-time"
                  type="datetime-local"
                  value={resultTime}
                  onChange={(event) => setResultTime(event.target.value)}
                />
              </>
            )}

            {category === "urine_output" && (
              <>
                <label htmlFor="clinical-observation-period">Observation period (optional)</label>
                <input
                  id="clinical-observation-period"
                  maxLength={80}
                  value={observationPeriod}
                  onChange={(event) => setObservationPeriod(event.target.value)}
                  placeholder="e.g. 1 hour"
                />
              </>
            )}

            {category === "other_lab" && (
              <>
                <label htmlFor="clinical-reference-range">Reference range (optional)</label>
                <input
                  id="clinical-reference-range"
                  maxLength={120}
                  value={referenceRange}
                  onChange={(event) => setReferenceRange(event.target.value)}
                />
              </>
            )}

            <label htmlFor="clinical-entered-by">
              Operator identifier or display name
            </label>
            <input
              id="clinical-entered-by"
              required
              maxLength={120}
              value={enteredBy}
              onChange={(event) => setEnteredBy(event.target.value)}
            />
            <small>This operator identifier is not authenticated or verified.</small>

            {editingRecord && (
              <>
                <label htmlFor="clinical-change-reason">Correction reason</label>
                <input
                  id="clinical-change-reason"
                  required
                  maxLength={500}
                  value={changeReason}
                  onChange={(event) => setChangeReason(event.target.value)}
                />
              </>
            )}

            <label htmlFor="clinical-notes">Notes (optional)</label>
            <textarea
              id="clinical-notes"
              maxLength={5000}
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
            />

            {saveError && <p className="clinical-error" role="alert">{saveError}</p>}
            {conflictRecordId && (
              <button type="button" onClick={reloadConflictRecord}>
                Reload latest record
              </button>
            )}

            {activePreview && !latestInterpretation && (
              <div
                className={`clinical-interpretation-banner preview status-${activePreview.status.toLowerCase()}`}
                role="region"
                aria-label="Live Clinical Interpretation Preview"
              >
                <div className="interpretation-header">
                  <span className="interpretation-live-tag">LIVE PREVIEW</span>
                  <span className={`interpretation-badge badge-${activePreview.status.toLowerCase()}`}>
                    {activePreview.status.replace("_", " ")}
                  </span>
                  {activePreview.severity && (
                    <span className={`interpretation-severity severity-${activePreview.severity.toLowerCase()}`}>
                      {activePreview.severity}
                    </span>
                  )}
                </div>
                <p className="interpretation-explanation">{activePreview.explanation}</p>
                <div className="interpretation-meta">
                  {activePreview.reference_range && (
                    <span><strong>Ref Range:</strong> {activePreview.reference_range}</span>
                  )}
                  {activePreview.critical_threshold && (
                    <span><strong>Critical:</strong> {activePreview.critical_threshold}</span>
                  )}
                  {activePreview.reference_source && (
                    <span><strong>Guideline:</strong> {activePreview.reference_source}</span>
                  )}
                </div>
              </div>
            )}

            {latestInterpretation && (
              <div
                className={`clinical-interpretation-banner saved status-${latestInterpretation.status.toLowerCase()}`}
                role="region"
                aria-label="Clinical Interpretation Result"
              >
                <div className="interpretation-header">
                  <span className="interpretation-confirmed-tag">EVALUATED</span>
                  <span className={`interpretation-badge badge-${latestInterpretation.status.toLowerCase()}`}>
                    {latestInterpretation.status.replace("_", " ")}
                  </span>
                  {latestInterpretation.severity && (
                    <span className={`interpretation-severity severity-${latestInterpretation.severity.toLowerCase()}`}>
                      {latestInterpretation.severity}
                    </span>
                  )}
                  {latestAlert && (
                    <span className="interpretation-alert-badge">
                      🚨 Alert Triggered
                    </span>
                  )}
                </div>
                <p className="interpretation-explanation">
                  <strong>{latestInterpretation.measurement_name}:</strong> {latestInterpretation.explanation}
                </p>
                <div className="interpretation-meta">
                  {latestInterpretation.reference_range && (
                    <span><strong>Reference:</strong> {latestInterpretation.reference_range}</span>
                  )}
                  {latestInterpretation.critical_threshold && (
                    <span><strong>Critical:</strong> {latestInterpretation.critical_threshold}</span>
                  )}
                  {latestInterpretation.reference_source && (
                    <span><strong>Evidence Basis:</strong> {latestInterpretation.reference_source}</span>
                  )}
                </div>
                {latestInterpretation.status === "UNINTERPRETABLE" && (
                  <p className="interpretation-uninterpretable-note">
                    ⚠️ Stored as narrative observation without false alarms.
                  </p>
                )}
              </div>
            )}

            <button
              className="clinical-save-button"
              type="submit"
              disabled={saving || !selectedPatientId}
            >
              {saving
                ? "Saving…"
                : editingRecord
                  ? "Save correction"
                  : "Save clinical record"}
            </button>
            {editingRecord && (
              <button
                type="button"
                onClick={() => {
                  setEditingRecord(null);
                  setConflictRecordId(null);
                  setValue("");
                  setUnit("");
                  setMeasurementTime(localDateTimeValue());
                  setSampleCollectionTime("");
                  setResultTime("");
                  setObservationPeriod("");
                  setReferenceRange("");
                  setNotes("");
                  setChangeReason("");
                  setSaveError("");
                  setSuccessMessage("");
                }}
              >
                Cancel correction
              </button>
            )}
          </form>
        </section>

        <section className="clinical-panel clinical-history" aria-labelledby="clinical-history-title">
          <h2 id="clinical-history-title" ref={historyHeadingRef} tabIndex={-1}>
            Patient-specific record history
          </h2>
          <p className="clinical-history-scope">
            Manual records only. Urine output is stored independently; no fluid
            balance is calculated.
          </p>
          <div className="clinical-filters">
            <label htmlFor="clinical-history-category">Category filter</label>
            <select
              id="clinical-history-category"
              value={historyCategory}
              onChange={(event) => setHistoryCategory(event.target.value)}
            >
              <option value="">All categories</option>
              {Object.entries(CATEGORY_LABELS).map(([key, label]) => (
                <option key={key} value={key}>{label}</option>
              ))}
            </select>
            <label htmlFor="clinical-start-date">From date</label>
            <input
              id="clinical-start-date"
              type="date"
              value={startDate}
              onChange={(event) => setStartDate(event.target.value)}
            />
            <label htmlFor="clinical-end-date">To date</label>
            <input
              id="clinical-end-date"
              type="date"
              value={endDate}
              onChange={(event) => setEndDate(event.target.value)}
            />
          </div>
          {recordsLoading && <p role="status">Loading clinical records…</p>}
          {recordsError && <p className="clinical-error" role="alert">{recordsError}</p>}
          {!selectedPatientId && (
            <p className="clinical-empty">Select an existing ICU stay to view manual records.</p>
          )}
          {selectedPatientId && !recordsLoading && !recordsError && records.length === 0 && (
            <p className="clinical-empty">No manual clinical records for this stay and filter.</p>
          )}
          <div className="clinical-record-list">
            {selectedPatientId && !recordsLoading && !recordsError && records.map((record) => (
              <article className={`clinical-record-card${record.voided ? " clinical-record-voided" : ""}`} key={record.record_id}>
                <div className="clinical-record-heading">
                  <div>
                    <p className="eyebrow">{CATEGORY_LABELS[record.category]}</p>
                    <h3>{record.measurement_name}</h3>
                    {record.interpretation && (
                      <span className={`interpretation-badge badge-${record.interpretation.status.toLowerCase()}`}>
                        {record.interpretation.status.replace("_", " ")}
                      </span>
                    )}
                  </div>
                  <strong className="clinical-record-value">
                    {record.value}{record.unit ? ` ${record.unit}` : ""}
                  </strong>
                </div>
                <dl>
                  <div><dt>Measurement time</dt><dd>{displayTimestamp(record.measurement_time)}</dd></div>
                  {record.sample_collection_time && (
                    <div><dt>Sample collection</dt><dd>{displayTimestamp(record.sample_collection_time)}</dd></div>
                  )}
                  {record.result_time && (
                    <div><dt>Result time</dt><dd>{displayTimestamp(record.result_time)}</dd></div>
                  )}
                  <div><dt>Entered</dt><dd>{displayTimestamp(record.entry_time)}</dd></div>
                  <div><dt>Operator identifier</dt><dd>{record.entered_by}</dd></div>
                  <div><dt>Source</dt><dd>{record.source}</dd></div>
                  {record.observation_period && (
                    <div><dt>Observation period</dt><dd>{record.observation_period}</dd></div>
                  )}
                  {record.reference_range && (
                    <div><dt>Reference range</dt><dd>{record.reference_range}</dd></div>
                  )}
                  {record.notes && <div><dt>Notes</dt><dd>{record.notes}</dd></div>}
                  {record.updated_at && <div><dt>Corrected</dt><dd>{displayTimestamp(record.updated_at)}</dd></div>}
                  {record.voided && (
                    <div>
                      <dt>Voided</dt>
                      <dd>{displayTimestamp(record.voided_at)} by {record.voided_by}: {record.void_reason}</dd>
                    </div>
                  )}
                </dl>
                <p className="clinical-record-id">Record ID: {record.record_id}</p>
                {!record.voided && (
                  <div className="clinical-record-actions">
                    <button
                      type="button"
                      onClick={() => beginEditingRecord(record)}
                    >
                      Correct record
                    </button>
                    <button
                      type="button"
                      onClick={(event) => openVoidDialog(record, event.currentTarget)}
                    >
                      Void record
                    </button>
                  </div>
                )}
              </article>
            ))}
          </div>
          {successMessage && (
            <p className="clinical-success" role="status">{successMessage}</p>
          )}
          {voidingRecord && (
            <div className="clinical-dialog-backdrop">
              <div
                className="clinical-void-dialog"
                role="dialog"
                aria-modal="true"
                aria-labelledby="clinical-void-title"
                aria-describedby="clinical-void-description"
                ref={voidDialogRef}
                onKeyDown={handleVoidDialogKeyDown}
              >
                <h2 id="clinical-void-title">Void clinical record?</h2>
                <p id="clinical-void-description">
                  This record will be removed from active measurements but retained
                  with its audit history. This does not permanently delete it.
                </p>
                <p className="clinical-void-record">
                  {voidingRecord.measurement_name}: {voidingRecord.value}
                  {voidingRecord.unit ? ` ${voidingRecord.unit}` : ""} · Version{" "}
                  {voidingRecord.version}
                </p>
                <form onSubmit={submitVoid}>
                  <label htmlFor="clinical-voided-by">Operator identifier or display name</label>
                  <input
                    id="clinical-voided-by"
                    required
                    maxLength={120}
                    value={voidedBy}
                    onChange={(event) => setVoidedBy(event.target.value)}
                  />
                  <small>This identifier is not authenticated or verified.</small>
                  <label htmlFor="clinical-void-reason">Reason for voiding</label>
                  <textarea
                    id="clinical-void-reason"
                    ref={voidReasonInputRef}
                    required
                    maxLength={500}
                    value={voidReason}
                    onChange={(event) => setVoidReason(event.target.value)}
                  />
                  {voidError && (
                    <p className="clinical-error" role="alert">{voidError}</p>
                  )}
                  {voidConflictRecordId && (
                    <button
                      type="button"
                      ref={reloadVoidButtonRef}
                      onClick={reloadLatestVoidRecord}
                      disabled={voiding}
                    >
                      Reload latest record
                    </button>
                  )}
                  <div className="clinical-dialog-actions">
                    <button type="button" onClick={closeVoidDialog} disabled={voiding}>
                      Cancel
                    </button>
                    <button type="submit" disabled={voiding}>
                      {voiding ? "Voiding…" : "Confirm void"}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}
        </section>
      </div>
    </section>
  );
}
