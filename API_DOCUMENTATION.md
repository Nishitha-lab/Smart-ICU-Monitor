# API Documentation: Smart ICU Patient Safety System

Base URL: `http://127.0.0.1:8000`

Interactive OpenAPI Documentation: `http://127.0.0.1:8000/docs`

> **Prototype security limitation:** API endpoints, including clinical-record routes, are unauthenticated. Caller-supplied `entered_by` values are not verified identities. Use only on localhost with synthetic data; public or shared-network deployment is prohibited until proper authentication, authorization, and audit-identity verification are implemented.

---

## 1. System & Health

### `GET /`
Returns service name, online status, and version.

### `GET /health`
Returns service health status, whether the eICU CSV dataset loaded, its row count, and the hardware telemetry status snapshot.

### `GET /api/summary`
Returns eICU triage cohort summary: total patient-hour rows, unique ICU stays, priority counts, anomaly and clinical counts, risk-threshold counts, and multi-signal alert count.

---

## 2. Patients API

### `POST /api/patients`
Registers a new patient in the hospital system.
- **Request Body:**
  ```json
  {
    "name": "Alice Walker",
    "patient_code": "P001",
    "age": 58,
    "sex": "female",
    "date_of_birth": "1968-04-12",
    "contact_information": "+1-555-0101"
  }
  ```
- **Responses:**
  - `201 Created`: Returns created patient record.
  - `400 Bad Request`: Blank name, duplicate `patient_code`, or invalid age (<0 or >130).

### `GET /api/patients`
Lists ICU patients. Supports both live registered patients and historical eICU demo rows.
- **Query Parameters:**
  - `source=registered`: List registered patients from SQLite. Without this option, and without `search` or `status`, the endpoint returns the historical eICU patient list.
  - `status`: `'active'`, `'archived'`, or `'all'`; providing this option selects the registered-patient list.
  - `search`: Case-insensitive substring match against `name` and `patient_code`.
  - `priority`: Optional historical eICU priority filter (`critical`, `high`, `moderate`, `low`, `normal`).
  - `limit`: Maximum results (1-500, default 50).

### `GET /api/registered-patients/{patient_ref}`
Retrieves complete profile for a registered patient, including:
- Demographics and active status
- Active ICU stay details
- All historical ICU admissions
- Active device assignment and hardware connection status
- Complete device assignment history
- Recent telemetry packet history
- Latest vital signs (`fluid_ml`, `bpm`, `spo2`, `device_status`)

### `GET /api/patients/{patient_ref}`
Unified patient / stay detail endpoint.
- For patient codes (e.g., `P001`), resolves to the registered patient profile.
- For integer IDs, resolves to the registered patient or historical eICU stay.
- **Ambiguity Guard:** If an integer ID exists in both registered patients and historical eICU stays, the endpoint returns `HTTP 400 Bad Request` requiring an explicit `?source=registered` or `?source=historical` query parameter.
- Invalid source query parameters return `HTTP 422 Unprocessable Entity`.

### `PATCH /api/patients/{patient_ref}`
Updates mutable demographic fields (`name`, `age`, `sex`, `date_of_birth`, `contact_information`).

### `POST /api/patients/{patient_ref}/archive`
Archives a patient. An archived patient cannot receive new ICU admissions or device assignments, but all historical records remain intact.

### `DELETE /api/patients/{patient_ref}`
Deletes a patient record.
- **Guarded:** Returns `HTTP 400 Bad Request` if the patient has associated ICU stays, clinical records, or telemetry records.

---

## 3. ICU Admissions (Stays) API

### `POST /api/patients/{patient_ref}/stays`
Admits an active patient to the ICU.
- **Request Body:**
  ```json
  {
    "bed_number": "ICU-Bed-04",
    "ward_or_unit": "Cardiothoracic ICU",
    "admission_reason": "Post-op coronary artery bypass observation"
  }
  ```
- **Responses:**
  - `201 Created`: Returns created ICU stay with generated `stay_code` (e.g. `S001`).
  - `400 Bad Request`: Patient is archived or already has an active ICU admission (`uq_active_patient_stay`).

### `GET /api/patients/{patient_ref}/stays`
Returns all historical and active ICU admission episodes for the patient.

### `GET /api/stays/{stay_ref}`
Returns details for a specific stay by ID or stay code.

### `PATCH /api/stays/{stay_ref}`
Updates stay details (`bed_number`, `ward_or_unit`, `admission_reason`).

### `POST /api/stays/{stay_ref}/discharge`
Discharges an active ICU stay.
- Sets `discharge_datetime = UTC now` and `discharge_status = 'discharged'`.
- Automatically releases and unassigns any active bedside monitoring device.
- Preserves all historical telemetry and clinical records.

---

## 4. Bedside Devices & Patient Assignment API

### `POST /api/devices`
Registers a bedside monitoring device.
- **Request Body:**
  ```json
  {
    "device_id": "ESP32-001",
    "device_name": "ICU Bed 4 Monitor",
    "connection_type": "serial",
    "port": "COM9"
  }
  ```

### `GET /api/devices`
Returns all registered devices with hardware connection status, active patient assignment, and latest telemetry packet.

### `POST /api/devices/{device_id}/assign`
Explicitly assigns a device to an active patient and their active ICU stay.
- **Request Body:**
  ```json
  {
    "patient_id": 1,
    "stay_id": 101
  }
  ```
- **Validation:**
  - Patient must exist and be active.
  - ICU stay must exist, be active, and belong to the specified patient.
  - Automatically terminates any previous active assignment for this device.
- **Responses:**
  - `200 OK`: Device assignment activated.
  - `400 Bad Request`: Discharged stay, mismatched patient/stay, or inactive patient.

### `POST /api/devices/{device_id}/unassign`
Ends monitoring and releases the device from the patient.
- Closes the active assignment with `unassigned_at = UTC now`.
- Subsequent telemetry from this device is not attributed to any patient.

### `GET /api/devices/{device_id}/assignments`
Returns the chronological audit log of all historical patient assignments for this device.

---

## 5. Telemetry & Timeline API

### `POST /api/devices/{device_id}/telemetry`
Ingests or simulates an incoming sensor reading packet.
- **Request Body:**
  ```json
  {
    "fluid_ml": 420.5,
    "bpm": 76.0,
    "spo2": 98.0,
    "device_status": "NORMAL",
    "reading_time": "2026-11-08T01:30:00Z"
  }
  ```
- **Attribution Behavior:**
  - If the device has an active assignment, the reading is saved with `patient_id` and `stay_id`.
  - If the device is unassigned, `patient_id = NULL` and `stay_id = NULL`.
  - Readings older than assignment start time are stored without patient attribution.
- **Metric validity:** The backend derives validity from the values and ignores caller-supplied validity flags. Fluid values must be non-negative, BPM must be within the telemetry parser's technical bounds of 40–200, and SpO₂ must be within its technical bounds of 70–100. These are data-quality bounds, not clinical reference ranges. Raw values remain stored for traceability; invalid metric fields are returned as `null` with validity metadata in telemetry views.

### `GET /api/patients/{patient_ref}/telemetry?limit=100`
Returns chronological telemetry readings attributed to this patient.

### `GET /api/stays/{stay_ref}/telemetry?limit=100`
Returns telemetry readings attributed to a specific ICU admission.

### `GET /api/patients/{patient_ref}/timeline`
Returns an auditable unified chronological timeline of all events for a patient:
- Patient registration
- ICU admission and discharge events
- Bedside device assignments and unassignments
- Clinical observation and laboratory records
- Sensor alerts

---

## 6. Clinical Records & Observations API

### `POST /api/clinical-records`
Creates an audited clinical observation or laboratory record. The category must be `urine_output`, `blood_test`, `other_lab`, or `clinical_observation`.

**Identifier Integrity Rules:**
- **Registered ICU Admission:** Send `stay_id` and optional `patient_id`. Do **NOT** send `patientunitstayid`. The stay must be active and must belong to the patient.
- **Historical eICU Admission:** Send `patientunitstayid` only. Do **NOT** send `patient_id` or `stay_id`.
- **Rejection Policy:** Supplying both `patientunitstayid` and `stay_id`/`patient_id` is strictly rejected with `HTTP 422 Unprocessable Entity`. Mismatched patient and stay identifiers are rejected with `HTTP 400 Bad Request`. Discharged stays are rejected with `HTTP 400 Bad Request`. Nonexistent patients or stays are rejected with `HTTP 404 Not Found`.

- **Request Body (Registered Admission):**
  ```json
  {
    "patient_id": 1,
    "stay_id": 101,
    "category": "urine_output",
    "measurement_name": "Urine output",
    "value": 150.0,
    "unit": "mL",
    "measurement_time": "2026-11-08T01:00:00Z",
    "entered_by": "Nurse J. Smith",
    "observation_period": "1 hour"
  }
  ```

- **Request Body (Historical eICU Record):**
  ```json
  {
    "patientunitstayid": 242040,
    "category": "urine_output",
    "measurement_name": "Urine output",
    "value": 85.0,
    "unit": "mL",
    "measurement_time": "2026-11-08T01:00:00Z",
    "entered_by": "eICU Researcher"
  }
  ```

### `GET /api/clinical-records/patient-stays`
Searches historical eICU stay IDs by prefix. Supports `query` and `limit`.

### `GET /api/patients/{patient_ref}/clinical-records`
Retrieves clinical records for a patient with optional filters:
- `category`: `urine_output`, `blood_test`, `other_lab`, or `clinical_observation`
- `start_time`, `end_time`: ISO UTC filter bounds
- `include_voided`: `true` or `false` (default: false)
- `stay_id`: optional registered ICU stay filter
- `source`: `registered` or `historical`; use this to disambiguate numeric registered patient IDs from historical eICU stay IDs. The default retains legacy numeric-ID lookup behavior.

### `GET /api/stays/{stay_ref}/clinical-records`
Retrieves records for a registered ICU stay, with the same category, time, and void filters.

### `GET /api/clinical-records/{record_id}`
Retrieves an individual record, including its current version and void status.

### `GET /api/clinical-records/{record_id}/revisions`
Retrieves preserved pre-correction snapshots and revision reasons.

### `PUT /api/clinical-records/{record_id}`
Updates/corrects an existing record.
- Requires `expected_version` (optimistic concurrency locking).
- Requires `change_reason` for clinical audit trail.
- Preserves previous state in revision history.
- **Alert Synchronization:** Automatically supersedes obsolete active alerts linked to this record and re-evaluates the revised reading.

### `DELETE /api/clinical-records/{record_id}`
Performs a reasoned clinical void.
- Requires `expected_version`, `voided_by`, and `void_reason`.
- Record remains stored with a void timestamp and audit reason; the response exposes `voided: true`. Voiding is not permanent deletion.
- **Alert Synchronization:** Automatically supersedes any active alerts linked to this voided record.

### `POST /api/clinical-records/evaluate`
Evaluates a clinical record candidate without persisting it (live preview).
- **Request Body:**
  ```json
  {
    "category": "blood_test",
    "measurement_name": "Potassium",
    "value": 6.8,
    "unit": "mmol/L",
    "patient_id": 1
  }
  ```
- **Response:**
  ```json
  {
    "status": "CRITICAL_HIGH",
    "severity": "CRITICAL",
    "category": "electrolyte",
    "measurement_name": "Potassium",
    "measured_value": 6.8,
    "unit": "mmol/L",
    "reference_range": "3.5 - 5.0 mmol/L",
    "critical_threshold": ">= 6.0 mmol/L",
    "reference_source": "Tietz Textbook of Clinical Chemistry",
    "explanation": "CRITICAL HIGH: Potassium 6.8 mmol/L exceeds critical threshold >= 6.0 mmol/L (cardiac arrhythmia/arrest risk).",
    "requires_alert": true
  }
  ```

---

## 6. Unified Clinical Alerts API

> **Prototype Security & Identity Limitation:** Alert endpoints, including acknowledgment and resolution, are unauthenticated in this development release. Operator identities (`acknowledged_by` / `clinician_name` and `resolved_by`) are self-reported and not verified by an authentication service. Do not expose these endpoints on public networks without deploying reverse-proxy authentication, JWT verification, and role-based access control.

### `GET /api/alerts`
Lists clinical alerts from all sources (`ESP32_TELEMETRY`, `LAB_RESULT`, `CLINICAL_OBSERVATION`).
- **Query Parameters:**
  - `status`: Filter by `'ACTIVE'`, `'ACKNOWLEDGED'`, `'RESOLVED'`, `'SUPERSEDED'`, or `'ALL'` (default: `'ACTIVE'`).
  - `severity`: Filter by `'CRITICAL'`, `'HIGH'`, `'MODERATE'`, `'LOW'`, or `'ALL'`.
  - `source`: Filter by source stream (`'ESP32_TELEMETRY'`, `'LAB_RESULT'`, `'CLINICAL_OBSERVATION'`).
  - `stay_id`, `patient_id`: Scope to specific ICU stay or registered patient.

### `GET /api/alerts/summary`
Returns statistical counts of clinical alerts:
```json
{
  "total": 12,
  "active": 3,
  "acknowledged": 1,
  "resolved": 7,
  "superseded": 1,
  "critical": 1,
  "high": 2,
  "moderate": 1,
  "low": 0
}
```

### `GET /api/alerts/{alert_id}`
Retrieves an alert by its UUID.

### `GET /api/alerts/{alert_id}/history`
Retrieves chronological audit log entries for an alert (`CREATED`, `ESCALATED`, `ACKNOWLEDGED`, `RESOLVED`, `SUPERSEDED`), including actor, timestamp, and clinical notes.

### `POST /api/alerts/{alert_id}/acknowledge`
Transitions an active alert to `ACKNOWLEDGED`.
- **Request Body (Canonical):**
  ```json
  {
    "acknowledged_by": "Dr. Sarah Chen, MD",
    "note": "Notified attending physician, initiating repeat lab panel."
  }
  ```
- **Accepted Alias:**
  ```json
  {
    "clinician_name": "Dr. Sarah Chen, MD"
  }
  ```

### `POST /api/alerts/{alert_id}/resolve`
Transitions an alert to `RESOLVED`.
- **Request Body (Canonical):**
  ```json
  {
    "resolved_by": "Dr. Sarah Chen, MD",
    "note": "Potassium corrected to 4.2 mmol/L following insulin-glucose infusion."
  }
  ```
- **Accepted Alias:**
  ```json
  {
    "clinician_name": "Dr. Sarah Chen, MD",
    "resolution_note": "Potassium corrected to 4.2 mmol/L following insulin-glucose infusion."
  }
  ```

### `GET /api/patients/{patient_ref}/alerts`
Retrieves alerts attributed to a registered patient.

### `GET /api/stays/{stay_ref}/alerts`
Retrieves alerts attributed to an ICU stay.

---

## 7. Machine Learning Status API

### `GET /api/ml/status`
Inspects the filesystem state of machine learning artifacts and reports deployment status safely without synthesizing probabilities or mock model files.
- **Response:**
  ```json
  {
    "status": "UNAVAILABLE",
    "deterioration_model": {
      "artifact": "ml/models/deterioration_logistic_v1.joblib",
      "target": "ICU discharge marked expired within 24 hours",
      "status": "UNAVAILABLE",
      "loaded": false,
      "missing_path": "ml\\models\\deterioration_logistic_v1.joblib"
    },
    "anomaly_model": {
      "artifact": "ml/models/anomaly_isolation_forest_v2.joblib",
      "status": "UNAVAILABLE",
      "loaded": false,
      "missing_path": "ml\\models\\anomaly_isolation_forest_v2.joblib"
    },
    "diagnostic_notes": [
      "Missing required artifact: ml\\models\\deterioration_logistic_v1.joblib",
      "Missing required artifact: ml\\models\\anomaly_isolation_forest_v2.joblib",
      "ML inference is offline. Deterministic clinical rules (KDIGO, Tietz, AHA) and hardware safeguards remain fully operational."
    ]
  }
  ```
