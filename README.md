# Smart ICU Patient Safety System

Smart ICU is a full-stack patient-centred ICU monitoring and clinical safety platform. The system bridges live IoT telemetry from bedside ESP32 sensor units with structured clinical records, ICU admissions management, and historical machine-learning risk intelligence derived from the eICU Collaborative Research Database.

> [!IMPORTANT]
> **Academic Research & Demonstration Prototype:** This application is developed for research and demonstration purposes. It is not a certified medical device and is not validated for clinical diagnosis, patient triage, or direct medical treatment decisions.

---

## 1. System Architecture & Patient-Centred Design

The system implements a strict patient-centred architecture organized around four core entities:

```
+-----------------------------------------------------------------------------------+
|                                  PATIENT                                          |
|  - id (PK)                                                                        |
|  - patient_code (Unique, e.g. P001)                                               |
|  - name, age, sex, date_of_birth, contact_information                             |
|  - status: 'active' | 'archived'                                                  |
+-----------------------------------------+-----------------------------------------+
                                          | 1:N
                                          v
+-----------------------------------------------------------------------------------+
|                                  ICU STAY                                         |
|  - id (PK)                                                                        |
|  - stay_code (Unique, e.g. S001)                                                  |
|  - patient_id (FK -> patients.id)                                                |
|  - bed_number, ward_or_unit, admission_reason                                     |
|  - admission_datetime, discharge_datetime                                         |
|  - discharge_status: 'active' | 'discharged' | 'transferred'                      |
|  * Constraint: At most ONE active stay per patient (uq_active_patient_stay)        |
+-------------------+-------------------------------------+-------------------------+
                    | 1:N                                 | 1:N
                    v                                     v
+--------------------------------------+   +----------------------------------------+
|          DEVICE ASSIGNMENT           |   |            CLINICAL RECORDS            |
|  - id (PK)                           |   |  - record_id (UUID PK)                 |
|  - device_id (FK -> devices.device_id|   |  - patient_id (FK -> patients.id)      |
|  - patient_id (FK -> patients.id)    |   |  - stay_id (FK -> icu_stays.id)        |
|  - stay_id (FK -> icu_stays.id)      |   |  - category: urine_output, blood_test, |
|  - assigned_at, unassigned_at        |   |    clinical_observation, notes         |
|  - assignment_status: active/closed  |   |  - versioned audit trail & reasoned    |
|  * Constraint: Max ONE active patient|   |    voiding (never hard deleted)        |
|    per device (uq_active_device_asn) |   +----------------------------------------+
+-------------------+------------------+
                    | 1:N
                    v
+-----------------------------------------------------------------------------------+
|                                 TELEMETRY RECORD                                  |
|  - id (PK)                                                                        |
|  - device_id (FK -> devices.device_id)                                            |
|  - patient_id (Nullable FK; NULL if device is unassigned)                          |
|  - stay_id (Nullable FK; NULL if device is unassigned)                             |
|  - fluid_ml, heart_rate_bpm, spo2_percent, device_status                           |
|  - reading_timestamp, ingestion_timestamp                                         |
+-----------------------------------------------------------------------------------+
```

### Complete Separation of Live Patient Data & Historical eICU Dataset
- **Registered ICU Patients:** Stored locally in SQLite (`data/clinical_records.sqlite3`), managing live patients, admissions, ESP32 device assignments, audited clinical records, and incoming telemetry.
- **Historical eICU Research Data:** Read-only reference dataset loaded at startup from `data/processed/smart_icu_final_alerts.csv` used exclusively by the eICU risk triage research dashboard. Live registered patients and live telemetry are completely isolated from historical research records.

---

## 2. Hardware Specification & IoT Serial Integration

The platform integrates bedside ESP32 microcontrollers streaming real-time vitals and IV bag weight.

### Bedside Hardware Pinout
- **HX711 Load Cell Amplifier (IV Fluid Monitoring):**
  - `DT` (Data): **GPIO 4**
  - `SCK` (Clock): **GPIO 5**
- **Pulse Oximeter (MAX30100 / MAX30102 via I2C):**
  - `SDA`: **GPIO 21**
  - `SCL`: **GPIO 22**
- **Auditory Alert:**
  - `Buzzer`: **GPIO 18**
- **Serial Configuration:**
  - Baud Rate: `115200`
  - Default Serial Port: `COM9` (configurable via `SMART_ICU_SERIAL_PORT` and `SMART_ICU_DEVICE_ID`)
- **Firmware Location:** `hardware/esp32/TBP_final.ino`
- **Serial Message Format:**
  ```text
  Fluid: 450.00 mL | BPM: 74 | SpO2: 98% | STATUS: NORMAL
  ```

### Firmware Limitation & Backend Device Abstraction
Physical ESP32 firmware transmitting the standard serial string does not transmit a unique hardware MAC or burned identifier. To ensure enterprise traceability:
1. The backend provides a stable registered device abstraction (`devices` table).
2. The active serial port monitor maps the physical serial stream to a backend-configured stable identifier (`SMART_ICU_DEVICE_ID`, default: `ESP32-001`).
3. Additional devices (e.g. `ESP32-002`, `ESP32-003`) can be registered and simulated simultaneously via the API and frontend device management interface.

---

## 3. Strict Telemetry Attribution & Patient Safety Integrity

A core safety requirement of the Smart ICU architecture is preventing telemetry misattribution:

> [!CAUTION]
> **CRITICAL PATIENT SAFETY RULE:**
> The backend resolves patient attribution **strictly from the validated, active device assignment stored in the database**. The frontend's currently selected patient or open tab is **NEVER** used to guess or attribute incoming sensor telemetry.

### Attribution Rules Enforced by Backend
1. **Active Assignment Required:** When a reading arrives from `ESP32-001`, the backend verifies whether an active assignment exists (`assignment_status = 'active'`).
2. **Unassigned Device Handling:** If `ESP32-001` is connected but unassigned, the reading is logged to `telemetry_records` with `patient_id = NULL` and `stay_id = NULL`. The frontend displays a prominent warning: `⚠️ Device Not Assigned`. Unassigned readings are **never** attached to any patient, and **never** fed into patient charts or alerts.
3. **Delayed Reading Policy:** If a reading carries a sensor timestamp prior to the assignment's `assigned_at` timestamp, it is rejected from attribution to prevent historical data leakage across patients.
4. **Reassignment Boundary:** Reassigning a device automatically terminates the previous assignment (`unassigned_at = UTC now`, status = `'reassigned'`). Subsequent readings immediately attribute to the new patient with zero telemetry leakage.
5. **Database-Level Concurrency Guards:** Partial unique SQLite indexes prevent multiple concurrent active admissions per patient (`uq_active_patient_stay`) and multiple concurrent active assignments per device (`uq_active_device_assignment`).
6. **Telemetry Quality:** The backend derives validity flags from the received values rather than trusting caller-supplied flags. Out-of-range values are retained in the database for traceability, but invalid metric fields are returned as `null` with validity metadata in patient/device/timeline telemetry views. These bounds are technical data-quality checks, not clinical reference ranges or diagnoses.

---

## 4. Workflows

### A. Patient Registration & Admission
1. Open the **Patients** tab in the top navigation.
2. Click **+ Register Patient**. Provide full legal name, optional patient code (e.g. `P001`), age, sex, date of birth, and contact information.
3. In the patient's detail view, click **+ Admit to ICU**. Specify bed number (e.g. `ICU-Bed-04`), ward/unit (e.g. `Cardiothoracic ICU`), and admission reason.
4. The patient now has an active ICU stay (e.g. `S001`).

### B. Device Registration & Assignment
1. Open the **Device Management** tab.
2. Click **+ Register New ESP32** to register a device (e.g. `ESP32-001`).
3. On an unassigned device card, click **Assign to Patient**.
4. Select the active patient and active ICU stay. Confirm the assignment.
5. The device is now actively bound to the patient and stay. Incoming serial readings or simulated telemetry packets are automatically attributed to this patient.

### C. Clinical Record Documentation
1. In the Patient Detail view, scroll to the **Clinical Records & Observations** section.
2. The form preselects the active patient and stay. Record urine output, blood tests, vital observations, or clinical notes.
3. Records feature optimistic concurrency control (`version`), revision histories, and auditable reasons for corrections or voids.

### D. Patient Discharge & De-escalation
1. In the Patient Detail view, click **Discharge Stay**.
2. Confirming discharge records the discharge timestamp, marks the stay as `discharged`, and automatically unassigns any active monitoring device.
3. All historical telemetry, stays, device assignments, and clinical records are permanently preserved for auditable review in the patient's chronological timeline.

---

## 5. Local Setup & Execution (Windows)

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- Optional: ESP32 hardware connected via USB (default port `COM9`)

### 1. Backend Setup
From PowerShell:
```powershell
cd D:\Smart-ICU\backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Start the Backend API
```powershell
cd D:\Smart-ICU\backend
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```
- API Base: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

### 3. Frontend Setup & Launch
In a second PowerShell terminal:
```powershell
cd D:\Smart-ICU\frontend
npm ci
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 6. Demonstration Without Physical Hardware

You can demonstrate and test the full end-to-end attribution lifecycle without needing a physical ESP32 connected:

1. **Register Patient & Admit to ICU:**
   - Create Patient `Alice Walker` (`P001`) and admit to `ICU-Bed-04` (Stay `S001`).
   - Create Patient `Bob Smith` (`P002`) and admit to `ICU-Bed-08` (Stay `S002`).
2. **Assign Device:**
   - In **Device Management**, assign `ESP32-001` to `Alice Walker` (`S001`).
3. **Inject Simulated Telemetry:**
   - In the **Interactive Telemetry Simulator** panel on `ESP32-001`, submit a packet:
     - Fluid: `420 mL`, BPM: `78`, SpO2: `99%`, Status: `NORMAL`.
   - The UI confirms attribution: `✓ Reading accepted and attributed to Patient #1 (Stay #101)`.
   - In **Patients -> Alice Walker**, observe the live vital boxes update immediately.
   - In **Patients -> Bob Smith**, verify his vitals remain empty (`—`).
4. **Demonstrate Unassigned Device Isolation:**
   - Unassign `ESP32-001`. The card shows `⚠️ Device Not Assigned`.
   - Inject another simulated packet on `ESP32-001`.
   - Feedback confirms: `⚠️ Device is unassigned: Reading accepted by hardware log, but NOT attributed to any patient.`
   - Check Alice Walker's profile: her records do **NOT** receive this reading.
5. **Demonstrate Reassignment:**
   - Assign `ESP32-001` to `Bob Smith`.
   - Inject simulated telemetry. Bob's profile now reflects the live measurements.

---

## 7. REST API Reference

### Patients
- `POST /api/patients` - Register a patient (`name`, `patient_code`, `age`, `sex`, `date_of_birth`, `contact_information`).
- `GET /api/patients?source=registered&status=active&search=P001` - List registered patients with active stay and device metadata.
- `GET /api/registered-patients/{id}` - Retrieve patient profile, admission history, device assignments, and latest telemetry.
- `PATCH /api/patients/{id}` - Update patient demographics.
- `POST /api/patients/{id}/archive` - Archive patient (prevents new stays; preserves clinical history).
- `DELETE /api/patients/{id}` - Delete patient (rejected with HTTP 400 if admissions or telemetry exist).

### ICU Admissions (Stays)
- `POST /api/patients/{id}/stays` - Admit patient to ICU (`bed_number`, `ward_or_unit`, `admission_reason`).
- `GET /api/patients/{id}/stays` - List all historical and active stays for a patient.
- `GET /api/stays/{id}` - Retrieve stay details.
- `PATCH /api/stays/{id}` - Update stay metadata (bed number, unit, notes).
- `POST /api/stays/{id}/discharge` - Discharge stay (marks stay discharged and automatically releases active device).

### Devices & Assignments
- `POST /api/devices` - Register an ESP32 monitoring device (`device_id`, `device_name`, `connection_type`, `port`).
- `GET /api/devices` - List all registered devices with connection and active assignment status.
- `GET /api/devices/{device_id}` - Retrieve device details.
- `POST /api/devices/{device_id}/assign` - Assign device to patient and active stay (`patient_id`, `stay_id`).
- `POST /api/devices/{device_id}/unassign` - Unassign device.
- `GET /api/devices/{device_id}/assignments` - Retrieve complete chronological assignment audit history.

### Telemetry Ingestion & Timeline
- `POST /api/devices/{device_id}/telemetry` - Ingest incoming sensor packet (`fluid_ml`, `bpm`, `spo2`, `device_status`, `reading_time`). Resolves attribution strictly from backend assignment and derives metric validity on the server.
- `GET /api/patients/{id}/telemetry` - Retrieve historical telemetry attributed to patient.
- `GET /api/stays/{id}/telemetry` - Retrieve telemetry attributed to a specific ICU stay.
- `GET /api/patients/{id}/timeline` - Retrieve unified chronological audit timeline (admissions, discharges, assignments, clinical records).

### Clinical Observations & Lab Records
- `POST /api/clinical-records` - Record urine output, blood tests, or observations.
- `GET /api/patients/{patient_id}/clinical-records` - Retrieve clinical records for patient.
- `GET /api/stays/{stay_id}/clinical-records` - Retrieve clinical records for stay.
- `GET /api/clinical-records/{id}` - Retrieve record.
- `PUT /api/clinical-records/{id}` - Correct record with version check and audit reason.
- `DELETE /api/clinical-records/{id}` - Reasoned void of record.

---

## 8. Automated Testing & Verification

The codebase includes comprehensive unit, integration, attribution safety, and UI component tests:

### Run Backend Tests (86 tests)
```powershell
cd D:\Smart-ICU\backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
- The current suite contains 86 tests covering identifier separation, patient-stay ownership, clinical records, patient lifecycle, device assignment, telemetry attribution, concurrency, and serial reliability.

### Run Frontend Tests & Checks (32 tests)
```powershell
cd D:\Smart-ICU\frontend
npm test -- --run
npm run lint
npm run build
```
- 9 tests in `App.test.tsx` (eICU dashboard, triage, hardware banners).
- 13 tests in `ClinicalRecords.test.tsx` (record creation, identifier integrity, optimistic locking, voiding).
- 5 tests in `PatientManagement.test.tsx` (patient list, register modal, detail view, discharge, stale request handling).
- 5 tests in `DeviceManagement.test.tsx` (device list, registration modal, assignment modal, unassign, simulator).
- **ESLint:** 0 errors, 0 warnings.
- **TypeScript & Vite Build:** 100% clean production bundle.

---

## 9. Academic Limitations & Boundary Notice

1. **Authentication:** All endpoints, including clinical-record routes, are unauthenticated; caller-supplied `entered_by` values are not verified identities. Use only on localhost for synthetic demonstrations. Deployment on public or shared networks is prohibited until proper authentication, authorization, and audit-identity verification are implemented.
2. **Clinical Reference Ranges:** Displayed flags and validation bounds are technical sanity checks, not diagnostic algorithms.
3. **No Automatic Clinical Diagnosis:** The platform does not prescribe medication, calculate dynamic drug dosages, or diagnose disease.
4. **Hardware Identity:** ESP32 firmware lacks cryptographically signed hardware identity; the system uses backend-managed device identifiers.
