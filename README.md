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

## 5. Docker & Docker Compose Quickstart (Recommended)

Smart ICU can be launched with a single command using Docker and Docker Compose, eliminating manual Python, Node.js, and package version conflicts across host operating systems.

### 1. Prerequisites & Docker Desktop Setup
- **Operating System:** Windows 10/11 64-bit (WSL2 recommended), macOS (Intel or Apple Silicon), or Linux (Ubuntu 20.04+, Debian 11+).
- **System Memory:** Minimum 4 GB RAM (8 GB recommended).
- **Docker Installation:**
  - Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine + Docker Compose Plugin (Linux).
  - Verify Docker in your terminal:
    ```bash
    docker --version
    docker compose version
    ```

### 2. Repository Cloning
```bash
git clone https://github.com/Nishitha-lab/Smart-ICU-Monitor.git Smart-ICU
cd Smart-ICU
```

### 3. Required Datasets & Model Artifacts
Before launching containers, verify that required data and model artifacts are present:
- **Historical Reference Population JSON (`data/processed/eicu_population_reference_v1.json`):**
  - Included and tracked in the repository. Automatically mounted into the container.
- **Machine Learning Models (`ml/models/`):**
  - Baseline models (`deterioration_logistic_v1.joblib`, `anomaly_isolation_forest_v2.joblib`, `deterioration_gradient_boosting_v1.joblib`, `deterioration_random_forest_v1.joblib`) and `model_comparison_metrics.json` are committed to the repository and mounted read-only (`:ro`).
- **Final Alert Research Dataset (`data/processed/smart_icu_final_alerts.csv`):**
  - **Required for historical eICU triage dashboard.**
  - Due to eICU data use restrictions, patient-level derived CSVs are excluded from Git.
  - If you have an authorized copy of `smart_icu_final_alerts.csv`, place it at:
    ```text
    data/processed/smart_icu_final_alerts.csv
    ```
  - *If absent:* The backend starts up and clinical CRUD/telemetry operations operate normally, but the historical eICU dataset endpoints return a clear `FileNotFoundError` explaining that the CSV is missing.

### 4. Environment Configuration
The system uses sensible defaults out of the box in `docker-compose.yml`. You may optionally configure environment variables via a root `.env` file or directly in `docker-compose.yml`:
```ini
SMART_ICU_SERIAL_PORT=NONE
SMART_ICU_BAUDRATE=115200
SMART_ICU_DEVICE_ID=ESP32-001
SMART_ICU_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://localhost:80,http://localhost
```

### 5. One-Command Application Startup
Build and start both backend and frontend containers:
```bash
docker compose up --build
```
Or run in the background (detached mode):
```bash
docker compose up -d --build
```

### 6. Accessing the Application
Once the containers are running and healthy, open your web browser:
- **Frontend Web Dashboard:** [http://localhost:5173](http://localhost:5173)
  *(Served via Nginx with built-in SPA routing and automatic reverse proxy for `/api/` and `/health`)*
- **Backend Direct Access & Health Check:** [http://localhost:8000/health](http://localhost:8000/health)
- **Interactive Swagger / OpenAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Alternative ReDoc API Documentation:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 7. Stopping, Rebuilding & Viewing Logs
- **Stop containers:**
  ```bash
  docker compose down
  ```
- **Stop containers and delete volumes (clean reset):**
  ```bash
  docker compose down -v
  ```
- **Rebuild images cleanly:**
  ```bash
  docker compose build --no-cache
  docker compose up -d
  ```
- **View aggregated real-time logs:**
  ```bash
  docker compose logs -f
  ```
- **View logs for a specific service:**
  ```bash
  docker compose logs -f backend
  docker compose logs -f frontend
  ```
- **Inspect container status and health:**
  ```bash
  docker compose ps
  ```

### 8. Running Automated Tests Inside Containers
Run the backend test suite directly inside the running backend container:
```bash
docker compose exec backend pytest tests -v
```

### 9. Troubleshooting Common Errors
- **Port 5173 or 8000 already in use:**
  - Check if another application or development server is using port 5173 or 8000. Stop the competing service, or adjust the port mapping in `docker-compose.yml` (e.g. `"5174:80"` or `"8001:8000"`).
- **Backend logs show `Smart-ICU dataset not found`:**
  - Confirm that `data/processed/smart_icu_final_alerts.csv` is present in your host `data/processed/` directory.
- **Frontend displays connection or network error:**
  - Verify that the backend container is healthy: run `docker compose ps` and verify `smart-icu-backend` status is `(healthy)`.
- **Hardware Serial Warning / `SerialException`:**
  - Physical ESP32 hardware is completely optional. The container environment defaults to `SMART_ICU_SERIAL_PORT=NONE`, ensuring the server starts and operates in software mode without errors.
- **Container health check failing:**
  - Inspect backend startup trace: `docker compose logs backend`.

---

## 6. Local Manual Setup & Execution (Alternative)

For developers actively modifying code who prefer running directly on the host machine without Docker:

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- Optional: ESP32 hardware connected via USB (default port `COM9`)

### 1. Backend Setup
From PowerShell:
```powershell
cd D:\Smart-ICU\backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> [!IMPORTANT]
> **Dedicated Virtual Environment Required:**
> Always execute backend services and tests using the dedicated virtual environment (`D:\Smart-ICU\backend\.venv\Scripts\python.exe` or by activating `.\backend\.venv\Scripts\Activate.ps1`). Do **NOT** invoke backend services or tests using unactivated system Python directly, as model artifacts are pinned to `scikit-learn==1.9.0`.

### 2. Start the Backend API
```powershell
cd D:\Smart-ICU\backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
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

## 7. Demonstration Without Physical Hardware

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

## 8. Unified Clinical Alert & ML Intelligence Engine

The platform integrates a continuous clinical interpretation and deterministic alert engine synthesizing multiple data streams:

```
+-----------------------------------------------------------------------------------+
|                        CLINICAL DATA INGESTION STREAMS                            |
|   1. ESP32 Sensor Telemetry   2. Lab Results   3. Urine Output   4. Observations  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     CLINICAL ALERT ENGINE (DETERMINISTIC)                         |
|  - KDIGO 2012 AKI Criteria (Duration-dependent urine output; weight-adjusted)      |
|  - Canonical Lab Registry (Electrolytes, Renal, Hematology, Metabolic; Tietz/AHA) |
|  - Vital Signs Rules (Bradycardia/Tachycardia, Hypoxemia, IV Fluid Depletion)     |
|  - Quality Masking: Disconnected sensors (BPM/SpO2 == 0) never trigger false alarm|
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     ALERT MANAGER & AUDIT LIFECYCLE                               |
|  - Fingerprint Deduplication (SHA-256: stay_id + source + measurement)             |
|  - Severity Escalation (Increments counter and elevates severity when deteriorating)|
|  - Auto-Resolution (Subsequent normal reading auto-resolves ACTIVE alert with log)|
|  - Record Revision Synchronization (Correcting/voiding records supersedes alerts)  |
|  - Clinician Action (Acknowledge & Resolve with mandatory clinician audit note)   |
+-----------------------------------------------------------------------------------+
```

### KDIGO 2012 Urine Output Criteria
- Evaluates rate in $mL/kg/h$ using patient's weight from admissions record:
  - **Stage 1 (Moderate Risk):** $< 0.5\text{ mL/kg/h}$ for $\ge 6$ hours.
  - **Stage 2 (High Risk):** $< 0.5\text{ mL/kg/h}$ for $\ge 12$ hours.
  - **Stage 3 (Critical / Anuria):** $< 0.3\text{ mL/kg/h}$ for $\ge 24$ hours OR total anuria ($\le 10\text{ mL}$) for $\ge 12$ hours.
- **Weight Fallback:** If patient body weight is unrecorded, evaluates volumetric rate ($mL/h$) using $30\text{ mL/h}$ (oliguria) and $10\text{ mL/h}$ (anuria) with explicit warning in the explanation.

### Canonical Laboratory Reference Registry
- 15 Standard canonical tests: Potassium, Sodium, Chloride, Bicarbonate, Creatinine, BUN, Urea, Hemoglobin, Hematocrit, WBC, Platelets, Glucose, Calcium, Magnesium, Lactate.
- Each test defines normal reference ranges, critical high/low thresholds, acceptable clinical units with automatic mathematical normalization (e.g. $\text{WBC/Platelets } /\mu L \rightarrow K/\mu L$, $\text{Lactate } mg/dL \rightarrow mmol/L$, $\text{Urea } mmol/L \rightarrow mg/dL$), and canonical evidence sources (Tietz Textbook of Clinical Chemistry, Surviving Sepsis 2021, KDIGO 2012, AHA, AABB).
- Additional panels (e.g. eGFR, Arterial Blood Gases, Troponin) are dynamically evaluated using clinician-provided custom reference ranges, or preserved safely as uninterpreted results without false alarms.

### Telemetry Alert Evaluation & Hardware Safeguards
- Evaluates valid bedside sensor telemetry attributed to registered ICU stays:
  - **IV Fluid Level:** $\le 200\text{ mL}$ (Critical), $\le 350\text{ mL}$ (Moderate).
  - **SpO2:** $< 88\%$ (Critical Hypoxemia), $< 92\%$ (High Hypoxemia).
  - **Heart Rate:** $< 40$ or $> 140\text{ BPM}$ (Critical), $< 50$ or $> 120\text{ BPM}$ (Moderate).
- **Sensor Quality Masking:** Hardware disconnections ($0\text{ BPM}$ or $0\%\text{ SpO2}$) are masked to `None` and do NOT trigger false physiological bradycardia or hypoxemia alerts. Unassigned telemetry is never attributed to patients.

### Machine Learning Prediction Engine (Retrospective Research Prototype)
- Trained models are serialized and deployed in `ml/models/` (`deterioration_logistic_v1.joblib` and `anomaly_isolation_forest_v2.joblib`).
- **Prediction Design:** Retrospective rolling 24-hour prediction model. At each observation timestamp $t$, it uses features available at or before $t$ to estimate whether ICU discharge marked expired will occur within the next 24 hours. The 24-hour prediction horizon is strictly distinguished from the feature-observation period (accumulated history $\le t$).
- **Target Definition:** Strictly `"ICU discharge marked expired within the next 24 hours from the prediction timestamp"` (derived from eICU `unitdischargestatus == "Expired"` and `0 <= hours_to_discharge <= 24`). General in-hospital mortality is not assumed.
- **Model Architecture & Calibration:** Unweighted Logistic Regression operating at natural training prevalence (~1.5-1.6%).
  - Test Calibration Intercept: `-0.4605` (ideal 0.0), Calibration Slope: `0.8360` (ideal 1.0).
  - Test Model Brier Score: `0.015443` vs Prevalence-Only Baseline Brier Score: `0.016369`. The low Brier score is driven by the 98.34% negative class prevalence and does not alone prove good calibration.
  - Test Discrimination: ROC-AUC: `0.6875` (95% Clustered CI: `[0.5699, 0.8021]`), PR-AUC: `0.1257` (95% Clustered CI: `[0.0531, 0.2258]`).
  - Validation-Tuned Operating Threshold: $\theta^* = 0.0823$.
  - Hourly Observations Confusion Matrix: $\text{TN}=20,803, \text{FP}=286, \text{FN}=296, \text{TP}=61$ (Specificity: $98.64\%$, Recall: $17.09\%$, Precision: $17.58\%$).
- **False-Positive Clarification:** The 286 false-positive hourly observations occur across 69 unique ICU stays, representing mathematical threshold crossings on isolated hourly rows, not 286 independent clinical alarms or pager alerts.
- **Patient/Stay-Level Event Detection:**
  - Unique positive ICU stays in test set: **18**
  - Unique positive ICU stays flagged at least once within 24 hours prior to ICU discharge marked expired: **13**
  - Patient/stay-level sensitivity: **72.22%** (13 / 18)
  - Detection lead-time distribution: Median **9.00 hours**, Mean **10.15 hours** (min 0.0h, max 23.0h, IQR: 2.0h - 19.0h) before final observation.
  - Positive ICU stays never flagged: **5** (Stays: 176730, 827085, 1073098, 1109136, 2901384).
- **Experimental Risk Bands:** Low ($p < 0.05$), Moderate ($0.05 \le p < 0.15$), High ($0.15 \le p < 0.30$), Extreme ($p \ge 0.30$). These are presentation categories, not validated clinical severity classifications.
- **Explainability:** Exact signed log-odds attribution ($\text{logit}(p) = \beta_0 + \sum \phi_j$) exposing `model_intercept` and `predicted_log_odds` down to the API.
- **Offline Isolation:** Historical eICU models do not ingest live serial telemetry, preserving deterministic clinical rule safety.

### Model Serialization & Runtime Compatibility Matrix

For complete technical specifications, see [docs/MODEL_COMPATIBILITY.md](docs/MODEL_COMPATIBILITY.md).

| Artifact | Artifact Path | Training `scikit-learn` | Model Type | Deployment Status | Dedicated Env (`backend\.venv`, 1.9.0) | System Python (`1.8.0`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Deterioration v1** | `ml/models/deterioration_logistic_v1.joblib` | **`1.9.0`** | `LogisticRegression` Pipeline | **ACTIVE DEFAULT** | **SUPPORTED** (0 warnings) | **WARNING** (`InconsistentVersionWarning` $\times 4$; numerically equivalent) |
| **Anomaly Isolation Forest v2** | `ml/models/anomaly_isolation_forest_v2.joblib` | **`1.9.0`** | `IsolationForest` Pipeline | **ACTIVE DEFAULT** | **SUPPORTED** (0 warnings) | **WARNING** (`InconsistentVersionWarning` $\times 304$; bit-for-bit identical) |
| **Candidate Deterioration v2** | `ml/candidate_artifacts_v2/deterioration_candidate_v2.joblib` | **`1.8.0`** | `HistGradientBoosting` + Platt | **UNPROMOTED CANDIDATE** | **UNSUPPORTED** (Fails: `ModuleNotFoundError: No module named '_loss'`) | **SUPPORTED** (0 warnings; 13/13 candidate tests pass) |

#### Key Technical Context:
- **Dedicated Virtual Environment:** The backend environment (`D:\Smart-ICU\backend\.venv`) uses `scikit-learn==1.9.0`. Both baseline production models load cleanly without warnings.
- **System Environment:** Unactivated system Python uses `scikit-learn 1.8.0`. Loading baseline models in 1.8.0 produces native `InconsistentVersionWarning`s.
- **Empirical Numerical Equivalence:** Side-by-side empirical testing confirmed that on tested cross-version inputs, Isolation Forest scores and decision functions are identical bit-for-bit to 17 decimal places (variance $= 0.0$), and Logistic Regression v1 probabilities match within floating-point epsilon ($\le 3 \times 10^{-17}$).
- **Candidate v2 Deserialization Limitation:** Candidate v2 was serialized under 1.8.0 and cannot be deserialized under 1.9.0 due to internal Cython loss module restructuring (`ModuleNotFoundError: No module named '_loss'`). Candidate v2 must be tested in a 1.8.0 environment until formally retrained under 1.9.0 upon promotion.
- **Upstream Caveat:** Cross-version model persistence via pickle/joblib is officially unsupported by scikit-learn. Numerical equivalence on selected test inputs is not a guarantee of universal compatibility.
- **Warning Visibility:** `InconsistentVersionWarning` is intentionally not suppressed to preserve auditability when running in non-standard environments.

---

## 9. REST API Reference

### Patients
- `POST /api/patients` - Register a patient (`name`, `patient_code`, `age`, `sex`, `date_of_birth`, `contact_information`, `weight_kg`).
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
- `POST /api/clinical-records` - Record urine output, blood tests, or observations. Automatically evaluates interpretation and triggers alerts.
- `GET /api/patients/{patient_id}/clinical-records` - Retrieve clinical records for patient.
- `GET /api/stays/{stay_id}/clinical-records` - Retrieve clinical records for stay.
- `GET /api/clinical-records/{id}` - Retrieve record.
- `PUT /api/clinical-records/{id}` - Correct record with version check; supersedes obsolete alerts and re-evaluates.
- `DELETE /api/clinical-records/{id}` - Reasoned void of record; supersedes associated active alerts.
- `POST /api/clinical-records/evaluate` - Live interpretation preview before saving.

### Unified Alerts & ML Status
- `GET /api/alerts` - List clinical alerts with optional `status`, `severity`, `source`, `patient_id`, `stay_id` filters.
- `GET /api/alerts/summary` - Statistical breakdown (active, critical, acknowledged, resolved).
- `GET /api/alerts/{alert_id}` - Retrieve alert record.
- `GET /api/alerts/{alert_id}/history` - Retrieve audit log history for alert.
- `POST /api/alerts/{alert_id}/acknowledge` - Clinician acknowledgment (`clinician_name`).
- `POST /api/alerts/{alert_id}/resolve` - Clinician resolution (`clinician_name`, `resolution_note`).
- `GET /api/patients/{patient_ref}/alerts` - Patient-filtered alerts.
- `GET /api/stays/{stay_ref}/alerts` - Stay-filtered alerts.
- `GET /api/ml/status` - Inspection of ML artifacts, loaded state, and diagnostic notes.

---

## 10. Automated Testing & Verification

The codebase includes comprehensive unit, integration, attribution safety, lifecycle, and UI component tests:

### Run Backend Tests (225 tests)
```powershell
cd D:\Smart-ICU
.\backend\.venv\Scripts\Activate.ps1
pytest backend/tests -v -o pythonpath="backend ."
```
Or directly:
```powershell
& "D:\Smart-ICU\backend\.venv\Scripts\python.exe" -m pytest backend/tests -v -o pythonpath="backend ."
```

> [!CAUTION]
> **Do NOT run backend tests using unactivated system Python.**
> The backend test suite relies on `scikit-learn==1.9.0` pinned in `backend/.venv`. Running under system Python (1.8.0) triggers `InconsistentVersionWarning` on baseline models and causes version mismatches.

- **225 automated backend tests** covering:
  - `test_clinical_alert_engine.py`: KDIGO urine output, canonical lab registry, vitals rules, custom reference range parsing, duration parsing.
  - `test_alert_manager.py`: Alert lifecycle, deduplication fingerprinting, escalation, auto-resolution on normal reading, manual acknowledge/resolve, record revision/voiding synchronization.
  - `test_data_consistency_engine.py` & `test_consistency_api.py`: Multi-source clinical data consistency, ESP32 vs manual entry verification, tolerance checking, unit conversion.
  - `test_historical_reference_engine.py` & `test_patient_trend_analyzer.py`: Historical eICU population baseline comparisons, moving patient trend analysis, rate-of-change boundaries, and Isolation Forest anomaly scoring.
  - `test_ml_engine.py`: Serialized model inference, probability bounds, mathematical log-odds attribution, anomaly scoring, REST prediction endpoint.
  - `test_ml_safety.py`: ML status reporting, missing artifact diagnostics, deterrence of synthetic score fabrication.
  - `test_telemetry_alerts.py`: Bedside IV fluid and vital alerts, quality masking, patient attribution isolation.
  - `test_telemetry_timestamp_validation.py`: P1 timestamp validation, UTC normalization, rejection of unparseable/future timestamps.
  - Plus all regression tests for patient lifecycle, admissions, device assignment, concurrency, and stress bursts.
  - *Candidate v2 Evaluation Suite (1.8.0):* Run via dedicated candidate evaluation environment:
    ```powershell
    .\ml\.venv_candidate\Scripts\Activate.ps1
    pytest backend/tests/test_ml_candidate_inference.py -v -o pythonpath="backend ."
    ```
    *(All 13 candidate inference tests pass with 100% success)*


### Run Frontend Tests & Checks (46 tests)
```powershell
cd D:\Smart-ICU\frontend
npm test -- --run
npm run build
```
- **46 automated tests** covering:
  - `AlertPanel.test.tsx`: Stat cards, status/severity/source filtering, 7-question diagnostic inspection, clinician acknowledgment, clinician resolution.
  - `ClinicalInterpretation.test.tsx`: Live interpretation preview while typing, evaluated interpretation banner with evidence upon submission.
  - `ClinicalRecords.test.tsx`: Manual record entry, unit normalization, optimistic concurrency locking, voiding.
  - `App.test.tsx`: Command center, eICU triage, hardware telemetry banner, ML status banner, patient ML explainability card.
  - `PatientManagement.test.tsx` & `DeviceManagement.test.tsx`: Lifecycle and assignment operations.
- **TypeScript & Vite Build:** 100% clean production bundle (`tsc -b && vite build` in ~180ms).

---

## 11. Academic Limitations & Boundary Notice

1. **Authentication:** All endpoints, including clinical-record routes, are unauthenticated; caller-supplied `entered_by` values are not verified identities. Use only on localhost for synthetic demonstrations. Deployment on public or shared networks is prohibited until proper authentication, authorization, and audit-identity verification are implemented.
2. **Clinical Reference Ranges:** Displayed flags and validation bounds are technical sanity checks, not diagnostic algorithms.
3. **No Automatic Clinical Diagnosis:** The platform does not prescribe medication, calculate dynamic drug dosages, or diagnose disease.
4. **Hardware Identity:** ESP32 firmware lacks cryptographically signed hardware identity; the system uses backend-managed device identifiers.
5. **Model Artifact Provenance:** The five current model artifacts were hashed and their file sizes and SHA-256 checksums recorded. These values match the current release manifest. Historical identity with an earlier trained baseline could not be independently established because the artifacts are not present in the available Git history.
