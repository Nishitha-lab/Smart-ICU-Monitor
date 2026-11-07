# Architecture Specification: Smart ICU Patient Safety System

## 1. Architectural Overview

Smart ICU implements a multi-tier, patient-centred clinical monitoring architecture designed to prevent patient data misattribution while integrating IoT sensor telemetry, structured clinical observations, and retrospective machine-learning risk intelligence.

```
+-----------------------------------------------------------------------------------+
|                                 USER INTERFACE                                    |
|                         React 19 + TypeScript + Vite                              |
|                                                                                   |
|  [ ICU Risk Dashboard ]  [ Patient Management ]  [ Device Management ]  [ Clinical Records ]
+----------------------------------------+------------------------------------------+
                                         | HTTP / REST (Localhost)
                                         v
+-----------------------------------------------------------------------------------+
|                                FASTAPI BACKEND                                    |
|                                                                                   |
|  +--------------------+  +----------------------+  +---------------------------+  |
|  |   PatientStore     |  |   ClinicalRecords    |  |    SerialMonitorService   |  |
|  |  (Patients, Stays, |  |  (Observations, Labs,|  |   (ESP32 Serial Parser &  |  |
|  |   Assignments,     |  |   Revisions, Voids)  |  |    Ingestion Hook)        |  |
|  |   Telemetry)       |  +----------------------+  +---------------------------+  |
|  +--------------------+             |                                             |
|            |                        |                                             |
|            +-----------+------------+                                             |
|                        v                                                          |
|          SQLite Database (WAL Mode)                                               |
|          data/clinical_records.sqlite3                                            |
|          Foreign Keys ON, Partial Unique Indexes                                  |
+-----------------------------------------------------------------------------------+
        ^                                                    ^
        | loads once at startup                              | reads COM9 / 115200 baud
+----------------------------------+            +-----------------------------------+
|  Historical eICU Alerts Dataset  |            |        Physical Bedside ESP32     |
|  smart_icu_final_alerts.csv      |            |  - HX711 Load Cell (GPIO 4/5)     |
|  (Retrospective 24h Risk ML)     |            |  - Pulse Oximeter I2C (GPIO 21/22)|
+----------------------------------+            +-----------------------------------+
```

---

## 2. Core Entities & Relational Schema

All patient, admission, device, assignment, telemetry, and clinical record data are stored in a unified SQLite database (`data/clinical_records.sqlite3`) configured with Write-Ahead Logging (`PRAGMA journal_mode = WAL;`) and enforced referential integrity (`PRAGMA foreign_keys = ON;`).

### 1. `patients`
Represents an individual admitted to the hospital.
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `patient_code` (TEXT UNIQUE NOT NULL) - Human-readable unique code (e.g. `P001`)
- `name` (TEXT NOT NULL) - Patient full legal or display name
- `age` (INTEGER NULL) - Patient age (bounded 0..130)
- `sex` (TEXT NULL) - Patient sex (`male`, `female`, `other`, `unknown`)
- `date_of_birth` (TEXT NULL) - ISO 8601 date
- `contact_information` (TEXT NULL) - Emergency contact / phone
- `status` (TEXT NOT NULL DEFAULT 'active') - `'active'` or `'archived'`
- `created_at` (TEXT NOT NULL), `updated_at` (TEXT NOT NULL)

### 2. `icu_stays`
Represents an ICU admission episode. A patient may have multiple historical stays, but at most **one** active stay at any given time.
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `stay_code` (TEXT UNIQUE NOT NULL) - Human-readable stay identifier (e.g. `S001`)
- `patient_id` (INTEGER NOT NULL, FK -> `patients(id)`)
- `admission_datetime` (TEXT NOT NULL ISO UTC)
- `discharge_datetime` (TEXT NULL ISO UTC)
- `bed_number` (TEXT NULL)
- `ward_or_unit` (TEXT NULL)
- `admission_reason` (TEXT NULL)
- `discharge_status` (TEXT NOT NULL DEFAULT 'active') - `'active'`, `'discharged'`, `'transferred'`
- `created_at` (TEXT NOT NULL), `updated_at` (TEXT NOT NULL)
- **Constraint:** Partial unique index:
  ```sql
  CREATE UNIQUE INDEX IF NOT EXISTS uq_active_patient_stay
  ON icu_stays(patient_id) WHERE discharge_status = 'active';
  ```

### 3. `devices`
Represents a bedside telemetry device registered in the hospital system.
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `device_id` (TEXT UNIQUE NOT NULL) - Stable device identifier (e.g. `ESP32-001`)
- `device_name` (TEXT NOT NULL) - Display name (e.g. `Bed 4 Monitor`)
- `connection_type` (TEXT NOT NULL DEFAULT 'serial')
- `serial_port` (TEXT NULL) - Configured hardware COM port (e.g. `COM9`)
- `status` (TEXT NOT NULL DEFAULT 'disconnected')
- `created_at` (TEXT NOT NULL), `updated_at` (TEXT NOT NULL)

### 4. `device_assignments`
Maintains an immutable historical log of device assignments to patients and active ICU admissions.
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `device_id` (TEXT NOT NULL, FK -> `devices(device_id)`)
- `patient_id` (INTEGER NOT NULL, FK -> `patients(id)`)
- `stay_id` (INTEGER NOT NULL, FK -> `icu_stays(id)`)
- `assigned_at` (TEXT NOT NULL ISO UTC)
- `unassigned_at` (TEXT NULL ISO UTC)
- `assignment_status` (TEXT NOT NULL DEFAULT 'active') - `'active'`, `'unassigned'`, `'reassigned'`
- `created_at` (TEXT NOT NULL), `updated_at` (TEXT NOT NULL)
- **Constraint:** Partial unique index guaranteeing no device is ever assigned to multiple patients simultaneously:
  ```sql
  CREATE UNIQUE INDEX IF NOT EXISTS uq_active_device_assignment
  ON device_assignments(device_id) WHERE assignment_status = 'active';
  ```

### 5. `telemetry_records`
Chronological telemetry readings ingested from bedside devices or hardware simulators.
- `id` (INTEGER PRIMARY KEY AUTOINCREMENT)
- `device_id` (TEXT NOT NULL, FK -> `devices(device_id)`)
- `patient_id` (INTEGER NULL, FK -> `patients(id)`)
- `stay_id` (INTEGER NULL, FK -> `icu_stays(id)`)
- `reading_timestamp` (TEXT NOT NULL ISO UTC)
- `ingestion_timestamp` (TEXT NOT NULL ISO UTC)
- `fluid_ml` (REAL NULL)
- `heart_rate_bpm` (REAL NULL)
- `spo2_percent` (REAL NULL)
- `device_status` (TEXT NULL)
- `raw_message` (TEXT NULL)

### 6. `clinical_records` & `clinical_record_revisions`
Versioned, auditable clinical measurements (urine output, lab tests, nurse observations).
- **Strict Separation of Identifiers:** Registered ICU records link via foreign keys to `patients(id)` (`patient_id`) and `icu_stays(id)` (`stay_id`), leaving `patientunitstayid` `NULL`. Historical eICU records populate `patientunitstayid` and leave `patient_id` / `stay_id` `NULL`. The system never substitutes `stay_id` into `patientunitstayid`.
- Referential integrity enforced via SQLite `PRAGMA foreign_keys = ON;` with `ON DELETE RESTRICT`.
- Features optimistic concurrency locking via an incrementing `version` column.
- Corrections preserve immutable previous versions in `clinical_record_revisions`.
- Voids require an explicit clinical reason and operator identifier (never hard deleted).

---

## 3. Telemetry Attribution & Safety Architecture

In ICU settings, attributing a sensor measurement to the wrong patient is a severe clinical hazard. The Smart ICU telemetry ingestion pipeline implements rigorous attribution rules:

```
Incoming Serial Packet
("Fluid: 450.00 mL | BPM: 74 | SpO2: 98% | STATUS: NORMAL")
         |
         v
Resolve device_id from Serial Monitor ("ESP32-001")
         |
         v
Query device_assignments WHERE device_id = 'ESP32-001' AND assignment_status = 'active'
         |
         +-----------------------------+-----------------------------+
         | Active assignment FOUND     | No active assignment (NULL) |
         v                             v                             |
Verify Patient & Stay:                 Record Telemetry:             |
- Patient is 'active'                  - patient_id = NULL           |
- Stay is 'active' & belongs to patient- stay_id = NULL              |
- reading_timestamp >= assigned_at     - device_id = 'ESP32-001'     |
         |                                                           |
         | YES                         Flag Frontend:                |
         v                             "⚠️ Device Not Assigned"      |
Record Telemetry:                      NEVER attach to any patient.  |
- patient_id = assigned.patient_id     NEVER display in chart/alerts.|
- stay_id = assigned.stay_id                                         |
- device_id = 'ESP32-001'                                            |
         |                                                           |
Attributed to Patient!                                               |
```

### Safety Principles
1. **Frontend Independence:** Telemetry attribution is completely decoupled from the frontend UI state. Whether a clinician is looking at Patient A, Patient B, or the Dashboard, incoming telemetry from `ESP32-001` is attributed strictly according to the active assignment in SQLite.
2. **Delayed Reading Quarantine:** Readings with timestamps prior to `assigned_at` are treated as belonging to a prior period and are rejected from patient attribution.
3. **Reassignment Atomicity:** Reassigning a device automatically terminates the previous assignment (`unassigned_at = now()`, `status = 'reassigned'`) and establishes the new active assignment in a single transaction.
4. **Audit Trail Preservation:** Historical telemetry records are immutable. De-escalating or discharging a patient preserves all historical sensor data and assignment history for clinical retrospective audits.

---

## 4. Hardware Integration & IoT Limitations

### ESP32 Bedside Hardware
- **HX711 Load Cell Interface:** Digital amplifier connected via GPIO 4 (DT) and GPIO 5 (SCK) measuring hanging IV fluid bag mass.
- **MAX30100/MAX30102:** Pulse oximeter communicating over I2C on GPIO 21 (SDA) and GPIO 22 (SCL).
- **Local Audible Warning:** Piezo buzzer on GPIO 18 triggers on critical telemetry thresholds.
- **Serial Protocol:** 115200 baud, 8N1, line-delimited ASCII frames:
  `Fluid: <val> mL | BPM: <val> | SpO2: <val>% | STATUS: <val>`

### Identity Abstraction
The physical ESP32 firmware (`hardware/esp32/TBP_final.ino`) streams formatted telemetry strings without an integrated cryptographic device certificate or unique hardware identifier. The backend bridges this limitation by maintaining registered device entities and mapping physical COM ports to stable logical device identifiers (`SMART_ICU_DEVICE_ID`, default: `ESP32-001`).
