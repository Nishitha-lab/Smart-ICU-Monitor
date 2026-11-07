# Demonstration Script: Smart ICU Patient Safety System (5-10 Minutes)

This script provides a structured demonstration of the Smart ICU system, highlighting the patient-centred architecture, ICU admissions, strict ESP32 patient assignment, telemetry attribution safety, and clinical record documentation.

---

## 1. Preparation Before the Demo (2 minutes)

1. Open PowerShell Terminal 1 and start the backend:
   ```powershell
   cd D:\Smart-ICU\backend
   .\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```
2. Open PowerShell Terminal 2 and start the frontend:
   ```powershell
   cd D:\Smart-ICU\frontend
   npm run dev
   ```
3. Open `http://localhost:5173` in a web browser.
4. Verify backend health at `http://127.0.0.1:8000/health`.

---

## 2. Demonstration Flow

### Step 1: Patient Registration & Management (2 minutes)
1. In the top navigation bar, click **Patients**.
2. Explain the patient-centred model:
   - "Previously, records and sensors had no unified patient registry. Now, every patient has a verified demographic identity and unique patient code."
3. Click **+ Register Patient**:
   - Full Name: `Alice Walker`
   - Patient Code: `P001`
   - Age: `58`, Sex: `Female`
   - Click **Register Patient**.
4. In the list, register a second patient:
   - Full Name: `Bob Smith`
   - Patient Code: `P002`
   - Age: `62`, Sex: `Male`
   - Click **Register Patient**.
5. Show the searchable, filterable registered patient table.

### Step 2: ICU Admissions & Admission Integrity (2 minutes)
1. Click **View Details →** on `Alice Walker`.
2. Notice her admission state: *"No active ICU admission."*
3. Click **+ Admit to ICU**:
   - Bed Number: `ICU-Bed-04`
   - Ward / Unit: `Cardiothoracic ICU`
   - Reason: `Post-op coronary bypass monitoring`
   - Click **Admit Patient**.
4. Show that Stay `S001` is now active.
5. Point out the architectural constraint:
   - "A patient can have multiple historical stays, but the system enforces at most one active admission episode at a time."

### Step 3: Device Management & Unassigned Device Warning (2 minutes)
1. In the top navigation bar, click **Device Management**.
2. Notice the registered device `ESP32-001` (ICU Bed 4 Telemetry Unit).
3. Point out the prominent warning banner:
   - `⚠️ Device Not Assigned`
   - *"Readings from this device will be monitored but NOT attributed to any patient."*
4. Use the **Interactive Telemetry Simulator** on `ESP32-001` to inject a test reading:
   - Fluid: `450 mL`, BPM: `75`, SpO2: `98%`, Status: `NORMAL`.
   - Click **⚡ Send Synthetic Reading**.
5. Observe the feedback message:
   - `⚠️ Device is unassigned: Reading accepted by hardware log, but NOT attributed to any patient.`
6. Return to **Patients -> Alice Walker**:
   - Confirm her vitals remain empty (`—`). The unassigned packet was never leaked to her record!

### Step 4: Explicit Device Assignment & Real-Time Telemetry (2 minutes)
1. Return to **Device Management**.
2. Click **Assign to Patient** on `ESP32-001`.
3. Select `Alice Walker (P001) — Stay S001 (ICU-Bed-04)`.
4. Click **Confirm Assignment**.
5. Notice the status card updates immediately:
   - `● Active Patient Assignment: Alice Walker (P001) - Stay S001`
6. In the **Interactive Telemetry Simulator**, inject a new packet:
   - Fluid: `380 mL`, BPM: `82`, SpO2: `99%`, Status: `NORMAL`.
   - Click **⚡ Send Synthetic Reading**.
7. Feedback confirms:
   - `✓ Reading accepted and attributed to Patient #1 (Stay #101).`
8. Return to **Patients -> Alice Walker**:
   - Her live vital boxes now show: **380.0 mL Fluid**, **82 BPM**, **99% SpO2**, **NORMAL**.
9. Check **Bob Smith**:
   - His vitals remain empty. Strict backend assignment prevents any cross-patient contamination.

### Step 5: Clinical Documentation & Auditable Timeline (1 minute)
1. In Alice Walker's detail view, scroll down to **Clinical Records & Observations**.
2. Note that her active stay `S001` is preselected.
3. Record a Urine Output measurement:
   - Volume: `140 mL`, Period: `1 hour`, Entered by: `Nurse Johnson`.
   - Click **Save Record**.
4. Scroll to **Patient Chronological Timeline**:
   - View the unified auditable history:
     - 1. Patient Registration (`P001`)
     - 2. ICU Admission (`S001`)
     - 3. Device Assignment (`ESP32-001`)
     - 4. Clinical Record Entry (`140 mL Urine output`)

### Step 6: Discharge & Safe De-escalation (1 minute)
1. At the top of Alice Walker's active stay card, click **Discharge Stay**.
2. Confirm the discharge in the modal.
3. Observe:
   - Stay `S001` moves into the historical table with its discharge timestamp.
   - Device `ESP32-001` is automatically released and unassigned.
   - In **Device Management**, `ESP32-001` returns to the `⚠️ Device Not Assigned` state, ready for the next patient.
   - All historical data remains intact in Alice's permanent record.

---

## 3. Key Talking Points for Evaluators

1. **Safety Rule Enforcement:**
   The backend resolves patient attribution from the database assignment, never from whichever patient is open in the web browser.
2. **Database Integrity:**
   Partial unique indexes prevent overlapping active admissions per patient and overlapping active assignments per device.
3. **Audit Compliance:**
   Clinical records use optimistic locking, revision histories, and reasoned voids. Stays and device assignments maintain complete chronological audit histories.
4. **Hardware-Free Testing:**
   The built-in telemetry simulation tool enables testing and grading all IoT attribution workflows without requiring a physical microcontroller.
