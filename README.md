# Smart ICU Patient Safety System

Smart ICU is a research prototype that displays precomputed hourly analyses
from the eICU Collaborative Research Database demo and, when available, a
separate ESP32 serial telemetry feed. It is not a medical device and is not
validated for patient-care use.

## Data and model scope

The backend reads `data/processed/smart_icu_final_alerts.csv` at startup. The
processed file and raw eICU data are local prerequisites and are intentionally
not tracked in Git. Obtain and prepare data only through the appropriate
research access and licensing process; do not commit patient-level exports or
raw datasets.

The dashboard serves precomputed CSV values; it does not run model inference
for each API request. The retrospective model target is expired ICU discharge
within the next 24 hours, not general clinical deterioration. Experimental
SpO2 and patient-level cross-validation work is research-only and does not
change production preprocessing, alerts, or inference. These results do not
establish clinical utility, safety, or production readiness.

Live ESP32 telemetry is separate from historical eICU records. Missing,
invalid, disconnected, or stale sensor readings are unavailable; they must not
be interpreted as normal or as a zero measurement. The current telemetry
protocol does not provide calibrated remaining bag volume or flow rate.

## Run locally on Windows

Prerequisites: Python 3.11 or newer, Node.js/npm supported by the installed
Vite version, and the local processed CSV described above.

1. In a terminal, create and install the backend environment:

   ```powershell
   cd backend
   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Start the API from the `backend` directory:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```

   The documented demo binds to localhost only. Do not bind this prototype to
   a public or shared network.

3. In a second terminal, install and start the frontend:

   ```powershell
   cd frontend
   npm ci
   npm run dev
   ```

4. Open the Vite URL printed in the frontend terminal. The API is available at
   `http://127.0.0.1:8000`; `/health` reports dataset and hardware status.
   The frontend currently targets this exact API base URL
   (`frontend/src/api.ts`); Vite has no proxy or API URL override. Start the
   backend from this checkout on port 8000 so the browser uses the same build.

If port 8000 appears to serve an older backend, inspect its listener and
OpenAPI routes before stopping anything. These PowerShell commands identify
the process owner, executable, command line, and whether the clinical routes
are present:

```powershell
$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
  Select-Object -First 1
if ($listener) {
  $ownerPid = $listener.OwningProcess
  Get-CimInstance Win32_Process -Filter "ProcessId = $ownerPid" |
    Format-List ProcessId, ExecutablePath, CommandLine
  $openApi = Invoke-RestMethod http://127.0.0.1:8000/openapi.json
  $openApi.paths.PSObject.Properties.Name -contains "/api/clinical-records/patient-stays"
} else {
  Write-Output "No process is listening on port 8000."
}
```

Only if you have confirmed that the listener is an obsolete Smart ICU
`uvicorn main:app` process from this repository and its OpenAPI lacks the
clinical-record route, replace the example PID below with the inspected PID.
The guard refuses to stop a process unless its executable or command line
belongs to this checkout and its command line runs this app:

```powershell
$pidToStop = 12345
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidToStop"
$openApi = Invoke-RestMethod http://127.0.0.1:8000/openapi.json
$hasClinicalRoute = $openApi.paths.PSObject.Properties.Name -contains "/api/clinical-records/patient-stays"
$isThisBackend = ($process.ExecutablePath -like "D:\Smart-ICU\backend\.venv\Scripts\python.exe") -or
  ($process.CommandLine -like "*D:\Smart-ICU\backend*")
if ($process -and $isThisBackend -and $process.CommandLine -match "uvicorn\s+main:app" -and -not $hasClinicalRoute) {
  Stop-Process -Id $pidToStop
} else {
  throw "Process was not stopped. Confirm its identity and routes manually."
}
```

Then start the current checkout using the documented backend command above.
Do not stop a process solely because it owns port 8000.

The ESP32 feed is optional. The backend defaults to `COM9` at 115200 baud;
configure `SMART_ICU_SERIAL_PORT` and `SMART_ICU_BAUDRATE` before starting the
backend when using another port or baud rate. Close any serial monitor that
has the port open.

## Clinical Observations & Lab Results

The application includes a separate prototype module for manually entered
clinical observations and laboratory results. It supports:

- Urine output volumes (mL), measurement time, optional observation period,
  and notes.
- Hemoglobin, creatinine, BUN, sodium, potassium, and blood glucose, with
  their displayed fixed units.
- Other named laboratory tests with a supplied unit, optional reference
  range, sample collection time, result time, and notes.
- Body temperature, respiratory rate, systolic/diastolic blood pressure,
  consciousness/general observations, and free-text clinical notes.

Select an ICU stay using its ID from the historical dataset. The API checks
that the stay exists; manual records remain separate from historical eICU
rows, live ESP32 readings, and the ML/alert pipeline. Every saved record gets
a unique ID, server-generated entry timestamp, and `MANUAL_ENTRY` source.
The “entered by” value is an operator identifier/display name only; the
application has no authentication and does not verify the operator or the
entered measurement. Units are validated as provided and are never silently
converted. Measurement, sample collection, laboratory result, and entry times
are shown separately where applicable, with displayed times labelled as local
time.

Records are stored locally in SQLite at
`data/clinical_records.sqlite3` (override with
`SMART_ICU_CLINICAL_DB_PATH`; relative override paths resolve from the
repository root). The database schema is migrated additively, preserving
existing records. This file is ignored by Git and is not included in version
control; back it up only through an approved secure process. Corrections
require the retrieved record version and a reason; stale concurrent
corrections receive HTTP 409 and do not overwrite newer records. Each
successful correction increments the version and preserves the previous
snapshot in the same transaction. The delete operation is a reasoned void, not
a hard delete. Repeated measurements at the same time are permitted and
receive different record IDs.

Clinical-record API:

- `GET /api/clinical-records/patient-stays?query=<stay-id-prefix>&limit=25`
- `POST /api/clinical-records`
- `GET /api/patients/{patientunitstayid}/clinical-records`
  (`category`, `start_time`, `end_time`, and `include_voided` filters)
- `GET /api/clinical-records/{record_id}`
- `PUT /api/clinical-records/{record_id}` (requires `expected_version` and
  correction reason)
- `GET /api/clinical-records/{record_id}/revisions`
- `DELETE /api/clinical-records/{record_id}` (requires `expected_version`,
  void reason, and operator identifier)

For records, the core fields are patient ICU stay ID, category, measurement
name, value, unit when applicable, timestamp(s), operator identifier, source,
and optional notes/context. Example request:

```json
{
  "patientunitstayid": 123456,
  "category": "urine_output",
  "measurement_name": "Urine output",
  "value": 125,
  "unit": "mL",
  "measurement_time": "2026-11-07T12:00:00Z",
  "entered_by": "Operator display name",
  "observation_period": "1 hour",
  "notes": null
}
```

This is a local prototype, not an authenticated or clinically validated
medical-record system. Do not enter real patient information unless the
environment is approved for it. It does not generate diagnoses, treatment
recommendations, or threshold-based alerts, and it does not calculate fluid
balance from urine output alone. Future integration would require approved
identity/access controls, privacy/security safeguards, clinical data
governance, validated interoperability and clinical review.

All clinical-record endpoints are unauthenticated. Deployment on public or
shared networks is prohibited until proper authentication, authorization, and
verified audit identity are implemented. Numeric input limits are technical
data-integrity safeguards only; they are not clinical reference ranges or
interpretation rules.

## Validation

Run backend tests from `backend`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run frontend checks from `frontend`:

```powershell
npm test
npm run lint
npm run build
```

The SpO2 raw-data audit is a separate research task and requires the relevant
local eICU demo files:

If you previously changed into the `frontend` directory, return to the
repository root before running the audit:

```powershell
cd ..
python ml\audit_spo2_dataset_wide.py
```

## API

- `GET /health`
- `GET /api/hardware/telemetry`
- `GET /api/summary`
- `GET /api/patients`
- `GET /api/patients/{patientunitstayid}`

## Project documentation

- [Architecture](./ARCHITECTURE.md)
- [Setup guide](./SETUP_GUIDE.md)
- [API documentation](./API_DOCUMENTATION.md)
- [End-to-end verification](./END_TO_END_VERIFICATION.md)
- [ML and alert audit](./ML_ALERT_FINAL_AUDIT.md)
- [Dependency security review](./DEPENDENCY_SECURITY_REVIEW.md)
- [Final test report](./FINAL_TEST_REPORT.md)
- [Demonstration script](./FINAL_DEMO_SCRIPT.md)
- [Known limitations](./KNOWN_LIMITATIONS.md)
- [Final completion report](./FINAL_PROJECT_COMPLETION_REPORT.md)

## Deferred work

Clinical validation, general deterioration modeling, downstream ML retraining,
live inference, alert-system redesign, deployment, and hardware calibration
are separate milestones. Do not treat this prototype or its outputs as
clinical advice or production-ready software.
