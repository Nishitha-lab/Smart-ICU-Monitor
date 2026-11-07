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

3. In a second terminal, install and start the frontend:

   ```powershell
   cd frontend
   npm ci
   npm run dev
   ```

4. Open the Vite URL printed in the frontend terminal. The API is available at
   `http://127.0.0.1:8000`; `/health` reports dataset and hardware status.

The ESP32 feed is optional. The backend defaults to `COM9` at 115200 baud;
configure `SMART_ICU_SERIAL_PORT` and `SMART_ICU_BAUDRATE` before starting the
backend when using another port or baud rate. Close any serial monitor that
has the port open.

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
