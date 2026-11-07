# Setup Guide

## Requirements

- Windows PowerShell.
- Python 3.11 or newer.
- Node.js supported by the locked Vite toolchain.
- The locally prepared `data/processed/smart_icu_final_alerts.csv`.
- Raw eICU demo files only if rebuilding the local processed dataset. Obtain
  them through the appropriate access/licensing process.

The repository intentionally does not include raw or patient-level processed
data. The ESP32 device is optional and is not required to run the software
demo.

## Backend environment

From the repository root:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the approved, locally prepared final-alert CSV at:

```text
data\processed\smart_icu_final_alerts.csv
```

Start the API from the `backend` directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The backend loads the CSV once at startup. Check
`http://127.0.0.1:8000/health` for dataset and hardware status.

## Frontend environment

Open a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open the URL Vite prints (normally `http://localhost:5173`). Keep the backend
running at `http://127.0.0.1:8000`.

## Optional serial telemetry

The backend defaults to `COM9` at 115200 baud. Before startup, configure
another port if needed:

```powershell
$env:SMART_ICU_SERIAL_PORT = "COM4"
$env:SMART_ICU_BAUDRATE = "115200"
```

Close other serial monitors before starting. Hardware absence does not prevent
the dashboard from loading; readings remain unavailable until valid, fresh
telemetry is received. Hardware calibration and physical operation have not
been verified.

## Checks

Backend tests:

```powershell
cd backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Experimental robustness tests:

```powershell
cd ..
.\backend\.venv\Scripts\python.exe -m unittest discover -s .\ml\experiments -p "test_*.py" -v
```

Frontend:

```powershell
cd frontend
npm test
npm run lint
npm run build
```

See [END_TO_END_VERIFICATION.md](./END_TO_END_VERIFICATION.md) for the latest
local verification results.
