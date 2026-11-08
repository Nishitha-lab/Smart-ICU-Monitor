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
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> [!IMPORTANT]
> **Virtual Environment Alignment:**
> The backend requires `scikit-learn==1.9.0` (pinned in `backend/requirements.txt`) to maintain binary serialization compatibility with baseline models. Always activate `.venv` or run via `.\.venv\Scripts\python.exe`. Do not use system Python. See [docs/MODEL_COMPATIBILITY.md](./docs/MODEL_COMPATIBILITY.md).

Place the approved, locally prepared final-alert CSV at:

```text
data\processed\smart_icu_final_alerts.csv
```

Start the API from the `backend` directory:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
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

Backend tests (225 tests):

```powershell
cd D:\Smart-ICU
.\backend\.venv\Scripts\Activate.ps1
pytest backend/tests -v -o pythonpath="backend ."
```

Or directly via the dedicated virtual environment binary:

```powershell
& "D:\Smart-ICU\backend\.venv\Scripts\python.exe" -m pytest backend/tests -v -o pythonpath="backend ."
```

> [!CAUTION]
> Do **NOT** execute tests using system Python. Always run within `.venv` where `scikit-learn==1.9.0` is pinned. Candidate v2 tests (`test_ml_candidate_inference.py`) require a 1.8.0 environment due to upstream loss module changes. See [docs/MODEL_COMPATIBILITY.md](./docs/MODEL_COMPATIBILITY.md).

Candidate v2 evaluation tests (dedicated 1.8.0 environment):

```powershell
cd D:\Smart-ICU
.\ml\.venv_candidate\Scripts\Activate.ps1
pytest backend/tests/test_ml_candidate_inference.py -v -o pythonpath="backend ."
```
*(All 13 candidate inference tests pass with 100% success)*


Experimental robustness tests:

```powershell
cd D:\Smart-ICU
.\backend\.venv\Scripts\Activate.ps1
pytest ml/experiments -k "test_" -v
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
