# End-to-End Verification

## Environment and scope

Verification used the local processed eICU demo dataset and the actual FastAPI
and Vite applications on Windows. The backend loaded
`data/processed/smart_icu_final_alerts.csv`; this file is local and is not
committed. No synthetic clinical records were added. ESP32 hardware was not
connected.

## Workflow results

| Workflow | Result | Evidence / limitation |
|---|---|---|
| Backend startup and dataset load | PASS | Uvicorn started; 139,432 patient-hour rows across 2,419 ICU stays loaded. |
| Health | PASS | `GET /health` returned healthy, dataset loaded, and the disconnected COM9 hardware state. |
| Summary | PASS | `GET /api/summary` returned 139,432 rows, 2,419 stays, and alert-signal counts. |
| Patient listing | PASS | `GET /api/patients?limit=1` returned the latest row with priority, risk, clinical rule explanation, and alert explanation. |
| Priority filtering | PASS | Earlier independent CSV/API verification found the 110 critical latest-row stays matched exactly; invalid priority returned HTTP 400. |
| Patient selection and history | PASS | The browser selected a row from the filtered critical queue, opened its patient detail, and loaded chronological history from the API. |
| Vital-sign visualization and historical trends | PASS WITH LIMITATION | Historical vital fields and the trend charts rendered in the browser. Data are retrospective hourly eICU records, not live patient readings. |
| Alert history and explanations | PASS | Alert details render with the source clinical explanation and precomputed final explanation. Full-row independent reconstruction of saved priorities and explanations had zero mismatches across 139,432 rows. |
| IV telemetry unavailable | PASS | Hardware telemetry returned `connected: false`, null measurements, and field validity false when COM9 was unavailable; the UI presents unavailable rather than normal/zero values. |
| Hardware disconnected state | PASS | The API reports `DISCONNECTED`; no physical ESP32 was tested. |
| Invalid patient ID | PASS | Unknown stay returned HTTP 404. |
| Invalid filter | PASS | Unknown priority returned HTTP 400 with accepted values. |
| Frontend startup and rendering | PASS | Vite 8.3.6 served the React application and the dashboard consumed the actual local API. |
| Responsive layout | PASS | At 320, 390, 768, 1024, and 1440 px, the rendered page had no horizontal overflow. |
| Frontend/backend error handling | PASS WITH SCOPE LIMITATION | Invalid API requests returned expected HTTP 4xx responses and the frontend API-failure state passed component tests. A forced live upstream-outage browser scenario was not separately exercised in this run. |

## Startup and access

From the repository root, use separate PowerShell terminals:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

```powershell
cd frontend
npm ci
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). The backend health
endpoint is `http://127.0.0.1:8000/health`.

## Limitations

- The API serves a precomputed CSV; requests do not execute model inference.
- Historical eICU rows must not be presented as live telemetry.
- Hardware was disconnected; physical serial connectivity and calibration were
  not verified.
- The risk model target is expired ICU discharge within 24 hours, not general
  clinical deterioration.
- This workflow check is software verification on a demo dataset, not clinical
  validation, clinical safety evidence, or production-readiness evidence.
