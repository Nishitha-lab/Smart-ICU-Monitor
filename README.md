# Smart ICU integrated source package

This source package consolidates the supplied backend, frontend, ML scripts and ESP32 firmware. The cleaned datasets are intentionally not duplicated in this source ZIP because your original `processed.zip` is large. Extract that archive into `data/processed/` so `data/processed/smart_icu_final_alerts.csv` exists.

## Setup (Windows)
1. Extract this ZIP to `D:\Smart-ICU` (or another folder).
2. Extract your `processed.zip` contents into `data/processed/`. If the archive contains a top-level `processed` directory, copy its CSV files into `data/processed/` (not `data/processed/processed/`).
3. In `backend`: `python -m venv .venv`, activate it, then `pip install -r requirements.txt`.
4. Start API: `uvicorn main:app --reload --host 127.0.0.1 --port 8000`.
5. In `frontend`: `npm install` then `npm run dev`.
6. Close Arduino Serial Monitor before backend starts; COM9 can be opened by only one application at a time.

## Train and save models
From the project root, with backend venv active:
`python ml/train_anomaly_model_v2.py`
`python ml/train_deterioration_probability_model.py`
Artifacts are written to `ml/models/`.

## API
`GET /health`, `GET /api/hardware/telemetry`, `GET /api/summary`, `GET /api/patients`, `GET /api/patients/{patientunitstayid}`.

The historical eICU patient analytics and live ESP32 telemetry are intentionally kept separate. Live HX711/oximeter values are not silently treated as eICU clinical features. This is a research prototype, not a medical device; clinical validation is required before patient-care use.
