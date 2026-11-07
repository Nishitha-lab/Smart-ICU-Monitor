# Architecture

## System overview

```text
Browser (React/Vite)
  | HTTP requests to local FastAPI
  v
FastAPI backend
  |-- loads data/processed/smart_icu_final_alerts.csv once at startup
  |-- serves summary, latest-per-stay queue, history, and latest state
  `-- reads a separate optional ESP32 serial monitor

Offline research/data pipeline
  raw eICU demo data -> hourly preprocessing/cleaning -> state/trend features
      -> clinical rules + anomaly/risk scores -> fused hourly alert CSV
      -> FastAPI reads the precomputed final CSV
```

## Application components

- **Frontend:** React and TypeScript in `frontend/src/`. It requests the
  summary, patient queue, selected-stay history, and hardware telemetry. The
  browser does not calculate the ML risk score.
- **Backend:** FastAPI in `backend/main.py`. It loads the local final-alert CSV
  at startup and provides read-only HTTP endpoints. The default local serial
  monitor reads separately configured ESP32 telemetry; it is not joined to an
  eICU stay.
- **Data:** Raw and processed datasets are local prerequisites and are
  ignored by Git. The API serves precomputed historical rows.
- **Hardware:** Optional serial input is parsed by
  `backend/services/serial_service.py`. Missing, invalid, disconnected, or
  stale readings are represented as unavailable.
- **Offline ML and rules:** Scripts under `ml/` prepare features, labels,
  rules, anomaly scores, risk scores, and fused alert rows. They are not
  invoked by API requests.
- **Experimental research:** SpO2 models/replays are isolated under the
  existing experimental paths and are not production inference artifacts.

## Data pipeline

1. `ml/preprocessing.py` reads local eICU patient, periodic vital, aperiodic
   blood-pressure, and laboratory files and creates hourly tables.
2. `ml/cleaning.py` applies the current production cleaning rules.
3. Patient state, fluid, infusion, and trend scripts produce the processed
   feature tables.
4. `ml/clinical_rules.py` builds base rule flags; `ml/add_hypertension_rule.py`
   creates the v2 rule file.
5. The deployment scoring scripts fit an IsolationForest and logistic
   regression from local training tables and write scores.
6. `ml/build_deployment_master.py` joins state, v2 rules, anomaly scores, and
   risk predictions. `ml/build_final_smart_icu_alerts.py` assigns priority,
   explanations, and categories.
7. FastAPI loads the final CSV once and serves the saved values.

Reproduction requires the correct licensed raw eICU inputs and the local
processed training/output tables expected by each script. Outputs are not
stored in Git.

## Trust and operating boundaries

- Historical eICU rows and ESP32 telemetry are separate streams.
- `risk_probability` is an offline score for expired ICU discharge within
  0-24 hours of the prediction point. It is not a general deterioration
  probability.
- The app has no model inference route or notification delivery service.
- The offline notification cooldown output is not consumed by the API.
- The application is a research/demo prototype, not a medical device.

See [API_DOCUMENTATION.md](./API_DOCUMENTATION.md),
[KNOWN_LIMITATIONS.md](./KNOWN_LIMITATIONS.md), and
[ML_ALERT_FINAL_AUDIT.md](./ML_ALERT_FINAL_AUDIT.md).
