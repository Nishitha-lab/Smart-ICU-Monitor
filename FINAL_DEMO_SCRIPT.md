# Final Demonstration Script (5-10 minutes)

## Before the demonstration

1. Obtain the eICU demo data through its applicable access and licensing
   process. Do not use fabricated clinical measurements.
2. Confirm that
   `data/processed/smart_icu_final_alerts.csv` exists locally.
3. Start the backend and frontend in separate PowerShell terminals:

   ```powershell
   cd backend
   .\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```

   ```powershell
   cd frontend
   npm ci
   npm run dev
   ```

4. Open the Vite URL and confirm `/health` reports the dataset loaded. ESP32
   hardware is optional; when disconnected, the interface should display
   unavailable telemetry.

## Walkthrough

1. **Dashboard overview (1 minute):** Point out that the application displays
   historical hourly eICU demo records and precomputed outputs, not live
   patient monitoring.
2. **Patient queue and priority filter (1 minute):** Show the queue and select
   the critical filter. Explain that the list uses each stay's latest saved
   row; the priority is an experimental combination of rules, risk score, and
   anomaly output, not a clinical triage recommendation.
3. **Patient detail (1 minute):** Select an actual row from the local dataset.
   Identify the displayed historical hour and vital signs without presenting
   them as current patient measurements.
4. **Trend charts (1-2 minutes):** Show the historical heart rate, SpO2, MAP,
   respiration, fluid balance, and urine-output charts. Explain that charts
   are limited by the observations present in the retrospective extract.
5. **Alert explanation (1-2 minutes):** Read the recorded clinical-rule
   explanation and distinguish it from the anomaly score and model risk
   probability. State the target precisely: expired ICU discharge within the
   next 24 hours, not general deterioration.
6. **Priority rationale (1 minute):** Describe the fixed risk cutoffs and
   combination logic at a high level. Do not suggest that the thresholds are
   clinically validated.
7. **IV/hardware state (1 minute):** Show the disconnected/unavailable ESP32
   state. Historical fluid context is a separate dataset field; it is not a
   sensor reading, remaining bag volume, or live flow rate.
8. **Research limitations (1 minute):** Explain that the SpO2 experiments are
   isolated and research-only, the experimental alert replay omitted the
   anomaly signal, and neither the model nor alert behavior is validated for
   patient-care use.

## Do not claim

- Clinical validation, patient-care safety, or production readiness.
- General deterioration prediction.
- That the dashboard is a live patient monitor.
- That experimental models are deployed.
- That the offline six-hour notification simulation is an active notification
  system.
- That hardware has been physically tested or calibrated.
