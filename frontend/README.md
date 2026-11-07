# Smart ICU frontend

The React/Vite dashboard displays historical patient-hour records and
precomputed research alerts from the Smart ICU backend. Live ESP32 telemetry,
when available, is presented separately from the historical eICU dataset.
See the [project README](../README.md) for data prerequisites, setup steps,
scope, and safety limitations.

From this directory:

```powershell
npm ci
npm run dev
npm test
npm run lint
npm run build
```

The backend must be running at `http://127.0.0.1:8000` and must have the local
processed dataset available. This research prototype is not validated for
patient-care use.
