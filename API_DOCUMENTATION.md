# API Documentation

Base URL for the local development setup: `http://127.0.0.1:8000`.
All data routes are read-only and use the precomputed
`data/processed/smart_icu_final_alerts.csv`.

## `GET /`

Returns system name, online status, and API version.

## `GET /health`

Returns service status, whether the CSV dataset loaded, row count, and a
hardware telemetry snapshot. Hardware fields may be null when disconnected or
unavailable.

## `GET /api/hardware/telemetry`

Returns the latest serial telemetry snapshot separately from historical eICU
records. The response includes connection state, port, timestamp, last error,
measurement values, and per-field validity. A disconnected device does not
produce fabricated zero readings.

## `GET /api/summary`

Returns total patient-hour rows, unique ICU stays, priority counts, anomaly and
clinical counts, risk-threshold counts, and multi-signal alert count.

## `GET /api/patients`

Returns one latest hourly row per ICU stay, ordered by severity and risk.

| Query | Default | Constraints |
|---|---:|---|
| `priority` | omitted | `normal`, `low`, `moderate`, `high`, or `critical` |
| `limit` | 50 | Integer from 1 through 500 |

Priority filtering is applied after selecting each stay's latest row.
Each returned item includes the stay ID, saved hour, final priority, alert
category, risk probability/band, anomaly score/flag, clinical severity and
explanation, and final explanation.

An invalid priority returns HTTP 400. Invalid query types or bounds return
FastAPI validation errors (HTTP 422).

## `GET /api/patients/{patientunitstayid}`

Returns chronological hourly history for one ICU stay. Unknown stay IDs return
HTTP 404.

| Query | Default | Constraints |
|---|---:|---|
| `limit` | 200 | Integer from 1 through 2000 |

History items include available vital signs, fluid/context fields, risk,
anomaly, clinical rule explanation, priority, category, and final explanation.

## `GET /api/patients/{patientunitstayid}/latest`

Returns the latest saved hour for a stay, grouped into vitals, fluid,
treatment, and intelligence fields. Unknown stay IDs return HTTP 404.

## Data and safety notes

- This API does not perform ML inference; it serves saved CSV values.
- Risk is associated with expired ICU discharge within 24 hours, not general
  clinical deterioration.
- Historical eICU data are not current patient observations.
- Hardware telemetry is an independent optional stream and is not attached to
  an eICU stay.
- The local development CORS allowlist is limited to localhost/127.0.0.1 on
  port 5173. The application is not configured as a public service.
- No authentication, write API, or notification delivery API is implemented.
