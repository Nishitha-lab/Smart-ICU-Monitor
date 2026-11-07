# ML and Alert Final Audit

## Executive result

The dashboard serves precomputed hourly results from
`data/processed/smart_icu_final_alerts.csv`. API requests do not run inference.
The local output contains 139,432 patient-hours across 2,419 ICU stays.
Independent reconstruction of final priorities and explanations from their
source fields produced zero mismatches; no clinically critical rule state was
downgraded.

This audit is a code/data consistency check on a retrospective demo dataset.
It does not establish clinical validity, safety, or production readiness.

## Executed clinical rules

`ml/clinical_rules.py` builds the base rule flags and explanations from the
hourly trend/state dataset. `ml/add_hypertension_rule.py` adds the severe
hypertension flag and writes the v2 clinical-rule output consumed downstream.
The applied rule families include:

- Heart rate: tachycardia above 120, bradycardia below 50, with severe
  thresholds above 140 and below 40 beats/min.
- Oxygenation: SpO2 below 92; severe below 88.
- Respiration: below 10 or above 30; severe below 8 or above 35 breaths/min.
- Blood pressure: systolic below 90 (severe below 80), MAP below 65 (severe
  below 55), and severe hypertension when systolic exceeds 180 or diastolic
  exceeds 120 mmHg.
- Short-window changes: rapid heart-rate rise/fall, SpO2 fall, and MAP fall
  flags derived from the one-hour features.
- Context: vasoactive support, multiple active infusions, fluid balance at or
  beyond +/-500 mL, and zero recorded hourly urine output.

Warning/severe counts are converted to `normal`, `low`, `moderate`, `high`, or
`critical`; severe hypertension upgrades the result to at least `high`.
Threshold flags are rule outputs, not clinician-adjudicated diagnoses.

## Risk and anomaly signals

- `ml/build_deployment_deterioration.py` fits a median-imputation,
  standardization, and logistic-regression pipeline from the local training
  table, then scores every hourly state. The target label is expired ICU
  discharge within 0-24 hours of the prediction hour. It is not general
  clinical deterioration. The model uses `class_weight=None`; the risk output
  is not evidence of clinically acceptable calibration.
- `ml/build_deployment_anomaly.py` fits a median-imputation and
  IsolationForest pipeline with 300 trees, 2% contamination, and random seed
  42, then writes anomaly scores and flags.
- Both scripts refit from local processed inputs. They do not persist a
  production serving artifact in `ml/models/`. That directory is empty;
  experimental SpO2 bundles are isolated under
  `models/experimental/spo2_cross_validation/` and are not loaded by the app.
- `ml/predict.py` expects production artifacts that are not present in this
  checkout. There is no live model-inference endpoint.

## Fusion, priority, and explanations

`ml/build_final_smart_icu_alerts.py` combines clinical severity, anomaly
flags, and the risk-probability cutoffs 0.10, 0.25, and 0.50. Existing
critical clinical severity always yields critical priority. Severe clinical
states and combinations of very-high risk/anomaly/clinical-warning signals can
also yield critical priority. Lower combinations yield high, moderate, low,
or normal according to the ordered conditions in that script.

The final explanation is composed from the v2 clinical explanation, an
anomaly score when flagged, risk when at least 0.10, active infusion count,
vasoactive support, and fluid balance when its absolute value is at least 500
mL. A full-data reconstruction reproduced every saved priority and final
explanation (zero mismatches across 139,432 rows). All rows whose clinical
rule severity was critical remained critical.

These are deterministic consistency results for the current exported file,
not independent proof that thresholds or model outputs are clinically correct.

## Cooldown and notification boundary

`ml/build_alert_notification_layer.py` is an offline batch processor with a
six-hour cooldown. It re-emits new severe events, rank escalations, changed
clinical reasons, and expired-cooldown reminders. Its local output contains
30,913 rows: 472 are marked `SUPPRESS`, including 165 repeated critical rows.
No new severe event or critical escalation was suppressed in that output.

This script/output is separate from the API and is not the notification
mechanism used by the dashboard. The dashboard reads hourly final-alert rows;
it does not expose notification actions or send notifications. Do not present
the offline cooldown as a deployed alert-delivery policy. In particular, the
existing batch behavior suppresses some unchanged critical rows within the
cooldown and needs a separately approved policy review before any live use.

## Experimental replay limitation

The experimental SpO2 alert replay omitted the anomaly signal because a frozen
anomaly artifact was unavailable for that replay. It is therefore not a full
reproduction of the three-signal saved alert pipeline. Experimental model
bundles remain research-only and are not promoted into the application.

## Audit disposition

Preserve current production-facing behavior. No alert rules, model inference,
notification scripts, or model artifacts were changed in this audit.
Clinical workflow review, calibration assessment, external validation, and
separate approval are required before any operational use.
