# Model integration readiness

## Status: not ready for application integration

The SpO2 experiments in this repository are isolated retrospective research
artifacts. They are not approved or suitable for use as clinical decision
support. No backend, frontend, hardware, deployment, or production alert change
was made for this milestone.

## What exists

- Experimental scikit-learn logistic pipelines and IsolationForest pipelines
  are stored only under `models/experimental/spo2_cross_validation/`.
- The supervised input contract has 22 ordered numeric features; the anomaly
  input contract has 17. Logistic pipelines include median imputation,
  standardization, and regression. IsolationForest pipelines include median
  imputation and a fixed 2% contamination setting.
- The supervised target is expired ICU discharge within 0–24 hours of an
  hourly prediction point. It is not a general deterioration target.
- The input cohort is a retrospective eICU demo extract with patient-level
  train/validation/test partitions.
- The test-set alert replay omits anomaly scores because a frozen anomaly model
  artifact was unavailable.

## Application boundary

Repository inspection found no supported model-serving integration in the
existing application. Existing backend/frontend and serial-monitor behavior
remain unchanged. The experimental bundles are not a stable inference API and
must not be loaded by the current application.

## Required before any future integration decision

1. Define and approve the intended population, outcome, prediction horizon,
   clinical workflow, and acceptable error trade-offs.
2. Obtain an independent, representative external evaluation with patient-level
   separation and sufficient outcome events.
3. Assess calibration, missingness, subgroup performance, temporal/site drift,
   and sensitivity to the SpO2 cleaning policy.
4. Have qualified clinical stakeholders review false positives, false
   negatives, alert burden, and escalation/cooldown behavior.
5. Establish monitoring, fallback behavior, auditability, privacy/security,
   ownership, versioning, and rollback requirements.
6. Conduct a separately authorized integration and safety review before any
   application, alert, or deployment modification.

No item above is satisfied by the current experiment alone. The result is
**research-only; integration readiness is not established**.
