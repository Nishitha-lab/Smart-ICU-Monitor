# SpO2 cross-validation and quality-policy milestone status

## Status

**Experimental analysis complete with limitations.** No production behavior
was changed. This report does not claim clinical validation, clinical utility,
or deployment readiness.

## Workstreams

| Workstream | Status | Evidence |
|---|---|---|
| Common patient-level cohort | PASS | `leakage_safe/split_validation_report.md` |
| Independent cohort validation | PASS (5 checks) | `leakage_safe/validation_report.md` |
| Paired supervised models | PASS; effect uncertain | `paired_models/experiment_report.md` |
| Descriptive anomaly comparison | PASS; no ground truth | `paired_anomaly/experiment_report.md` |
| Isolated alert replay | PASS with anomaly signal omitted | `alert_replay/alert_replay_report.md` |
| Robustness tests | PASS (5 tests) | `ml/experiments/test_experiment_robustness.py` |
| Independent artifact/metric/replay validation | PASS (13 checks) | `paired_models/validation_report.md` |
| Application integration readiness | NOT READY | `docs/model_integration_readiness.md` |

## Overall interpretation

Baseline and corrected SpO2 versions shared 139,432 patient-hours from 1,816
people and 2,419 stays. The held-out test set contained 21,446 hours and 357
positive hours from 18 people. Patient identifiers were disjoint across the
train, validation, and test partitions.

The paired patient-cluster intervals do not establish a reliable overall
performance improvement from corrected SpO2. Anomaly scores were highly
correlated between versions, but there are no anomaly ground-truth labels.
Alert replay showed small or mixed priority shifts and excluded the anomaly
signal. These are exploratory dataset observations, not clinical or workflow
evidence.

The evaluated label is expired ICU discharge within 24 hours of an hourly
prediction point, not general deterioration. External validation, calibration,
clinical utility, safety assessment, and an approved model-serving path remain
unavailable. The experimental model files are not approved for production.

## Stop condition

This milestone ends after analysis, validation, and reporting. No downstream
model promotion/retraining, production preprocessing or cleaning change, alert
modification, backend/frontend/hardware change, or deployment work was
performed.
