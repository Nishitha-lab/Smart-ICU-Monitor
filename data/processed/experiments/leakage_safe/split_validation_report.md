# Leakage-safe split report

## Status

Patient-level split generated with `uniquepid` groups, seed 20261107, and
stratification by whether each person has any positive hourly target.

- Patients: 1,816
- ICU stays: 2,419
- Matched patient-hours per dataset version: 139,432
- Train / validation / test patient counts: 1,271 /
  272 / 273
- Patient overlap by partition: {'train_validation': 0, 'train_test': 0, 'validation_test': 0}
- Baseline/corrected key sets and target labels match exactly.

## Target

`deterioration_label_24h` marks an hourly row positive when `unitdischargestatus`
is `expired` and ICU discharge occurs 0–24 hours after the prediction hour.
Multiple hourly rows per stay can be positive. This target is not general
clinical deterioration.

## Constraints

- The dataset represents the supplied eICU demo extract and is retrospective.
- Splitting is grouped by natural patient ID, so all of a person's ICU stays
  remain together.
- The target label and prediction hours are taken from the previously generated
  label file and are identical across dataset versions.
- The experiment feature table reuses non-SpO2 state variables and regenerates
  SpO2 current value, one-hour change, three-hour rolling mean, and dependent
  rule counts with the same gap-aware temporal rules.
- No production files or prior experiment outputs were overwritten.
