# Experimental SpO2 cross-validation milestone

## Outcome

**Completed with limitations.** Leakage-safe patient-level cohorts, paired
supervised experiments, a descriptive anomaly comparison, and an isolated
test-partition alert replay were produced and independently checked. This work
does not validate clinical use, authorize model promotion, or establish
deployment readiness.

No raw data, production preprocessing/cleaning, existing model or alert logic,
backend, frontend, hardware, or deployment behavior was changed. All new model
bundles are under `models/experimental/spo2_cross_validation/`.

## Cohort and target

The common baseline/corrected cohort contains 1,816 unique people, 2,419 ICU
stays, and 139,432 matched patient-hours. The patient-level stratified split
contains 1,271 train, 272 validation, and 273 test people, with no person
overlap. There are 96,731 training, 21,255 validation, and 21,446 test rows.
The test set contains 357 positive hours across 18 people.

The target is `deterioration_label_24h`: expired ICU discharge occurring within
0–24 hours of an hourly prediction point. It is not a general deterioration
label. Baseline/corrected keys, patient assignments, and targets match.
SpO2-derived one-hour changes, gap-aware three-hour means, and SpO2-dependent
rule counts were reconstructed independently.

## Supervised model results

Thresholds were selected by maximizing hourly F1 on validation, separately per
model and dataset version. Imputation/scaling were fitted on training only.
Each test set was evaluated once. Test PR-AUC is compared with the test positive
prevalence of 0.01665.

| Model | Version | Validation threshold | Test ROC-AUC | Test PR-AUC | Brier | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Balanced logistic | Baseline | 0.8431 | 0.6732 | 0.1277 | 0.1765 | 0.1834 | 0.1737 |
| Balanced logistic | Corrected | 0.8402 | 0.6742 | 0.1273 | 0.1764 | 0.1786 | 0.1821 |
| Probability logistic | Baseline | 0.0821 | 0.6842 | 0.1301 | 0.0154 | 0.1900 | 0.1709 |
| Probability logistic | Corrected | 0.0823 | 0.6875 | 0.1257 | 0.0154 | 0.1758 | 0.1709 |

Paired 1,000-replicate patient-cluster bootstrap 95% intervals for corrected
minus baseline included zero for the principal discrimination and probability
metrics. For example, probability-model ROC-AUC difference was +0.0031
(95% CI −0.0057 to +0.0131) and PR-AUC difference was −0.0046
(95% CI −0.0152 to +0.0051). Results do not show a clear, reliable quality
improvement in this held-out sample. The balanced model's Brier score also
shows that its weighted scores are not interchangeable with calibrated risks.

## Unsupervised anomaly comparison

IsolationForest v2 was fitted separately on training partitions with fixed 2%
contamination and evaluated descriptively on test data. Baseline/corrected flag
rates were 2.20% and 2.33%; flag Jaccard similarity was 0.862 and score Spearman
correlation was 0.997. No ground-truth anomaly labels were available, so no
accuracy, predictive, or clinical-performance metrics were computed.

## Isolated alert replay

The existing fixed 0.10/0.25/0.50 risk cutoffs and six-hour notification
cooldown were replayed on held-out test hours. The existing two-hour severe
episode gap rule was used. No production alert or notification outputs were
modified. The anomaly signal was disabled because no frozen anomaly artifact
was recovered; therefore this is not a complete reproduction of deployed
signal fusion.

For probability logistic, corrected SpO2 changed priority on 4 of 21,446
test-hours; critical rows were 287 baseline and 289 corrected. For balanced
logistic, 239 priorities changed; critical rows were 4,246 baseline and 4,226
corrected. Notification suppression and episode counts are recorded in
`data/processed/experiments/alert_replay/alert_replay_summary.csv`. Counts
describe an experimental replay only and are not evidence of clinical utility.

## Validation and tests

- Leakage-safe cohort independent validator: **PASS, 5 checks**.
- Paired experiment independent validator: **PASS, 13 checks**.
- Robustness unit tests: **PASS, 5 tests**.
- All reported model metrics, validation thresholds, patient-level split
  assignments, alert priority counts, cooldown safety counts, episode totals,
  and paired anomaly stability statistics were independently reconciled.

Validation reports:

- `data/processed/experiments/leakage_safe/validation_report.md`
- `data/processed/experiments/paired_models/validation_report.md`

## Limits and stop point

- The eICU demo extract is retrospective and is not external validation.
- The target concerns death/discharge status within 24 hours, not general
  deterioration.
- Test positives are clustered within only 18 people; results are uncertain.
- F1-selected operating points are not clinical utility thresholds.
- No prospective workflow evaluation, clinician adjudication, calibration
  study, safety review, or deployment evaluation was performed.
- The anomaly signal was omitted from replay; alert replay is incomplete.
- Existing model-serving integration was not found. The experimental bundles
  are not production-ready or approved.

This milestone stops here. No downstream retraining, alert-system change,
backend/frontend work, or deployment change is included.
