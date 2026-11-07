# Paired SpO2 model experiment

This isolated experiment trains balanced and unweighted logistic regression
pipelines separately on baseline and corrected SpO2 model inputs. Imputation
and scaling are fitted only on the patient-level training partition. Each
threshold is selected by validation F1; the test partition is evaluated once.

## Test metrics

```text
               model dataset_version  threshold_selected_on_validation  validation_f1_at_selected_threshold  test_rows  test_positive_hours  test_positive_prevalence_baseline_ap  roc_auc  average_precision    brier  precision   recall
   balanced_logistic        baseline                          0.843114                             0.188782      21446                  357                              0.016646 0.673232           0.127735 0.176463   0.183432 0.173669
   balanced_logistic       corrected                          0.840235                             0.188425      21446                  357                              0.016646 0.674182           0.127324 0.176421   0.178571 0.182073
probability_logistic        baseline                          0.082119                             0.189152      21446                  357                              0.016646 0.684187           0.130068 0.015423   0.190031 0.170868
probability_logistic       corrected                          0.082265                             0.193906      21446                  357                              0.016646 0.687452           0.125653 0.015443   0.175793 0.170868
```

## Paired patient-cluster bootstrap deltas

Corrected-minus-baseline differences use the same patient resamples and paired
patient-hour rows. Intervals are percentile 95% confidence intervals.

```text
               model            metric  corrected_minus_baseline    ci_2_5  ci_97_5  valid_patient_bootstrap_replicates
   balanced_logistic           roc_auc                  0.000899 -0.002756 0.005148                                1000
   balanced_logistic average_precision                 -0.000482 -0.004679 0.003403                                1000
   balanced_logistic             brier                 -0.000031 -0.000748 0.000560                                1000
   balanced_logistic         precision                 -0.004324 -0.016022 0.010276                                1000
   balanced_logistic            recall                  0.008496  0.000000 0.022926                                1000
probability_logistic           roc_auc                  0.003142 -0.005679 0.013082                                1000
probability_logistic average_precision                 -0.004576 -0.015181 0.005140                                1000
probability_logistic             brier                  0.000021 -0.000098 0.000122                                1000
probability_logistic         precision                 -0.013764 -0.030232 0.001944                                1000
probability_logistic            recall                 -0.000169 -0.011498 0.011462                                1000
```

## Limitations

- Retrospective demo-dataset results; no clinical validity or deployment claim.
- The target is expired ICU discharge within 24 hours, not generic deterioration.
- Thresholds optimize hourly F1 on validation, not clinical utility.
- Results are exploratory and do not authorize model promotion.
