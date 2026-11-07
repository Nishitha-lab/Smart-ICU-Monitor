# Paired SpO2 anomaly-model experiment

Separate IsolationForest pipelines were fitted on the training partition for
each dataset version and scored only on the paired held-out test partition.
Preprocessing is fitted on training data. The contamination setting is held
fixed at 2%, matching the existing v2 research configuration.

No ground-truth anomaly labels exist in this cohort. Therefore, this report
contains descriptive flag/score stability and agreement with rule counts only;
it does not report anomaly accuracy, clinical performance, or utility.

## Per-version summary

```text
dataset_version  train_rows  test_rows  anomaly_rows  anomaly_rate  anomalous_patients  anomaly_with_clinical_warning  anomaly_with_clinical_severe_rule
       baseline       96731      21446           471      0.021962                  55                            395                                123
      corrected       96731      21446           499      0.023268                  59                            427                                135
```

## Paired stability

```text
 test_patient_hours  baseline_anomaly_rows  corrected_anomaly_rows  flag_overlap_rows  flag_union_rows  flag_jaccard_similarity  score_spearman_rho  score_spearman_pvalue  baseline_only_flags  corrected_only_flags
              21446                    471                     499                449              521                 0.861804            0.997357                    0.0                   22                    50
```

No clinical validation, deployment, or model-promotion claim is made.
