# Isolated experimental alert replay

The replay substitutes held-out experimental logistic risk scores into the
existing alert-priority thresholds and 6-hour notification cooldown. Clinical
rule counts and severity are reconstructed from each version's test features.
The anomaly signal is disabled because no frozen anomaly artifact was recovered.
Only held-out test hours are included; production outputs are untouched.

## Summary

```text
               model dataset_version  test_patient_hours  alert_rows  critical_rows  new_severe_events  severe_notifications_emitted  severe_rows_suppressed  critical_rows_suppressed  critical_target_positive_hours  severe_alert_episodes  episodes_with_target_positive_hours
   balanced_logistic        baseline               21446        7845           4246               2518                          5625                    2220                      1339                             159                   2522                                   36
   balanced_logistic       corrected               21446        7861           4226               2511                          5631                    2230                      1334                             159                   2515                                   36
probability_logistic        baseline               21446        2004            287               1153                          1750                     254                        24                              51                   1153                                   38
probability_logistic       corrected               21446        2004            289               1153                          1749                     255                        25                              51                   1153                                   38
```

## Baseline/corrected priority changes

```text
               model  test_patient_hours  priority_changed_hours  priority_changed_fraction  baseline_critical_to_corrected_noncritical  baseline_noncritical_to_corrected_critical
   balanced_logistic               21446                     239                   0.011144                                          46                                          26
probability_logistic               21446                       4                   0.000187                                           1                                           3
```

Severe-alert episodes use the existing gap rule: high/critical rows in a stay
remain one episode when the gap is at most two hours. Per-episode outcomes
describe overlap with the historical expired-discharge label only; they are
not clinical-validation results.

## Limitations

- This is not a full deployed-fusion replay: the anomaly signal is disabled.
- Results are retrospective and based on the eICU demo extract.
- The target is expired ICU discharge within 24 hours, not generic deterioration.
- No clinical utility, safety, deployment, or model-promotion claim is made.
