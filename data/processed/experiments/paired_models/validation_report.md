# Independent paired-experiment validation

**Result: PASS (13 checks, 0 failures).**

| Check | Result | Detail |
|---|---|---|
| cohort_validation | PASS | Independent cohort validator reports 5 PASS checks. |
| patient_disjointness | PASS | No natural-person overlap across train/validation/test. |
| balanced_logistic_baseline | PASS | Bundle reproduces test probabilities; validation threshold and test metrics independently recomputed. |
| balanced_logistic_corrected | PASS | Bundle reproduces test probabilities; validation threshold and test metrics independently recomputed. |
| probability_logistic_baseline | PASS | Bundle reproduces test probabilities; validation threshold and test metrics independently recomputed. |
| probability_logistic_corrected | PASS | Bundle reproduces test probabilities; validation threshold and test metrics independently recomputed. |
| paired_bootstrap | PASS | Ten paired metrics each use 1,000 valid patient-cluster replicates. |
| alert_replay_balanced_logistic_baseline | PASS | Clinical counts, fixed-threshold priority, and notification safety counts independently checked. |
| alert_replay_balanced_logistic_corrected | PASS | Clinical counts, fixed-threshold priority, and notification safety counts independently checked. |
| alert_replay_probability_logistic_baseline | PASS | Clinical counts, fixed-threshold priority, and notification safety counts independently checked. |
| alert_replay_probability_logistic_corrected | PASS | Clinical counts, fixed-threshold priority, and notification safety counts independently checked. |
| alert_episodes | PASS | All reported severe-alert episode counts independently reconstructed with the <=2-hour gap rule. |
| anomaly_stability | PASS | Paired anomaly keys, flag overlap, score rank correlation, and descriptive rates reconcile. |

The validator independently recomputes validation thresholds, held-out
metrics, replay priorities, notification safety counts, severe-episode
counts, and anomaly stability statistics. It does not assert clinical
validity, alert utility, or deployment readiness.
