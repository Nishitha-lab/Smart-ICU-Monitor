# Independent leakage-safe experiment validation

**Result: PASS (5 checks, 0 failures).**

| Check | Result | Detail |
|---|---|---|
| input_hashes | PASS | All source inputs match recorded SHA-256 hashes. |
| patient_splits | PASS | 1816 unique patients; deterministic 70/15/15 split; no overlap. |
| baseline_reconstruction | PASS | Keys, patient mapping, targets, all features, and split assignments independently reconstructed. |
| corrected_reconstruction | PASS | Keys, patient mapping, targets, all features, and split assignments independently reconstructed. |
| paired_cohort | PASS | Baseline and corrected versions have identical keys, labels, and splits. |

Validation independently reconstructs cohort source joins, the patient-level
split, the row-to-person mapping, SpO2 temporal features, rule counts, targets,
and all reported model-input feature values. It does not evaluate model
performance or clinical validity.
