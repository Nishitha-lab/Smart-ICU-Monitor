# SMART ICU PATIENT SAFETY SYSTEM
# FINAL INDEPENDENT ENGINEERING AUDIT REPORT (PHASES 4–7)

**Document ID:** `FINAL_PHASE4_TO_PHASE7_INDEPENDENT_AUDIT.md`
**Audit Date:** November 9, 2026
**Auditor Role:** Senior Software Engineer, Clinical-Software Logic Auditor & Independent ML Reviewer
**Target Repository:** `D:\Smart-ICU`
**Active Git Branch:** `experiment/ml-candidate-v2`
**Head Commit:** `2d32767bab30408fd09eacbf7d807e2996bf3fb9`
**Baseline Git Commit:** `767f3b88d434142d2ffd200671a306442ef5a4e7`
**Audit Policy:** Strict Audit-First / Zero Code Modification Protocol

---

## 1. Executive Summary

This independent engineering and scientific audit evaluated the deliverables, test suites, clinical rule engines, machine learning evaluation pipelines, and architectural integration plans produced across **Phases 4 through 7** of the Smart ICU Patient Safety System:

- **Phase 4:** End-to-end software validation (patient lifecycle, device assignment, delayed telemetry packets, clinical record optimistic locking, soft-voiding, alert lifecycle, and database integrity).
- **Phase 5:** Clinical alert logic verification (KDIGO 2012 urine output staging, canonical laboratory registry, unit conversions, and reference range provenance).
- **Phase 6:** Retrospective ML model evaluation (Baseline v1 vs Candidate v2, patient-cluster bootstrap uncertainty, alert episode burden, calibration reliability, and PR-AUC vs ROC-AUC reconciliation).
- **Phase 7:** ML integration planning (feature availability matrix, candidate v2 compatibility, proposed shadow mode, and data contracts).

### High-Level Audit Decision: CONDITIONAL PASS / PROMOTION BLOCKED
1. **Software Mechanics & Integrity (PASS):**
   All **185 backend tests** and **46 frontend tests** reproduced with **zero failures**. The SQLite database has **zero foreign key violations**. Delayed telemetry attribution accurately isolates historical packets, suppresses live alert generation on retrospective data, and prevents cross-patient contamination.
2. **Clinical Alert Logic (PASS WITH LIMITATIONS):**
   KDIGO AKI staging (Stages 1–3, anuria) and laboratory molecular conversions (lactate $\div 9.008$, urea $\times 6.006$, WBC/platelets $\div 1000$) are mathematically sound. However, a **critical provenance defect** was identified: polyuria ($> 2.5\text{ mL/kg/h}$) is erroneously attributed to the KDIGO 2012 guideline, and Arterial Blood Gas (ABG) / eGFR calculations are absent from the canonical registry.
3. **Machine Learning Reproducibility (PASS WITH LIMITATIONS):**
   The metric discrepancy between earlier reports (ROC-AUC 0.7476 / PR-AUC 0.0886) and latest reports (ROC-AUC 0.7373 / PR-AUC 0.1019) was fully reconciled. It was caused by the methodology corrections (timestamp-based as-of join from raw `lab.csv.gz` replacing naive shift/ffill, segment-based gap-reset for rolling std, Platt scaling, and threshold re-tuning). On the held-out test benchmark, **Candidate v2 suffers an 18.9% degradation in PR-AUC** ($0.1257 \to 0.1019$), a **worse Brier score** ($0.015443 \to 0.015710$), and nearly **doubles false alarm hours** ($286 \to 560$).
4. **Promotion Governance Verdict (BLOCKED):**
   **Candidate v2 MUST NOT BE PROMOTED.** Baseline v1 (`deterioration_logistic_v1.joblib`) must remain the sole production model. Candidate v2 remains an unpromoted evaluation candidate.
5. **Phase 7 Integration Claims (PASS WITH LIMITATIONS):**
   Runtime isolation in `ml_service.py` is verified: candidate v2 strictly rejects live telemetry. However, the proposed **"Silent Shadow Mode"** and `ml_shadow_predictions` database table are **PROPOSED ARCHITECTURAL SPECIFICATIONS ONLY** and are not yet implemented in runtime code.

---

## 2. Exact Git State

Before performing any test execution or audit commands, the local Git environment was inspected and recorded:

```powershell
# git branch --show-current
experiment/ml-candidate-v2

# git rev-parse HEAD
2d32767bab30408fd09eacbf7d807e2996bf3fb9

# git status
On branch experiment/ml-candidate-v2
Your branch is up to date with 'origin/experiment/ml-candidate-v2'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   backend/services/patient_service.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	PHASE4_END_TO_END_VALIDATION.md
	PHASE5_CLINICAL_RULE_VERIFICATION.md
	PHASE6_ML_EVALUATION_REPORT.md
	PHASE7_ML_INTEGRATION_PLAN.md
	backend/tests/test_e2e_integration_workflows.py
	ml/baseline_artifacts_v1/
	ml/research_phase6/

no changes added to commit (use "git add" and/or "git commit -a")
```

- `git diff --cached`: **Empty** (zero staged files).
- `git diff`: Only contains the delayed-packet historical assignment logic in `backend/services/patient_service.py` (lines 1479, 1492, 1494–1520, 1552).
- Zero commits, pushes, or branch switches were executed during this audit.

---

## 3. Phase 4 Independent Test Results

### Automated Test Suite Execution

| Test Suite | Command Executed | Tests Passed | Tests Failed | Execution Time | Audit Classification |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Backend Suite** | `python -m pytest backend/tests -o pythonpath=backend` | **185** | **0** | $55.41\text{ s}$ | **PASS** |
| **Frontend Suite** | `npm test -- --run` (in `frontend/`) | **46** | **0** | $5.58\text{ s}$ | **PASS** |
| **E2E Integration**| `python -m pytest backend/tests/test_e2e_integration_workflows.py -o pythonpath=backend` | **6** | **0** | $3.28\text{ s}$ | **PASS** |

### Database Referential Integrity
Executed `PRAGMA foreign_key_check;` against the active SQLite database (`data/clinical_records.sqlite3`):
```python
conn.execute('PRAGMA foreign_keys = ON;')
violations = conn.execute('PRAGMA foreign_key_check;').fetchall()
# Result: 0 violations []
```
**Classification:** **PASS**. Referential integrity across `patients`, `icu_stays`, `devices`, `device_assignments`, `telemetry_records`, `clinical_records`, and `alerts` is 100% compliant.

### Workflow Verification Summary
- **Workflow A (Patient & Stay Lifecycle):** Verified patient creation, demographic update, admission, discharge, 409 Conflict duplicate active stay rejection, and discharged stay telemetry rejection. (**PASS**)
- **Workflow B (Device Assignment & Transfer):** Verified device assignment to active stay, atomic reassignment to another patient (closing previous assignment with `unassigned_at`), and unassignment. (**PASS**)
- **Workflow C (Telemetry Ingestion & Quality Masking):** Numerical sanity checks confirmed (`0 BPM` and `0% SpO2` masked to `None`), malformed serial frames skipped, stale telemetry detection ($> 5\text{s}$) confirmed. (**PASS**)
- **Workflow D (Clinical Record Lifecycle):** Verified record creation, optimistic concurrency control (409 Conflict on stale version update), soft-voiding with mandatory audit reason, and namespace separation between eICU research data and bedside patients. (**PASS**)
- **Workflow E (Alert Notification Lifecycle):** Verified alert triggering, fingerprint deduplication, manual acknowledgment, resolution with clinical notes, automatic resolution on normal observations, and superseding when parent records are updated or voided. (**PASS**)

---

## 4. Delayed Packet Attribution Findings

### Implementation Inspection (`backend/services/patient_service.py:1464-1566`)

```python
# 1. Active Assignment Check:
assignment = connection.execute(
    "SELECT a.id, a.patient_id, a.stay_id, a.assigned_at, s.discharge_status ... "
    "WHERE a.device_id = ? AND a.assignment_status = 'active'", (device_id,)
).fetchone()
if assignment and assignment["discharge_status"] == "active":
    if reading_time >= assignment["assigned_at"]:
        patient_id = assignment["patient_id"]
        stay_id = assignment["stay_id"]
        is_active_attribution = True

# 2. Historical Assignment Fallback:
if patient_id is None:
    hist_assignment = connection.execute(
        "SELECT a.id, a.patient_id, a.stay_id, a.assigned_at, a.unassigned_at ... "
        "WHERE a.device_id = ? AND a.assignment_status = 'ended' "
        "  AND a.assigned_at <= ? AND a.unassigned_at IS NOT NULL AND a.unassigned_at >= ? "
        "ORDER BY a.assigned_at DESC LIMIT 1",
        (device_id, reading_time, reading_time),
    ).fetchone()
    if hist_assignment and hist_assignment["stay_patient_id"] == hist_assignment["patient_id"]:
        patient_id = hist_assignment["patient_id"]
        stay_id = hist_assignment["stay_id"]
        # is_active_attribution remains FALSE

# 3. Alert Suppression for Historical Packets:
if is_active_attribution and patient_id is not None and stay_id is not None:
    self._evaluate_telemetry_alerts(...)
```

### Detailed Verification Findings

| Scenario / Condition | Expected Behavior | Actual Observed Behavior | Classification |
| :--- | :--- | :--- | :---: |
| **Packets before first assignment** | Unassigned (`patient_id = NULL`) | Neither active nor historical match; stored with `patient_id = NULL`. | **PASS** |
| **Packets arriving during active stay** | Attributed to active stay, live alerts evaluate | `reading_time >= assigned_at` matches; alerts fire. | **PASS** |
| **Packets arriving after reassignment** | Attributed to previous patient, NOT new patient | Historical query matches prior ended assignment; new patient unaffected. | **PASS** |
| **Historical packet alert evaluation** | Suppressed (no live alerts for old packets) | `is_active_attribution == False`; `_evaluate_telemetry_alerts` bypassed. | **PASS** |
| **Current live patient state pollution** | Historical packet must not update live patient | `WHERE patient_id = ?` queries isolate data by patient ID; no state pollution. | **PASS** |
| **Exact boundary conditions ($t = \text{boundary}$)** | Handled deterministically | Inclusive $\ge$ on `assigned_at` and `unassigned_at`. Active assignment takes precedence. | **PASS** |
| **Missing timestamp** | Default to server ingestion time | `reading_time = now_iso` applied safely. | **PASS** |
| **Malformed timestamp string** | Return HTTP 400 / 422 Bad Request | **DEFECT:** `_utc_iso` raises unhandled `ValueError`, causing HTTP 500 error. | **PASS WITH LIMITATIONS** |
| **Future timestamp ($t > \text{now}$)** | Reject or flag clock drift | No future timestamp validation; accepted into database as-is. | **PASS WITH LIMITATIONS** |

---

## 5. Phase 5 Clinical Rule Findings

### KDIGO 2012 Urine Output Staging Verification

Formula tested: $\text{rate} = \frac{\text{urine\_volume\_ml}}{\text{patient\_weight\_kg} \times \text{duration\_hours}}$

| Test Condition | Input Values | KDIGO Criterion | Expected Interpretation | Engine Output | Classification |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Stage 1 (Threshold)** | Vol: 210 mL, Wt: 70 kg, Dur: 6h | $< 0.5$ mL/kg/h for $\ge 6$h | Rate 0.500 $\to$ NORMAL | `NORMAL (0.5 mL/kg/h)` | **PASS** |
| **Stage 1 (Oliguria)** | Vol: 200 mL, Wt: 70 kg, Dur: 6h | $< 0.5$ mL/kg/h for $\ge 6$h | Rate 0.476 $\to$ LOW / HIGH | `LOW / HIGH (KDIGO Stage 1)` | **PASS** |
| **Stage 2 (Oliguria)** | Vol: 400 mL, Wt: 70 kg, Dur: 12h | $< 0.5$ mL/kg/h for $\ge 12$h | Rate 0.476 $\to$ LOW / HIGH | `LOW / HIGH (KDIGO Stage 2)` | **PASS** |
| **Stage 3 (Oliguria)** | Vol: 500 mL, Wt: 70 kg, Dur: 24h | $< 0.3$ mL/kg/h for $\ge 24$h | Rate 0.298 $\to$ CRITICAL_LOW | `CRITICAL_LOW (KDIGO Stage 3)` | **PASS** |
| **Stage 3 (Anuria)** | Vol: 0 mL, Wt: 70 kg, Dur: 12h | Anuria for $\ge 12$h | Anuria $\to$ CRITICAL_LOW | `CRITICAL_LOW (KDIGO Stage 3)` | **PASS** |
| **Short Duration ($< 6$h)** | Vol: 10 mL, Wt: 70 kg, Dur: 2h | $< 6$h screening | Rate 0.071 $\to$ Screening Warn | `LOW / MODERATE (Screening)` | **PASS** |
| **Polyuria ($> 2.5$ mL/kg/h)**| Vol: 1200 mL, Wt: 70 kg, Dur: 6h | $> 2.5$ mL/kg/h | Rate 2.857 $\to$ HIGH | `HIGH / LOW (Polyuria)` | **PASS (Logic)** |
| **Missing Weight** | Vol: 100 mL, Wt: None, Dur: 6h | Unadjusted volumetric | Rate 16.67 mL/h $\to$ $< 30$ mL/h | `LOW / HIGH (Unadjusted)` | **PASS** |
| **Negative Volume** | Vol: -50 mL, Wt: 70 kg, Dur: 6h | Physical impossibility | `UNINTERPRETABLE` | `UNINTERPRETABLE` | **PASS** |
| **Zero Duration** | Vol: 100 mL, Wt: 70 kg, Dur: 0h | Division by zero safety | `UNINTERPRETABLE` | `UNINTERPRETABLE` | **PASS** |

### Critical Provenance Defect Identified: Polyuria Reference Source
> [!WARNING]
> **PROVENANCE DEFECT in `backend/services/clinical_alert_engine.py:566`:**
> ```python
> if weight_adj_rate > 2.5:
>     return ClinicalInterpretation(
>         ...
>         critical_threshold="> 2.5 mL/kg/h",
>         reference_source="KDIGO Clinical Practice Guideline for AKI (2012)",
>         explanation=f"POLYURIA — measured {rate_disp}..."
>     )
> ```
> **Auditor Finding:** KDIGO 2012 AKI Guidelines do **NOT** define or mention a 2.5 mL/kg/h threshold for polyuria. KDIGO is strictly a guideline for Acute Kidney Injury (hypoperfusion, oliguria, anuria, and elevated serum creatinine). Polyuria ($> 2.5\text{ mL/kg/h}$ or $> 3\text{ L/day}$) is a general clinical monitoring threshold for post-obstructive diuresis, diabetes insipidus, or hypervolemia.
> **Verdict:** **FAIL ON PROVENANCE CITATION**. Must be corrected to `"General Clinical Nephrology / ICU Monitoring Guideline"`.

---

## 6. KDIGO and Laboratory Conversion Findings

### Molecular Unit Conversions Independent Verification

1. **Lactate Conversion ($\text{mg/dL} \to \text{mmol/L}$):**
   - Molecular Formula: Lactic acid ($C_3H_6O_3$)
   - Molar Mass: $3 \times 12.011 + 6 \times 1.008 + 3 \times 15.999 = 90.078\text{ g/mol}$.
   - Exact conversion: $1\text{ mmol/L} = 90.078\text{ mg/L} = 9.0078\text{ mg/dL} \approx 9.008\text{ mg/dL}$.
   - Engine implementation (`clinical_alert_engine.py:796`): `round(numeric_val / 9.008, 2)`.
   - Numerical test: $18.016\text{ mg/dL} \to 2.00\text{ mmol/L}$ (Upper normal limit).
   - **Verdict:** **PASS (Mathematically exact)**.

2. **Urea Conversion ($\text{mmol/L} \to \text{mg/dL}$):**
   - Molecular Formula: Urea ($CO(NH_2)_2$)
   - Molar Mass: $12.011 + 15.999 + 2 \times (14.007 + 2 \times 1.008) = 60.056\text{ g/mol}$.
   - Exact conversion: $1\text{ mmol/L} = 60.056\text{ mg/L} = 6.0056\text{ mg/dL} \approx 6.006\text{ mg/dL}$.
   - Engine implementation (`clinical_alert_engine.py:800`): `round(numeric_val * 6.006, 2)`.
   - Numerical test: $10.0\text{ mmol/L} \to 60.06\text{ mg/dL}$.
   - **Verdict:** **PASS (Mathematically exact)**.

3. **Urea vs Blood Urea Nitrogen (BUN) Distinction:**
   - Nitrogen molar mass in urea: $2 \times 14.007 = 28.014\text{ g/mol}$.
   - Molecular ratio: $\frac{\text{Urea}}{\text{BUN}} = \frac{60.056}{28.014} \approx 2.1438$.
   - Engine handling: The registry maintains separate entries for `BUN` (`standard_unit: "mg/dL"`, normal 7–20 mg/dL) and `UREA` (`standard_unit: "mg/dL"`, normal 15–45 mg/dL).
   - Conversion safety: `BUN` acceptable units strictly allow only `"mg/dL"`. Submitting `BUN` in `mmol/L` is rejected with `UNINTERPRETABLE` rather than misinterpreted, preventing confusion between urea nitrogen and whole urea.
   - **Verdict:** **PASS**.

4. **WBC and Platelets Conversion ($/\mu\text{L} \to \text{K}/\mu\text{L}$):**
   - Engine implementation (`clinical_alert_engine.py:792`): `round(numeric_val / 1000.0, 2)`.
   - Numerical test: $4,500/\mu\text{L} \to 4.5\text{ K}/\mu\text{L}$.
   - **Verdict:** **PASS**.

5. **Arterial Blood Gas (ABG) & eGFR Handling:**
   - `LAB_REGISTRY` contains **zero entries** for ABG tests (`pH`, `pO2`, `pCO2`, `Base Excess`, `FiO2`, `sO2`) or `eGFR`.
   - Submissions without custom reference ranges return `UNINTERPRETABLE` with `canonical_test_id="UNKNOWN"`.
   - No automated eGFR calculation (e.g., CKD-EPI 2021) is implemented.
   - **Verdict:** **PASS WITH LIMITATIONS (Documented Feature Boundary)**.

---

## 7. Clinical Reference Provenance Findings

Every claimed reference source in `backend/services/clinical_alert_engine.py` was evaluated against clinical literature:

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                        CLINICAL REFERENCE PROVENANCE AUDIT TABLE                                      |
+----+-------------+------------------------------------+--------------------------------+-----------------------------+
| ID | Test Name   | Claimed Reference Source in Code   | Configured Thresholds          | Independent Audit Status    |
+----+-------------+------------------------------------+--------------------------------+-----------------------------+
| 1  | Potassium   | Tietz / Mayo Clinic Laboratories   | 3.5-5.0, crit: < 2.8, > 6.0    | VERIFIED (Supported)        |
| 2  | Sodium      | Tietz Textbook of Clinical Chem.   | 135-145, crit: < 120, > 160    | VERIFIED (Supported)        |
| 3  | Chloride    | Tietz Textbook of Clinical Chem.   | 96-106, crit: < 80, > 125      | VERIFIED (Supported)        |
| 4  | Bicarbonate | Tietz Textbook of Clinical Chem.   | 22-29, crit: < 10, > 40        | VERIFIED (Venous total CO2) |
| 5  | Creatinine  | KDIGO Clinical Practice Guideline  | 0.6-1.2 mg/dL, crit: >= 4.0    | UNVERIFIED (KDIGO defines   |
|    |             | for AKI (2012)                     |                                | relative rise, not single   |
|    |             |                                    |                                | static population normal)   |
| 6  | BUN         | Tietz Textbook of Clinical Chem.   | 7-20 mg/dL, crit: > 60         | VERIFIED (Adult standard)   |
| 7  | Urea        | Tietz Textbook of Clinical Chem.   | 15-45 mg/dL, crit: > 100       | VERIFIED (Supported)        |
| 8  | Hemoglobin  | Surviving Sepsis / AABB Guidelines | Crit low: < 7.0 g/dL           | VERIFIED (TRICC trial base) |
| 9  | Platelets   | AABB Transfusion Guidelines        | Crit low: < 20 K/uL            | VERIFIED (Supported)        |
| 10 | Glucose     | ADA / Critical Care Guidelines     | 70-140, crit: < 50, > 400      | VERIFIED (Supported)        |
| 11 | Calcium     | Tietz Textbook of Clinical Chem.   | 8.5-10.5, crit: < 6.5, > 13.0  | CUSTOM APPROXIMATION        |
|    |             |                                    |                                | (Unadjusted for albumin)    |
| 12 | Magnesium   | Tietz Textbook of Clinical Chem.   | 1.7-2.2, crit: < 1.0, > 4.5    | VERIFIED (Supported)        |
| 13 | Lactate     | Surviving Sepsis Campaign 2021     | 0.5-2.0, crit: >= 4.0 mmol/L   | VERIFIED (Exact match)      |
| 14 | Urine: KDIGO| KDIGO AKI Guidelines (2012)        | <0.5 (6h/12h), <0.3 (24h/anur) | VERIFIED (Exact match)      |
| 15 | Urine: Poly | KDIGO AKI Guidelines (2012)        | > 2.5 mL/kg/h                  | INVALID PROVENANCE (KDIGO   |
|    |             |                                    |                                | does not define polyuria)   |
+----+-------------+------------------------------------+--------------------------------+-----------------------------+
```

---

## 8. Phase 6 Model Artifact Integrity

Independent SHA-256 cryptographic hashes and byte sizes were calculated for all model artifacts:

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                           MODEL ARTIFACT INTEGRITY AUDIT                                              |
+-------------------------------------------------------+---------+----------------------------------------------------+
| File Path                                             | Size    | Calculated SHA-256 Hash                            |
+-------------------------------------------------------+---------+----------------------------------------------------+
| ml/models/deterioration_logistic_v1.joblib            | 5,604 B | 26d79474d38521bb251f3c0d456e099e038736c6c2f1bc43...|
| ml/models/deterioration_random_forest_v1.joblib       | 1.56 MB | b470d94c9aea9183a6be9362ea492c89886168b0c5016e56...|
| ml/models/deterioration_gradient_boosting_v1.joblib   | 238 KB  | 2deacbbb4a429746055b672256a09f7a092822eed5b6af39...|
| ml/models/anomaly_isolation_forest_v2.joblib          | 2.57 MB | 20d75831a41a78cf042be5127afb0d7f943f2b831f87a6f6...|
| ml/models/model_comparison_metrics.json               | 5,391 B | 3001c5ed030d3e46771e62b4674a6e3d5d9eed350721010f...|
| ml/candidate_artifacts_v2/deterioration_candidate...  | 156 KB  | ff5b8f3f7528d6a3d3fabaf655f4986526243d7ad3345eaeb...|
| ml/candidate_artifacts_v2/optimization_metrics.json   | 4,673 B | 7c0090a1bc117feab5aacd004e1182144b2d30df84e0486f...|
| ml/research_phase6/phase6_evaluation_results.json    | 65.6 KB | 3b1b99aee854a05f23a93618461129e29fd116dd703b0c3d...|
+-------------------------------------------------------+---------+----------------------------------------------------+
```

### Baseline Immutability Check
All five baseline files in `ml/models/` were compared bit-for-bit against `ml/baseline_artifacts_v1/`:
- `deterioration_logistic_v1.joblib`: **100% IDENTICAL**
- `deterioration_random_forest_v1.joblib`: **100% IDENTICAL**
- `deterioration_gradient_boosting_v1.joblib`: **100% IDENTICAL**
- `anomaly_isolation_forest_v2.joblib`: **100% IDENTICAL**
- `model_comparison_metrics.json`: **100% IDENTICAL**

### Documentation Discrepancy Found in Report
In `PHASE6_ML_EVALUATION_REPORT.md` (lines 11–12), placeholder hashes (`a93766a2...` and `6601b0ea...`) were erroneously recorded. The physical files in `ml/models/` and `ml/candidate_artifacts_v2/` match `26d79474...` and `ff5b8f3f...` exactly.

---

## 9. ML Metric Reconciliation

An in-depth code audit of `ml/optimize_ml_engine.py` and `ml/research_phase6/evaluate_phase6_models.py` resolved the discrepancy between earlier candidate numbers and the latest numbers:

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                            METRIC RECONCILIATION TABLE                                                |
+----------------------+---------------+---------------+------------------+---------------------------------------------+
| Metric Dimension     | Earlier Value | Latest Value  | Reproduced Value | Technical Root Cause / Reconciliation       |
+----------------------+---------------+---------------+------------------+---------------------------------------------+
| Candidate ROC-AUC    | 0.7476        | 0.7373        | 0.7373           | Earlier run used naive shift(1).ffill(23)   |
|                      |               |               |                  | lab join; latest uses exact timestamp as-of |
|                      |               |               |                  | join with 60m lag and 24h staleness cutoff. |
+----------------------+---------------+---------------+------------------+---------------------------------------------+
| Candidate PR-AUC     | 0.0886        | 0.1019        | 0.1019           | Platt sigmoid scaling fitted on OOF logits  |
|                      |               |               |                  | restored probability curvature, raising test|
|                      |               |               |                  | PR-AUC from 0.0886 to 0.1019.               |
+----------------------+---------------+---------------+------------------+---------------------------------------------+
| Candidate Brier Score| 0.017891      | 0.015710      | 0.015710         | OOF Platt scaling reduced probability error;|
|                      |               |               |                  | lower is better (0.017891 -> 0.015710).     |
+----------------------+---------------+---------------+------------------+---------------------------------------------+
| Candidate Threshold  | 0.0550        | 0.0600        | 0.0600           | Multi-objective search on validation set    |
|                      |               |               |                  | shifted operating point to satisfy FAR < 0.5|
+----------------------+---------------+---------------+------------------+---------------------------------------------+
| Baseline Threshold   | ~0.0816       | 0.0823        | 0.0823           | Exact decision threshold serialized in      |
|                      |               |               |                  | deterioration_logistic_v1.joblib artifact.  |
+----------------------+---------------+---------------+------------------+---------------------------------------------+
| Baseline Test ROC-AUC| 0.6875        | 0.6875        | 0.6875           | 100% Verified Identical                     |
+----------------------+---------------+---------------+------------------+---------------------------------------------+
| Baseline Test PR-AUC | 0.1257        | 0.1257        | 0.1257           | 100% Verified Identical                     |
+----------------------+---------------+---------------+------------------+---------------------------------------------+
| Baseline Test Brier  | 0.015443      | 0.015443      | 0.015443         | 100% Verified Identical                     |
+----------------------+---------------+---------------+------------------+---------------------------------------------+
```

**Reconciliation Conclusion:** The latest numbers in `PHASE6_ML_EVALUATION_REPORT.md` represent the **audited, leakage-safe pipeline** incorporating timestamp-based as-of joins and segment-reset rolling std. The earlier values reflected the pre-correction naive shift/ffill pipeline. All latest numbers are **100% verified and reproducible**.

---

## 10. Bootstrap Methodology Audit

### Algorithm Inspection (`ml/research_phase6/evaluate_phase6_models.py:45-98`)
1. **Cluster Granularity:** Resamples `df_split["uniquepid"].unique()` with replacement ($N = 272$ on validation, $N = 273$ on test). All chronological observations belonging to each selected patient are drawn together. This correctly preserves temporal autocorrelation.
2. **Cluster Multiplicity:** Handled appropriately via concatenated index arrays.
3. **Empty Cluster Protection:** Validates `pos_count > 0 and pos_count < len(y_boot)` to prevent undefined ROC-AUC exceptions.
4. **Percentile Computation:** Empirical 2.5th and 97.5th percentiles computed correctly.

### Statistical Sample Size Limitations
- In the validation partition, only **18 ICU stays** (341 hourly observations across 18 patients) deteriorate to discharge expired.
- In the test partition, only **18 ICU stays** (357 hourly observations across 18 patients) deteriorate to discharge expired.
- While 1,000 bootstrap resamples are computationally stable, the effective sample size of positive clusters ($N = 18$) produces wide confidence intervals:
  - Baseline v1 Test PR-AUC: $[0.0534, 0.2245]$
  - Candidate v2 Test PR-AUC: $[0.0448, 0.1848]$
  - Baseline v1 Test PPV: $[0.0812, 0.2959]$
  - Candidate v2 Test PPV: $[0.0549, 0.2033]$
- **Conclusion:** The wide overlap in 95% CIs demonstrates that Candidate v2 does **not** have a statistically significant advantage over Baseline v1 on precision or PR-AUC.

---

## 11. Alert Episode and Lead-Time Audit

The 2-state episode simulator (`cooldown_hours=6`, `reset_hours=2`) was executed independently on the exploratory test set ($N = 21,446$ hours, 359 stays):

```
+----------------------------------------------------------------------------------------------------+
|                                  ALERT EPISODE AUDIT COMPARISON                                    |
+------------------------------------+--------------------------+------------------------------------+
| Metric                             | Baseline v1 (θ = 0.0823) | Candidate v2 (θ = 0.0600)          |
+------------------------------------+--------------------------+------------------------------------+
| True Positive Observations         | 61 hours                 | 77 hours                           |
| False Positive Observations        | 286 hours                | 560 hours (+95.8% increase)        |
| Hourly PPV (Precision)            | 17.58%                   | 12.09% (-5.49% degradation)        |
| Total False Alert Episodes         | 123 episodes             | 166 episodes (+35.0% increase)     |
| Episode FAR per 100 Patient-Hours  | 0.5735 / 100h            | 0.7740 / 100h (+34.9% increase)    |
| False Alarms per True Alarm Ratio  | 4.69 : 1                 | 7.27 : 1 (+55.0% alarm fatigue)    |
| Negative-Stay False Alarm Rate     | 19.35% (66/341 stays)    | 22.29% (76/341 stays)              |
| Stay-Level Sensitivity             | 72.22% (13/18 stays)     | 77.78% (14/18 stays) (+1 stay)     |
| Median Lead Time to Discharge      | 9.0 hours                | 6.5 hours (-2.5 hours warning)     |
+------------------------------------+--------------------------+------------------------------------+
```

### Reporting Discrepancy Found in Phase 6 Report
In `PHASE6_ML_EVALUATION_REPORT.md` (scorecard and text), Baseline v1 stay sensitivity was reported as **14/18 stays (77.78%)**.
Our independent reproduction proves that at Baseline v1's exact default threshold ($\theta = 0.0823$), it detects **13 / 18 stays (72.22%)**. It detects 14/18 stays only at $\theta \le 0.0800$.
Candidate v2 detects **14 / 18 stays (77.78%)** at $\theta = 0.0600$. Candidate v2 detects 1 additional stay, but at the cost of **274 additional false-positive hours** and **2.5 hours shorter lead time**.

---

## 12. Phase 7 Feature Compatibility Findings

All 52 candidate features were audited against live streaming and historical data availability:

```
+----------------------------------------------------------------------------------------------------+
|                                    FEATURE COMPATIBILITY AUDIT                                     |
+----+-----------------------------+---------------------+-------------------+-----------------------+
| ID | Feature Name                | Live Serial Telemetry| Manual Entry      | Runtime Compatibility |
+----+-----------------------------+---------------------+-------------------+-----------------------+
| 1  | heart_rate                  | YES (bpm)           | YES (Vitals)      | Compatible            |
| 2  | spo2                        | YES (spo2)          | YES (Vitals)      | Compatible            |
| 3  | respiration                 | NO (Serial lacks RR)| YES (Vitals)      | Degraded on Serial    |
| 4-6| BP (Sys, Dia, MAP)          | NO (Serial lacks BP)| YES (Vitals)      | Missing on Serial     |
|7-16| 1h / 3h Derived Dynamics    | Partially available | Requires history  | Missing on Serial     |
| 17 | net_fluid_balance_ml        | NO                  | YES (Clinical Rec)| Requires nursing chart|
| 18 | urine_output_ml             | YES (ESP32 load cell| YES (Urine Record)| Compatible            |
|19-20| Infusion / Vasoactive      | NO                  | NO direct table   | Missing in Runtime    |
|21-22| Warning / Severe Counts     | YES (Rule Engine)   | YES (Rule Engine) | Compatible            |
|23-32| Missingness Flags           | Computed            | Computed          | Compatible            |
|33-40| Shock Index, Pulse Press, Std| Partial            | Partial           | Missing on Serial     |
|41-52| 6 Lagged Labs + Indicators  | NO (LIMS not linked)| YES (Blood Test)  | Latency: 45-90 min    |
+----+-----------------------------+---------------------+-------------------+-----------------------+
```

### Critical Architecture Confirmation
- In the current bedside serial monitor implementation (`serial_service.py`), only `fluid_ml`, `bpm`, and `spo2` are transmitted.
- **Over 30 candidate features cannot be populated from serial telemetry.**
- Feeding live serial telemetry into Candidate v2 would cause $60\%+$ feature absence, forcing the model into missingness branches that have not been prospectively evaluated.
- **Auditor Conclusion:** The restriction in `ml_service.py` rejecting live telemetry for Candidate v2 is **technically mandatory and clinically essential**.

---

## 13. Implemented vs Proposed Architecture

To prevent misleading claims of completion, this audit strictly delineates what is implemented in code versus what is proposed in documentation:

| Architectural Component | Status in Codebase | Evidence / File Path |
| :--- | :---: | :--- |
| **Baseline v1 Synchronous Inference** | **IMPLEMENTED** | `backend/services/ml_service.py:predict_deterioration` |
| **Candidate v2 Historical eICU Inference**| **IMPLEMENTED** | `backend/services/ml_service.py:predict_deterioration_candidate` |
| **Live Telemetry Rejection for Candidate v2**| **IMPLEMENTED** | `backend/services/ml_service.py:446` (`UNSUPPORTED_DATA_SOURCE`) |
| **Transparent Fallback to Baseline v1** | **IMPLEMENTED** | `backend/services/ml_service.py:450` (`fallback_applied=True`) |
| **Deterministic Clinical Alert Engine** | **IMPLEMENTED** | `backend/services/clinical_alert_engine.py` (KDIGO, labs, vitals) |
| **2-State Alert Episode Manager** | **IMPLEMENTED** | `backend/services/alert_manager.py` |
| **Silent Candidate Shadow Inference Worker**| **PROPOSED ONLY** | Described in `PHASE7_ML_INTEGRATION_PLAN.md`; no Python code exists. |
| **`ml_shadow_predictions` Audit Table** | **PROPOSED ONLY** | Table schema does not exist in `data/clinical_records.sqlite3`. |
| **Automated Daily Reconciliation Worker** | **PROPOSED ONLY** | Described in `PHASE7_ML_INTEGRATION_PLAN.md`; no cron/worker exists. |
| **Sub-Second Latency Circuit Breaker SLA** | **PROPOSED ONLY** | No SLA monitoring or automated fail-safe circuit breaker in code. |

---

## 14. Confirmed Bugs & Documentation Discrepancies

### Bug 1 (Clinical Logic / Provenance): Polyuria Reference Source
- **Location:** `backend/services/clinical_alert_engine.py:566`
- **Defect:** Attributing polyuria ($> 2.5\text{ mL/kg/h}$) to `"KDIGO Clinical Practice Guideline for AKI (2012)"`.
- **Impact:** Misrepresents clinical guideline provenance in alert metadata.
- **Proposed Correction:** Change `reference_source` to `"General Clinical Nephrology / ICU Monitoring Guideline"`.

### Bug 2 (API Robustness): Malformed Timestamp Causes HTTP 500
- **Location:** `backend/services/patient_service.py:1436` and `backend/main.py:1090`
- **Defect:** A malformed ISO timestamp string (e.g., `"2026-99-99"`) in a telemetry payload causes `_utc_iso` to raise an unhandled `ValueError`, crashing the endpoint with HTTP 500 Internal Server Error instead of returning HTTP 400 or HTTP 422.
- **Proposed Correction:** Wrap `_utc_iso` in a `try/except ValueError` block in `ingest_reading`, raising `HTTPException(status_code=422, detail="Invalid reading_time ISO format.")`.

### Bug 3 (Reporting Discrepancy): Baseline v1 Stay Sensitivity
- **Location:** `PHASE6_ML_EVALUATION_REPORT.md` (Executive Scorecard & Section 4)
- **Defect:** Baseline v1 stay sensitivity was reported as $14 / 18$ ($77.78\%$).
- **Actual Reproduced Value:** At the default threshold ($\theta = 0.0823$), Baseline v1 detects **$13 / 18$ stays ($72.22\%$)**. It detects 14/18 only at $\theta \le 0.0800$.
- **Proposed Correction:** Correct report text to accurately state $13 / 18$ ($72.22\%$).

### Bug 4 (Documentation Discrepancy): Model Artifact Hashes
- **Location:** `PHASE6_ML_EVALUATION_REPORT.md:11-12`
- **Defect:** Placeholder SHA-256 hashes (`a93766a2...` and `6601b0ea...`) were printed instead of physical file hashes (`26d79474...` and `ff5b8f3f...`).
- **Proposed Correction:** Update report with verified physical hashes.

---

## 15. Potential Risks Requiring Investigation

1. **Future Timestamp Ingestion Risk:**
   In `PatientStore.ingest_reading`, there is no validation preventing readings dated in the future. A misconfigured device sending timestamps 24 hours ahead would be accepted, distorting `ORDER BY reading_time DESC` queries for active patients.
2. **Unadjusted Total Calcium Risk:**
   In `clinical_alert_engine.py:231`, total calcium is evaluated without albumin correction. Hypoalbuminemic ICU patients will trigger false hypocalcemia alerts.
3. **eICU `labresultoffset` Semantics:**
   The eICU database provides `labresultoffset` (reporting time), not specimen collection time. Claiming "collection time alignment" is technically imprecise.

---

## 16. Unresolved Limitations

1. **Test Set Prior Exposure:**
   The test split ($N = 21,446$ rows) was exposed in earlier benchmark iterations. It cannot be considered a virgin independent test set. Final validation requires external databases (e.g., MIMIC-IV).
2. **Low Positive Cohort Count:**
   Only 18 deteriorating stays exist in the test partition. This limits statistical power for subgroup analysis.
3. **High-Risk Tail Over-Prediction:**
   Candidate v2 has an absolute calibration error of $0.0195$ in the highest-risk decile (Bin 10), driving elevated false alarms.

---

## 17. Final Acceptance Decision

```
+----------------------------------------------------------------------------------------------------+
|                                    FINAL AUDIT SCORECARD                                           |
+----------------------------------------+-------------------+---------------------------------------+
| Audit Dimension                        | Decision          | Key Evidence / Justification          |
+----------------------------------------+-------------------+---------------------------------------+
| Phase 4: End-to-End Software Validation| PASS              | 185 Pytest passed, 46 Vitest passed,  |
|                                        |                   | 0 FK errors, delayed packets isolated |
+----------------------------------------+-------------------+---------------------------------------+
| Phase 5: Clinical Alert Logic          | PASS WITH         | KDIGO and unit conversions verified;  |
|                                        | LIMITATIONS       | Polyuria reference citation defect    |
+----------------------------------------+-------------------+---------------------------------------+
| Phase 6: Machine Learning Evaluation   | PASS WITH         | Metrics reconciled and reproduced;    |
|                                        | LIMITATIONS       | Candidate v2 PR-AUC degrades (-18.9%) |
+----------------------------------------+-------------------+---------------------------------------+
| Model Promotion Gate                   | BLOCKED / REJECTED| Candidate v2 unpromoted; Baseline v1  |
|                                        |                   | retained as default production model  |
+----------------------------------------+-------------------+---------------------------------------+
| Phase 7: ML Integration Plan           | PASS WITH         | Matrix verified; Shadow Mode flagged  |
|                                        | LIMITATIONS       | as proposed architecture, not code    |
+----------------------------------------+-------------------+---------------------------------------+
| OVERALL AUDIT VERDICT                  | CONDITIONAL PASS  | PROMOTION BLOCKED; PROVENANCE &       |
|                                        |                   | REPORTING CORRECTIONS REQUIRED        |
+----------------------------------------+-------------------+---------------------------------------+
```

---
*Audit completed under strict zero-code-modification policy. Certified by Independent Clinical ML Reviewer.*
