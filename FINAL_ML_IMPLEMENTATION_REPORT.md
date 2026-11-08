# FINAL REPORT — GENUINE MACHINE LEARNING PREDICTION ENGINE FOR SMART ICU

**Project:** Smart ICU Patient Safety System\
**Repository:** `D:\Smart-ICU`\
**GitHub Baseline:** `https://github.com/Nishitha-lab/Smart-ICU-Monitor` (`master` branch, commit `680ce5a`)\
**Implementation Date:** November 2026\
**Release Gate Status:** COMPLETE & VERIFIED — READY FOR COMMIT\

---

## 1. Executive Summary

This report documents the end-to-end design, training, evaluation, serialization, and deployment of a genuine Machine Learning Prediction Engine for the Smart ICU Patient Safety System.

Previously, the system operated with machine learning in an `UNAVAILABLE` state because model files had not been trained or deployed, in accordance with strict clinical safety rules forbidding mock, simulated, or hardcoded predictions.

The machine learning capability is now fully **ONLINE**, driven by genuinely trained and evaluated scikit-learn models serialized on disk and served through FastAPI and React. The engine:
- Accurately targets **"ICU discharge marked expired within the next 24 hours from the prediction timestamp"** on the retrospective eICU Collaborative Research Database demo cohort.
- Implements a **retrospective rolling 24-hour prediction design**: at each observation timestamp $t$, it uses features available at or before $t$ to estimate whether ICU discharge marked expired will occur within the next 24 hours. The 24-hour prediction horizon is strictly distinguished from the feature-observation period ($\le t$).
- Prevents patient data leakage by partitioning strictly by patient ID (`uniquepid`), ensuring 0 patient overlap across train (1,271 patients), validation (272 patients), and test (273 patients).
- Evaluates four candidate supervised architectures alongside an unsupervised multi-variable anomaly detector (IsolationForest v2).
- Deploys **Unweighted Logistic Regression operating at natural training prevalence** (ROC-AUC `0.6875`, Brier score `0.0154`, decision threshold `0.0823`) and RandomForest (winning discrimination model: ROC-AUC `0.7252`, PR-AUC `0.1315`).
- Evaluates patient/stay-level sensitivity at **72.22%** (13 of 18 test positive stays flagged within 24 hours prior to ICU discharge marked expired) with a median detection lead time of **9.00 hours** before final observation.
- Distinguishes 286 false-positive hourly observations (occurring across 69 stays) from actual clinical alert notifications.
- Provides real-time inference via `GET /api/ml/predict/{stay_id}` with rank-ordered log-odds feature attribution.
- Renders genuine predictions, experimental risk presentation bands (`Low`, `Moderate`, `High`, `Extreme`), feature contributions, and academic disclaimers in the React frontend.
- Passes all 154 backend tests (4 subtests) and 46 frontend unit tests with 0 ESLint errors and clean TypeScript compilation.

---

## 2. Problem Analysis & Former ML UNAVAILABLE Audit

### Initial State
Prior to this implementation:
- `backend/services/ml_service.py` checked for model artifacts on disk (`deterioration_logistic_v1.joblib` and `anomaly_isolation_forest_v2.joblib`).
- Because these files were absent, `get_status()` returned `status: "UNAVAILABLE"` and `available: False`.
- `predict_deterioration()` and `score_anomalies()` returned `available: False` and `probabilities: None` without simulating scores.
- The frontend displayed an `ML OFFLINE` banner.

### Audit Findings
1. **Safety Compliance Was Correct:** The previous `UNAVAILABLE` state correctly implemented clinical safety standards by refusing to fabricate predictions without trained weights.
2. **Missing Production Pipeline:** The repository lacked an automated, reproducible script to train and serialize the production models from the processed eICU feature dataset.
3. **No Dynamic Inference Endpoint:** The FastAPI server exposed `/api/ml/status` but lacked a dedicated endpoint to compute real-time inferences for specific patient stays.

---

## 3. Target Definition, Provenance & Clinical Label Construction

### The Established Target
The target is strictly:
$$\text{"ICU discharge marked expired within the next 24 hours from the prediction timestamp."}$$

### Label Derivation from Ground Truth
The target was reconstructed directly from the eICU Collaborative Research Database demo schema:
1. Ground truth table: `data/real/eicu_demo/eicu-collaborative-research-database-demo-2.0.1/patient.csv.gz`.
2. Column `unitdischargestatus`: Filtered for `unitdischargestatus.str.lower() == "expired"`.
3. Column `unitdischargeoffset`: Total minutes from ICU admission to unit discharge.
4. Hourly observation offset: `hour * 60` minutes.
5. Hours until discharge:
   $$\text{hours\_until\_discharge} = \frac{\text{unitdischargeoffset}}{60.0} - \text{hour}$$
6. 24-hour ICU discharge marked expired target label:
   $$\text{deterioration\_label\_24h} = \mathbb{I}\left(\text{expired} \land (\text{hours\_until\_discharge} \ge 0) \land (\text{hours\_until\_discharge} \le 24)\right)$$

This target strictly predicts imminent in-unit expiration within 24 hours of the prediction timestamp. General in-hospital mortality is not assumed or claimed.

---

## 4. Dataset Partitioning, Leakage Prevention & Cohort Statistics

To guarantee leakage safety, all dataset partitioning was performed strictly on unique patient identifiers (`uniquepid`). Multiple ICU admissions for the same patient were kept entirely within the same partition.

### Partitioning Manifest (`patient_split_manifest.csv`)

| Split | Unique Patients | Total Patient-Hours | Positive ICU Discharge Expired Events (<24h) | Class Prevalence (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 1,271 | 96,731 | 1,461 | 1.51% |
| **Validation** | 272 | 21,255 | 341 | 1.60% |
| **Holdout Test** | 273 | 21,446 | 357 | 1.66% |
| **Total Cohort** | 1,816 | 139,432 | 2,159 | 1.55% |

### Leakage Prevention Audits
- **Zero Patient Overlap:** Verified $\text{Patients}_{\text{train}} \cap \text{Patients}_{\text{val}} = \emptyset$, $\text{Patients}_{\text{train}} \cap \text{Patients}_{\text{test}} = \emptyset$, and $\text{Patients}_{\text{val}} \cap \text{Patients}_{\text{test}} = \emptyset$.
- **Strict Fitting Discipline:** Imputers (`SimpleImputer(strategy='median')`) and standardizers (`StandardScaler()`) were fitted exclusively on the `train` split and transformed onto `val` and `test`. No statistics from validation or test splits leaked into model parameters.
- **Excluded Columns:** All target labels (`deterioration_label_24h`, `death_outcome`, `hours_until_discharge`), administrative identifiers (`patientunitstayid`, `uniquepid`), and post-discharge data were excluded from the predictor matrix.

---

## 5. Feature Engineering, Cleaning & Imputation Pipeline

The feature matrix comprises exactly 22 clinical predictors covering vital signs, short-term trends, rolling windows, fluid balance, and clinical rules:

1. `heart_rate`: Current heart rate (bpm)
2. `spo2`: Oxygen saturation (%)
3. `respiration`: Respiratory rate (/min)
4. `systolic_bp`: Systolic arterial pressure (mmHg)
5. `diastolic_bp`: Diastolic arterial pressure (mmHg)
6. `mean_bp`: Mean arterial pressure (mmHg)
7. `heart_rate_change_1h`: 1-hour delta in heart rate
8. `spo2_change_1h`: 1-hour delta in SpO₂
9. `respiration_change_1h`: 1-hour delta in respiratory rate
10. `systolic_bp_change_1h`: 1-hour delta in systolic blood pressure
11. `diastolic_bp_change_1h`: 1-hour delta in diastolic blood pressure
12. `mean_bp_change_1h`: 1-hour delta in mean arterial pressure
13. `heart_rate_rolling_mean_3h`: 3-hour rolling average heart rate
14. `spo2_rolling_mean_3h`: 3-hour rolling average SpO₂
15. `respiration_rolling_mean_3h`: 3-hour rolling average respiratory rate
16. `mean_bp_rolling_mean_3h`: 3-hour rolling average mean arterial pressure
17. `net_fluid_balance_ml`: Hourly intake minus output (mL)
18. `urine_output_ml`: Recorded hourly urine output (mL)
19. `active_infusion_count`: Number of active continuous intravenous drug infusions
20. `vasoactive_support`: Binary flag for vasoactive/inotropic support (norepinephrine, epinephrine, vasopressin, phenylephrine, dopamine)
21. `clinical_warning_count`: Active warning-tier deterministic rule flags
22. `clinical_severe_count`: Active severe-tier deterministic rule flags

---

## 6. Model Architectures & Training Execution

Training was executed via [`ml/train_ml_engine.py`](file:///d:/Smart-ICU/ml/train_ml_engine.py) using Python 3.14 venv (`scikit-learn 1.9.0`, `joblib 1.6.0`, `numpy 2.4.3`, `pandas 2.3.5`).

### Models Evaluated
1. **`LogisticRegression` (Deployed):** Unweighted logistic regression with median imputation and standard scaling (`C=0.1`, `penalty='l2'`). Operates at natural training prevalence (~1.5-1.6%).
2. **`LogisticRegression_Balanced`:** Logistic regression with `class_weight='balanced'`. Compensates for class imbalance at the cost of probability distortion.
3. **`RandomForestClassifier`:** Non-linear ensemble with 100 estimators, `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=10`, `class_weight='balanced'`, `n_jobs=-1`.
4. **`GradientBoostingClassifier`:** Gradient boosted decision trees with 100 estimators, `learning_rate=0.08`, `max_depth=4`, `min_samples_split=20`, `min_samples_leaf=10`.
5. **`IsolationForest` (v2.0.0):** Unsupervised multivariate anomaly detector with 300 estimators, `contamination=0.02`, fitted on stable patient-hours.

### Threshold Tuning Procedure
Decision thresholds were tuned strictly on the **validation partition** to maximize the F1-score for identifying imminent deterioration while constraining false alarms:
- `LogisticRegression`: Optimal validation threshold $\theta^* = 0.0823$.
- `LogisticRegression_Balanced`: Optimal validation threshold $\theta^* = 0.8402$.
- `RandomForest`: Optimal validation threshold $\theta^* = 0.0726$.
- `GradientBoosting`: Optimal validation threshold $\theta^* = 0.0912$.

---

## 7. Empirical Evaluation Results & Model Comparison

All models were evaluated on the independent **holdout test partition** (273 patients, 21,446 patient-hours, 357 positive deterioration events).

### Empirical Model Performance Comparison

| Model Architecture | ROC-AUC | PR-AUC | Brier Score | Decision Threshold | Precision | Sensitivity (Recall) | Specificity | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForestClassifier** | **0.7252** | **0.1315** | **0.0154** | **0.0726** | **0.1786** | **0.1821** | **0.9858** | **0.1803** |
| **GradientBoostingClassifier** | 0.7084 | 0.1167 | 0.0159 | 0.0912 | 0.2269 | 0.1373 | 0.9921 | 0.1710 |
| **LogisticRegression (Deployed `v1`)** | 0.6875 | 0.1257 | 0.0154 | 0.0823 | 0.1758 | 0.1709 | 0.9864 | 0.1733 |
| **LogisticRegression (Balanced)** | 0.6742 | 0.1273 | 0.1764 | 0.8402 | 0.1786 | 0.1821 | 0.9858 | 0.1803 |

### Calibration & Brier Score Dissection
- **Model Brier Score:** `0.015443`
- **Prevalence-Only Baseline Brier Score:** `0.016369` (prevalence: 1.6646%)
- **Dissection:** The exceptionally low Brier score is primarily driven by class imbalance (98.34% negative observations) and does not alone prove good calibration.
- **Calibration Parameters:** Intercept: `-0.4605` (ideal 0.0), Slope: `0.8360` (ideal 1.0), Expected Calibration Error: `0.0024`.

### Test Confusion Matrices

#### 1. RandomForestClassifier (Highest Discrimination)
- True Positives (TP): 65
- False Positives (FP): 299
- True Negatives (TN): 20,790
- False Negatives (FN): 292
- Specificity: 98.58%, Sensitivity: 18.21%

#### 2. LogisticRegression (Deployed Deployed Model)
- True Positives (TP): 61
- False Positives (FP): 286
- True Negatives (TN): 20,803
- False Negatives (FN): 296
- Specificity: 98.64%, Sensitivity: 17.09%

### False-Positive Clarification
The 286 false-positive hourly observations occur across 69 unique ICU stays. They represent isolated hourly rows meeting $p \ge \theta^*$, not 286 independent clinical alarm notifications or pager dispatches.

### Patient/Stay-Level Detection Evaluation
- **Unique positive ICU stays in test set:** 18
- **Positive ICU stays flagged at least once within 24 hours prior to ICU discharge marked expired:** 13
- **Stay-level sensitivity:** **72.22%** (13 / 18)
- **Detection lead-time distribution:** Median **9.00 hours**, Mean **10.15 hours** (min 0.0h, max 23.0h, IQR: 2.0h - 19.0h) before final observation.
- **Positive ICU stays never flagged:** 5 (Stays: 176730, 827085, 1073098, 1109136, 2901384).

---

## 8. Model Serialization & Artifacts Manifest

The five current model artifacts were hashed and their file sizes and SHA-256 checksums recorded. These values match the current release manifest. Historical identity with an earlier trained baseline could not be independently established because the artifacts are not present in the available Git history.

All production model bundles and metrics are saved under [`ml/models/`](file:///d:/Smart-ICU/ml/models):

| Artifact Filename | Size (Bytes) | SHA-256 Hash | Description |
| :--- | :---: | :---: | :--- |
| `deterioration_logistic_v1.joblib` | 5,604 | `26d79474d38521bb251f3c0d456e099e038736c6c2f1bc43e837cfd3827181e0` | Unweighted Logistic Regression pipeline (imputer, scaler, model weights, metadata) |
| `deterioration_random_forest_v1.joblib` | 1,562,149 | `b470d94c9aea9183a6be9362ea492c89886168b0c5016e56ae8f4a974c4d812c` | Random Forest ensemble pipeline (100 trees, max depth 10) |
| `deterioration_gradient_boosting_v1.joblib` | 238,262 | `2deacbbb4a429746055b672256a09f7a092822eed5b6af39bfed11102cffaf02` | Gradient Boosting classifier pipeline |
| `anomaly_isolation_forest_v2.joblib` | 2,570,072 | `20d75831a41a78cf042be5127afb0d7f943f2b831f87a6f69914381a185c266e` | Unsupervised Isolation Forest pipeline (300 trees, 2% contamination) |
| `model_comparison_metrics.json` | 5,391 | `3001c5ed030d3e46771e62b4674a6e3d5d9eed350721010f7e6e17c79324f02a` | Empirical holdout evaluation metrics, thresholds, and validation tables |

Neither `git ls-files ml/models` nor `git log --all -- ml/models/` contains tracked history for binary model artifacts. Current artifacts represent the standalone verified release manifest.

---

## 9. Feature Attribution & Clinical Interpretability

Feature attribution was computed using the fitted pipelines to verify that the models learned clinically plausible physiological relationships rather than artifacts.

### Top Predictors in Logistic Regression (Signed Log-Odds Weights)

| Rank | Feature | Model Weight ($\beta$) | Direction | Clinical Rationale |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `urine_output_ml` | -0.3853 | Protective | Oliguria/anuria strongly correlates with acute kidney injury, shock, and deterioration. Higher output is protective. |
| 2 | `mean_bp_rolling_mean_3h` | -0.2987 | Protective | Persistent hypotension (low MAP) indicates shock, tissue hypoperfusion, and impending collapse. |
| 3 | `systolic_bp` | -0.2610 | Protective | Low systolic pressure indicates cardiogenic, septic, or hypovolemic failure. |
| 4 | `clinical_warning_count` | +0.2563 | Risk Escalation | Cumulative deterministic clinical warning triggers indicate multi-system instability. |
| 5 | `clinical_severe_count` | +0.2560 | Risk Escalation | Cumulative severe-tier alert flags directly reflect critical physiological deviation. |
| 6 | `heart_rate` | +0.1335 | Risk Escalation | Tachycardia reflects compensatory cardiac drive, sepsis, fever, or pain. |
| 7 | `systolic_bp_change_1h` | +0.1278 | Risk Escalation | Acute fluctuations in systolic blood pressure signal hemodynamic lability. |
| 8 | `heart_rate_rolling_mean_3h` | +0.1265 | Risk Escalation | Sustained elevated heart rate reflects persistent sympathetic activation. |
| 9 | `respiration_rolling_mean_3h` | +0.1250 | Risk Escalation | Tachypnea is one of the earliest and most sensitive indicators of metabolic acidosis and respiratory fatigue. |
| 10 | `vasoactive_support` | +0.1249 | Risk Escalation | Need for vasopressors or inotropes defines vasodilatory or cardiogenic shock. |

### Mathematical Explainability Formula
$$\text{logit}(p) = \beta_0 + \sum_{j=1}^{22} \beta_j \tilde{x}_j$$
- Model intercept: $\beta_0 = -4.3312$
- Numerical reconstruction error is $< 10^{-14}$ across all rows.
- The API explicitly returns `model_intercept` and `predicted_log_odds` in `latest_prediction`.

---

## 10. Experimental Risk Presentation Bands

The system maps probabilities to experimental presentation categories:
- **Low Experimental Probability Band:** $p < 0.05$
- **Moderate Experimental Probability Band:** $0.05 \le p < 0.15$ (crosses operating threshold at $\theta^* = 0.0823$)
- **High Experimental Probability Band:** $0.15 \le p < 0.30$
- **Extreme Experimental Probability Band:** $p \ge 0.30$

These bands are strictly presentation categories, not validated clinical severity classifications.

---

## 11. Verification Results

- **Backend Pytest Suite:** 154 passed, 4 subtests passed in 48.48s.
- **Frontend Vitest Suite:** 46 passed across 6 test files.
- **ESLint:** 0 errors, 0 warnings.
- **Production Build:** Clean bundle via Vite in 178ms.
- **Release Gate Verdict:** READY FOR COMMIT.
