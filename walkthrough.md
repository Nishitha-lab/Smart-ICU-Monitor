# Walkthrough — Genuine Machine Learning Prediction Engine for Smart ICU

## Overview

The Smart ICU Patient Safety System previously operated with machine learning in an `UNAVAILABLE` state due to missing model artifacts and safety restrictions against fabricating unvalidated predictions.

We have engineered, evaluated, and deployed a genuine, leakage-safe, empirically trained Machine Learning Prediction Engine without mocks, simulated probabilities, or fabricated risk scores.

---

## 1. Prediction Design & Methodology

We created and executed [`ml/train_ml_engine.py`](file:///d:/Smart-ICU/ml/train_ml_engine.py), training supervised classifiers for ICU discharge marked expired within 24 hours and an unsupervised multi-variable anomaly detector on the eICU Collaborative Research Database demo cohort.

- **Prediction Design:** Retrospective rolling 24-hour prediction model. At each observation timestamp $t$, the model uses features available at or before $t$ to estimate whether ICU discharge marked expired will occur within the next 24 hours. The 24-hour prediction horizon is strictly distinguished from the feature-observation period (accumulated history $\le t$).
- **Target Definition:** Strictly `"ICU discharge marked expired within the next 24 hours from the prediction timestamp"`, derived from eICU `patient.csv.gz` (`unitdischargestatus == "Expired"` and $0 \le \text{hours\_to\_discharge} \le 24$). General in-hospital mortality is not assumed.
- **Patient-Level Leakage Prevention:** Partitioned strictly by natural patient identifier (`uniquepid`) with zero patient or stay overlap:
  - **Train:** 1,271 patients · 96,731 patient-hours · 1,461 positive events (1.51%)
  - **Validation:** 272 patients · 21,255 patient-hours · 341 positive events (1.60%)
  - **Holdout Test:** 273 patients · 21,446 patient-hours · 357 positive events (1.66%)
- **Predictor Set:** Exactly 22 physiological and clinical features (instantaneous vitals, 1h deltas, 3h rolling averages, fluid balance, and hourly rule flags; all outcome, label, and identifier columns strictly excluded).

---

## 2. Empirical Holdout Test Results & Calibration

### Hourly Observation Metrics (Held-Out Test Set: 21,446 observations, 273 patients)

| Model Architecture | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier Score | Decision Threshold ($\theta^*$) | Sensitivity (Recall) | Specificity | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForestClassifier** | **0.7252** `[0.64, 0.81]` | **0.1315** `[0.06, 0.23]` | 0.0154 | 0.0726 | 18.21% | 98.58% | 0.1803 |
| **GradientBoostingClassifier** | 0.7084 | 0.1167 | 0.0159 | 0.0912 | 13.73% | 99.21% | 0.1710 |
| **LogisticRegression (Deployed `v1`)** | **0.6875** `[0.57, 0.80]` | **0.1257** `[0.05, 0.23]` | 0.0154 | 0.0823 | 17.09% | 98.64% | 0.1733 |

- **Calibration Reality:** The deployed model is **Unweighted Logistic Regression operating at natural training prevalence** (~1.5-1.6%).
  - Calibration Intercept: `-0.4605` (ideal 0.0)
  - Calibration Slope: `0.8360` (ideal 1.0)
  - Model Brier Score: `0.015443` vs Prevalence-Only Baseline Brier Score: `0.016369`. The low Brier score is primarily driven by the 98.34% negative class prevalence and does not alone prove good calibration.
- **Hourly Confusion Matrix (Logistic Regression):** $\text{TN}=20,803, \text{FP}=286, \text{FN}=296, \text{TP}=61$.
- **False-Positive Clarification:** The 286 false-positive hourly observations occur across 69 unique ICU stays, representing mathematical threshold crossings on isolated hourly rows, not 286 independent clinical alarms or pager alerts.
- **Patient/Stay-Level Event Detection:**
  - Unique positive ICU stays in test set: **18**
  - Unique positive ICU stays flagged at least once within 24 hours prior to ICU discharge marked expired: **13**
  - Patient/stay-level sensitivity: **72.22%** (13 / 18)
  - Detection lead-time distribution: Median **9.00 hours**, Mean **10.15 hours** (min 0.0h, max 23.0h, IQR: 2.0h - 19.0h) before final observation.
  - Positive ICU stays never flagged: **5** (Stays: 176730, 827085, 1073098, 1109136, 2901384).

---

## 3. Serialized Artifacts Manifest

The five current model artifacts were hashed and their file sizes and SHA-256 checksums recorded. These values match the current release manifest. Historical identity with an earlier trained baseline could not be independently established because the artifacts are not present in the available Git history.

Saved in [`ml/models/`](file:///d:/Smart-ICU/ml/models):
- `deterioration_logistic_v1.joblib` (5,604 bytes | SHA-256: `26d79474d38521bb251f3c0d456e099e038736c6c2f1bc43e837cfd3827181e0`) — Fitted `Pipeline([imputer, scaler, model])`
- `deterioration_random_forest_v1.joblib` (1,562,149 bytes | SHA-256: `b470d94c9aea9183a6be9362ea492c89886168b0c5016e56ae8f4a974c4d812c`) — Fitted `Pipeline([imputer, scaler, model])`
- `deterioration_gradient_boosting_v1.joblib` (238,262 bytes | SHA-256: `2deacbbb4a429746055b672256a09f7a092822eed5b6af39bfed11102cffaf02`) — Fitted `Pipeline([imputer, scaler, model])`
- `anomaly_isolation_forest_v2.joblib` (2,570,072 bytes | SHA-256: `20d75831a41a78cf042be5127afb0d7f943f2b831f87a6f69914381a185c266e`) — Fitted `Pipeline([imputer, scaler, model])` (300 trees, 2% contamination)
- `model_comparison_metrics.json` (5,391 bytes | SHA-256: `3001c5ed030d3e46771e62b4674a6e3d5d9eed350721010f7e6e17c79324f02a`) — Detailed evaluation metrics across all splits and architectures.

---

## 4. Backend Service & API Endpoints

- **[`backend/services/ml_service.py`](file:///d:/Smart-ICU/backend/services/ml_service.py)**:
  - `get_status()`: Inspects models on disk and returns `ONLINE` status, detailed validation metrics (ROC-AUC, PR-AUC, Brier score, threshold), training sample counts, and provenance metadata.
  - `predict_deterioration()`: Safe inference interface calculating probabilities, experimental risk bands (`Low`, `Moderate`, `High`, `Extreme`), decision threshold comparison (`0.0823`), model intercept, predicted log-odds, and top 6 feature contributions.
  - `score_anomalies()`: Unsupervised anomaly scoring via IsolationForest v2.
- **[`backend/main.py`](file:///d:/Smart-ICU/backend/main.py)**:
  - `GET /api/ml/status`: Exposes status and validation metrics.
  - `GET /api/ml/predict/{stay_id}`: Executes genuine ML prediction on an ICU stay, returning probabilities, experimental risk band, decision threshold status, log-odds feature contributions, anomaly flags, and academic prototype notice.

---

## 5. Frontend Integration & UI

- **[`frontend/src/App.tsx`](file:///d:/Smart-ICU/frontend/src/App.tsx)**:
  - **ML Online Banner:** Displays model name (`deterioration_logistic_v1`), version (`1.0.0`), unweighted natural prevalence specification, target, validation metrics (ROC-AUC `0.6875`, PR-AUC `0.1257`, Brier score `0.0154`, decision threshold `0.0823`), and academic prototype disclaimer.
  - **Patient Detail Attribution:** In the patient detail modal, fetches `/api/ml/predict/{stay_id}` and renders:
    - Estimated probability of ICU discharge marked expired within the next 24 hours and Experimental Risk Band.
    - Decision threshold status (`Threshold Exceeded` vs `Nominal`).
    - Top Contributing Features table with observed values, model weights, log-odds contributions, and directional badges (▲ Escalates Risk vs ▼ Mitigates Risk).
    - Academic prototype notice clarifying that risk bands are experimental presentation categories, not validated clinical severity classifications.
- **[`frontend/src/App.css`](file:///d:/Smart-ICU/frontend/src/App.css)**: Clean styling for `.ml-status-banner-online`, `.ml-model-card`, `.ml-feature-table`, `.impact-badge`, and research pills.

---

## 6. Verification Results

### Backend Automated Tests
Ran full test suite (`$env:PYTHONPATH="." ; & .venv\Scripts\pytest tests -v`):
- **154 passed, 4 subtests passed in 48.48s** (100% pass rate).
- Test module [`backend/tests/test_ml_engine.py`](file:///d:/Smart-ICU/backend/tests/test_ml_engine.py): 7 tests covering model loading, probability bounds `[0.0, 1.0]`, feature attribution math, log-odds reconstruction, anomaly scoring, `/api/ml/predict/{stay_id}` endpoint, 404 handling, and determinism.
- Test module [`backend/tests/test_ml_safety.py`](file:///d:/Smart-ICU/backend/tests/test_ml_safety.py): 5 tests verifying online status with real models and strict unavailable fallback without fabrication when models are missing.

### Frontend Automated Tests & Build
Ran frontend validation pipeline:
- **Vitest:** 6 test files, **46 passed** (including tests for ML Online banner and patient detail explainability).
- **ESLint:** 0 errors, 0 warnings.
- **Vite Build:** `tsc -b && vite build` succeeded in 178ms.
- **Git diff hygiene:** `git diff --check` passed. Changes remain uncommitted and unstaged for final review.
