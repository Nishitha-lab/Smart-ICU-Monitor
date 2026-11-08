# Model Compatibility Matrix & Runtime Environment Guide

**Project:** Smart ICU Patient Safety System
**Repository:** `D:\Smart-ICU`
**Branch:** `experiment/ml-candidate-v2`
**Last Updated:** November 2026

---

## 1. Executive Summary

This document formalizes the runtime compatibility matrix, environment alignment standards, empirical verification results, and operational limitations for machine learning model artifacts in the Smart ICU Patient Safety System.

The repository contains three primary serialized model artifacts across two distinct scikit-learn minor versions:
1. **Baseline Deterioration Model v1 (`deterioration_logistic_v1.joblib`)**: Serialized under **`scikit-learn 1.9.0`**. Active default production inference model.
2. **Anomaly Isolation Forest v2 (`anomaly_isolation_forest_v2.joblib`)**: Serialized under **`scikit-learn 1.9.0`**. Active multi-variable outlier detection model.
3. **Candidate Deterioration Model v2 (`deterioration_candidate_v2.joblib`)**: Serialized under **`scikit-learn 1.8.0`**. Unpromoted candidate evaluation artifact.

---

## 2. Model Artifact Compatibility Matrix

| Artifact | Artifact Path | Training `scikit-learn` | Model Type | Deployment Status | Dedicated Env (`backend\.venv`, 1.9.0) | System Python (`1.8.0`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Deterioration v1** | `ml/models/deterioration_logistic_v1.joblib` | **`1.9.0`** | `LogisticRegression` Pipeline | **ACTIVE DEFAULT** | **SUPPORTED** (0 warnings) | **WARNING** (`InconsistentVersionWarning` $\times 4$; numerically equivalent) |
| **Anomaly Isolation Forest v2** | `ml/models/anomaly_isolation_forest_v2.joblib` | **`1.9.0`** | `IsolationForest` Pipeline | **ACTIVE DEFAULT** | **SUPPORTED** (0 warnings) | **WARNING** (`InconsistentVersionWarning` $\times 304$; bit-for-bit identical) |
| **Candidate Deterioration v2** | `ml/candidate_artifacts_v2/deterioration_candidate_v2.joblib` | **`1.8.0`** | `HistGradientBoosting` + Platt | **UNPROMOTED CANDIDATE** | **UNSUPPORTED** (Fails: `ModuleNotFoundError: No module named '_loss'`) | **SUPPORTED** (0 warnings; 13/13 candidate tests pass) |

---

## 3. Environment Bifurcation & Execution Standardization

### 3.1 Two Formal Environments
To maintain complete stability of active production baseline models while enabling reproducible evaluation of candidate models, the repository formalizes two isolated virtual environments:

1. **Main Backend Environment (`scikit-learn 1.9.0`):**
   - **Path:** `D:\Smart-ICU\backend\.venv`
   - **Binary:** `D:\Smart-ICU\backend\.venv\Scripts\python.exe`
   - **Installed Stack:** `scikit-learn==1.9.0`, `fastapi 0.141.1`, `numpy 2.5.3`, `pandas 3.0.5`, `joblib 1.6.0`, `pytest 9.2.0`, `httpx 0.28.1`.
   - **Role:** Main application runtime and full automated backend test suite.
   - **Model Support:** Baseline Logistic Regression v1 and Anomaly Isolation Forest v2 deserialize and infer with **0 warnings**.
   - **Candidate v2 Handling:** Candidate v2 is present on disk but gracefully reported as `UNAVAILABLE_INCOMPATIBLE_RUNTIME`. Explicit fallback to baseline v1 operates when requested.

2. **Dedicated Candidate Evaluation Environment (`scikit-learn 1.8.0`):**
   - **Path:** `D:\Smart-ICU\ml\.venv_candidate`
   - **Binary:** `D:\Smart-ICU\ml\.venv_candidate\Scripts\python.exe`
   - **Installed Stack:** `scikit-learn==1.8.0`, `numpy 2.5.4`, `pandas 3.1.1`, `joblib 1.6.0`, `scipy 1.18.1`, `pytest 9.2.0`.
   - **Role:** Isolated evaluation harness for Candidate v2 without modifying or downgrading the main backend environment.
   - **Model Support:** Candidate v2 deserializes cleanly; all 13 candidate inference tests pass with **100% success**.

### 3.2 Reproducible Setup Instructions for Candidate Environment
If recreating `ml/.venv_candidate` on a clean system:
```powershell
cd D:\Smart-ICU
py -m venv ml\.venv_candidate
.\ml\.venv_candidate\Scripts\Activate.ps1
pip install scikit-learn==1.8.0 "joblib>=1.4" "pandas>=2.2" "numpy>=1.26" scipy pytest
```

### 3.3 Standardized Developer Execution Instructions

> [!IMPORTANT]
> **MANDATORY EXECUTION RULES:**
> 1. Run main backend services and the primary backend test suite in `backend\.venv` (`scikit-learn 1.9.0`).
> 2. Run Candidate v2 evaluation in `ml\.venv_candidate` (`scikit-learn 1.8.0`).
> 3. **DO NOT** execute tests or services using unactivated system Python.

#### Running Main Backend Test Suite (under 1.9.0):
```powershell
cd D:\Smart-ICU
.\backend\.venv\Scripts\Activate.ps1
pytest backend/tests -v -o pythonpath="backend ."
```
*Expected Result:* **220 passed, 7 skipped, 0 failed** (4 subtests passed). Exactly 7 tests that require candidate model deserialization skip with explicit reason.

#### Running Candidate v2 Evaluation Suite (under 1.8.0):
```powershell
cd D:\Smart-ICU
.\ml\.venv_candidate\Scripts\Activate.ps1
pytest backend/tests/test_ml_candidate_inference.py -v -o pythonpath="backend ."
```
*Expected Result:* **13 passed, 2 skipped, 0 failed** (100% of candidate inference tests pass).

#### Running the Backend API:
```powershell
cd D:\Smart-ICU\backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```


---

## 4. Empirical Cross-Version Verification Results

An empirical, side-by-side verification was conducted across both environments on identical multi-variable test inputs:

### 4.1 Isolation Forest v2 (`anomaly_isolation_forest_v2.joblib`)
Evaluated across 4 representative clinical input vectors (normal stable vitals, severe acute deterioration, all-zeros baseline, and extreme multi-channel outlier):

| Test Case | Metric | Dedicated Env (1.9.0) | System Python (1.8.0) | Absolute Difference |
| :--- | :--- | :---: | :---: | :---: |
| **Normal Vitals** | Prediction (`+1` normal, `-1` anomaly) | `1` | `1` | 0 |
| | Sample Score (`score_samples`) | `-0.36171085393449603` | `-0.36171085393449603` | **0.0000000000000000** |
| | Decision Function | `0.17643245066125918` | `0.17643245066125918` | **0.0000000000000000** |
| **Deteriorating** | Prediction | `-1` | `-1` | 0 |
| | Sample Score | `-0.6975149552745036` | `-0.6975149552745036` | **0.0000000000000000** |
| | Decision Function | `-0.15937165067874837` | `-0.15937165067874837` | **0.0000000000000000** |
| **All Zeros** | Prediction | `-1` | `-1` | 0 |
| | Sample Score | `-0.5701301703882463` | `-0.5701301703882463` | **0.0000000000000000** |
| | Decision Function | `-0.031986865792491104` | `-0.031986865792491104` | **0.0000000000000000** |
| **Extreme Outlier** | Prediction | `-1` | `-1` | 0 |
| | Sample Score | `-0.7981994393135773` | `-0.7981994393135773` | **0.0000000000000000** |
| | Decision Function | `-0.2600561347178221` | `-0.2600561347178221` | **0.0000000000000000** |

*Verification Summary:* Outputs are **identical bit-for-bit to 17 decimal places** across all inputs.

### 4.2 Baseline Logistic Regression v1 (`deterioration_logistic_v1.joblib`)

| Test Case | Metric | Dedicated Env (1.9.0) | System Python (1.8.0) | Absolute Difference |
| :--- | :--- | :---: | :---: | :---: |
| Normal Vitals | Prediction / Probability | `0` / `0.004226213853787253` | `0` / `0.004226213853787253` | 0.0 |
| Deteriorating | Prediction / Probability | `0` / `0.11544885070215272` | `0` / `0.11544885070215269` | $\le 3.0 \times 10^{-17}$ |
| All Zeros | Prediction / Probability | `1` / `0.79571317320922000` | `1` / `0.79571317320922000` | 0.0 |
| Extreme Outlier | Prediction / Probability | `1` / `1.00000000000000000` | `1` / `1.00000000000000000` | 0.0 |

*Verification Summary:* Predictions are identical; probabilities match within floating-point epsilon ($\le 3 \times 10^{-17}$), which is governed by low-level BLAS/LAPACK floating-point accumulation.

### 4.3 Candidate Model v2 Deserialization Failure Under 1.9.0
When attempting to load `deterioration_candidate_v2.joblib` under `scikit-learn 1.9.0`:
```text
InconsistentVersionWarning: Trying to unpickle estimator LabelEncoder from version 1.8.0 when using version 1.9.0.
ModuleNotFoundError: No module named '_loss'
```
*Technical Root Cause:* `HistGradientBoostingClassifier` in `scikit-learn 1.8.0` references internal private modules (`_loss`) that were reorganized/refactored in version `1.9.0`. Unlike linear models or tree regressor structures whose pickle format remained backwards-compatible, complex Cython-backed estimators with private class references cannot be unpickled across minor versions.

---

## 5. Upstream Policy and Scientific Caveat

> [!WARNING]
> **NO GUARANTEE OF UNIVERSAL COMPATIBILITY:**
> According to official scikit-learn documentation ([Security & Maintainability Limitations](https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations)):
> *"While models saved with one version of scikit-learn might load in other versions, this is entirely unsupported and not recommended. It should also be kept in mind that operations performed on such estimators could give different and unexpected results."*
>
> Numerical equivalence on selected empirical test inputs does **NOT** guarantee universal equivalence across all possible edge cases, extreme values, or unvisited feature regions.

---

## 6. Model Governance and Promotion Requirements

1. **Baseline Model Preservation:**
   Under strict clinical software governance, existing baseline model artifacts (`deterioration_logistic_v1.joblib` and `anomaly_isolation_forest_v2.joblib`) are permanently preserved. They must **not** be modified, retrained, or replaced.
2. **Warning Visibility Policy:**
   `InconsistentVersionWarning` is intentionally **NOT** globally suppressed. Retaining native warnings ensures transparent auditability whenever an engineer inadvertently runs code in an unpinned or mismatched Python environment.
3. **Future Candidate Promotion Protocol:**
   Before Candidate v2 (or any future candidate) can be formally considered for promotion:
   - It must be regenerated and serialized within the standardized backend environment (`scikit-learn==1.9.0`).
   - All tests in `backend/tests/test_ml_candidate_inference.py` must pass within the dedicated virtual environment.
   - SHA-256 manifest records must be updated accordingly.

---

## 7. Model Artifact Checksum Manifest

All model artifacts are cryptographically tracked. The following SHA-256 hashes must remain invariant:

| File Path | SHA-256 Checksum |
| :--- | :--- |
| `ml/models/deterioration_logistic_v1.joblib` | `26d79474d38521bb251f3c0d456e099e038736c6c2f1bc43e837cfd3827181e0` |
| `ml/models/anomaly_isolation_forest_v2.joblib` | `20d75831a41a78cf042be5127afb0d7f943f2b831f87a6f69914381a185c266e` |
| `ml/candidate_artifacts_v2/deterioration_candidate_v2.joblib` | `ff5b8f3f7528d6a3d3fabaf655f4986526243d7ad3345eaeba7d03146e17de62` |
| `ml/baseline_artifacts_v1/deterioration_logistic_v1.joblib` | `26d79474d38521bb251f3c0d456e099e038736c6c2f1bc43e837cfd3827181e0` |
| `ml/baseline_artifacts_v1/anomaly_isolation_forest_v2.joblib` | `20d75831a41a78cf042be5127afb0d7f943f2b831f87a6f69914381a185c266e` |
| `ml/baseline_artifacts_v1/deterioration_gradient_boosting_v1.joblib` | `2deacbbb4a429746055b672256a09f7a092822eed5b6af39bfed11102cffaf02` |
| `ml/baseline_artifacts_v1/deterioration_random_forest_v1.joblib` | `b470d94c9aea9183a6be9362ea492c89886168b0c5016e56ae8f4a974c4d812c` |
| `ml/baseline_artifacts_v1/model_comparison_metrics.json` | `3001c5ed030d3e46771e62b4674a6e3d5d9eed350721010f7e6e17c79324f02a` |
