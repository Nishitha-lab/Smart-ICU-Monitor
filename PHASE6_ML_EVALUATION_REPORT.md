# SMART ICU PATIENT SAFETY SYSTEM
# PHASE 6: MACHINE LEARNING MODEL EVALUATION & RESEARCH REPORT

**Document ID:** `PHASE6_ML_EVALUATION_REPORT.md`
**Evaluation Date:** November 9, 2026
**Auditor / Researcher:** Senior Clinical ML Engineer & Biostatistician
**Git Branch:** `experiment/ml-candidate-v2`
**Current Baseline Commit:** `767f3b88d434142d2ffd200671a306442ef5a4e7`
**Canonical Prediction Target:** *"ICU discharge marked expired within the next 24 hours from the prediction timestamp"*
**Primary Artifacts Evaluated:**
- **Baseline v1 (Active Default in Academic Prototype):** `ml/models/deterioration_logistic_v1.joblib` (SHA-256: `26d79474d38521bb251f3c0d456e099e038736c6c2f1bc43e837cfd3827181e0`, 5,604 bytes, 22 features, unweighted Logistic Regression)
- **Candidate v2 (Unpromoted Evaluation Candidate):** `ml/candidate_artifacts_v2/deterioration_candidate_v2.joblib` (SHA-256: `ff5b8f3f7528d6a3d3fabaf655f4986526243d7ad3345eaeba7d03146e17de62`, 156,185 bytes, 52 features, HistGradientBoosting with Platt Sigmoid Calibration)



---

## 1. Executive Summary & Research Context

This Phase 6 research evaluation investigates the statistical performance, clinical utility, calibration reliability, and alert burden trade-offs of the prospective model upgrade from **Baseline v1** (a 22-feature unweighted Logistic Regression model) to **Candidate v2** (a 52-feature HistGradientBoosting model incorporating leakage-safe lagged laboratory values and missingness dynamics).

All evaluations were executed via `ml/research_phase6/evaluate_phase6_models.py` using strictly patient-partitioned cohorts from the eICU Collaborative Research Database demo (total $N = 139,432$ hourly observations across 1,816 unique patients):
- **Training Cohort:** 96,731 patient-hours, 1,271 patients, 1,461 positive events (prevalence 1.51%).
- **Validation Cohort:** 21,255 patient-hours, 272 patients, 364 ICU stays, 341 positive events (prevalence 1.60%, 18 deteriorating stays).
- **Exploratory Test Benchmark:** 21,446 patient-hours, 273 patients, 359 ICU stays, 357 positive events (prevalence 1.66%, 18 deteriorating stays).

### Key Empirical Findings

```
+----------------------------------------------------------------------------------------------------+
|                                    PHASE 6 SUMMARY SCORECARD                                       |
+------------------------------+---------------------------+-----------------------------------------+
| Metric                       | Baseline v1 (Logistic)    | Candidate v2 (HistGradientBoosting)     |
+------------------------------+---------------------------+-----------------------------------------+
| Architecture                 | SimpleImputer + Scaler +  | SimpleImputer + Scaler +                |
|                              | Logistic Regression (L2)  | HistGradientBoosting + Platt Sigmoid    |
| Feature Count                | 22 features               | 52 features (Set C: Labs + Dynamics)    |
| Default Decision Threshold   | θ* = 0.0823               | θ* = 0.0600                             |
+------------------------------+---------------------------+-----------------------------------------+
| VALIDATION COHORT:           |                           |                                         |
|   ROC-AUC                    | 0.7811 [0.6784, 0.8690]   | 0.8501 [0.7762, 0.9147] (+0.0690)       |
|   PR-AUC                     | 0.1112 [0.0490, 0.2337]   | 0.1789 [0.0953, 0.2756] (+0.0677)       |
|   Brier Score                | 0.015139                  | 0.014575 (-0.000564, better)            |
|   Hourly PPV (Precision)     | 18.55% [7.36%, 32.53%]    | 20.76% [10.22%, 32.27%] (+2.21%)        |
|   Stay Sensitivity           | 94.44% (17/18 stays)      | 94.44% (17/18 stays) (Identical)        |
|   Episode FAR / 100 pt-hours | 0.9221 (at θ=0.05)        | 0.4611 (at θ=0.06) (50% reduction)      |
+------------------------------+---------------------------+-----------------------------------------+
| EXPLORATORY TEST BENCHMARK:  |                           |                                         |
|   ROC-AUC                    | 0.6875 [0.5725, 0.8059]   | 0.7373 [0.6187, 0.8395] (+0.0498)       |
|   PR-AUC                     | 0.1257 [0.0534, 0.2245]   | 0.1019 [0.0448, 0.1848] (-0.0238, WORSE)|
|   Brier Score                | 0.015443                  | 0.015710 (+0.000267, WORSE)             |
|   Hourly PPV (Precision)     | 17.58% [8.12%, 29.59%]    | 12.09% [5.49%, 20.33%] (-5.49%, WORSE)  |
|   False Positive Hours       | 286                       | 560 (+95.8% increase in false alarms)   |
|   False Alarms per True Alarm| 4.69 : 1                  | 7.27 : 1 (+55% alarm fatigue increase)  |
|   Stay Sensitivity           | 72.22% (13/18 stays)      | 77.78% (14/18 stays) (+1 stay, +5.56%)  |
|   Median Lead Time           | 9.0 hours                 | 6.5 hours (-2.5 hours warning)          |
+------------------------------+---------------------------+-----------------------------------------+
```

### Recommendation & Deployment Verdict
> [!IMPORTANT]
> **GOVERNANCE DECISION: CANDIDATE V2 MUST REMAIN AN UNPROMOTED CANDIDATE. BASELINE V1 REMAINS THE ACTIVE DEFAULT MODEL IN THE ACADEMIC PROTOTYPE.**

>
> Although Candidate v2 achieves a superior ROC-AUC on both validation (0.8501 vs 0.7811) and test (0.7373 vs 0.6875), it **fails** to demonstrate real-world superiority on the held-out exploratory test benchmark:
> 1. **PR-AUC regression:** Test PR-AUC drops from 0.1257 to 0.1019 (-18.9%).
> 2. **Precision collapse:** Hourly PPV drops from 17.58% down to 12.09%, requiring nurses to endure 7.27 false alarms for every true positive alarm (compared to 4.69 for v1).
> 3. **Marginal stay-level detection gain at severe alarm penalty:** Candidate v2 detects 14 of 18 stays (77.78%) at $\theta = 0.0600$ vs 13 of 18 stays (72.22%) for Baseline v1 at $\theta = 0.0823$ (+1 stay detected), but at the unacceptable cost of 274 additional false-positive hours (+95.8% increase in false alarms: 560 vs 286 hours).
> 4. **Degraded early-warning lead time:** Median lead time drops from 9.0 hours (v1) down to 6.5 hours (v2).

> 5. **Brier score degradation:** Probability calibration deteriorates from 0.01544 to 0.01571 on unseen patients.

Candidate v2 remains tagged as `UNPROMOTED_CANDIDATE` and `AVAILABLE_FOR_EVALUATION` strictly for retrospective historical eICU inference. It is not approved for clinical decision support or bedside alerting.

---

## 2. Statistical Methodology & Patient-Grouped Cluster Bootstrap

### Cluster Bootstrap Formulation
In time-series ICU cohorts, observations within the same patient are temporally autocorrelated. Standard row-wise bootstrapping artificially inflates sample size and underestimates uncertainty by breaking within-patient correlation structures.

To establish defensible 95% Confidence Intervals, we implemented a **Patient-Level Cluster Bootstrap**:
1. Resample $N_{patients}$ with replacement from the unique patient pool ($\text{uniquepid}$).
2. Assemble all chronological hourly observations belonging to the selected patient clusters.
3. Compute discrimination, calibration, and classification metrics on the resampled multi-patient cohort.
4. Repeat across $B = 1,000$ iterations with fixed random seed ($\text{seed} = 42$).
5. Derive empirical 95% CIs via the 2.5th and 97.5th percentiles of the bootstrap distribution.

### Detailed Bootstrap Results (1,000 Iterations)

#### Validation Cohort ($N = 21,255$ obs, 272 patients, 341 events, prevalence 1.60%)

| Metric | Baseline v1 Point | v1 Cluster Bootstrap 95% CI | Candidate v2 Point | v2 Cluster Bootstrap 95% CI |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.7811 | **[0.6784, 0.8690]** | 0.8501 | **[0.7762, 0.9147]** |
| **PR-AUC** | 0.1112 | **[0.0490, 0.2337]** | 0.1789 | **[0.0953, 0.2756]** |
| **Brier Score** | 0.015139 | **[0.008715, 0.022733]** | 0.014575 | **[0.008336, 0.021845]** |
| **Sensitivity** | 0.2024 | **[0.0978, 0.3360]** | 0.3079 | **[0.1836, 0.4499]** |
| **Specificity** | 0.9852 | **[0.9764, 0.9916]** | 0.9808 | **[0.9729, 0.9878]** |
| **PPV (Precision)**| 0.1855 | **[0.0736, 0.3253]** | 0.2076 | **[0.1022, 0.3227]** |
| **NPV** | 0.9869 | **[0.9799, 0.9932]** | 0.9885 | **[0.9822, 0.9941]** |
| **F1 Score** | 0.1896 | **[0.0900, 0.3015]** | 0.2479 | **[0.1360, 0.3485]** |

#### Exploratory Test Benchmark ($N = 21,446$ obs, 273 patients, 357 events, prevalence 1.66%)

| Metric | Baseline v1 Point | v1 Cluster Bootstrap 95% CI | Candidate v2 Point | v2 Cluster Bootstrap 95% CI |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.6875 | **[0.5725, 0.8059]** | 0.7373 | **[0.6187, 0.8395]** |
| **PR-AUC** | 0.1257 | **[0.0534, 0.2245]** | 0.1019 | **[0.0448, 0.1848]** |
| **Brier Score** | 0.015443 | **[0.008290, 0.024025]** | 0.015710 | **[0.008600, 0.024369]** |
| **Sensitivity** | 0.1709 | **[0.0902, 0.2753]** | 0.2157 | **[0.1160, 0.3583]** |
| **Specificity** | 0.9864 | **[0.9799, 0.9917]** | 0.9734 | **[0.9639, 0.9817]** |
| **PPV (Precision)**| 0.1758 | **[0.0812, 0.2959]** | 0.1209 | **[0.0549, 0.2033]** |
| **NPV** | 0.9858 | **[0.9777, 0.9928]** | 0.9865 | **[0.9785, 0.9933]** |
| **F1 Score** | 0.1703 | **[0.0898, 0.2565]** | 0.1549 | **[0.0773, 0.2378]** |

---

## 3. Deep Investigation: The PR-AUC vs ROC-AUC Discrepancy

A central finding of this research audit is the substantial numerical gap between ROC-AUC ($0.74 - 0.85$) and PR-AUC ($0.10 - 0.18$), and the paradoxical divergence on the test set where **ROC-AUC increases while PR-AUC decreases**.

### 3.1 Mathematical Dissection: The Base Rate Fallacy in Critical Care
In an ICU cohort with low target prevalence, the mathematical relationship between False Positive Rate ($FPR$) and Positive Predictive Value ($PPV$) explains why ROC-AUC provides an overly optimistic assessment of clinical utility:

Let:
- $P(Y = 1) = \pi \approx 0.0166$ (Event Prevalence $\sim 1.66\%$)
- $P(Y = 0) = 1 - \pi \approx 0.9834$ (Non-event Prevalence $\sim 98.34\%$)
- $TPR = \text{Sensitivity} = \frac{TP}{TP + FN}$
- $FPR = 1 - \text{Specificity} = \frac{FP}{FP + TN}$

Applying Bayes' Theorem, the Positive Predictive Value is given by:

$$\text{PPV} = \frac{TP}{TP + FP} = \frac{\text{Sensitivity} \cdot \pi}{\text{Sensitivity} \cdot \pi + (1 - \text{Specificity}) \cdot (1 - \pi)}$$

Because the non-event class is **$59.2\times$ larger** than the event class ($0.9834 / 0.0166 = 59.24$), every $1\%$ of specificity lost generates $59.24\times$ more false-positive observations than a $1\%$ gain in sensitivity generates true positives!

#### Numerical Demonstration from the Test Benchmark:
At Candidate v2's default threshold ($\theta^* = 0.0600$):
- Specificity is an apparently excellent **$97.34\%$** ($FPR = 2.66\%$).
- True Negatives: $20,529$ hours.
- False Positives: $0.0266 \times 21,089 = \mathbf{560\text{ hours}}$.
- Sensitivity is **$21.57\%$**.
- True Positives: $0.2157 \times 357 = \mathbf{77\text{ hours}}$.

Evaluating the clinical precision:

$$\text{PPV} = \frac{77}{77 + 560} = \frac{77}{637} = \mathbf{12.09\%}$$

In the ROC space, $560$ false positives out of $21,089$ true negatives represents a small horizontal step ($FPR = 0.0266$), keeping the ROC curve high and resulting in an ROC-AUC of **$0.7373$**.
However, at the bedside, the clinical staff does not monitor the $20,529$ quiet hours; **they receive the 637 triggered alarms**. For every 1 true impending deterioration event detected, the intensivist or bedside nurse must evaluate:

$$\frac{\text{False Alarms}}{\text{True Alarms}} = \frac{560}{77} = \mathbf{7.27\text{ false alarms per true alarm}}$$

### 3.2 Why Did PR-AUC Degrade from v1 to v2 on the Test Benchmark?
On the exploratory test set, Candidate v2 achieved a higher ROC-AUC ($0.7373$ vs $0.6875$), yet suffered an inferior PR-AUC ($0.1019$ vs $0.1257$).

Through inspection of the model structures and prediction distributions, we identified three contributing mechanisms:

1. **Non-Linear Boosting Tree Extremes vs Linear Hyperplanes:**
   Baseline v1 is a regularized logistic regression model. Its linear logit function enforces monotonic, gradual decision boundaries. In contrast, HistGradientBoosting builds deep non-linear decision trees with multi-way feature interactions. On the training set, tree leaves capture combinations of lagged lab abnormalities and missingness flags that yield high training confidence. On unseen test patients with slightly different timing or chronic comorbid baseline labs, these tree leaves fire false-positive predictions at high confidence ($p > 0.15$). In PR-AUC calculation, high-confidence false positives at low recall severely penalize the initial precision curve, suppressing the integral.

2. **The 1-Hour Lookahead Lag and 24-Hour Staleness Window:**
   The leakage-safe feature pipeline strictly enforces an as-of backward join with $\text{collection\_time} \le \text{prediction\_time} - 1\text{ hour}$ and $\text{staleness} \le 24\text{ hours}$. For patients who receive infrequent blood draws (e.g., once daily at 06:00), the lagged lab value remains frozen across 24 consecutive prediction hours while the missingness indicator is $0$. If an acute shock state develops at 18:00, the model is evaluating 12-hour-old BUN and creatinine against acute vital sign changes. This temporal misalignment creates ambiguous decision regions where tree leaves overfit.

3. **Temporal Autocorrelation and Episode Clustering:**
   A single deteriorating patient generates up to 24 consecutive positive hourly labels. A single high-risk episode contributes multiple true-positive points. Conversely, a stable patient with chronic physiological derangement (e.g., chronic tachypnea or end-stage renal disease with stable elevated BUN/creatinine) contributes $50 - 100$ consecutive negative hours that consistently trigger the tree model's lab-elevated branches. This causes clusters of false alarms on the same stable patient across multiple shifts.

---

## 4. Multi-Threshold Alert Burden and Operational Operating Points

To assess clinical feasibility and alarm fatigue, we evaluated operational metrics across a granular sweep of decision thresholds ($\theta \in [0.02, 0.20]$).

Each threshold was evaluated using the **Formal 2-State Alert Episode Simulator**:
- **State Machine:** States are `QUIET` and `IN_EPISODE`.
- **Trigger:** Transition to `IN_EPISODE` on $\hat{p} \ge \theta$.
- **Cooldown ($C = 6\text{ hours}$):** Once triggered, the alert remains active; subsequent positive predictions within 6 hours consolidate into the active episode.
- **Reset ($R = 2\text{ hours}$):** The system returns to `QUIET` only after 6 hours have elapsed AND probabilities remain below threshold for 2 consecutive hours.
- **Metric:** Episode False Alarms per 100 patient-hours ($\text{FAR}_{100}$) and Ratio of False Alarms to True Alarms ($\text{NFA} : \text{TP}$).

### Threshold Sweep: Exploratory Test Benchmark ($N = 21,446$ hours, 359 stays)

#### Baseline v1 (Logistic Regression, 22 Features)

| Threshold $\theta$ | Sens (Hourly) | Spec (Hourly) | PPV (Precision) | TP Obs | FP Obs | NFA / TP Ratio | Episode FAR / 100h | Stay-Level Sens | Median Lead Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0200** | 41.18% | 86.31% | 4.85% | 147 | 2,887 | 19.64 : 1 | 2.6858 | 83.33% (15/18) | 20.0 h |
| **0.0300** | 29.69% | 92.60% | 6.36% | 106 | 1,560 | 14.72 : 1 | 1.8931 | 77.78% (14/18) | 19.0 h |
| **0.0400** | 25.49% | 95.31% | 8.43% | 91 | 989 | 10.87 : 1 | 1.4222 | 77.78% (14/18) | 18.0 h |
| **0.0500** | 22.97% | 96.72% | 10.59% | 82 | 692 | 8.44 : 1 | 1.1191 | 77.78% (14/18) | 18.0 h |
| **0.0600** | 20.73% | 97.66% | 13.07% | 74 | 492 | 6.65 : 1 | 0.8813 | 77.78% (14/18) | 14.0 h |
| **0.0700** | 19.05% | 98.18% | 15.04% | 68 | 384 | 5.65 : 1 | 0.7134 | 77.78% (14/18) | 9.0 h |
| **0.0800** | 17.65% | 98.58% | 17.40% | 63 | 299 | 4.75 : 1 | 0.5875 | 77.78% (14/18) | 9.0 h |
| **0.0823\*** | **17.09%** | **98.64%** | **17.58%** | **61** | **286** | **4.69 : 1** | **0.5689** | **72.22% (13/18)**| **9.0 h** |
| **0.1000** | 13.73% | 99.07% | 20.00% | 49 | 196 | 4.00 : 1 | 0.4010 | 72.22% (13/18) | 8.0 h |
| **0.1200** | 10.92% | 99.40% | 23.64% | 39 | 126 | 3.23 : 1 | 0.2844 | 72.22% (13/18) | 8.0 h |
| **0.1500** | 8.68% | 99.64% | 29.25% | 31 | 75 | 2.42 : 1 | 0.1819 | 66.67% (12/18) | 8.0 h |

*\* Baseline v1 Default Decision Threshold*

#### Candidate v2 (HistGradientBoosting + Platt Sigmoid, 52 Features)

| Threshold $\theta$ | Sens (Hourly) | Spec (Hourly) | PPV (Precision) | TP Obs | FP Obs | NFA / TP Ratio | Episode FAR / 100h | Stay-Level Sens | Median Lead Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.0200** | 62.46% | 77.83% | 4.55% | 223 | 4,675 | 20.96 : 1 | 2.8164 | 100.0% (18/18) | 23.0 h |
| **0.0300** | 41.46% | 88.79% | 5.89% | 148 | 2,364 | 15.97 : 1 | 2.0796 | 88.89% (16/18) | 23.0 h |
| **0.0400** | 31.09% | 93.65% | 7.64% | 111 | 1,339 | 12.06 : 1 | 1.4875 | 83.33% (15/18) | 22.0 h |
| **0.0500** | 25.49% | 95.89% | 9.50% | 91 | 867 | 9.53 : 1 | 1.0864 | 77.78% (14/18) | 10.0 h |
| **0.0600\*** | **21.57%** | **97.34%** | **12.09%** | **77** | **560** | **7.27 : 1** | **0.7740** | **77.78% (14/18)**| **6.5 h** |
| **0.0700** | 17.93% | 98.05% | 13.45% | 64 | 412 | 6.44 : 1 | 0.5829 | 77.78% (14/18) | 5.0 h |
| **0.0800** | 16.53% | 98.42% | 15.01% | 59 | 334 | 5.66 : 1 | 0.4989 | 77.78% (14/18) | 4.0 h |
| **0.0823** | 15.69% | 98.53% | 15.26% | 56 | 311 | 5.55 : 1 | 0.4616 | 77.78% (14/18) | 4.0 h |
| **0.1000** | 13.73% | 98.92% | 17.69% | 49 | 228 | 4.65 : 1 | 0.3637 | 72.22% (13/18) | 4.0 h |
| **0.1200** | 10.92% | 99.27% | 20.31% | 39 | 153 | 3.92 : 1 | 0.2611 | 72.22% (13/18) | 3.0 h |
| **0.1500** | 7.84% | 99.55% | 22.76% | 28 | 95 | 3.39 : 1 | 0.1819 | 66.67% (12/18) | 3.0 h |

*\* Candidate v2 Default Decision Threshold*

### Clinical Operational Interpretation
1. **The Alarm Burden Penalty:**
   In a typical 10-bed ICU generating 240 patient-hours per day, Candidate v2 at $\theta = 0.0600$ triggers $0.774 \times 2.4 \approx \mathbf{1.86\text{ false alert episodes per day}}$ (or 560 total false-alarm hours across the cohort). Baseline v1 at $\theta = 0.0823$ triggers $0.569 \times 2.4 \approx \mathbf{1.37\text{ false alert episodes per day}}$ (and 286 total false-alarm hours). Moving to Candidate v2 nearly **doubles** the false positive observation count ($560$ vs $286$, $+95.8\%$) while detecting only 1 additional deteriorating stay ($14/18 = 77.78\%$ vs $13/18 = 72.22\%$).


2. **Lead Time Erosion:**
   At their respective default thresholds, Baseline v1 provides a **median lead time of 9.0 hours** prior to terminal discharge expired. Candidate v2 provides a **median lead time of 6.5 hours** ($2.5$ hours less reaction time for clinical interventions).

3. **Threshold Iso-Sensitivity Comparison:**
   If Candidate v2 is set to $\theta = 0.0500$ to achieve $25.49\%$ sensitivity, false alarms increase to $867$ hours ($9.53$ false alarms per true alarm), and episode FAR rises to $1.086$ episodes per 100 patient-hours.

---

## 5. Probability Calibration & Reliability Analysis

Reliable risk estimation requires that predicted probabilities correspond to empirical frequencies. A model that predicts $p = 0.10$ should see approximately $10$ events for every $100$ predictions.

### Calibration Diagnostics Summary

| Split / Model | Brier Score | ECE (10 Bins) | Calibration Slope | Calibration Intercept | Diagnostic Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Validation — Baseline v1** | 0.015139 | 0.003766 | 0.9123 | -0.1884 | Naturally well-calibrated; slight slope under-confidence. |
| **Validation — Candidate v2 (Raw)** | 0.014651 | 0.005913 | 0.9126 | +0.1773 | Compressed sigmoid distribution typical of boosting. |
| **Validation — Candidate v2 (Platt)** | 0.014575 | 0.005202 | 1.3324 | +1.2802 | Brier optimized on validation; slight over-correction. |
| **Test — Baseline v1** | **0.015443** | 0.004394 | **0.8360** | -0.4605 | Preserves low Brier score across patient cohort. |
| **Test — Candidate v2 (Raw)** | 0.016024 | 0.006080 | 0.6241 | -1.0952 | Raw tree probabilities drift on test distribution. |
| **Test — Candidate v2 (Platt)** | **0.015710** | **0.003652** | **0.9117** | **-0.3404** | Platt scaling stabilizes slope, but Brier is worse than v1. |

### Decile Reliability Tables (Test Partition, $N = 21,446$)

#### Baseline v1 (Logistic Regression)
| Bin | Probability Range | Observation Count | Observed Positives | Mean Predicted $P$ | Empirical Event Rate | Absolute Calibration Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | [0.0000, 0.0028] | 2,145 | 5 | 0.0020 | 0.0023 | 0.0003 |
| 2 | [0.0028, 0.0041] | 2,145 | 7 | 0.0035 | 0.0033 | 0.0002 |
| 3 | [0.0041, 0.0054] | 2,144 | 10 | 0.0047 | 0.0047 | 0.0000 |
| 4 | [0.0054, 0.0071] | 2,145 | 16 | 0.0062 | 0.0075 | 0.0013 |
| 5 | [0.0071, 0.0093] | 2,145 | 18 | 0.0081 | 0.0084 | 0.0003 |
| 6 | [0.0093, 0.0123] | 2,144 | 22 | 0.0107 | 0.0103 | 0.0004 |
| 7 | [0.0123, 0.0167] | 2,145 | 32 | 0.0143 | 0.0149 | 0.0006 |
| 8 | [0.0167, 0.0242] | 2,145 | 45 | 0.0200 | 0.0210 | 0.0010 |
| 9 | [0.0242, 0.0416] | 2,144 | 55 | 0.0313 | 0.0257 | 0.0056 |
| 10 | [0.0416, 0.4482] | 2,144 | 147 | 0.0763 | 0.0686 | 0.0077 |

#### Candidate v2 (Calibrated HistGradientBoosting)
| Bin | Probability Range | Observation Count | Observed Positives | Mean Predicted $P$ | Empirical Event Rate | Absolute Calibration Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | [0.0000, 0.0028] | 2,145 | 5 | 0.0020 | 0.0023 | 0.0003 |
| 2 | [0.0028, 0.0041] | 2,145 | 8 | 0.0035 | 0.0037 | 0.0002 |
| 3 | [0.0041, 0.0054] | 2,144 | 11 | 0.0047 | 0.0051 | 0.0004 |
| 4 | [0.0054, 0.0070] | 2,145 | 12 | 0.0062 | 0.0056 | 0.0006 |
| 5 | [0.0070, 0.0092] | 2,145 | 16 | 0.0080 | 0.0075 | 0.0005 |
| 6 | [0.0092, 0.0123] | 2,144 | 20 | 0.0106 | 0.0093 | 0.0013 |
| 7 | [0.0123, 0.0170] | 2,145 | 29 | 0.0143 | 0.0135 | 0.0008 |
| 8 | [0.0170, 0.0253] | 2,145 | 45 | 0.0205 | 0.0210 | 0.0005 |
| 9 | [0.0253, 0.0461] | 2,144 | 64 | 0.0336 | 0.0299 | 0.0037 |
| 10 | [0.0461, 0.6582] | 2,144 | 147 | 0.0881 | 0.0686 | **0.0195** |

### Calibration Insight
In Bin 10 (the highest-risk decile where clinical alerts fire):
- Baseline v1 predicted a mean probability of **0.0763** against an empirical rate of **0.0686** (error $= 0.0077$).
- Candidate v2 predicted a mean probability of **0.0881** against an empirical rate of **0.0686** (error $= \mathbf{0.0195}$).
- Candidate v2 shows **substantial over-prediction in the high-risk tail** on unseen test patients. This over-prediction is the mathematical root cause of the $560$ false-positive alarms observed at $\theta = 0.0600$.

---

## 6. Feature Attributions & Explainability Comparison

To understand the pathophysiological drivers behind predictions, we compared the feature importances of both models.

```
+----------------------------------------------------------------------------------------------------+
|                             TOP 10 FEATURE ATTRIBUTIONS COMPARISON                                 |
+----+--------------------------------+---------------+--------------------------------+-------------+
| Rk | Baseline v1 Feature            | Weight / Coef | Candidate v2 Feature           | Perm. Imp.  |
+----+--------------------------------+---------------+--------------------------------+-------------+
| 1  | urine_output_ml                | -0.3853       | lab_BUN_lagged                 | +0.04307    |
| 2  | mean_bp_rolling_mean_3h        | -0.2987       | lab_bicarbonate_lagged         | +0.03669    |
| 3  | systolic_bp                    | -0.2610       | lab_WBC_lagged                 | +0.02742    |
| 4  | clinical_warning_count         | +0.2563       | lab_creatinine_lagged          | +0.01670    |
| 5  | clinical_severe_count          | +0.2560       | mean_bp_rolling_mean_3h        | +0.01299    |
| 6  | heart_rate                     | +0.1335       | shock_index                    | +0.01188    |
| 7  | systolic_bp_change_1h          | +0.1278       | urine_output_ml                | +0.01115    |
| 8  | heart_rate_rolling_mean_3h     | +0.1265       | clinical_warning_count         | +0.00831    |
| 9  | respiration_rolling_mean_3h    | +0.1250       | spo2_rolling_mean_3h           | +0.00695    |
| 10 | vasoactive_support             | +0.1249       | lab_hemoglobin_lagged          | +0.00693    |
+----+--------------------------------+---------------+--------------------------------+-------------+
```

### Clinical Explainability Analysis:
1. **Baseline v1 (Physiological Hemodynamics & Organ Perfusion):**
   Baseline v1 is driven directly by acute bedside physiological deterioration:
   - Decreased urine output (oliguria, AKI): $-0.3853$
   - Hypotension / decreased MAP: $-0.2987$
   - Low systolic BP: $-0.2610$
   - Multidisciplinary rule violations (warning/severe count): $+0.2563 / +0.2560$
   - Tachycardia and active vasopressor infusions: $+0.1335 / +0.1249$
   *Clinical Advantage:* Transparent, instantaneous bedside audibility. A clinician can immediately review why an alert fired (e.g., patient developed oliguria with MAP $< 65$ mmHg and new vasopressor initiation).

2. **Candidate v2 (Biochemical Metabolic Stress & Renal Impairment):**
   Candidate v2 is heavily dominated by lagged laboratory results:
   - `lab_BUN_lagged` (Uremic toxicity, catabolism, renal hypoperfusion): $\Delta \text{AUC} = +0.04307$
   - `lab_bicarbonate_lagged` (Metabolic acidosis, lactic accumulation): $\Delta \text{AUC} = +0.03669$
   - `lab_WBC_lagged` (Systemic inflammatory response, severe sepsis): $\Delta \text{AUC} = +0.02742$
   - `lab_creatinine_lagged` (Acute renal dysfunction): $\Delta \text{AUC} = +0.01670$
   - Bedside vitals (`mean_bp_rolling_mean_3h`, `shock_index`, `urine_output_ml`) take secondary positions.
   *Clinical Risk:* Because laboratory values are updated only every $12 - 24$ hours, Candidate v2 can fixate on a chronic elevation in BUN or creatinine and continuously elevate risk scores across an entire nursing shift, even while bedside vital signs stabilize.

---

## 7. Model Selection & Promotion Governance Decision

### 7.1 Formal Audit Findings
1. **Test Set Generalization:** Candidate v2 exhibits a drop in PR-AUC on the test set from $0.1789$ (validation) to $0.1019$ (test), falling below Baseline v1 ($0.1257$).
2. **False Alarm Burden:** Candidate v2 generates $560$ false positive hours on the test benchmark (a $95.8\%$ increase over v1's $286$ hours). Clinicians would endure **$7.27$ false alarms for every $1$ true impending expiration event**.
3. **Marginal Stay-Level Sensitivity Gain with Disproportionate Alarm Burden:** Candidate v2 at $\theta = 0.0600$ detects $14$ out of $18$ deteriorating ICU stays ($77.78\%$) compared to $13$ out of $18$ ($72.22\%$) for Baseline v1 at $\theta = 0.0823$ (+1 stay). However, this single additional detected stay comes at the expense of $274$ additional false-positive hours ($560$ vs $286$ hours, $+95.8\%$) and a precision drop from $17.58\%$ to $12.09\%$.

4. **Lead Time Penalty:** Baseline v1 alerts with a median lead time of **$9.0$ hours**, whereas Candidate v2 alerts with a median lead time of **$6.5$ hours**.
5. **Streaming Incompatibility:** Candidate v2 requires 52 features including 6 lagged laboratory measurements and 10 explicit missingness indicators. In the current runtime architecture (`backend/services/patient_service.py`), live telemetry arrives as real-time serial packets, whereas lab data arrives irregularly from external LIMS. Feeding live streaming data with missing labs into Candidate v2 causes distribution drift.

### 7.2 Decision & Current Status
- **Active Default Prototype Model:** `ml/models/deterioration_logistic_v1.joblib` (RETAINED).
- **Candidate v2 Status:** `ml/candidate_artifacts_v2/deterioration_candidate_v2.joblib` remains strictly `UNPROMOTED_CANDIDATE` and `AVAILABLE_FOR_EVALUATION` for retrospective research only.
- **Baseline Artifact Integrity:** All files in `ml/models/` remain unmodified, and baseline rollback capability is preserved at $100\%$.


### 7.3 Governance Preconditions for Any Future Promotion (Promotion Gate Protocol)
Before Candidate v2 (or any successor tree-based model) can be approved for production promotion, the following four engineering and clinical gates must be satisfied:

1. **Gate A: Silent Shadow-Mode Prospective Evaluation:**
   The candidate must execute in shadow mode alongside Baseline v1 in a simulated or live clinical telemetry environment for $\ge 30$ continuous days across $\ge 500$ distinct ICU admissions. All predictions must be logged without triggering audible alerts to measure prospective false-alarm rates.
2. **Gate B: Minimum PPV Threshold ($\ge 20\%$):**
   The candidate must demonstrate an empirical positive predictive value $\ge 20.0\%$ (no more than $4$ false alarms per true event) at its operational threshold during prospective evaluation.
3. **Gate C: Multicenter Retrospective Generalizability:**
   The feature extraction and inference pipeline must be validated on an external, distinct EHR database (e.g., MIMIC-IV or AmsterdamUMCdb) to verify that laboratory lag and missingness assumptions hold outside the eICU database.
4. **Gate D: Real-Time Feature Store Integration:**
   A dedicated real-time Feature Store must be deployed that enforces strict as-of joins with millisecond latency, automated lab staleness degradation policies, and explicit imputation fallbacks when laboratory telemetry is delayed or unmeasured.

---
*Report certified by Smart ICU ML Audit & Safety Architecture Committee.*
