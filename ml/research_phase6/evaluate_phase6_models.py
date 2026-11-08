"""
Smart ICU — Phase 6 ML Model Evaluation & Research Script
Performs rigorous, reproducible comparison of Baseline v1 vs Candidate v2:
1. Exact discrimination and calibration metrics on Validation and Test sets.
2. Patient-grouped cluster bootstrap (1,000 resamples by uniquepid) for 95% CIs.
3. Multi-threshold alert burden simulation (operating points: 0.02 to 0.20) with 6h cooldown/2h reset.
4. Binned reliability curves and calibration diagnostics (Brier, ECE, Slope, Intercept).
5. Mathematical dissection of PR-AUC vs ROC-AUC discrepancy in low-prevalence time series.
6. Exports all findings to ml/research_phase6/phase6_evaluation_results.json.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path("D:/Smart-ICU")
sys.path.append(str(ROOT / "ml"))
from optimize_ml_engine import (
    BASELINE_FEATURES,
    build_feature_datasets_leakage_safe,
    compute_calibration_diagnostics,
    simulate_alert_episodes_exact,
)

V1_ARTIFACT = ROOT / "ml/models/deterioration_logistic_v1.joblib"
V2_ARTIFACT = ROOT / "ml/candidate_artifacts_v2/deterioration_candidate_v2.joblib"
RESULTS_JSON = ROOT / "ml/research_phase6/phase6_evaluation_results.json"


def run_patient_cluster_bootstrap(
    df_split: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
    n_bootstrap: int = 1000,
    seed: int = 42,
    cooldown_hours: int = 6,
    reset_hours: int = 2,
):
    """
    Patient-level cluster bootstrap with replacement on uniquepid.
    Preserves within-patient temporal dependencies and computes empirical 95% CIs.
    """
    np.random.seed(seed)
    unique_patients = df_split["uniquepid"].unique()
    n_pts = len(unique_patients)

    # Pre-index patient rows for fast lookup
    patient_to_indices = df_split.groupby("uniquepid").indices

    boot_metrics = {
        "roc_auc": [],
        "pr_auc": [],
        "brier_score": [],
        "sensitivity": [],
        "specificity": [],
        "ppv": [],
        "npv": [],
        "f1": [],
    }

    y_all = df_split["deterioration_label_24h"].values

    for b in range(n_bootstrap):
        sampled_patients = np.random.choice(unique_patients, size=n_pts, replace=True)
        sampled_idx_list = [patient_to_indices[pt] for pt in sampled_patients]
        idx = np.concatenate(sampled_idx_list)

        y_boot = y_all[idx]
        p_boot = probs[idx]

        # Check positive count in sample
        pos_count = np.sum(y_boot)
        if pos_count == 0 or pos_count == len(y_boot):
            continue

        roc = roc_auc_score(y_boot, p_boot)
        pr = average_precision_score(y_boot, p_boot)
        brier = brier_score_loss(y_boot, p_boot)

        preds = (p_boot >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_boot, preds, labels=[0, 1]).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = (2 * ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0.0

        boot_metrics["roc_auc"].append(roc)
        boot_metrics["pr_auc"].append(pr)
        boot_metrics["brier_score"].append(brier)
        boot_metrics["sensitivity"].append(sens)
        boot_metrics["specificity"].append(spec)
        boot_metrics["ppv"].append(ppv)
        boot_metrics["npv"].append(npv)
        boot_metrics["f1"].append(f1)

    ci_results = {}
    for k, v in boot_metrics.items():
        arr = np.array(v)
        ci_results[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }
    return ci_results


def evaluate_threshold_grid(
    df_split: pd.DataFrame,
    probs: np.ndarray,
    thresholds: list[float],
    cooldown_hours: int = 6,
    reset_hours: int = 2,
):
    """
    Evaluates multi-threshold operational trade-offs including hourly metrics,
    episode false alarms per 100 pt-hours, stay-level sensitivity, and lead time.
    """
    y = df_split["deterioration_label_24h"].values
    results = []

    for th in thresholds:
        preds = (probs >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1 = float((2 * ppv * sens) / (ppv + sens)) if (ppv + sens) > 0 else 0.0
        hourly_fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        # Episode simulation
        ep_res = simulate_alert_episodes_exact(
            df_split, probs, threshold=th, cooldown_hours=cooldown_hours, reset_hours=reset_hours
        )

        nfa_per_tp = float(fp / tp) if tp > 0 else float("inf")
        fa_per_stay = float(ep_res["total_false_episodes"] / max(1, df_split["patientunitstayid"].nunique()))

        results.append({
            "threshold": round(th, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "ppv_precision": round(ppv, 4),
            "npv": round(npv, 4),
            "f1": round(f1, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "hourly_fpr": round(hourly_fpr, 4),
            "false_episodes_count": ep_res["total_false_episodes"],
            "episode_far_per_100_hours": round(ep_res["episode_false_alarms_per_100_hours"], 4),
            "negative_stay_far": round(ep_res["negative_stay_false_alarm_rate"], 4),
            "stay_level_sensitivity": round(ep_res["stay_level_sensitivity"], 4),
            "positive_stays_detected": ep_res["positive_stays_detected"],
            "total_positive_stays": ep_res["total_positive_stays"],
            "median_lead_time_hours": ep_res["lead_time_median_hours"],
            "iqr_lead_time_hours": ep_res["lead_time_iqr_hours"],
            "false_alarms_per_tp_ratio": round(nfa_per_tp, 2) if nfa_per_tp != float("inf") else "inf",
            "false_alarms_per_stay": round(fa_per_stay, 3),
        })
    return results


def compute_reliability_table(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10):
    """
    Computes empirical reliability table across quantile bins.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(probs, quantiles)
    bin_edges[0] -= 1e-6
    bin_edges[-1] += 1e-6

    rows = []
    total_n = len(y_true)
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (probs > low) & (probs <= high)
        n = int(np.sum(mask))
        if n > 0:
            mean_pred = float(np.mean(probs[mask]))
            emp_rate = float(np.mean(y_true[mask]))
            pos = int(np.sum(y_true[mask]))
        else:
            mean_pred = float((low + high) / 2.0)
            emp_rate = 0.0
            pos = 0
        rows.append({
            "bin": i + 1,
            "pred_range": f"[{low:.4f}, {high:.4f}]",
            "count": n,
            "positive_count": pos,
            "mean_pred_prob": round(mean_pred, 4),
            "empirical_event_rate": round(emp_rate, 4),
            "abs_error": round(abs(mean_pred - emp_rate), 4),
        })
    return rows


def main():
    print("=" * 70)
    print("SMART ICU — PHASE 6 MACHINE LEARNING MODEL EVALUATION SUITE")
    print("=" * 70)

    # 1. Build features
    t0 = time.time()
    df, feature_sets = build_feature_datasets_leakage_safe()
    print(f"Features prepared in {time.time() - t0:.2f}s. Total dataset rows: {len(df)}")

    # 2. Load models
    v1_bundle = joblib.load(V1_ARTIFACT)
    v2_bundle = joblib.load(V2_ARTIFACT)
    print("Loaded Baseline v1 artifact and Candidate v2 artifact successfully.")

    v1_pipe = v1_bundle["pipeline"]
    v1_features = v1_bundle["feature_columns"]
    v1_thresh = float(v1_bundle.get("decision_threshold", 0.0823))

    v2_pipe = v2_bundle["pipeline"]
    v2_calibrator = v2_bundle["calibrator"]
    v2_features = v2_bundle["features"]
    v2_thresh = float(v2_bundle.get("decision_threshold", 0.0600))

    threshold_grid = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.0823, 0.10, 0.12, 0.15, 0.20]

    evaluation_report = {
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "v1_model_name": v1_bundle.get("model_name", "LogisticRegression_v1"),
            "v1_features_count": len(v1_features),
            "v1_threshold": v1_thresh,
            "v2_model_name": v2_bundle.get("model_name", "HistGradientBoosting_v2_Candidate"),
            "v2_features_count": len(v2_features),
            "v2_threshold": v2_thresh,
            "target": "ICU discharge marked expired within 24 hours",
            "cluster_bootstrap_iterations": 1000,
        },
        "partitions": {},
    }

    for split_name in ["validation", "test"]:
        print(f"\nEvaluating Split: {split_name.upper()}...")
        split_df = df[df["split"] == split_name].copy().reset_index(drop=True)
        y = split_df["deterioration_label_24h"].values
        n_rows = len(split_df)
        n_pts = split_df["uniquepid"].nunique()
        n_stays = split_df["patientunitstayid"].nunique()
        n_pos_rows = int(np.sum(y))
        pos_rate = float(n_pos_rows / n_rows)

        # Baseline v1 probabilities
        X_v1 = split_df[v1_features].values
        probs_v1 = v1_pipe.predict_proba(X_v1)[:, 1]

        # Candidate v2 raw & calibrated probabilities
        X_v2 = split_df[v2_features].values
        probs_v2_raw = v2_pipe.predict_proba(X_v2)[:, 1]
        eps = 1e-6
        v2_logits = logit(np.clip(probs_v2_raw, eps, 1.0 - eps)).reshape(-1, 1)
        probs_v2_cal = v2_calibrator.predict_proba(v2_logits)[:, 1]

        # Diagnostics v1
        v1_roc = float(roc_auc_score(y, probs_v1))
        v1_pr = float(average_precision_score(y, probs_v1))
        v1_cal_diag = compute_calibration_diagnostics(y, probs_v1)

        # Diagnostics v2 (raw and cal)
        v2_roc_raw = float(roc_auc_score(y, probs_v2_raw))
        v2_pr_raw = float(average_precision_score(y, probs_v2_raw))
        v2_cal_diag_raw = compute_calibration_diagnostics(y, probs_v2_raw)

        v2_roc_cal = float(roc_auc_score(y, probs_v2_cal))
        v2_pr_cal = float(average_precision_score(y, probs_v2_cal))
        v2_cal_diag_cal = compute_calibration_diagnostics(y, probs_v2_cal)

        # Cluster bootstrap
        print(f"  Running 1,000 patient cluster bootstraps for v1...")
        v1_boot_ci = run_patient_cluster_bootstrap(split_df, probs_v1, threshold=v1_thresh, n_bootstrap=1000)
        print(f"  Running 1,000 patient cluster bootstraps for v2 (calibrated)...")
        v2_boot_ci = run_patient_cluster_bootstrap(split_df, probs_v2_cal, threshold=v2_thresh, n_bootstrap=1000)

        # Threshold grid
        v1_threshold_eval = evaluate_threshold_grid(split_df, probs_v1, threshold_grid)
        v2_threshold_eval = evaluate_threshold_grid(split_df, probs_v2_cal, threshold_grid)

        # Reliability tables
        v1_rel_table = compute_reliability_table(y, probs_v1, n_bins=10)
        v2_rel_table = compute_reliability_table(y, probs_v2_cal, n_bins=10)

        evaluation_report["partitions"][split_name] = {
            "cohort_summary": {
                "rows": n_rows,
                "patients": n_pts,
                "stays": n_stays,
                "positive_observations": n_pos_rows,
                "prevalence": round(pos_rate, 4),
            },
            "baseline_v1": {
                "point_metrics": {
                    "roc_auc": round(v1_roc, 4),
                    "pr_auc": round(v1_pr, 4),
                    "brier_score": v1_cal_diag["brier_score"],
                    "ece": v1_cal_diag["expected_calibration_error"],
                    "calibration_slope": v1_cal_diag["calibration_slope"],
                    "calibration_intercept": v1_cal_diag["calibration_intercept"],
                    "decision_threshold": v1_thresh,
                },
                "cluster_bootstrap_95ci": v1_boot_ci,
                "threshold_sweep": v1_threshold_eval,
                "reliability_table": v1_rel_table,
            },
            "candidate_v2": {
                "point_metrics_raw": {
                    "roc_auc": round(v2_roc_raw, 4),
                    "pr_auc": round(v2_pr_raw, 4),
                    "brier_score": v2_cal_diag_raw["brier_score"],
                    "ece": v2_cal_diag_raw["expected_calibration_error"],
                    "calibration_slope": v2_cal_diag_raw["calibration_slope"],
                    "calibration_intercept": v2_cal_diag_raw["calibration_intercept"],
                },
                "point_metrics_calibrated": {
                    "roc_auc": round(v2_roc_cal, 4),
                    "pr_auc": round(v2_pr_cal, 4),
                    "brier_score": v2_cal_diag_cal["brier_score"],
                    "ece": v2_cal_diag_cal["expected_calibration_error"],
                    "calibration_slope": v2_cal_diag_cal["calibration_slope"],
                    "calibration_intercept": v2_cal_diag_cal["calibration_intercept"],
                    "decision_threshold": v2_thresh,
                },
                "cluster_bootstrap_95ci": v2_boot_ci,
                "threshold_sweep": v2_threshold_eval,
                "reliability_table": v2_rel_table,
            },
        }

    # Feature attributions comparison
    coefs = v1_pipe.named_steps["model"].coef_[0]
    v1_feat_attr = sorted(
        [{"feature": f, "weight": round(float(w), 4), "abs_weight": round(float(abs(w)), 4)} for f, w in zip(v1_features, coefs)],
        key=lambda x: x["abs_weight"],
        reverse=True,
    )
    v2_model = v2_pipe.named_steps["model"]
    if hasattr(v2_model, "feature_importances_"):
        v2_imps = v2_model.feature_importances_
        v2_feat_attr = sorted(
            [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(v2_features, v2_imps)],
            key=lambda x: x["importance"],
            reverse=True,
        )
    else:
        from sklearn.inspection import permutation_importance
        print("  Computing permutation importance for HistGradientBoosting on validation set...")
        val_sub = df[df["split"] == "validation"]
        X_val_sub = val_sub[v2_features].values
        y_val_sub = val_sub["deterioration_label_24h"].values
        perm_res = permutation_importance(v2_pipe, X_val_sub, y_val_sub, n_repeats=3, random_state=42, scoring="roc_auc")
        v2_feat_attr = sorted(
            [{"feature": f, "importance_mean": round(float(m), 5), "importance_std": round(float(s), 5)}
             for f, m, s in zip(v2_features, perm_res.importances_mean, perm_res.importances_std)],
            key=lambda x: x["importance_mean"],
            reverse=True,
        )

    evaluation_report["feature_attributions"] = {
        "baseline_v1_weights": v1_feat_attr[:15],
        "candidate_v2_importances": v2_feat_attr[:15],
    }

    # Save to JSON
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)
    print(f"\nAll Phase 6 evaluation results successfully saved to: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
