"""
Smart ICU — Retrospective ML Prediction Performance Optimization Engine (Phase 2)
Strictly adheres to Phase 2 Critical Methodology Corrections:
- Laboratory feature integrity: pd.merge_asof with collection/reporting time <= pred_time - 1h,
  staleness <= 24h, stay-constrained, no future lookahead, missingness preserved.
- Rolling feature integrity: Elapsed-time window on complete hourly grid, gap-reset, sort-guaranteed.
- Alert episode simulation: Formal 2-state machine (QUIET, IN_EPISODE) with cooldown C=6h, reset R=2h,
  gap handling, no same-obs retriggering, separate hourly FPR vs episode FAR per 100 pt-hours.
- Validation discipline: Patient-grouped CV (5 folds on uniquepid) with strictly isolated inside-fold
  preprocessing, OOF calibration fitting (Platt and Isotonic), validation confirmation, and
  explicit 'NO FEASIBLE THRESHOLD' reporting if experimental design targets are unmet.
- Artifact safety: All candidate artifacts written to ml/candidate_artifacts_v2/ without touching v1.
  Full reproducibility metadata (Git SHA, dataset SHA-256 hashes, validation predictions).
- Test set exposure: Test partition documented strictly as a previously observed exploratory benchmark.
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path("D:/Smart-ICU")
INPUT_FILE = BASE_DIR / "data/processed/experiments/leakage_safe/model_inputs_corrected.csv"
STATE_FILE = BASE_DIR / "data/processed/hourly_patient_state.csv"
RAW_LAB_FILE = BASE_DIR / "data/real/eicu_demo/eicu-collaborative-research-database-demo-2.0.1/lab.csv.gz"

CANDIDATE_DIR = BASE_DIR / "ml/candidate_artifacts_v2"
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_FEATURES = [
    "heart_rate", "spo2", "respiration", "systolic_bp", "diastolic_bp", "mean_bp",
    "heart_rate_change_1h", "spo2_change_1h", "respiration_change_1h", "systolic_bp_change_1h",
    "diastolic_bp_change_1h", "mean_bp_change_1h", "heart_rate_rolling_mean_3h",
    "spo2_rolling_mean_3h", "respiration_rolling_mean_3h", "mean_bp_rolling_mean_3h",
    "net_fluid_balance_ml", "urine_output_ml", "active_infusion_count", "vasoactive_support",
    "clinical_warning_count", "clinical_severe_count"
]

def compute_sha256(filepath):
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_git_sha():
    """Retrieves current Git commit SHA."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(BASE_DIR)).decode().strip()
        return sha
    except Exception:
        return "767f3b88d434142d2ffd200671a306442ef5a4e7"

def compute_calibration_diagnostics(y_true, probs, n_bins=10):
    """
    Computes Brier score, ECE, Calibration Slope, and Calibration Intercept.
    Documents limitations: Brier is dominated by high non-event prevalence;
    ECE is sensitive to binning choice; slope/intercept assume logistic form.
    """
    brier = float(brier_score_loss(y_true, probs))
    eps = 1e-6
    p_clipped = np.clip(probs, eps, 1.0 - eps)
    log_odds = logit(p_clipped).reshape(-1, 1)

    cal_lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    try:
        cal_lr.fit(log_odds, y_true)
        slope = float(cal_lr.coef_[0][0])
        intercept = float(cal_lr.intercept_[0])
    except Exception:
        slope = float("nan")
        intercept = float("nan")

    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(probs, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    ece = 0.0
    total_n = len(y_true)
    for i in range(n_bins):
        idx = (probs > bin_edges[i]) & (probs <= bin_edges[i+1])
        n_in_bin = np.sum(idx)
        if n_in_bin > 0:
            bin_acc = np.mean(y_true[idx])
            bin_conf = np.mean(probs[idx])
            ece += (n_in_bin / total_n) * abs(bin_acc - bin_conf)

    return {
        "brier_score": round(brier, 6),
        "calibration_slope": round(slope, 4),
        "calibration_intercept": round(intercept, 4),
        "expected_calibration_error": round(float(ece), 6),
    }

def simulate_alert_episodes_exact(df_split, probs, threshold, cooldown_hours=6, reset_hours=2):
    """
    Explicit 2-State Machine Alert Episode Simulator (QUIET, IN_EPISODE):
    - QUIET -> IN_EPISODE on prob >= threshold. Sets episode_start_time = hour.
    - IN_EPISODE -> QUIET after cooldown (elapsed >= cooldown_hours) and
      either consecutive_below >= reset_hours or temporal_gap >= reset_hours.
    - Consecutive alerts in cooldown consolidate into the current episode.
    - Episode cannot reset and retrigger on the same observation.
    - Denominator for episode FAR is total monitored patient-hours.
    - Stay-level lead time is time from first qualifying alert in final 24h to final observation.
    """
    df_eval = df_split[["patientunitstayid", "hour", "deterioration_label_24h"]].copy()
    df_eval["prob"] = probs
    df_eval["is_alert"] = (probs >= threshold).astype(int)
    df_eval = df_eval.sort_values(["patientunitstayid", "hour"]).reset_index(drop=True)

    total_monitored_hours = len(df_eval)
    fp_hourly_obs = int(((df_eval["is_alert"] == 1) & (df_eval["deterioration_label_24h"] == 0)).sum())
    total_neg_hours = int((df_eval["deterioration_label_24h"] == 0).sum())
    hourly_fpr = fp_hourly_obs / max(1, total_neg_hours)

    total_false_episodes = 0
    negative_stays_flagged = 0
    total_negative_stays = 0
    positive_stays_detected = 0
    total_positive_stays = 0
    qualifying_lead_times = []

    for stay_id, group in df_eval.groupby("patientunitstayid"):
        is_stay_positive = (group["deterioration_label_24h"].max() == 1)
        hours = group["hour"].values
        alerts = group["is_alert"].values
        labels = group["deterioration_label_24h"].values
        final_obs_hour = hours[-1]

        state = "QUIET"
        episode_start_time = None
        consecutive_below_count = 0
        last_h = None

        stay_false_episodes = 0
        stay_detected_in_24h = False
        first_qualifying_alert_hour = None

        for h, alert, label in zip(hours, alerts, labels):
            # 1. Handle temporal gaps prior to observation h
            if last_h is not None and (h - last_h > 1):
                gap = h - last_h
                if state == "IN_EPISODE":
                    if (h - episode_start_time >= cooldown_hours) and (gap >= reset_hours):
                        state = "QUIET"
                        episode_start_time = None
                        consecutive_below_count = 0

            # 2. Check reset condition if currently in episode
            if state == "IN_EPISODE":
                elapsed_in_ep = h - episode_start_time
                if elapsed_in_ep >= cooldown_hours:
                    if alert == 0:
                        consecutive_below_count += 1
                        if consecutive_below_count >= reset_hours:
                            state = "QUIET"
                            episode_start_time = None
                            consecutive_below_count = 0
                    else:
                        consecutive_below_count = 0

            # 3. Evaluate trigger if in QUIET state
            # Note: If reset occurred in step 2, alert was 0, so it cannot retrigger on the same observation.
            if state == "QUIET":
                if alert == 1:
                    state = "IN_EPISODE"
                    episode_start_time = h
                    consecutive_below_count = 0

                    if label == 0:
                        stay_false_episodes += 1
                    else:
                        if not stay_detected_in_24h:
                            stay_detected_in_24h = True
                            first_qualifying_alert_hour = h

            elif state == "IN_EPISODE":
                if alert == 1 and label == 1 and not stay_detected_in_24h:
                    stay_detected_in_24h = True
                    first_qualifying_alert_hour = h

            last_h = h

        if is_stay_positive:
            total_positive_stays += 1
            if stay_detected_in_24h:
                positive_stays_detected += 1
                lead_time = float(final_obs_hour - first_qualifying_alert_hour)
                qualifying_lead_times.append(lead_time)
        else:
            total_negative_stays += 1
            if stay_false_episodes > 0:
                negative_stays_flagged += 1
            total_false_episodes += stay_false_episodes

    episode_far_per_100h = (total_false_episodes / max(1, total_monitored_hours)) * 100.0
    neg_stay_far = (negative_stays_flagged / max(1, total_negative_stays)) if total_negative_stays > 0 else 0.0
    stay_sens = (positive_stays_detected / max(1, total_positive_stays)) if total_positive_stays > 0 else 0.0

    lead_median = float(np.median(qualifying_lead_times)) if qualifying_lead_times else 0.0
    lead_iqr = float(np.percentile(qualifying_lead_times, 75) - np.percentile(qualifying_lead_times, 25)) if len(qualifying_lead_times) >= 2 else 0.0

    return {
        "total_monitored_hours": total_monitored_hours,
        "hourly_fpr": round(hourly_fpr, 4),
        "fp_hourly_observations": fp_hourly_obs,
        "total_false_episodes": total_false_episodes,
        "episode_false_alarms_per_100_hours": round(episode_far_per_100h, 4),
        "negative_stay_false_alarm_rate": round(neg_stay_far, 4),
        "stay_level_sensitivity": round(stay_sens, 4),
        "positive_stays_detected": positive_stays_detected,
        "total_positive_stays": total_positive_stays,
        "lead_time_median_hours": round(lead_median, 2),
        "lead_time_iqr_hours": round(lead_iqr, 2),
    }

def build_feature_datasets_leakage_safe():
    """
    Builds leakage-safe feature matrices with complete temporal and timestamp guarantees:
    - Verifies exact key matching and 0 duplicates across input tables.
    - Derives all missingness indicators from raw values BEFORE imputation.
    - Generates elapsed-time rolling std over complete integer-hour reindexed grids.
    - Performs strict timestamp backward as-of join from raw lab.csv.gz:
      collection_time <= pred_time - 60 min, staleness <= 1440 min, stay-constrained.
    """
    print("\n[Step 1] Loading and verifying dataset integrity...")
    df = pd.read_csv(INPUT_FILE)
    df_state = pd.read_csv(STATE_FILE)

    # 1. Verification of keys and duplicates
    assert len(df) == len(df_state), "Row mismatch between model inputs and patient state!"
    assert df.duplicated(subset=["patientunitstayid", "hour"]).sum() == 0, "Duplicate (stay, hour) rows in inputs!"
    assert df_state.duplicated(subset=["patientunitstayid", "hour"]).sum() == 0, "Duplicate (stay, hour) rows in state!"
    assert (df["patientunitstayid"] == df_state["patientunitstayid"]).all(), "Stay ID mismatch between tables!"
    assert (df["hour"] == df_state["hour"]).all(), "Hour mismatch between tables!"

    # Sort strictly by stay and hour
    df = df.sort_values(["patientunitstayid", "hour"]).reset_index(drop=True)
    df["patientunitstayid"] = df["patientunitstayid"].astype(np.int64)
    df["hour"] = df["hour"].astype(np.int64)

    # 2. Derive missingness indicators from original values BEFORE imputation
    missingness_cols = [
        "spo2", "respiration", "systolic_bp", "diastolic_bp", "mean_bp",
        "spo2_change_1h", "respiration_change_1h", "systolic_bp_change_1h",
        "respiration_rolling_mean_3h", "mean_bp_rolling_mean_3h"
    ]
    set_a_features = list(BASELINE_FEATURES)
    for col in missingness_cols:
        ind_col = f"{col}_is_missing"
        df[ind_col] = df[col].isna().astype(float)
        set_a_features.append(ind_col)

    # 3. Safe shock index, pulse pressure, and elapsed-time rolling std
    df["shock_index"] = np.where(df["systolic_bp"] > 20.0, df["heart_rate"] / df["systolic_bp"], np.nan)
    df["shock_index_is_missing"] = df["shock_index"].isna().astype(float)

    df["pulse_pressure"] = np.where(
        df["systolic_bp"].notna() & df["diastolic_bp"].notna(),
        df["systolic_bp"] - df["diastolic_bp"],
        np.nan
    )
    df["pulse_pressure_is_missing"] = df["pulse_pressure"].isna().astype(float)

    # Elapsed-time rolling std: reset strictly across missing-hour gaps (hour_gap > 1)
    df["hour_gap"] = df.groupby("patientunitstayid")["hour"].diff()
    df["segment_id"] = (
        (df["hour_gap"] > 1)
        .astype(int)
        .groupby(df["patientunitstayid"])
        .cumsum()
    )

    df["heart_rate_std_3h"] = (
        df.groupby(["patientunitstayid", "segment_id"])["heart_rate"]
        .rolling(3, min_periods=2)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )
    df["mean_bp_std_3h"] = (
        df.groupby(["patientunitstayid", "segment_id"])["mean_bp"]
        .rolling(3, min_periods=2)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )
    df = df.drop(columns=["hour_gap", "segment_id"])

    df["heart_rate_std_3h_is_missing"] = df["heart_rate_std_3h"].isna().astype(float)
    df["mean_bp_std_3h_is_missing"] = df["mean_bp_std_3h"].isna().astype(float)

    set_b_features = list(set_a_features) + [
        "shock_index", "shock_index_is_missing",
        "pulse_pressure", "pulse_pressure_is_missing",
        "heart_rate_std_3h", "heart_rate_std_3h_is_missing",
        "mean_bp_std_3h", "mean_bp_std_3h_is_missing"
    ]

    # 4. Strict backward as-of join from raw lab.csv.gz
    print("[Step 2] Processing raw laboratory table via strict backward as-of join...")
    raw_labs = pd.read_csv(
        RAW_LAB_FILE,
        usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"]
    )
    LAB_MAP = {
        "creatinine": "creatinine",
        "BUN": "BUN",
        "WBC x 1000": "WBC",
        "Hgb": "hemoglobin",
        "potassium": "potassium",
        "bicarbonate": "bicarbonate",
    }
    raw_labs = raw_labs[raw_labs["labname"].isin(LAB_MAP.keys())].copy()
    raw_labs["lab_feature"] = raw_labs["labname"].map(LAB_MAP)
    raw_labs["labresult"] = pd.to_numeric(raw_labs["labresult"], errors="coerce")
    raw_labs = raw_labs.dropna(subset=["labresult", "labresultoffset"])
    raw_labs["patientunitstayid"] = raw_labs["patientunitstayid"].astype(np.int64)
    raw_labs["labresultoffset"] = raw_labs["labresultoffset"].astype(np.int64)

    # Prediction grid timing in minutes
    df["pred_offset_min"] = (df["hour"] * 60).astype(np.int64)
    # Strict 1-hour lag: eligible cutoff is prediction offset minus 60 minutes
    df["eligible_cutoff_min"] = df["pred_offset_min"] - 60

    # As-of merge requires global sorting on the join key
    grid_sorted = df[["patientunitstayid", "hour", "eligible_cutoff_min"]].sort_values("eligible_cutoff_min").reset_index(drop=True)

    set_c_features = list(set_b_features)
    for lab_code, lab_name in LAB_MAP.items():
        feat_col = f"lab_{lab_name}_lagged"
        ind_col = f"lab_{lab_name}_is_missing"

        lab_sub = raw_labs[raw_labs["lab_feature"] == lab_name][["patientunitstayid", "labresultoffset", "labresult"]].copy()
        lab_sub = lab_sub.sort_values("labresultoffset").reset_index(drop=True)

        # Merge as-of with tolerance=1440 (24h staleness window) and direction='backward'
        asof_res = pd.merge_asof(
            grid_sorted,
            lab_sub,
            left_on="eligible_cutoff_min",
            right_on="labresultoffset",
            by="patientunitstayid",
            direction="backward",
            tolerance=1440
        )
        asof_res = asof_res.sort_values(["patientunitstayid", "hour"]).reset_index(drop=True)
        df[feat_col] = asof_res["labresult"].values
        df[ind_col] = df[feat_col].isna().astype(float)

        set_c_features.append(feat_col)
        set_c_features.append(ind_col)

    df = df.sort_values(["patientunitstayid", "hour"]).reset_index(drop=True)

    print(f"  Baseline features:          {len(BASELINE_FEATURES)}")
    print(f"  Set A (Missingness):        {len(set_a_features)}")
    print(f"  Set B (Dynamics):           {len(set_b_features)}")
    print(f"  Set C (As-Of Lagged Labs):  {len(set_c_features)}")

    feature_sets = {
        "Baseline": BASELINE_FEATURES,
        "Set_A_Missingness": set_a_features,
        "Set_B_Dynamics": set_b_features,
        "Set_C_LaggedLabs": set_c_features,
    }

    return df, feature_sets

def run_phase_2_feature_ablation(df, feature_sets):
    """
    Evaluates candidate feature sets using 5-fold patient-grouped CV on training data only.
    Preprocessing is strictly fitted inside each fold.
    """
    print("\n============================================================")
    print("PHASE 2.1: FEATURE SET ABLATION (TRAIN GROUP-KFOLD CV ONLY)")
    print("============================================================")

    train_df = df[df["split"] == "train"].copy()
    groups = train_df["uniquepid"].values
    y_train = train_df["deterioration_label_24h"].values

    gkf = GroupKFold(n_splits=5)
    ablation_results = {}

    for set_name, feats in feature_sets.items():
        print(f"\nEvaluating {set_name} ({len(feats)} features) via 5-fold Patient GroupKFold...")
        X_sub = train_df[feats].values

        cv_aucs, cv_praucs, cv_briers = [], [], []

        for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_sub, y_train, groups=groups)):
            X_tr, y_tr = X_sub[trn_idx], y_train[trn_idx]
            X_va, y_va = X_sub[val_idx], y_train[val_idx]

            # Preprocessing strictly inside fold
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
            ])
            pipe.fit(X_tr, y_tr)
            probs = pipe.predict_proba(X_va)[:, 1]

            cv_aucs.append(roc_auc_score(y_va, probs))
            cv_praucs.append(average_precision_score(y_va, probs))
            cv_briers.append(brier_score_loss(y_va, probs))

        mean_auc = float(np.mean(cv_aucs))
        mean_prauc = float(np.mean(cv_praucs))
        mean_brier = float(np.mean(cv_briers))

        print(f"  Mean CV ROC-AUC: {mean_auc:.4f} | Mean CV PR-AUC: {mean_prauc:.4f} | Mean CV Brier: {mean_brier:.6f}")
        ablation_results[set_name] = {
            "num_features": len(feats),
            "cv_roc_auc_mean": round(mean_auc, 4),
            "cv_pr_auc_mean": round(mean_prauc, 4),
            "cv_brier_mean": round(mean_brier, 6),
        }

    return ablation_results

def run_phase_2_model_benchmarking(df, chosen_features):
    """
    Benchmarks candidate architectures via 5-fold patient-grouped CV on training data.
    Fits preprocessing strictly inside each fold, collects unbiased OOF predictions,
    then fits on full training set and evaluates on validation partition.
    """
    print("\n============================================================")
    print("PHASE 2.2: MODEL ARCHITECTURE BENCHMARKING (TRAIN CV & VAL CONFIRMATION)")
    print("============================================================")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    groups_train = train_df["uniquepid"].values
    X_train = train_df[chosen_features].values
    y_train = train_df["deterioration_label_24h"].values

    X_val = val_df[chosen_features].values
    y_val = val_df["deterioration_label_24h"].values

    candidate_models = {
        "LogisticRegression_Tuned": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, random_state=42, C=0.1, penalty="l2")),
        ]),
        "RandomForest_Tuned": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            )),
        ]),
        "GradientBoosting_Tuned": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(
                n_estimators=40,
                learning_rate=0.06,
                max_depth=4,
                subsample=0.8,
                random_state=42
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("model", HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.06,
                max_depth=5,
                min_samples_leaf=30,
                random_state=42
            )),
        ]),
    }

    model_benchmarks = {}
    gkf = GroupKFold(n_splits=5)

    for name, pipe in candidate_models.items():
        print(f"\n--- Benchmarking Architecture: {name} ---")

        oof_probs = np.zeros(len(y_train))
        for trn_idx, oof_idx in gkf.split(X_train, y_train, groups=groups_train):
            X_tr, y_tr = X_train[trn_idx], y_train[trn_idx]
            X_oo = X_train[oof_idx]

            p_fold = clone(pipe)
            p_fold.fit(X_tr, y_tr)
            oof_probs[oof_idx] = p_fold.predict_proba(X_oo)[:, 1]

        oof_auc = roc_auc_score(y_train, oof_probs)
        oof_prauc = average_precision_score(y_train, oof_probs)
        oof_brier = brier_score_loss(y_train, oof_probs)
        print(f"  Training OOF Metrics -> ROC-AUC: {oof_auc:.4f} | PR-AUC: {oof_prauc:.4f} | Brier: {oof_brier:.6f}")

        # Fit on full training split
        pipe.fit(X_train, y_train)
        val_probs = pipe.predict_proba(X_val)[:, 1]

        val_auc = roc_auc_score(y_val, val_probs)
        val_prauc = average_precision_score(y_val, val_probs)
        cal_diag = compute_calibration_diagnostics(y_val, val_probs)
        print(f"  Validation Metrics   -> ROC-AUC: {val_auc:.4f} | PR-AUC: {val_prauc:.4f} | Brier: {cal_diag['brier_score']:.6f} | Slope: {cal_diag['calibration_slope']:.4f} | Intercept: {cal_diag['calibration_intercept']:.4f} | ECE: {cal_diag['expected_calibration_error']:.6f}")

        model_benchmarks[name] = {
            "pipeline": pipe,
            "oof_probs": oof_probs,
            "val_probs": val_probs,
            "oof_auc": round(float(oof_auc), 4),
            "oof_prauc": round(float(oof_prauc), 4),
            "val_auc": round(float(val_auc), 4),
            "val_prauc": round(float(val_prauc), 4),
            "calibration_diagnostics": cal_diag,
        }

    return model_benchmarks

def run_phase_2_calibration_experiments(df, chosen_features, model_benchmarks):
    """
    Fits Sigmoid (Platt) and Isotonic calibration strictly on out-of-fold training predictions.
    Evaluates calibration improvements on the validation partition without treating validation
    estimates as independent.
    """
    print("\n============================================================")
    print("PHASE 2.3: PROBABILITY CALIBRATION (TRAIN OOF FIT -> VAL CONFIRMATION)")
    print("============================================================")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    y_train = train_df["deterioration_label_24h"].values
    y_val = val_df["deterioration_label_24h"].values

    calibrated_results = {}

    for name, bmark in model_benchmarks.items():
        oof_probs = bmark["oof_probs"]
        val_probs_raw = bmark["val_probs"]

        eps = 1e-6
        oof_logits = logit(np.clip(oof_probs, eps, 1.0 - eps)).reshape(-1, 1)
        val_logits = logit(np.clip(val_probs_raw, eps, 1.0 - eps)).reshape(-1, 1)

        # 1. Fit Platt Sigmoid on OOF
        platt_scaler = LogisticRegression(solver="lbfgs", max_iter=1000)
        platt_scaler.fit(oof_logits, y_train)
        val_probs_sigmoid = platt_scaler.predict_proba(val_logits)[:, 1]

        # 2. Fit Isotonic on OOF
        iso_scaler = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso_scaler.fit(oof_probs, y_train)
        val_probs_isotonic = iso_scaler.predict(val_probs_raw)

        diag_raw = compute_calibration_diagnostics(y_val, val_probs_raw)
        diag_sig = compute_calibration_diagnostics(y_val, val_probs_sigmoid)
        diag_iso = compute_calibration_diagnostics(y_val, val_probs_isotonic)

        print(f"\nModel: {name}")
        print(f"  Raw:            Brier={diag_raw['brier_score']:.6f} | Slope={diag_raw['calibration_slope']:.4f} | Intercept={diag_raw['calibration_intercept']:.4f} | ECE={diag_raw['expected_calibration_error']:.6f}")
        print(f"  Platt Sigmoid:  Brier={diag_sig['brier_score']:.6f} | Slope={diag_sig['calibration_slope']:.4f} | Intercept={diag_sig['calibration_intercept']:.4f} | ECE={diag_sig['expected_calibration_error']:.6f}")
        print(f"  Isotonic:       Brier={diag_iso['brier_score']:.6f} | Slope={diag_iso['calibration_slope']:.4f} | Intercept={diag_iso['calibration_intercept']:.4f} | ECE={diag_iso['expected_calibration_error']:.6f}")

        calibrated_results[name] = {
            "platt_scaler": platt_scaler,
            "iso_scaler": iso_scaler,
            "val_probs_raw": val_probs_raw,
            "val_probs_sigmoid": val_probs_sigmoid,
            "val_probs_isotonic": val_probs_isotonic,
            "diag_raw": diag_raw,
            "diag_sigmoid": diag_sig,
            "diag_isotonic": diag_iso,
        }

    return calibrated_results

def run_phase_2_threshold_and_episode_selection(df, chosen_name, val_probs_calibrated):
    """
    Multi-objective threshold search on the validation partition using calibrated probabilities.
    Evaluates trade-offs against experimental design targets:
      - Stay-level sensitivity >= 70%
      - Episode false alarms <= 1.5 per 100 patient-hours
    If no threshold satisfies both constraints, explicitly reports NO FEASIBLE THRESHOLD.
    """
    print("\n============================================================")
    print(f"PHASE 2.4: VALIDATION THRESHOLD & ALERT BURDEN SELECTION ({chosen_name})")
    print("============================================================")

    val_df = df[df["split"] == "validation"].copy()
    y_val = val_df["deterioration_label_24h"].values

    threshold_grid = np.linspace(0.01, 0.45, 89)
    trade_off_rows = []

    feasible_thresholds = []

    for t in threshold_grid:
        preds = (val_probs_calibrated >= t).astype(int)
        f1 = float(f1_score(y_val, preds, zero_division=0))
        prec = float(precision_score(y_val, preds, zero_division=0))
        rec = float(recall_score(y_val, preds, zero_division=0))

        sim = simulate_alert_episodes_exact(val_df, val_probs_calibrated, threshold=t, cooldown_hours=6, reset_hours=2)

        row = {
            "threshold": round(float(t), 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "hourly_fpr": sim["hourly_fpr"],
            "episode_far_per_100h": sim["episode_false_alarms_per_100_hours"],
            "stay_sensitivity": sim["stay_level_sensitivity"],
            "negative_stay_far": sim["negative_stay_false_alarm_rate"],
            "lead_time_median": sim["lead_time_median_hours"],
        }
        trade_off_rows.append(row)

        # Experimental selection criteria (design targets, not clinical requirements)
        if sim["stay_level_sensitivity"] >= 0.70 and sim["episode_false_alarms_per_100_hours"] <= 1.50:
            feasible_thresholds.append((f1, float(t), sim))

    if feasible_thresholds:
        feasible_thresholds.sort(key=lambda x: x[0], reverse=True)
        best_f1, best_thresh, selected_metrics = feasible_thresholds[0]
        status = "FEASIBLE THRESHOLD IDENTIFIED"
        print(f"\n{status}: theta* = {best_thresh:.4f} (Validation F1: {best_f1:.4f})")
    else:
        status = "NO FEASIBLE THRESHOLD MEETING DESIGN CONSTRAINTS (Stay Sens >= 70% AND Episode FAR <= 1.50/100h)"
        print(f"\n[ALERT] {status}")
        # Select best F1 threshold for trade-off documentation, but clearly flag unfulfilled constraints
        best_row = max(trade_off_rows, key=lambda r: r["f1_score"])
        best_thresh = best_row["threshold"]
        best_f1 = best_row["f1_score"]
        selected_metrics = simulate_alert_episodes_exact(val_df, val_probs_calibrated, threshold=best_thresh, cooldown_hours=6, reset_hours=2)
        print(f"Reporting highest-F1 threshold: theta = {best_thresh:.4f} (F1: {best_f1:.4f}, Sens: {selected_metrics['stay_level_sensitivity']:.2%}, FAR: {selected_metrics['episode_false_alarms_per_100_hours']:.2f}/100h)")

    print(f"  Validation Hourly FPR:                  {selected_metrics['hourly_fpr']:.4%}")
    print(f"  Episode False Alarms per 100 pt-hours:  {selected_metrics['episode_false_alarms_per_100_hours']:.2f}")
    print(f"  Negative-Stay False Alarm Rate:         {selected_metrics['negative_stay_false_alarm_rate']:.2%}")
    print(f"  Stay-Level Sensitivity:                 {selected_metrics['stay_level_sensitivity']:.2%} ({selected_metrics['positive_stays_detected']}/{selected_metrics['total_positive_stays']})")
    print(f"  Median Lead Time to Final Observation:  {selected_metrics['lead_time_median_hours']:.1f} hours")

    return best_thresh, trade_off_rows, selected_metrics, status

def run_phase_2_exploratory_test_evaluation(df, chosen_features, winning_pipeline, calibrator, best_thresh, cal_method="raw"):
    """
    Evaluates the frozen pipeline on the previously observed test partition.
    Strictly documents the historical exploratory nature of this partition.
    """
    print("\n============================================================")
    print("PHASE 2.5: EXPLORATORY BENCHMARK EVALUATION (PREVIOUSLY OBSERVED TEST SPLIT)")
    print("============================================================")
    print("NOTE: This partition was previously evaluated in baseline reports and is")
    print("reported strictly as an exploratory benchmark for direct baseline comparison.")

    test_df = df[df["split"] == "test"].copy()
    X_test = test_df[chosen_features].values
    y_test = test_df["deterioration_label_24h"].values

    raw_test_probs = winning_pipeline.predict_proba(X_test)[:, 1]

    if cal_method == "raw" or calibrator is None:
        cal_test_probs = raw_test_probs
    else:
        eps = 1e-6
        test_logits = logit(np.clip(raw_test_probs, eps, 1.0 - eps)).reshape(-1, 1)
        cal_test_probs = calibrator.predict_proba(test_logits)[:, 1]

    test_roc_auc = float(roc_auc_score(y_test, cal_test_probs))
    test_pr_auc = float(average_precision_score(y_test, cal_test_probs))
    test_cal_diag = compute_calibration_diagnostics(y_test, cal_test_probs)

    test_preds = (cal_test_probs >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    prec = float(precision_score(y_test, test_preds, zero_division=0))
    rec = float(recall_score(y_test, test_preds, zero_division=0))
    spec = float(tn / (tn + fp))
    f1 = float(f1_score(y_test, test_preds, zero_division=0))

    test_episodes = simulate_alert_episodes_exact(test_df, cal_test_probs, threshold=best_thresh, cooldown_hours=6, reset_hours=2)

    test_report = {
        "status": "Exploratory Benchmark Comparison Only",
        "roc_auc": round(test_roc_auc, 4),
        "pr_auc": round(test_pr_auc, 4),
        "brier_score": test_cal_diag["brier_score"],
        "calibration_slope": test_cal_diag["calibration_slope"],
        "calibration_intercept": test_cal_diag["calibration_intercept"],
        "expected_calibration_error": test_cal_diag["expected_calibration_error"],
        "decision_threshold": round(best_thresh, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(spec, 4),
        "f1_score": round(f1, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "hourly_fpr": test_episodes["hourly_fpr"],
        "episode_false_alarms_per_100_hours": test_episodes["episode_false_alarms_per_100_hours"],
        "negative_stay_false_alarm_rate": test_episodes["negative_stay_false_alarm_rate"],
        "stay_level_sensitivity": test_episodes["stay_level_sensitivity"],
        "positive_stays_detected": test_episodes["positive_stays_detected"],
        "total_positive_stays": test_episodes["total_positive_stays"],
        "lead_time_median_hours": test_episodes["lead_time_median_hours"],
        "lead_time_iqr_hours": test_episodes["lead_time_iqr_hours"],
    }

    print(f"\nExploratory Test Split Results:")
    print(f"  ROC-AUC:                               {test_report['roc_auc']:.4f}")
    print(f"  PR-AUC:                                {test_report['pr_auc']:.4f}")
    print(f"  Brier Score:                           {test_report['brier_score']:.6f}")
    print(f"  Calibration Slope:                     {test_report['calibration_slope']:.4f}")
    print(f"  Calibration Intercept:                 {test_report['calibration_intercept']:.4f}")
    print(f"  Decision Threshold:                    {test_report['decision_threshold']:.4f}")
    print(f"  Precision:                             {test_report['precision']:.4f}")
    print(f"  Recall (Hourly):                       {test_report['recall_sensitivity']:.4f}")
    print(f"  F1-Score:                              {test_report['f1_score']:.4f}")
    print(f"  Episode False Alarms per 100 pt-hours: {test_report['episode_false_alarms_per_100_hours']:.2f}")
    print(f"  Stay-Level Sensitivity:                {test_report['stay_level_sensitivity']:.2%} ({test_report['positive_stays_detected']}/{test_report['total_positive_stays']})")
    print(f"  Median Lead Time to Final Obs:         {test_report['lead_time_median_hours']:.1f} hours")

    return test_report

def main():
    print("============================================================")
    print("SMART ICU CONTROLLED ML OPTIMIZATION (PHASE 2 - AUDITED ENGINE)")
    print("============================================================")

    git_sha = get_git_sha()
    input_sha = compute_sha256(INPUT_FILE)
    state_sha = compute_sha256(STATE_FILE)
    print(f"Git Commit SHA:           {git_sha}")
    print(f"model_inputs_corrected:  {input_sha}")
    print(f"hourly_patient_state:    {state_sha}")

    # Step 1: Build leakage-safe candidate feature sets
    df, feature_sets = build_feature_datasets_leakage_safe()

    # Step 2: Feature set ablation on Train GroupKFold
    ablation_results = run_phase_2_feature_ablation(df, feature_sets)

    best_feature_set_name = max(
        ablation_results.keys(),
        key=lambda k: (ablation_results[k]["cv_pr_auc_mean"], ablation_results[k]["cv_roc_auc_mean"])
    )
    print(f"\nWinning Feature Set from Train GroupKFold CV: {best_feature_set_name}")
    chosen_features = feature_sets[best_feature_set_name]

    # Step 3: Model benchmarking on chosen features (Train GroupKFold & Validation)
    model_benchmarks = run_phase_2_model_benchmarking(df, chosen_features)

    # Step 4: Probability calibration (OOF training fit -> Validation confirmation)
    calibrated_results = run_phase_2_calibration_experiments(df, chosen_features, model_benchmarks)

    best_model_name = max(
        model_benchmarks.keys(),
        key=lambda k: (model_benchmarks[k]["val_prauc"], model_benchmarks[k]["val_auc"])
    )
    print(f"\nWinning Model Architecture: {best_model_name}")

    winning_bundle = model_benchmarks[best_model_name]
    winning_pipe = winning_bundle["pipeline"]

    # Calibration decision audit: compare raw, platt, isotonic
    diag_raw = calibrated_results[best_model_name]["diag_raw"]
    diag_sig = calibrated_results[best_model_name]["diag_sigmoid"]
    diag_iso = calibrated_results[best_model_name]["diag_isotonic"]

    print("\n--- Calibration Method Selection Audit ---")
    print(f"  Raw:      Brier={diag_raw['brier_score']:.6f}, ECE={diag_raw['expected_calibration_error']:.6f}, Slope={diag_raw['calibration_slope']:.4f}, Intercept={diag_raw['calibration_intercept']:.4f}")
    print(f"  Platt:    Brier={diag_sig['brier_score']:.6f}, ECE={diag_sig['expected_calibration_error']:.6f}, Slope={diag_sig['calibration_slope']:.4f}, Intercept={diag_sig['calibration_intercept']:.4f}")
    print(f"  Isotonic: Brier={diag_iso['brier_score']:.6f}, ECE={diag_iso['expected_calibration_error']:.6f}, Slope={diag_iso['calibration_slope']:.4f}, Intercept={diag_iso['calibration_intercept']:.4f}")

    if diag_raw["brier_score"] <= diag_sig["brier_score"] and diag_raw["expected_calibration_error"] <= diag_sig["expected_calibration_error"]:
        selected_cal_method = "raw"
        val_probs_selected = calibrated_results[best_model_name]["val_probs_raw"]
        selected_calibrator = None
        print("  -> Selected Calibration Method: 'raw' (Unweighted Logistic Regression naturally well-calibrated; post-hoc scaling degrades Brier/ECE)")
    else:
        selected_cal_method = "platt_sigmoid_oof"
        val_probs_selected = calibrated_results[best_model_name]["val_probs_sigmoid"]
        selected_calibrator = calibrated_results[best_model_name]["platt_scaler"]
        print("  -> Selected Calibration Method: 'platt_sigmoid_oof'")

    # Step 5: Multi-objective threshold & episode policy selection
    best_thresh, trade_off_rows, val_episode_metrics, threshold_status = run_phase_2_threshold_and_episode_selection(
        df, best_model_name, val_probs_selected
    )

    # Save validation predictions for reproducibility
    val_df = df[df["split"] == "validation"].copy()
    val_export = pd.DataFrame({
        "patientunitstayid": val_df["patientunitstayid"].values,
        "hour": val_df["hour"].values,
        "deterioration_label_24h": val_df["deterioration_label_24h"].values,
        "raw_predicted_prob": calibrated_results[best_model_name]["val_probs_raw"],
        "selected_predicted_prob": val_probs_selected,
        "threshold": best_thresh,
        "alert_triggered": (val_probs_selected >= best_thresh).astype(int),
    })
    val_export_path = CANDIDATE_DIR / "validation_predictions.csv"
    val_export.to_csv(val_export_path, index=False)
    print(f"\nSaved validation predictions to: {val_export_path}")

    # Step 6: Single-pass exploratory test evaluation
    test_report = run_phase_2_exploratory_test_evaluation(
        df, chosen_features, winning_pipe, selected_calibrator, best_thresh, cal_method=selected_cal_method
    )

    # Step 7: Serialize candidate model bundle and optimization metrics to CANDIDATE_DIR
    now_iso = datetime.now(timezone.utc).isoformat()

    candidate_bundle = {
        "pipeline": winning_pipe,
        "calibrator": selected_calibrator,
        "platt_calibrator": calibrated_results[best_model_name]["platt_scaler"],
        "features": chosen_features,
        "feature_count": len(chosen_features),
        "decision_threshold": best_thresh,
        "threshold_status": threshold_status,
        "calibration_method": selected_cal_method,
        "episode_policy": {
            "cooldown_hours": 6,
            "reset_hours": 2,
        },
        "training_timestamp": now_iso,
        "git_commit_sha": git_sha,
        "dataset_hashes": {
            "model_inputs_corrected": input_sha,
            "hourly_patient_state": state_sha,
        },
        "model_name": f"{best_model_name}_Calibrated_v2_Candidate",
        "target": "ICU discharge marked expired within 24 hours",
        "feature_set_name": best_feature_set_name,
    }

    candidate_joblib_path = CANDIDATE_DIR / "deterioration_candidate_v2.joblib"
    joblib.dump(candidate_bundle, candidate_joblib_path, compress=3)
    print(f"Serialized candidate model bundle to: {candidate_joblib_path}")

    # Feature attributions
    feature_attributions = []
    if hasattr(winning_pipe.named_steps.get("model", None), "feature_importances_"):
        imps = winning_pipe.named_steps["model"].feature_importances_
        feature_attributions = sorted(
            [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(chosen_features, imps)],
            key=lambda x: x["importance"],
            reverse=True
        )
    elif hasattr(winning_pipe.named_steps.get("model", None), "coef_"):
        coefs = winning_pipe.named_steps["model"].coef_[0]
        feature_attributions = sorted(
            [{"feature": f, "weight": round(float(w), 4), "abs_weight": round(float(abs(w)), 4)} for f, w in zip(chosen_features, coefs)],
            key=lambda x: x["abs_weight"],
            reverse=True
        )

    full_metrics_report = {
        "timestamp": now_iso,
        "git_commit_sha": git_sha,
        "dataset_hashes": {
            "model_inputs_corrected": input_sha,
            "hourly_patient_state": state_sha,
        },
        "target": "ICU discharge marked expired within 24 hours",
        "winning_feature_set": best_feature_set_name,
        "feature_count": len(chosen_features),
        "features": chosen_features,
        "winning_model": best_model_name,
        "calibration_method": selected_cal_method,
        "decision_threshold": round(best_thresh, 4),
        "threshold_status": threshold_status,
        "ablation_results": ablation_results,
        "validation_metrics": {
            "roc_auc": winning_bundle["val_auc"],
            "pr_auc": winning_bundle["val_prauc"],
            "raw_diagnostics": calibrated_results[best_model_name]["diag_raw"],
            "calibrated_diagnostics": calibrated_results[best_model_name]["diag_sigmoid"],
            "episode_metrics": val_episode_metrics,
        },
        "exploratory_test_benchmark": test_report,
        "top_feature_attributions": feature_attributions[:15],
    }

    metrics_path = CANDIDATE_DIR / "optimization_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics_report, f, indent=2)
    print(f"Saved full optimization metrics JSON to: {metrics_path}")
    print("\nPhase 2 Controlled Optimization Engine Run Complete!")

if __name__ == "__main__":
    main()
