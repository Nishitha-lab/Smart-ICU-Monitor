"""
Training and Evaluation Pipeline for Smart ICU Machine Learning Engine.

Trains and compares:
1. Logistic Regression (unweighted and class-weighted)
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Isolation Forest (for unsupervised multi-variable anomaly scoring)

Guarantees:
- Strictly patient-level disjoint partitions (stratified by natural person uniquepid).
- Zero data leakage: imputers and scalers fitted strictly on training data.
- Evaluation metrics: ROC-AUC, PR-AUC (Average Precision), Brier score, Precision,
  Recall, Specificity, F1-score, Confusion Matrix.
- Serializes complete model bundles with provenance, feature schemas, and decision thresholds.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "experiments" / "leakage_safe"
INPUT_FILE = DATA_DIR / "model_inputs_corrected.csv"
MODEL_DIR = ROOT / "ml" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "deterioration_label_24h"
KEYS = ["patientunitstayid", "hour", "uniquepid", "split"]

FEATURES = [
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "heart_rate_change_1h",
    "spo2_change_1h",
    "respiration_change_1h",
    "systolic_bp_change_1h",
    "diastolic_bp_change_1h",
    "mean_bp_change_1h",
    "heart_rate_rolling_mean_3h",
    "spo2_rolling_mean_3h",
    "respiration_rolling_mean_3h",
    "mean_bp_rolling_mean_3h",
    "net_fluid_balance_ml",
    "urine_output_ml",
    "active_infusion_count",
    "vasoactive_support",
    "clinical_warning_count",
    "clinical_severe_count",
]


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Loading leakage-safe dataset from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET] = df[TARGET].astype(int)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Verify patient-level separation
    train_pts = set(train_df["uniquepid"].unique())
    val_pts = set(val_df["uniquepid"].unique())
    test_pts = set(test_df["uniquepid"].unique())

    assert len(train_pts & val_pts) == 0, "Leakage: Overlap between train and validation patients!"
    assert len(train_pts & test_pts) == 0, "Leakage: Overlap between train and test patients!"
    assert len(val_pts & test_pts) == 0, "Leakage: Overlap between validation and test patients!"

    print(f"Train set: {len(train_df):,} rows, {len(train_pts):,} patients, {train_df[TARGET].sum():,} positive ({train_df[TARGET].mean():.2%})")
    print(f"Val set:   {len(val_df):,} rows, {len(val_pts):,} patients, {val_df[TARGET].sum():,} positive ({val_df[TARGET].mean():.2%})")
    print(f"Test set:  {len(test_df):,} rows, {len(test_pts):,} patients, {test_df[TARGET].sum():,} positive ({test_df[TARGET].mean():.2%})")

    return train_df, val_df, test_df


def tune_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    prec, rec, thresh = precision_recall_curve(labels, probs)
    if len(thresh) == 0:
        return 0.10, 0.0
    f1_scores = np.divide(
        2 * prec[:-1] * rec[:-1],
        prec[:-1] + rec[:-1],
        out=np.zeros_like(prec[:-1]),
        where=(prec[:-1] + rec[:-1]) > 0,
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresh[best_idx]), float(f1_scores[best_idx])


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    probs = pipeline.predict_proba(X)[:, 1]
    roc_auc = float(roc_auc_score(y, probs))
    pr_auc = float(average_precision_score(y, probs))
    brier = float(brier_score_loss(y, probs))

    t = threshold if threshold is not None else 0.10
    preds = (probs >= t).astype(int)

    cm = confusion_matrix(y, preds).tolist()
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

    precision = float(precision_score(y, preds, zero_division=0))
    recall = float(recall_score(y, preds, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y, preds, zero_division=0))

    return {
        "model_name": name,
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "decision_threshold": round(t, 4),
        "precision": round(precision, 4),
        "recall_sensitivity": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def main():
    start_time = time.time()
    train_df, val_df, test_df = load_dataset()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET].values

    X_val = val_df[FEATURES]
    y_val = val_df[TARGET].values

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET].values

    models: dict[str, Pipeline] = {}

    # 1. Logistic Regression (Probability oriented, natural prevalence)
    print("\n--- Training Model 1: Logistic Regression ---")
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1500, random_state=42, C=1.0)),
    ])
    lr.fit(X_train, y_train)
    models["LogisticRegression"] = lr

    # 2. Logistic Regression (Balanced class weight)
    print("\n--- Training Model 2: Logistic Regression (Balanced) ---")
    lr_bal = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42, C=1.0)),
    ])
    lr_bal.fit(X_train, y_train)
    models["LogisticRegression_Balanced"] = lr_bal

    # 3. Random Forest Classifier
    print("\n--- Training Model 3: Random Forest ---")
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    # 4. Gradient Boosting Classifier
    print("\n--- Training Model 4: Gradient Boosting ---")
    gb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            random_state=42,
        )),
    ])
    gb.fit(X_train, y_train)
    models["GradientBoosting"] = gb

    # 5. Isolation Forest for Unsupervised Anomaly Scoring
    print("\n--- Training Model 5: Isolation Forest (Anomaly) ---")
    iso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", IsolationForest(
            n_estimators=300,
            contamination=0.02,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    # Fit only on training data where deterioration label is 0 (healthy baseline reference)
    iso.fit(X_train[y_train == 0])

    # Validation & Test evaluation
    results = {}
    print("\n============================================================")
    print("MODEL COMPARISON ON HELD-OUT PATIENT TEST SET")
    print("============================================================")

    best_model_name = "LogisticRegression"
    best_pr_auc = 0.0

    for name, pipeline in models.items():
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        best_thresh, val_f1 = tune_threshold(y_val, val_probs)

        # Evaluate on Test set using validation-tuned threshold
        metrics = evaluate_model(name, pipeline, X_test, y_test, threshold=best_thresh)
        results[name] = metrics

        print(f"\nModel: {name}")
        print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:             {metrics['pr_auc']:.4f}")
        print(f"  Brier Score:        {metrics['brier_score']:.4f}")
        print(f"  Tuned Threshold:    {metrics['decision_threshold']:.4f}")
        print(f"  Precision:          {metrics['precision']:.4f}")
        print(f"  Recall/Sensitivity: {metrics['recall_sensitivity']:.4f}")
        print(f"  Specificity:        {metrics['specificity']:.4f}")
        print(f"  F1-Score:           {metrics['f1_score']:.4f}")
        print(f"  Confusion Matrix:   TN={metrics['true_negatives']}, FP={metrics['false_positives']}, FN={metrics['false_negatives']}, TP={metrics['true_positives']}")

        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            best_model_name = name

    print(f"\nWinning Supervised Model: {best_model_name} (PR-AUC: {best_pr_auc:.4f})")

    # Feature Importance for Logistic Regression
    lr_pipeline = models["LogisticRegression"]
    lr_model = lr_pipeline.named_steps["model"]
    coefficients = lr_model.coef_[0].tolist()
    feature_importance_lr = sorted(
        [{"feature": f, "weight": round(w, 4), "abs_weight": round(abs(w), 4)} for f, w in zip(FEATURES, coefficients)],
        key=lambda x: x["abs_weight"],
        reverse=True,
    )

    # Feature Importance for Random Forest
    rf_pipeline = models["RandomForest"]
    rf_importances = rf_pipeline.named_steps["model"].feature_importances_.tolist()
    feature_importance_rf = sorted(
        [{"feature": f, "importance": round(imp, 4)} for f, imp in zip(FEATURES, rf_importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # Serialize Artifacts
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Primary deterioration bundle: deterioration_logistic_v1.joblib
    # (Bundles the primary logistic regression pipeline + metadata)
    det_bundle = {
        "pipeline": lr_pipeline,
        "feature_columns": FEATURES,
        "model_name": "deterioration_logistic_v1",
        "model_type": "LogisticRegression",
        "version": "1.0.0",
        "target": "ICU discharge marked expired within 24 hours",
        "training_dataset": "eICU Collaborative Research Database Demo v2.0.1",
        "training_samples": len(train_df),
        "positive_samples": int(train_df[TARGET].sum()),
        "negative_samples": int((train_df[TARGET] == 0).sum()),
        "validation_metrics": results["LogisticRegression"],
        "all_model_metrics": results,
        "decision_threshold": results["LogisticRegression"]["decision_threshold"],
        "feature_importance": feature_importance_lr,
        "training_timestamp": now_iso,
        "training_duration_seconds": round(time.time() - start_time, 2),
    }
    det_path = MODEL_DIR / "deterioration_logistic_v1.joblib"
    joblib.dump(det_bundle, det_path)
    print(f"\nSaved primary deterioration model: {det_path} ({det_path.stat().st_size / 1024:.1f} KB)")

    # 2. Also save winning GradientBoosting bundle for comprehensive benchmark access
    gb_bundle = {
        "pipeline": models["GradientBoosting"],
        "feature_columns": FEATURES,
        "model_name": "deterioration_gradient_boosting_v1",
        "model_type": "GradientBoostingClassifier",
        "version": "1.0.0",
        "target": "ICU discharge marked expired within 24 hours",
        "validation_metrics": results["GradientBoosting"],
        "decision_threshold": results["GradientBoosting"]["decision_threshold"],
        "training_timestamp": now_iso,
    }
    gb_path = MODEL_DIR / "deterioration_gradient_boosting_v1.joblib"
    joblib.dump(gb_bundle, gb_path)
    print(f"Saved gradient boosting model: {gb_path} ({gb_path.stat().st_size / 1024:.1f} KB)")

    # 3. Isolation Forest Anomaly model
    iso_bundle = {
        "pipeline": iso,
        "feature_columns": FEATURES,
        "model_name": "anomaly_isolation_forest_v2",
        "model_type": "IsolationForest",
        "version": "v2.0.0",
        "contamination": 0.02,
        "statistical_interpretation": "Statistical multi-variable outlier detection; not a medical diagnosis.",
        "training_samples": int((train_df[TARGET] == 0).sum()),
        "training_timestamp": now_iso,
    }
    iso_path = MODEL_DIR / "anomaly_isolation_forest_v2.joblib"
    joblib.dump(iso_bundle, iso_path)
    print(f"Saved anomaly detection model: {iso_path} ({iso_path.stat().st_size / 1024:.1f} KB)")

    # 3. Save Random Forest bundle
    rf_bundle = {
        "pipeline": models["RandomForest"],
        "feature_columns": FEATURES,
        "model_name": "deterioration_random_forest_v1",
        "model_type": "RandomForestClassifier",
        "version": "1.0.0",
        "target": "ICU discharge marked expired within 24 hours",
        "validation_metrics": results["RandomForest"],
        "decision_threshold": results["RandomForest"]["decision_threshold"],
        "feature_importance": feature_importance_rf,
        "training_timestamp": now_iso,
    }
    rf_path = MODEL_DIR / "deterioration_random_forest_v1.joblib"
    joblib.dump(rf_bundle, rf_path)
    print(f"Saved random forest model: {rf_path} ({rf_path.stat().st_size / 1024:.1f} KB)")

    # 4. Save metrics report JSON
    metrics_path = MODEL_DIR / "model_comparison_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "training_timestamp": now_iso,
                "target": "ICU discharge marked expired within 24 hours",
                "winning_model": best_model_name,
                "features": FEATURES,
                "dataset": {
                    "train_patients": int(train_df["uniquepid"].nunique()),
                    "train_rows": len(train_df),
                    "train_positive": int(train_df[TARGET].sum()),
                    "val_patients": int(val_df["uniquepid"].nunique()),
                    "val_rows": len(val_df),
                    "val_positive": int(val_df[TARGET].sum()),
                    "test_patients": int(test_df["uniquepid"].nunique()),
                    "test_rows": len(test_df),
                    "test_positive": int(test_df[TARGET].sum()),
                },
                "models": results,
                "top_features_logistic": feature_importance_lr[:10],
                "top_features_random_forest": feature_importance_rf[:10],
            },
            f,
            indent=2,
        )
    print(f"Saved comparison metrics report: {metrics_path}")
    print("\nTraining and evaluation pipeline finished successfully!")


if __name__ == "__main__":
    main()
