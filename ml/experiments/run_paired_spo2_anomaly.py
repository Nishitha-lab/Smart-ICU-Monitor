from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "processed" / "experiments" / "leakage_safe"
OUTPUT_DIR = ROOT / "data" / "processed" / "experiments" / "paired_anomaly"
MODEL_DIR = ROOT / "models" / "experimental" / "spo2_cross_validation"
FEATURES = [
    "heart_rate",
    "spo2",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "heart_rate_change_1h",
    "spo2_change_1h",
    "mean_bp_change_1h",
    "heart_rate_rolling_mean_3h",
    "spo2_rolling_mean_3h",
    "respiration_rolling_mean_3h",
    "mean_bp_rolling_mean_3h",
    "net_fluid_balance_ml",
    "urine_output_ml",
    "active_infusion_count",
    "vasoactive_support",
]
KEYS = ["patientunitstayid", "hour", "uniquepid"]


def load_split(version: str, split: str) -> pd.DataFrame:
    frame = pd.read_csv(
        INPUT_DIR / f"model_inputs_{version}.csv",
        low_memory=False,
    )
    frame = frame.loc[frame["split"].eq(split)].copy()
    if frame.empty:
        raise ValueError(f"No {split} rows in {version} input.")
    if frame.duplicated(KEYS[:2]).any():
        raise ValueError(f"Duplicate patient-hour rows in {version} input.")
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, pd.DataFrame] = {}
    summary: list[dict[str, Any]] = []
    for version in ("baseline", "corrected"):
        train = load_split(version, "train")
        test = load_split(version, "test")
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    IsolationForest(
                        n_estimators=300,
                        contamination=0.02,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipeline.fit(train[FEATURES])
        scores = -pipeline.decision_function(test[FEATURES])
        flags = (pipeline.predict(test[FEATURES]) == -1).astype(int)
        scored = test[KEYS + ["clinical_warning_count", "clinical_severe_count"]].copy()
        scored["anomaly_score"] = scores
        scored["anomaly_flag"] = flags
        scored.to_csv(OUTPUT_DIR / f"test_anomaly_{version}.csv", index=False)
        predictions[version] = scored
        summary.append(
            {
                "dataset_version": version,
                "train_rows": len(train),
                "test_rows": len(test),
                "anomaly_rows": int(flags.sum()),
                "anomaly_rate": float(flags.mean()),
                "anomalous_patients": int(
                    scored.loc[scored["anomaly_flag"].eq(1), "uniquepid"].nunique()
                ),
                "anomaly_with_clinical_warning": int(
                    (
                        scored["anomaly_flag"].eq(1)
                        & scored["clinical_warning_count"].gt(0)
                    ).sum()
                ),
                "anomaly_with_clinical_severe_rule": int(
                    (
                        scored["anomaly_flag"].eq(1)
                        & scored["clinical_severe_count"].gt(0)
                    ).sum()
                ),
            }
        )
        joblib.dump(
            {
                "pipeline": pipeline,
                "feature_columns": FEATURES,
                "model_name": "isolation_forest_v2",
                "dataset_version": version,
                "training_split": "patient-level train partition only",
                "random_state": 42,
                "n_estimators": 300,
                "contamination": 0.02,
                "input_cohort": str(INPUT_DIR),
                "production_approved": False,
                "evaluation_scope": "descriptive stability and rule agreement only",
            },
            MODEL_DIR / f"isolation_forest_v2_{version}.joblib",
            compress=3,
        )

    baseline = predictions["baseline"].sort_values(KEYS[:2]).reset_index(drop=True)
    corrected = predictions["corrected"].sort_values(KEYS[:2]).reset_index(drop=True)
    if not baseline[KEYS].equals(corrected[KEYS]):
        raise ValueError("Anomaly test cohorts are not paired.")
    overlap = int(
        (baseline["anomaly_flag"].eq(1) & corrected["anomaly_flag"].eq(1)).sum()
    )
    union = int(
        (baseline["anomaly_flag"].eq(1) | corrected["anomaly_flag"].eq(1)).sum()
    )
    rho = spearmanr(baseline["anomaly_score"], corrected["anomaly_score"])
    comparison = {
        "test_patient_hours": len(baseline),
        "baseline_anomaly_rows": int(baseline["anomaly_flag"].sum()),
        "corrected_anomaly_rows": int(corrected["anomaly_flag"].sum()),
        "flag_overlap_rows": overlap,
        "flag_union_rows": union,
        "flag_jaccard_similarity": float(overlap / union) if union else 1.0,
        "score_spearman_rho": float(rho.statistic),
        "score_spearman_pvalue": float(rho.pvalue),
        "baseline_only_flags": int(
            (baseline["anomaly_flag"].eq(1) & corrected["anomaly_flag"].eq(0)).sum()
        ),
        "corrected_only_flags": int(
            (baseline["anomaly_flag"].eq(0) & corrected["anomaly_flag"].eq(1)).sum()
        ),
    }
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUTPUT_DIR / "anomaly_summary.csv", index=False)
    pd.DataFrame([comparison]).to_csv(
        OUTPUT_DIR / "paired_stability_summary.csv", index=False
    )
    configuration = {
        "model": "IsolationForest",
        "feature_columns": FEATURES,
        "fit_partition": "patient-level train partition only",
        "contamination": 0.02,
        "n_estimators": 300,
        "random_state": 42,
        "evaluation": "descriptive paired score/flag stability and clinical-rule agreement",
        "supervised_anomaly_truth": "not available; no discrimination or clinical-performance metrics computed",
        "model_bundles": str(MODEL_DIR),
        "production_approved": False,
    }
    (OUTPUT_DIR / "experiment_configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    report = [
        "# Paired SpO2 anomaly-model experiment",
        "",
        "Separate IsolationForest pipelines were fitted on the training partition for",
        "each dataset version and scored only on the paired held-out test partition.",
        "Preprocessing is fitted on training data. The contamination setting is held",
        "fixed at 2%, matching the existing v2 research configuration.",
        "",
        "No ground-truth anomaly labels exist in this cohort. Therefore, this report",
        "contains descriptive flag/score stability and agreement with rule counts only;",
        "it does not report anomaly accuracy, clinical performance, or utility.",
        "",
        "## Per-version summary",
        "",
        "```text",
        summary_frame.to_string(index=False),
        "```",
        "",
        "## Paired stability",
        "",
        "```text",
        pd.DataFrame([comparison]).to_string(index=False),
        "```",
        "",
        "No clinical validation, deployment, or model-promotion claim is made.",
        "",
    ]
    (OUTPUT_DIR / "experiment_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))
    print(pd.DataFrame([comparison]).to_string(index=False))


if __name__ == "__main__":
    main()
