from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "processed" / "experiments" / "leakage_safe"
OUTPUT_DIR = ROOT / "data" / "processed" / "experiments" / "paired_models"
MODEL_DIR = ROOT / "models" / "experimental" / "spo2_cross_validation"
TARGET = "deterioration_label_24h"
KEYS = ["patientunitstayid", "hour", "uniquepid"]
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
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261109
MODEL_CONFIGS = {
    "balanced_logistic": {"class_weight": "balanced"},
    "probability_logistic": {"class_weight": None},
}


def load_split(version: str, split: str) -> pd.DataFrame:
    path = INPUT_DIR / f"model_inputs_{version}.csv"
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.loc[frame["split"].eq(split)].copy()
    if frame.empty:
        raise ValueError(f"No {split} rows in {path.name}.")
    if frame.duplicated(KEYS[:2]).any():
        raise ValueError(f"Duplicate patient-hour rows in {path.name}.")
    return frame.reset_index(drop=True)


def make_model(class_weight: str | None, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def numeric_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[FEATURES].copy()
    for column in FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def tune_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        raise ValueError("Validation scores did not produce threshold candidates.")
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])


def metric_values(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    return {
        "roc_auc": float(
            roc_auc_score(labels, probabilities, sample_weight=sample_weight)
        ),
        "average_precision": float(
            average_precision_score(labels, probabilities, sample_weight=sample_weight)
        ),
        "brier": float(
            brier_score_loss(labels, probabilities, sample_weight=sample_weight)
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                sample_weight=sample_weight,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                sample_weight=sample_weight,
                zero_division=0,
            )
        ),
    }


def paired_patient_bootstrap(
    baseline: pd.DataFrame,
    corrected: pd.DataFrame,
    model_name: str,
    baseline_threshold: float,
    corrected_threshold: float,
) -> list[dict[str, Any]]:
    left = baseline.sort_values(KEYS[:2]).reset_index(drop=True)
    right = corrected.sort_values(KEYS[:2]).reset_index(drop=True)
    if not left[KEYS[:2]].equals(right[KEYS[:2]]):
        raise ValueError("Paired bootstrap requires aligned test patient-hours.")
    if not left[TARGET].equals(right[TARGET]):
        raise ValueError("Paired bootstrap requires identical test labels.")
    patients, inverse = np.unique(left["uniquepid"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas: dict[str, list[float]] = {
        metric: []
        for metric in [
            "roc_auc",
            "average_precision",
            "brier",
            "precision",
            "recall",
        ]
    }
    labels = left[TARGET].to_numpy(dtype=int)
    base_scores = left["probability"].to_numpy(dtype=float)
    corrected_scores = right["probability"].to_numpy(dtype=float)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(patients), size=len(patients))
        patient_weights = np.bincount(sampled, minlength=len(patients))
        weights = patient_weights[inverse].astype(float)
        if weights.sum() == 0 or np.unique(labels[weights > 0]).size < 2:
            continue
        base_metrics = metric_values(
            labels, base_scores, baseline_threshold, sample_weight=weights
        )
        corrected_metrics = metric_values(
            labels, corrected_scores, corrected_threshold, sample_weight=weights
        )
        for metric in deltas:
            deltas[metric].append(corrected_metrics[metric] - base_metrics[metric])
    rows = []
    for metric, values in deltas.items():
        if len(values) < BOOTSTRAP_REPLICATES * 0.95:
            raise RuntimeError(
                f"Too many invalid patient-bootstrap replicates for {model_name}/{metric}: "
                f"{len(values)} of {BOOTSTRAP_REPLICATES}."
            )
        low, high = np.percentile(values, [2.5, 97.5])
        rows.append(
            {
                "model": model_name,
                "metric": metric,
                "corrected_minus_baseline": float(np.mean(values)),
                "ci_2_5": float(low),
                "ci_97_5": float(high),
                "valid_patient_bootstrap_replicates": len(values),
            }
        )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train = load_split("baseline", "train")
    validation = load_split("baseline", "validation")
    test_baseline = load_split("baseline", "test")
    corrected_splits = {
        split: load_split("corrected", split)
        for split in ("train", "validation", "test")
    }
    for split, baseline in [
        ("train", train),
        ("validation", validation),
        ("test", test_baseline),
    ]:
        corrected = corrected_splits[split]
        left = baseline.sort_values(KEYS[:2]).reset_index(drop=True)
        right = corrected.sort_values(KEYS[:2]).reset_index(drop=True)
        if not left[KEYS[:2]].equals(right[KEYS[:2]]):
            raise ValueError(f"Baseline/corrected {split} keys do not align.")
        if not left[TARGET].equals(right[TARGET]):
            raise ValueError(f"Baseline/corrected {split} targets do not align.")
        if not left["uniquepid"].equals(right["uniquepid"]):
            raise ValueError(f"Baseline/corrected {split} patient IDs do not align.")

    metrics_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    paired_test_results: dict[str, dict[str, pd.DataFrame]] = {}
    for model_name, config in MODEL_CONFIGS.items():
        paired_test_results[model_name] = {}
        thresholds: dict[str, float] = {}
        for version in ("baseline", "corrected"):
            version_train = train if version == "baseline" else corrected_splits["train"]
            version_validation = (
                validation if version == "baseline" else corrected_splits["validation"]
            )
            version_test = (
                test_baseline if version == "baseline" else corrected_splits["test"]
            )
            pipeline = make_model(config["class_weight"], seed=20261110)
            pipeline.fit(numeric_features(version_train), version_train[TARGET].astype(int))
            validation_probabilities = pipeline.predict_proba(
                numeric_features(version_validation)
            )[:, 1]
            threshold, validation_f1 = tune_f1_threshold(
                version_validation[TARGET].to_numpy(dtype=int),
                validation_probabilities,
            )
            thresholds[version] = threshold
            test_probabilities = pipeline.predict_proba(
                numeric_features(version_test)
            )[:, 1]
            test_labels = version_test[TARGET].to_numpy(dtype=int)
            scores = metric_values(test_labels, test_probabilities, threshold)
            metrics_rows.append(
                {
                    "model": model_name,
                    "dataset_version": version,
                    "threshold_selected_on_validation": threshold,
                    "validation_f1_at_selected_threshold": validation_f1,
                    "test_rows": len(version_test),
                    "test_positive_hours": int(test_labels.sum()),
                    "test_positive_prevalence_baseline_ap": float(test_labels.mean()),
                    **scores,
                }
            )
            scored = version_test[KEYS + ["split", TARGET]].copy()
            scored["probability"] = test_probabilities
            scored["predicted_at_validation_threshold"] = (
                test_probabilities >= threshold
            ).astype(int)
            scored.to_csv(
                OUTPUT_DIR / f"test_predictions_{model_name}_{version}.csv",
                index=False,
            )
            paired_test_results[model_name][version] = scored
            bundle = {
                "pipeline": pipeline,
                "feature_columns": FEATURES,
                "target": TARGET,
                "dataset_version": version,
                "model_name": model_name,
                "class_weight": config["class_weight"],
                "validation_threshold_max_f1": threshold,
                "training_split": "patient-level train partition only",
                "experiment_seed": 20261110,
                "input_cohort": str(INPUT_DIR),
                "production_approved": False,
            }
            joblib.dump(
                bundle,
                MODEL_DIR / f"{model_name}_{version}.joblib",
                compress=3,
            )
        bootstrap_rows.extend(
            paired_patient_bootstrap(
                paired_test_results[model_name]["baseline"],
                paired_test_results[model_name]["corrected"],
                model_name,
                thresholds["baseline"],
                thresholds["corrected"],
            )
        )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(OUTPUT_DIR / "paired_patient_bootstrap.csv", index=False)
    configuration = {
        "models": MODEL_CONFIGS,
        "feature_columns": FEATURES,
        "target": TARGET,
        "split_source": str(INPUT_DIR / "patient_split_manifest.csv"),
        "threshold_selection": "maximize hourly F1 on the validation partition",
        "test_usage": "one evaluation after training and threshold selection",
        "bootstrap": {
            "unit": "uniquepid",
            "replicates_requested": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "paired_versions": True,
        },
        "outputs": {
            "metrics": str(OUTPUT_DIR / "test_metrics.csv"),
            "bootstrap": str(OUTPUT_DIR / "paired_patient_bootstrap.csv"),
            "model_bundles": str(MODEL_DIR),
        },
        "limitations": [
            "Retrospective evaluation on the supplied eICU demo extract only.",
            "The hourly target is expired ICU discharge within 24 hours, not general deterioration.",
            "Operating thresholds were selected for F1, not clinical utility.",
            "No model is approved for production or clinical use.",
        ],
    }
    (OUTPUT_DIR / "experiment_configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    report_lines = [
        "# Paired SpO2 model experiment",
        "",
        "This isolated experiment trains balanced and unweighted logistic regression",
        "pipelines separately on baseline and corrected SpO2 model inputs. Imputation",
        "and scaling are fitted only on the patient-level training partition. Each",
        "threshold is selected by validation F1; the test partition is evaluated once.",
        "",
        "## Test metrics",
        "",
        "```text\n" + metrics.to_string(index=False) + "\n```",
        "",
        "## Paired patient-cluster bootstrap deltas",
        "",
        "Corrected-minus-baseline differences use the same patient resamples and paired",
        "patient-hour rows. Intervals are percentile 95% confidence intervals.",
        "",
        "```text\n" + bootstrap.to_string(index=False) + "\n```",
        "",
        "## Limitations",
        "",
        "- Retrospective demo-dataset results; no clinical validity or deployment claim.",
        "- The target is expired ICU discharge within 24 hours, not generic deterioration.",
        "- Thresholds optimize hourly F1 on validation, not clinical utility.",
        "- Results are exploratory and do not authorize model promotion.",
        "",
    ]
    (OUTPUT_DIR / "experiment_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(f"\nSaved paired bootstrap and experiment report under {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
