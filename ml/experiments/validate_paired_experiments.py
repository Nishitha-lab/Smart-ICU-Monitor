from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "experiments"
COHORT = PROCESSED / "leakage_safe"
PAIRED = PROCESSED / "paired_models"
ANOMALY = PROCESSED / "paired_anomaly"
ALERTS = PROCESSED / "alert_replay"
MODEL_DIR = ROOT / "models" / "experimental" / "spo2_cross_validation"
TARGET = "deterioration_label_24h"
KEYS = ["patientunitstayid", "hour", "uniquepid"]
PAIR_KEYS = ["patientunitstayid", "hour"]
MODELS = ("balanced_logistic", "probability_logistic")
ANOMALY_FEATURES = [
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_inputs(version: str) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(COHORT / f"model_inputs_{version}.csv", low_memory=False)
    if frame.duplicated(KEYS[:2]).any():
        raise AssertionError(f"Duplicate patient-hours in {version} cohort.")
    return {
        split: frame.loc[frame["split"].eq(split)].reset_index(drop=True)
        for split in ("train", "validation", "test")
    }


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])


def compare_float(actual: float, expected: float, label: str) -> None:
    require(
        np.isclose(actual, expected, rtol=1e-10, atol=1e-12),
        f"{label}: got {actual}, expected {expected}.",
    )


def main() -> None:
    checks: list[dict[str, str]] = []

    def passed(name: str, detail: str) -> None:
        checks.append({"check": name, "status": "PASS", "detail": detail})

    cohort_validation = pd.read_csv(COHORT / "validation_checks.csv")
    require(
        not cohort_validation["status"].ne("PASS").any(),
        "Independent cohort validation contains non-PASS results.",
    )
    passed(
        "cohort_validation",
        f"Independent cohort validator reports {len(cohort_validation)} PASS checks.",
    )

    manifest = pd.read_csv(COHORT / "patient_split_manifest.csv")
    split_sets = {
        split: set(manifest.loc[manifest["split"].eq(split), "uniquepid"])
        for split in ("train", "validation", "test")
    }
    require(
        not (split_sets["train"] & split_sets["validation"])
        and not (split_sets["train"] & split_sets["test"])
        and not (split_sets["validation"] & split_sets["test"]),
        "Patient overlap exists across cohort partitions.",
    )
    passed("patient_disjointness", "No natural-person overlap across train/validation/test.")

    metrics = pd.read_csv(PAIRED / "test_metrics.csv")
    for model_name in MODELS:
        baseline_inputs = load_inputs("baseline")
        corrected_inputs = load_inputs("corrected")
        for version, splits in [
            ("baseline", baseline_inputs),
            ("corrected", corrected_inputs),
        ]:
            bundle_path = MODEL_DIR / f"{model_name}_{version}.joblib"
            bundle = joblib.load(bundle_path)
            require(
                bundle["production_approved"] is False,
                f"{model_name}/{version} bundle is not marked experimental.",
            )
            require(
                bundle["training_split"] == "patient-level train partition only",
                f"{model_name}/{version} has unexpected training-split metadata.",
            )
            require(
                str(bundle_path).startswith(str(MODEL_DIR)),
                "Model bundle outside isolated model directory.",
            )
            pipeline = bundle["pipeline"]
            feature_columns = bundle["feature_columns"]
            require(
                TARGET not in feature_columns
                and set(feature_columns).issubset(splits["train"].columns),
                f"Target leakage or missing features in {model_name}/{version} bundle.",
            )
            scored_path = PAIRED / f"test_predictions_{model_name}_{version}.csv"
            scored = pd.read_csv(scored_path, low_memory=False)
            test = splits["test"]
            require(not scored.duplicated(KEYS[:2]).any(), f"Duplicate test predictions for {model_name}/{version}.")
            ordered_test = test.sort_values(KEYS).reset_index(drop=True)
            ordered_scored = scored.sort_values(KEYS).reset_index(drop=True)
            require(
                ordered_test[KEYS + [TARGET]].equals(ordered_scored[KEYS + [TARGET]]),
                f"Test prediction rows/labels differ from {version} held-out cohort.",
            )
            require(
                ordered_scored["probability"].between(0, 1).all(),
                f"Out-of-range probability in {model_name}/{version}.",
            )
            recomputed_test = pipeline.predict_proba(ordered_test[feature_columns])[:, 1]
            require(
                np.allclose(
                    recomputed_test,
                    ordered_scored["probability"].to_numpy(),
                    rtol=1e-12,
                    atol=1e-14,
                ),
                f"Saved test predictions differ from experimental bundle for {model_name}/{version}.",
            )
            validation = splits["validation"].sort_values(KEYS).reset_index(drop=True)
            validation_scores = pipeline.predict_proba(validation[feature_columns])[:, 1]
            threshold, validation_f1 = choose_threshold(
                validation[TARGET].to_numpy(dtype=int), validation_scores
            )
            row = metrics.loc[
                metrics["model"].eq(model_name)
                & metrics["dataset_version"].eq(version)
            ]
            require(len(row) == 1, f"Missing/duplicate metric row for {model_name}/{version}.")
            row = row.iloc[0]
            compare_float(
                float(row["threshold_selected_on_validation"]),
                threshold,
                f"{model_name}/{version} validation threshold",
            )
            compare_float(
                float(row["validation_f1_at_selected_threshold"]),
                validation_f1,
                f"{model_name}/{version} validation F1",
            )
            labels = ordered_test[TARGET].to_numpy(dtype=int)
            scores = ordered_scored["probability"].to_numpy(dtype=float)
            predictions = scores >= threshold
            require(
                ordered_scored["predicted_at_validation_threshold"].equals(
                    pd.Series(predictions.astype(int))
                ),
                f"Stored threshold predictions mismatch for {model_name}/{version}.",
            )
            require(
                int(row["test_rows"]) == len(labels)
                and int(row["test_positive_hours"]) == int(labels.sum()),
                f"Test cohort totals mismatch for {model_name}/{version}.",
            )
            calculated = {
                "roc_auc": roc_auc_score(labels, scores),
                "average_precision": average_precision_score(labels, scores),
                "brier": brier_score_loss(labels, scores),
                "precision": precision_score(labels, predictions, zero_division=0),
                "recall": recall_score(labels, predictions, zero_division=0),
            }
            for name, value in calculated.items():
                compare_float(float(row[name]), float(value), f"{model_name}/{version} {name}")
            passed(
                f"{model_name}_{version}",
                "Bundle reproduces test probabilities; validation threshold and test metrics independently recomputed.",
            )

    bootstrap = pd.read_csv(PAIRED / "paired_patient_bootstrap.csv")
    require(
        len(bootstrap) == 10
        and bootstrap["valid_patient_bootstrap_replicates"].eq(1000).all(),
        "Paired bootstrap metric/replicate counts are incomplete.",
    )
    require(
        (bootstrap["ci_2_5"] <= bootstrap["ci_97_5"]).all(),
        "Invalid paired-bootstrap confidence interval ordering.",
    )
    passed("paired_bootstrap", "Ten paired metrics each use 1,000 valid patient-cluster replicates.")

    rules = pd.read_csv(
        ROOT / "data" / "processed" / "hourly_patient_state_rules_v2.csv",
        low_memory=False,
    )
    warning_columns = [
        "rule_tachycardia", "rule_bradycardia", "rule_hypoxemia",
        "rule_abnormal_respiration", "rule_hypotension_sbp", "rule_low_map",
        "rule_rapid_hr_rise", "rule_rapid_hr_fall", "rule_falling_spo2",
        "rule_falling_map", "rule_vasoactive_support", "rule_multiple_infusions",
        "rule_large_positive_fluid_balance", "rule_large_negative_fluid_balance",
    ]
    severe_columns = [
        "rule_severe_tachycardia", "rule_severe_bradycardia",
        "rule_severe_hypoxemia", "rule_severe_respiration",
        "rule_severe_hypotension_sbp", "rule_severe_low_map",
        "rule_severe_hr_rise", "rule_severe_hr_fall",
        "rule_rapid_falling_spo2", "rule_rapid_falling_map",
    ]
    priority_rank = {"normal": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}

    def independent_priority(rank: int, warning_count: int, risk: float) -> str:
        risk10, risk25, risk50 = risk >= 0.10, risk >= 0.25, risk >= 0.50
        if rank >= 4 or (rank >= 3 and risk25) or (risk50 and warning_count >= 1):
            return "critical"
        if rank >= 3 or (rank >= 2 and risk25) or (rank >= 1 and risk25) or risk50:
            return "high"
        if rank >= 2 or (rank >= 1 and risk10) or risk25:
            return "moderate"
        if rank >= 1 or risk10:
            return "low"
        return "normal"

    replay_summary = pd.read_csv(ALERTS / "alert_replay_summary.csv")
    episode_summary = pd.read_csv(ALERTS / "severe_alert_episodes.csv")
    priority_comparison_summary = pd.read_csv(ALERTS / "priority_comparison_summary.csv")
    for model_name in MODELS:
        version_replays: dict[str, pd.DataFrame] = {}
        for version in ("baseline", "corrected"):
            replay = pd.read_csv(
                ALERTS / f"replay_{model_name}_{version}.csv", low_memory=False
            )
            if replay.duplicated(KEYS[:2]).any():
                raise AssertionError(f"Duplicate replay keys for {model_name}/{version}.")
            input_rows = load_inputs(version)["test"]
            joined = input_rows.merge(
                rules[PAIR_KEYS + warning_columns + severe_columns],
                on=PAIR_KEYS,
                how="left",
                validate="one_to_one",
            )
            joined["rule_hypoxemia"] = joined["spo2"].lt(92).astype(int)
            joined["rule_falling_spo2"] = joined["spo2_change_1h"].lt(-3).astype(int)
            joined["rule_severe_hypoxemia"] = joined["spo2"].lt(88).astype(int)
            joined["rule_rapid_falling_spo2"] = joined["spo2_change_1h"].lt(-5).astype(int)
            for column in warning_columns + severe_columns:
                joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(0)
            hypertension = (
                joined["systolic_bp"].gt(180) | joined["diastolic_bp"].gt(120)
            ).fillna(False).astype(int)
            warning_count = joined[warning_columns].sum(axis=1) + hypertension
            severe_count = joined[severe_columns].sum(axis=1) + hypertension
            rank = np.select(
                [severe_count.ge(2), severe_count.eq(1), warning_count.ge(2), warning_count.eq(1)],
                [4, 3, 2, 1],
                default=0,
            )
            rank = np.where(hypertension.eq(1), np.maximum(rank, 3), rank)
            joined["clinical_warning_count_v2"] = warning_count
            joined["clinical_severe_count_v2"] = severe_count
            joined["clinical_rank"] = rank
            expected = replay.merge(
                joined[
                    PAIR_KEYS
                    + [
                        TARGET,
                        "clinical_warning_count_v2",
                        "clinical_severe_count_v2",
                        "clinical_rank",
                    ]
                ],
                on=PAIR_KEYS,
                how="inner",
                suffixes=("", "_source"),
                validate="one_to_one",
            )
            require(
                len(expected) == len(replay)
                and expected[TARGET].equals(expected[f"{TARGET}_source"]),
                f"Replay target mismatch for {model_name}/{version}.",
            )
            require(
                expected["clinical_warning_count_v2"].equals(
                    expected["clinical_warning_count_v2_source"]
                )
                and expected["clinical_severe_count_v2"].equals(
                    expected["clinical_severe_count_v2_source"]
                ),
                f"Replay clinical rule counts mismatch for {model_name}/{version}.",
            )
            expected_priority = [
                independent_priority(int(row.clinical_rank), int(row.clinical_warning_count_v2_source), float(row.risk_probability))
                for row in expected.itertuples(index=False)
            ]
            require(
                expected_priority == expected["final_alert_priority"].tolist(),
                f"Replay priorities differ from independent priority logic for {model_name}/{version}.",
            )
            require(
                not (
                    expected["notification_action"].eq("SUPPRESS")
                    & (
                        expected["new_severe_event"]
                        | expected["critical_escalation"]
                    )
                ).any(),
                f"Replay suppressed a new severe event/critical escalation for {model_name}/{version}.",
            )
            summary_row = replay_summary.loc[
                replay_summary["model"].eq(model_name)
                & replay_summary["dataset_version"].eq(version)
            ]
            require(len(summary_row) == 1, "Missing alert replay summary row.")
            summary_row = summary_row.iloc[0]
            require(
                int(summary_row["alert_rows"])
                == int(expected["final_alert_priority"].isin(["high", "critical"]).sum())
                and int(summary_row["critical_rows"])
                == int(expected["final_alert_priority"].eq("critical").sum())
                and int(summary_row["critical_rows_suppressed"])
                == int(
                    (
                        expected["final_alert_priority"].eq("critical")
                        & expected["notification_action"].eq("SUPPRESS")
                    ).sum()
                ),
                f"Alert replay summary counts mismatch for {model_name}/{version}.",
            )
            version_replays[version] = expected
            passed(
                f"alert_replay_{model_name}_{version}",
                "Clinical counts, fixed-threshold priority, and notification safety counts independently checked.",
            )
        baseline = version_replays["baseline"].sort_values(PAIR_KEYS).reset_index(drop=True)
        corrected = version_replays["corrected"].sort_values(PAIR_KEYS).reset_index(drop=True)
        changes = baseline["final_alert_priority"].ne(corrected["final_alert_priority"])
        comparison_row = priority_comparison_summary.loc[
            priority_comparison_summary["model"].eq(model_name)
        ]
        require(len(comparison_row) == 1, "Missing priority-comparison summary row.")
        require(
            int(comparison_row.iloc[0]["priority_changed_hours"]) == int(changes.sum()),
            f"Priority comparison count mismatch for {model_name}.",
        )
        for version, replay in [("baseline", baseline), ("corrected", corrected)]:
            rows = replay.sort_values(PAIR_KEYS).copy()
            rows["severe_alert"] = rows["final_alert_priority"].isin(["high", "critical"])
            rows["previous_severe"] = rows.groupby("patientunitstayid")[
                "severe_alert"
            ].shift(1).fillna(False)
            rows["previous_hour"] = rows.groupby("patientunitstayid")["hour"].shift(1)
            rows["new_episode"] = rows["severe_alert"] & (
                rows["previous_severe"].eq(False)
                | (rows["hour"] - rows["previous_hour"]).gt(2)
            )
            episode_count = int(rows["new_episode"].sum())
            rows["episode_id"] = rows.groupby("patientunitstayid")[
                "new_episode"
            ].cumsum()
            severe_rows = rows.loc[rows["severe_alert"]]
            positive_episode_count = int(
                severe_rows.groupby(["patientunitstayid", "episode_id"])[TARGET]
                .sum()
                .gt(0)
                .sum()
            )
            expected_count = len(
                episode_summary.loc[
                    episode_summary["model"].eq(model_name)
                    & episode_summary["dataset_version"].eq(version)
                ]
            )
            require(episode_count == expected_count, f"Severe episode count mismatch for {model_name}/{version}.")
            report_row = replay_summary.loc[
                replay_summary["model"].eq(model_name)
                & replay_summary["dataset_version"].eq(version)
            ].iloc[0]
            require(
                int(report_row["severe_alert_episodes"]) == episode_count
                and int(report_row["episodes_with_target_positive_hours"])
                == positive_episode_count,
                f"Reported episode summary mismatch for {model_name}/{version}.",
            )
    passed(
        "alert_episodes",
        "All reported severe-alert episode counts independently reconstructed with the <=2-hour gap rule.",
    )

    anomaly_summary = pd.read_csv(ANOMALY / "anomaly_summary.csv")
    anomaly_baseline = pd.read_csv(ANOMALY / "test_anomaly_baseline.csv")
    anomaly_corrected = pd.read_csv(ANOMALY / "test_anomaly_corrected.csv")
    anomaly_baseline = anomaly_baseline.sort_values(KEYS[:2]).reset_index(drop=True)
    anomaly_corrected = anomaly_corrected.sort_values(KEYS[:2]).reset_index(drop=True)
    require(anomaly_baseline[KEYS].equals(anomaly_corrected[KEYS]), "Anomaly test keys are not paired.")
    require(
        anomaly_baseline["anomaly_score"].notna().all()
        and anomaly_corrected["anomaly_score"].notna().all()
        and anomaly_baseline["anomaly_flag"].isin([0, 1]).all()
        and anomaly_corrected["anomaly_flag"].isin([0, 1]).all(),
        "Invalid anomaly scores or flags.",
    )
    overlap = int(
        (
            anomaly_baseline["anomaly_flag"].eq(1)
            & anomaly_corrected["anomaly_flag"].eq(1)
        ).sum()
    )
    union = int(
        (
            anomaly_baseline["anomaly_flag"].eq(1)
            | anomaly_corrected["anomaly_flag"].eq(1)
        ).sum()
    )
    correlation = spearmanr(
        anomaly_baseline["anomaly_score"], anomaly_corrected["anomaly_score"]
    ).statistic
    stability = pd.read_csv(ANOMALY / "paired_stability_summary.csv").iloc[0]
    require(
        overlap == int(stability["flag_overlap_rows"])
        and union == int(stability["flag_union_rows"]),
        "Anomaly overlap counts do not reconcile.",
    )
    compare_float(
        float(stability["score_spearman_rho"]),
        float(correlation),
        "Anomaly score Spearman correlation",
    )
    require(
        (anomaly_summary["anomaly_rate"].between(0, 1)).all(),
        "Invalid anomaly rate.",
    )
    for version, scored in [
        ("baseline", anomaly_baseline),
        ("corrected", anomaly_corrected),
    ]:
        summary_row = anomaly_summary.loc[
            anomaly_summary["dataset_version"].eq(version)
        ]
        require(len(summary_row) == 1, f"Missing anomaly summary row for {version}.")
        summary_row = summary_row.iloc[0]
        compare_float(
            float(summary_row["anomaly_rate"]),
            float(scored["anomaly_flag"].mean()),
            f"{version} anomaly rate",
        )
        require(
            int(summary_row["anomaly_rows"]) == int(scored["anomaly_flag"].sum()),
            f"{version} anomaly count mismatch.",
        )
    passed(
        "anomaly_stability",
        "Paired anomaly keys, flag overlap, score rank correlation, and descriptive rates reconcile.",
    )

    results = pd.DataFrame(checks)
    results.to_csv(PAIRED / "validation_checks.csv", index=False)
    report_lines = [
        "# Independent paired-experiment validation",
        "",
        f"**Result: PASS ({len(results)} checks, 0 failures).**",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    report_lines.extend(
        f"| {row.check} | {row.status} | {row.detail} |"
        for row in results.itertuples(index=False)
    )
    report_lines.extend(
        [
            "",
            "The validator independently recomputes validation thresholds, held-out",
            "metrics, replay priorities, notification safety counts, severe-episode",
            "counts, and anomaly stability statistics. It does not assert clinical",
            "validity, alert utility, or deployment readiness.",
            "",
        ]
    )
    (PAIRED / "validation_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(f"Paired experiment validation: PASS ({len(results)} checks).")


if __name__ == "__main__":
    main()
