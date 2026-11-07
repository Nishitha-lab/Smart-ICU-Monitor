from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
INPUT_DIR = PROCESSED / "experiments" / "leakage_safe"
MODEL_OUTPUT_DIR = PROCESSED / "experiments" / "paired_models"
OUTPUT_DIR = PROCESSED / "experiments" / "alert_replay"
RULE_FILE = PROCESSED / "hourly_patient_state_rules_v2.csv"
KEYS = ["patientunitstayid", "hour"]
TARGET = "deterioration_label_24h"
COOLDOWN_HOURS = 6
PRIORITY_RANK = {"normal": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
RANK_PRIORITY = {value: key for key, value in PRIORITY_RANK.items()}
WARNINGS = [
    "rule_tachycardia",
    "rule_bradycardia",
    "rule_hypoxemia",
    "rule_abnormal_respiration",
    "rule_hypotension_sbp",
    "rule_low_map",
    "rule_rapid_hr_rise",
    "rule_rapid_hr_fall",
    "rule_falling_spo2",
    "rule_falling_map",
    "rule_vasoactive_support",
    "rule_multiple_infusions",
    "rule_large_positive_fluid_balance",
    "rule_large_negative_fluid_balance",
]
SEVERE = [
    "rule_severe_tachycardia",
    "rule_severe_bradycardia",
    "rule_severe_hypoxemia",
    "rule_severe_respiration",
    "rule_severe_hypotension_sbp",
    "rule_severe_low_map",
    "rule_severe_hr_rise",
    "rule_severe_hr_fall",
    "rule_rapid_falling_spo2",
    "rule_rapid_falling_map",
]
RULE_NAMES = {
    "rule_tachycardia": "tachycardia",
    "rule_bradycardia": "bradycardia",
    "rule_hypoxemia": "low SpO2",
    "rule_abnormal_respiration": "abnormal respiration",
    "rule_hypotension_sbp": "low systolic BP",
    "rule_low_map": "low MAP",
    "rule_rapid_hr_rise": "rapid HR rise",
    "rule_rapid_hr_fall": "rapid HR fall",
    "rule_falling_spo2": "falling SpO2",
    "rule_falling_map": "falling MAP",
    "rule_vasoactive_support": "vasoactive support",
    "rule_multiple_infusions": "multiple active infusions",
    "rule_large_positive_fluid_balance": "large positive fluid balance",
    "rule_large_negative_fluid_balance": "large negative fluid balance",
}


def clinical_state(frame: pd.DataFrame, version: str) -> pd.DataFrame:
    rules = pd.read_csv(RULE_FILE, low_memory=False)
    if rules.duplicated(KEYS).any():
        raise ValueError("Duplicate patient-hour rows in clinical rule input.")
    rule_columns = KEYS + list(dict.fromkeys(WARNINGS + SEVERE))
    data = frame.merge(
        rules[rule_columns], on=KEYS, how="left", validate="one_to_one"
    )
    for column in WARNINGS + SEVERE:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["rule_hypoxemia"] = data["spo2"].lt(92).astype(int)
    data["rule_falling_spo2"] = data["spo2_change_1h"].lt(-3).astype(int)
    data["rule_severe_hypoxemia"] = data["spo2"].lt(88).astype(int)
    data["rule_rapid_falling_spo2"] = data["spo2_change_1h"].lt(-5).astype(int)
    data["rule_severe_hypertension"] = (
        data["systolic_bp"].gt(180) | data["diastolic_bp"].gt(120)
    ).fillna(False).astype(int)
    data["clinical_warning_count_v2"] = (
        data[WARNINGS].sum(axis=1) + data["rule_severe_hypertension"]
    )
    data["clinical_severe_count_v2"] = (
        data[SEVERE].sum(axis=1) + data["rule_severe_hypertension"]
    )
    rank = np.select(
        [
            data["clinical_severe_count_v2"].ge(2),
            data["clinical_severe_count_v2"].eq(1),
            data["clinical_warning_count_v2"].ge(2),
            data["clinical_warning_count_v2"].eq(1),
        ],
        [4, 3, 2, 1],
        default=0,
    )
    data["clinical_severity_rank"] = np.where(
        data["rule_severe_hypertension"].eq(1), np.maximum(rank, 3), rank
    )
    data["clinical_rule_severity_v2"] = data["clinical_severity_rank"].map(
        RANK_PRIORITY
    )

    def explanation(row: pd.Series) -> str:
        reasons = [
            label
            for column, label in RULE_NAMES.items()
            if row[column] == 1
        ]
        if row["rule_severe_hypertension"] == 1:
            sbp = row["systolic_bp"]
            dbp = row["diastolic_bp"]
            if pd.notna(sbp) and pd.notna(dbp):
                reasons.append(f"severe hypertension (SBP {sbp:.1f}; DBP {dbp:.1f})")
            elif pd.notna(sbp):
                reasons.append(f"severe hypertension (SBP {sbp:.1f})")
            elif pd.notna(dbp):
                reasons.append(f"severe hypertension (DBP {dbp:.1f})")
        return "; ".join(reasons) if reasons else "No rule-based abnormality"

    data["clinical_rule_explanation_v2"] = data.apply(explanation, axis=1)
    data["dataset_version"] = version
    return data


def alert_priority(row: pd.Series) -> str:
    clinical_rank = int(row["clinical_severity_rank"])
    warning = row["clinical_warning_count_v2"] >= 1
    risk10 = row["risk_probability"] >= 0.10
    risk25 = row["risk_probability"] >= 0.25
    risk50 = row["risk_probability"] >= 0.50

    if clinical_rank >= 4:
        return "critical"
    if clinical_rank >= 3 and risk25:
        return "critical"
    if risk50 and warning:
        return "critical"
    if clinical_rank >= 3:
        return "high"
    if clinical_rank >= 2 and risk25:
        return "high"
    if clinical_rank >= 1 and risk25:
        return "high"
    if risk50:
        return "high"
    if clinical_rank >= 2:
        return "moderate"
    if clinical_rank >= 1 and risk10:
        return "moderate"
    if risk25:
        return "moderate"
    if clinical_rank >= 1 or risk10:
        return "low"
    return "normal"


def replay_notifications(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values(KEYS).reset_index(drop=True).copy()
    data["priority_rank"] = data["final_alert_priority"].map(PRIORITY_RANK).astype(int)
    data["severe_alert"] = data["final_alert_priority"].isin(["high", "critical"])
    actions = ["NO_ALERT"] * len(data)
    new_events = np.zeros(len(data), dtype=bool)
    critical_escalations = np.zeros(len(data), dtype=bool)
    for _, group in data.groupby("patientunitstayid", sort=False):
        last_emit_hour: float | None = None
        last_reason: str | None = None
        previous_severe = False
        previous_rank: int | None = None
        for index, row in group.iterrows():
            current_hour = float(row["hour"])
            current_severe = bool(row["severe_alert"])
            current_rank = int(row["priority_rank"])
            current_reason = str(row["clinical_rule_explanation_v2"])
            if not current_severe:
                previous_severe = False
                previous_rank = current_rank
                continue
            new_events[index] = not previous_severe
            critical_escalations[index] = (
                row["final_alert_priority"] == "critical"
                and previous_rank is not None
                and previous_rank < PRIORITY_RANK["critical"]
            )
            elapsed = (
                current_hour - last_emit_hour
                if last_emit_hour is not None
                else None
            )
            reason_changed = last_reason is not None and current_reason != last_reason
            if not previous_severe:
                action = "EMIT" if last_emit_hour is None else "NEW_EVENT"
            elif previous_rank is not None and current_rank > previous_rank:
                action = "ESCALATION"
            elif reason_changed:
                action = "NEW_REASON"
            elif last_emit_hour is None:
                action = "EMIT"
            elif elapsed is not None and elapsed >= COOLDOWN_HOURS:
                action = "COOLDOWN_EXPIRED"
            else:
                action = "SUPPRESS"
            if action != "SUPPRESS":
                last_emit_hour = current_hour
                last_reason = current_reason
            actions[index] = action
            previous_severe = True
            previous_rank = current_rank
    data["notification_action"] = actions
    data["new_severe_event"] = new_events
    data["critical_escalation"] = critical_escalations
    data["notification_emitted"] = data["notification_action"].ne("SUPPRESS") & data[
        "severe_alert"
    ]
    if (
        data["notification_action"].eq("SUPPRESS")
        & (data["new_severe_event"] | data["critical_escalation"])
    ).any():
        raise ValueError("Replay suppressed a new severe event or critical escalation.")
    return data


def extract_episodes(frame: pd.DataFrame, model: str, version: str) -> list[dict[str, Any]]:
    data = frame.sort_values(KEYS).copy()
    data["previous_hour"] = data.groupby("patientunitstayid")["hour"].shift(1)
    data["previous_severe"] = data.groupby("patientunitstayid")["severe_alert"].shift(1).fillna(False)
    data["hour_gap"] = data["hour"] - data["previous_hour"]
    data["new_episode"] = data["severe_alert"] & (
        data["previous_severe"].eq(False) | data["hour_gap"].gt(2)
    )
    data["episode_id"] = data.groupby("patientunitstayid")["new_episode"].cumsum()
    severe = data.loc[data["severe_alert"]].copy()
    if severe.empty:
        return []
    episodes = (
        severe.groupby(["patientunitstayid", "episode_id"], as_index=False)
        .agg(
            start_hour=("hour", "min"),
            end_hour=("hour", "max"),
            alert_hours=("hour", "size"),
            peak_priority=("final_alert_priority", lambda x: "critical" if x.eq("critical").any() else "high"),
            target_positive_hours=(TARGET, "sum"),
            notification_emissions=("notification_emitted", "sum"),
            suppressed_hours=("notification_action", lambda x: x.eq("SUPPRESS").sum()),
        )
    )
    episodes["episode_span_hours"] = episodes["end_hour"] - episodes["start_hour"] + 1
    episodes["intermittent_gap_hours"] = (
        episodes["episode_span_hours"] - episodes["alert_hours"]
    )
    episodes["model"] = model
    episodes["dataset_version"] = version
    return episodes.to_dict("records")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rules = pd.read_csv(RULE_FILE, usecols=KEYS)
    if rules.duplicated(KEYS).any():
        raise ValueError("Duplicate rule keys.")
    all_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    comparison_summaries: list[dict[str, Any]] = []
    for model in ("balanced_logistic", "probability_logistic"):
        model_versions: dict[str, pd.DataFrame] = {}
        for version in ("baseline", "corrected"):
            predictions = pd.read_csv(
                MODEL_OUTPUT_DIR / f"test_predictions_{model}_{version}.csv",
                low_memory=False,
            )
            if predictions.duplicated(KEYS).any():
                raise ValueError(f"Duplicate prediction keys for {model}/{version}.")
            features = pd.read_csv(
                INPUT_DIR / f"model_inputs_{version}.csv",
                usecols=KEYS + ["split", "uniquepid", TARGET, "spo2", "spo2_change_1h",
                                "systolic_bp", "diastolic_bp"],
                low_memory=False,
            )
            features = features.loc[features["split"].eq("test")].drop(columns="split")
            feature_keys = features.sort_values(KEYS).reset_index(drop=True)
            prediction_keys = predictions.sort_values(KEYS).reset_index(drop=True)
            if not feature_keys[KEYS + [TARGET]].equals(
                prediction_keys[KEYS + [TARGET]]
            ):
                raise ValueError(f"Prediction/source keys differ for {model}/{version}.")
            joined = features.merge(
                predictions[KEYS + ["probability"]],
                on=KEYS,
                how="inner",
                validate="one_to_one",
            )
            joined = clinical_state(joined, version)
            joined["risk_probability"] = joined.pop("probability")
            joined["final_alert_priority"] = joined.apply(alert_priority, axis=1)
            replay = replay_notifications(joined)
            path = OUTPUT_DIR / f"replay_{model}_{version}.csv"
            replay[
                KEYS
                + [
                    TARGET,
                    "risk_probability",
                    "clinical_rule_severity_v2",
                    "clinical_warning_count_v2",
                    "clinical_severe_count_v2",
                    "final_alert_priority",
                    "notification_action",
                    "notification_emitted",
                    "new_severe_event",
                    "critical_escalation",
                ]
            ].to_csv(path, index=False)
            model_versions[version] = replay
            model_episodes = extract_episodes(replay, model, version)
            severe = replay["severe_alert"]
            critical = replay["final_alert_priority"].eq("critical")
            summaries.append(
                {
                    "model": model,
                    "dataset_version": version,
                    "test_patient_hours": len(replay),
                    "alert_rows": int(severe.sum()),
                    "critical_rows": int(critical.sum()),
                    "new_severe_events": int(replay["new_severe_event"].sum()),
                    "severe_notifications_emitted": int(
                        (severe & replay["notification_emitted"]).sum()
                    ),
                    "severe_rows_suppressed": int(
                        (severe & replay["notification_action"].eq("SUPPRESS")).sum()
                    ),
                    "critical_rows_suppressed": int(
                        (critical & replay["notification_action"].eq("SUPPRESS")).sum()
                    ),
                    "critical_target_positive_hours": int(
                        replay.loc[critical, TARGET].sum()
                    ),
                    "severe_alert_episodes": len(model_episodes),
                    "episodes_with_target_positive_hours": sum(
                        episode["target_positive_hours"] > 0
                        for episode in model_episodes
                    ),
                }
            )
            episodes.extend(model_episodes)
            all_rows.append(replay)
        baseline = model_versions["baseline"].sort_values(KEYS).reset_index(drop=True)
        corrected = model_versions["corrected"].sort_values(KEYS).reset_index(drop=True)
        if not baseline[KEYS].equals(corrected[KEYS]):
            raise ValueError(f"Alert replay keys differ for {model}.")
        comparisons = baseline[KEYS + ["final_alert_priority"]].merge(
            corrected[KEYS + ["final_alert_priority"]],
            on=KEYS,
            suffixes=("_baseline", "_corrected"),
            validate="one_to_one",
        )
        comparisons["priority_changed"] = comparisons[
            "final_alert_priority_baseline"
        ].ne(comparisons["final_alert_priority_corrected"])
        comparison_summaries.append(
            {
                "model": model,
                "test_patient_hours": len(comparisons),
                "priority_changed_hours": int(comparisons["priority_changed"].sum()),
                "priority_changed_fraction": float(comparisons["priority_changed"].mean()),
                "baseline_critical_to_corrected_noncritical": int(
                    (
                        comparisons["final_alert_priority_baseline"].eq("critical")
                        & comparisons["final_alert_priority_corrected"].ne("critical")
                    ).sum()
                ),
                "baseline_noncritical_to_corrected_critical": int(
                    (
                        comparisons["final_alert_priority_baseline"].ne("critical")
                        & comparisons["final_alert_priority_corrected"].eq("critical")
                    ).sum()
                ),
            }
        )
        comparisons.to_csv(
            OUTPUT_DIR / f"priority_comparison_{model}.csv", index=False
        )

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / "alert_replay_summary.csv", index=False)
    comparison_summary = pd.DataFrame(comparison_summaries)
    comparison_summary.to_csv(
        OUTPUT_DIR / "priority_comparison_summary.csv", index=False
    )
    episode_frame = pd.DataFrame(episodes)
    episode_frame.to_csv(OUTPUT_DIR / "severe_alert_episodes.csv", index=False)
    config = {
        "source_alert_logic": str(ROOT / "ml" / "build_final_smart_icu_alerts.py"),
        "source_notification_logic": str(ROOT / "ml" / "build_alert_notification_layer.py"),
        "risk_thresholds_unchanged": [0.10, 0.25, 0.50],
        "cooldown_hours_unchanged": COOLDOWN_HOURS,
        "severe_episode_gap_rule": "high/critical rows grouped within stay; split when gap > 2 hours",
        "anomaly_signal": "disabled; no recovered frozen anomaly model was available",
        "population": "held-out test patient-hours only",
        "model_thresholds": "not used in priority logic; existing fixed risk cutoffs retained",
        "outputs": {
            "summary": str(OUTPUT_DIR / "alert_replay_summary.csv"),
            "episodes": str(OUTPUT_DIR / "severe_alert_episodes.csv"),
        },
        "limitations": [
            "Isolated research replay only; no production alert or notification files modified.",
            "Anomaly input omitted, so this is not a full reproduction of the deployed fusion system.",
            "The hourly label is expired ICU discharge within 24 hours, not general deterioration.",
            "Associations with this label do not establish clinical utility or safety.",
        ],
    }
    (OUTPUT_DIR / "replay_configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    report = [
        "# Isolated experimental alert replay",
        "",
        "The replay substitutes held-out experimental logistic risk scores into the",
        "existing alert-priority thresholds and 6-hour notification cooldown. Clinical",
        "rule counts and severity are reconstructed from each version's test features.",
        "The anomaly signal is disabled because no frozen anomaly artifact was recovered.",
        "Only held-out test hours are included; production outputs are untouched.",
        "",
        "## Summary",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Baseline/corrected priority changes",
        "",
        "```text",
        comparison_summary.to_string(index=False),
        "```",
        "",
        "Severe-alert episodes use the existing gap rule: high/critical rows in a stay",
        "remain one episode when the gap is at most two hours. Per-episode outcomes",
        "describe overlap with the historical expired-discharge label only; they are",
        "not clinical-validation results.",
        "",
        "## Limitations",
        "",
        "- This is not a full deployed-fusion replay: the anomaly signal is disabled.",
        "- Results are retrospective and based on the eICU demo extract.",
        "- The target is expired ICU discharge within 24 hours, not generic deterioration.",
        "- No clinical utility, safety, deployment, or model-promotion claim is made.",
        "",
    ]
    (OUTPUT_DIR / "alert_replay_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nWrote isolated alert replay to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
