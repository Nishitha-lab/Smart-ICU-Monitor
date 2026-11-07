from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED / "spo2_comparison"
KEYS = ["patientunitstayid", "hour"]

DATASETS = {
    "original_raw": PROCESSED / "hourly_vitals_labs.csv",
    "original_clean": PROCESSED / "hourly_vitals_labs_clean.csv",
    "corrected_raw": PROCESSED / "hourly_vitals_labs_spo2_experiment.csv",
    "corrected_clean": PROCESSED / "hourly_vitals_labs_spo2_experiment_clean.csv",
}

MEASUREMENT_COLUMNS = [
    "heart_rate",
    "respiration",
    "systolic_bp",
    "diastolic_bp",
    "mean_bp",
    "BUN",
    "WBC",
    "anion_gap",
    "bicarbonate",
    "calcium",
    "chloride",
    "creatinine",
    "glucose",
    "hemoglobin",
    "platelets",
    "potassium",
    "sodium",
]
METADATA_COLUMNS = {
    "original_spo2_raw_mean",
    "corrected_spo2_raw_mean",
    "original_spo2_valid_count",
    "corrected_spo2_valid_count",
    "original_spo2_zero_count",
    "corrected_spo2_zero_count",
    "original_spo2_missing_count",
    "corrected_spo2_missing_count",
    "original_spo2_quality_flag",
    "corrected_spo2_quality_flag",
}
THRESHOLDS = [20, 30, 40, 50, 70, 80]
EXPERIMENT_ID = "spo2-dataset-impact-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        "sha256": sha256(path),
    }


def require_schema(frame: pd.DataFrame, name: str) -> None:
    missing = [column for column in KEYS + ["spo2"] if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    valid_key_rows = frame[KEYS].notna().all(axis=1)
    duplicates = int(frame.loc[valid_key_rows].duplicated(KEYS).sum())
    if duplicates:
        raise ValueError(f"{name} contains {duplicates} duplicate patient-hour keys.")


def different(left: pd.Series, right: pd.Series) -> pd.Series:
    left_missing = left.isna()
    right_missing = right.isna()
    both_missing = left_missing & right_missing
    numeric_left = pd.to_numeric(left, errors="coerce")
    numeric_right = pd.to_numeric(right, errors="coerce")
    numeric_comparable = (
        (left.notna() & numeric_left.notna())
        | (right.notna() & numeric_right.notna())
        | both_missing
    )
    numeric_equal = pd.Series(False, index=left.index)
    numeric_equal.loc[both_missing] = True
    numeric_mask = numeric_comparable & ~both_missing
    if numeric_mask.any():
        numeric_equal.loc[numeric_mask] = np.isclose(
            numeric_left.loc[numeric_mask].astype(float),
            numeric_right.loc[numeric_mask].astype(float),
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )
    string_equal = (
        left.astype("string").fillna("<NA>")
        == right.astype("string").fillna("<NA>")
    )
    equal = numeric_equal.where(numeric_comparable, string_equal)
    return ~equal


def pair_data(
    original: pd.DataFrame,
    corrected: pd.DataFrame,
    original_name: str,
    corrected_name: str,
) -> pd.DataFrame:
    require_schema(original, original_name)
    require_schema(corrected, corrected_name)
    original_valid = original.loc[original[KEYS].notna().all(axis=1)]
    corrected_valid = corrected.loc[corrected[KEYS].notna().all(axis=1)]
    return original_valid.merge(
        corrected_valid,
        on=KEYS,
        how="outer",
        suffixes=("_original", "_corrected"),
        indicator=True,
        validate="one_to_one",
    )


def dataset_metrics(
    name: str,
    frame: pd.DataFrame,
    source_path: Path,
    removed_by_cleaning: int | None,
) -> list[dict[str, Any]]:
    spo2 = pd.to_numeric(frame["spo2"], errors="coerce")
    metrics: dict[str, Any] = {
        "rows": len(frame),
        "unique_icu_stays": frame["patientunitstayid"].nunique(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_patient_hour_keys": int(frame.duplicated(KEYS).sum()),
        "invalid_patient_hour_key_rows": int(frame[KEYS].isna().any(axis=1).sum()),
        "spo2_missing_count": int(spo2.isna().sum()),
        "spo2_missing_percent": float(spo2.isna().mean() * 100),
        "spo2_zero_count": int(spo2.eq(0).sum()),
        "spo2_nonzero_count": int((spo2.notna() & spo2.ne(0)).sum()),
        "spo2_min": float(spo2.min()) if spo2.notna().any() else np.nan,
        "spo2_max": float(spo2.max()) if spo2.notna().any() else np.nan,
        "spo2_mean": float(spo2.mean()),
        "spo2_median": float(spo2.median()),
        "spo2_std": float(spo2.std()),
        "rows_removed_during_cleaning": removed_by_cleaning,
        "file_size_bytes": source_path.stat().st_size,
        "file_modified_at": datetime.fromtimestamp(
            source_path.stat().st_mtime
        ).astimezone().isoformat(),
    }
    for threshold in THRESHOLDS:
        metrics[f"spo2_le_{threshold}_count"] = int(spo2.le(threshold).sum())
    metrics["spo2_gt_100_count"] = int(spo2.gt(100).sum())
    return [
        {"dataset_version": name, "metric": key, "value": value}
        for key, value in metrics.items()
    ]


def make_spo2_features(frame: pd.DataFrame) -> pd.DataFrame:
    timeline = frame[KEYS + ["spo2"]].copy()
    timeline["hour"] = pd.to_numeric(timeline["hour"], errors="coerce")
    timeline["spo2"] = pd.to_numeric(timeline["spo2"], errors="coerce")
    timeline = timeline.sort_values(KEYS).reset_index(drop=True)
    groups = timeline.groupby("patientunitstayid", sort=False)

    previous_value = groups["spo2"].shift(1)
    previous_hour = groups["hour"].shift(1)
    elapsed = timeline["hour"] - previous_hour
    timeline["spo2_change_1h"] = (timeline["spo2"] - previous_value).where(
        elapsed.eq(1)
    )

    lookup = timeline.set_index(KEYS)["spo2"]
    for hours in [3, 6]:
        target_keys = pd.MultiIndex.from_arrays(
            [timeline["patientunitstayid"], timeline["hour"] - hours],
            names=KEYS,
        )
        prior = lookup.reindex(target_keys).to_numpy()
        timeline[f"spo2_change_{hours}h"] = timeline["spo2"].to_numpy() - prior

    gap = groups["hour"].diff()
    segment_break = gap.isna() | gap.gt(1)
    segment_number = segment_break.groupby(
        timeline["patientunitstayid"], sort=False
    ).cumsum()
    timeline["_segment"] = (
        timeline["patientunitstayid"].astype(str)
        + "_"
        + segment_number.astype(str)
    )
    segmented = timeline.groupby("_segment", sort=False)["spo2"]
    for hours in [3, 6]:
        window = segmented.transform(
            lambda values: values.rolling(window=hours, min_periods=2).mean()
        )
        timeline[f"spo2_rolling_mean_{hours}h"] = window
        timeline[f"spo2_rolling_std_{hours}h"] = segmented.transform(
            lambda values: values.rolling(window=hours, min_periods=2).std()
        )

    timeline["flag_hypoxemia_lt_92"] = timeline["spo2"].lt(92).astype("Int64")
    timeline["flag_severe_hypoxemia_lt_88"] = timeline["spo2"].lt(88).astype(
        "Int64"
    )
    timeline["flag_falling_spo2_gt_3"] = timeline["spo2_change_1h"].lt(-3).astype(
        "Int64"
    )
    timeline["flag_rapid_falling_spo2_gt_5"] = (
        timeline["spo2_change_1h"].lt(-5).astype("Int64")
    )
    return timeline.drop(columns="_segment")


def add_rule_impact_features(
    spo2_features: pd.DataFrame, saved_rules: pd.DataFrame
) -> pd.DataFrame:
    warning_columns = [
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
    severe_columns = [
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
    output_columns = [
        *warning_columns,
        *severe_columns,
        "rule_severe_hypertension",
        "hypertension_explanation",
    ]
    missing = [column for column in KEYS + output_columns if column not in saved_rules]
    if missing:
        raise ValueError(f"Existing rule output is missing required columns: {missing}")
    if saved_rules.duplicated(KEYS).any():
        raise ValueError("Existing rule output contains duplicate patient-hour keys.")
    base = spo2_features.merge(
        saved_rules[KEYS + output_columns],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    base["rule_hypoxemia"] = base["flag_hypoxemia_lt_92"]
    base["rule_severe_hypoxemia"] = base["flag_severe_hypoxemia_lt_88"]
    base["rule_falling_spo2"] = base["flag_falling_spo2_gt_3"]
    base["rule_rapid_falling_spo2"] = base["flag_rapid_falling_spo2_gt_5"]
    for column in warning_columns + severe_columns + ["rule_severe_hypertension"]:
        base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0)
    warning_values = base[warning_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    severe_values = base[severe_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    base["clinical_warning_count"] = warning_values.sum(axis=1)
    base["clinical_severe_count"] = severe_values.sum(axis=1)
    hypertension = base["rule_severe_hypertension"]
    base["clinical_warning_count_v2"] = base["clinical_warning_count"] + hypertension
    base["clinical_severe_count_v2"] = base["clinical_severe_count"] + hypertension

    def severity(warnings: pd.Series, severe: pd.Series) -> np.ndarray:
        return np.select(
            [
                severe.ge(2),
                severe.eq(1),
                warnings.ge(2),
                warnings.eq(1),
            ],
            ["critical", "high", "moderate", "low"],
            default="normal",
        )

    base_severity = severity(
        base["clinical_warning_count"], base["clinical_severe_count"]
    )
    base["clinical_rule_severity"] = base_severity
    severity_rank = pd.Series(base_severity, index=base.index).map(
        {"normal": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
    )
    severity_rank = severity_rank.where(
        ~hypertension.eq(1), severity_rank.clip(lower=3)
    )
    base["clinical_rule_severity_v2"] = severity_rank.map(
        {0: "normal", 1: "low", 2: "moderate", 3: "high", 4: "critical"}
    )

    rule_labels = {
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
    labels = list(rule_labels.values())
    base["clinical_rule_explanation"] = [
        "; ".join(
            label
            for label, value in zip(labels, values)
            if value == 1
        )
        or "No rule-based abnormality"
        for values in warning_values.itertuples(index=False, name=None)
    ]
    hypertension_text = base["hypertension_explanation"].fillna("").astype(str)
    base["clinical_rule_explanation_v2"] = [
        (
            high
            if high and text in {"", "No rule-based abnormality", "nan"}
            else text + "; " + high
            if high
            else text
        )
        for text, high in zip(base["clinical_rule_explanation"], hypertension_text)
    ]
    return base[
        KEYS
        + [
            column
            for column in spo2_features.columns
            if column not in KEYS
        ]
        + [
            "clinical_warning_count",
            "clinical_severe_count",
            "clinical_rule_severity",
            "clinical_rule_explanation",
            "clinical_warning_count_v2",
            "clinical_severe_count_v2",
            "clinical_rule_severity_v2",
            "clinical_rule_explanation_v2",
        ]
    ]


def compare_features(
    original: pd.DataFrame,
    corrected: pd.DataFrame,
    saved_rules: pd.DataFrame,
) -> pd.DataFrame:
    left = add_rule_impact_features(
        make_spo2_features(original), saved_rules
    ).set_index(KEYS)
    right = add_rule_impact_features(
        make_spo2_features(corrected), saved_rules
    ).set_index(KEYS)
    both = left.join(right, how="outer", lsuffix="_original", rsuffix="_corrected")
    common = left.index.intersection(right.index)
    rows: list[dict[str, Any]] = []
    feature_columns = [
        column for column in left.columns if column not in KEYS
    ]
    for column in feature_columns:
        a = both[f"{column}_original"].reindex(common)
        b = both[f"{column}_corrected"].reindex(common)
        changed = different(a, b)
        numeric_a = pd.to_numeric(a, errors="coerce")
        numeric_b = pd.to_numeric(b, errors="coerce")
        numeric_change = (numeric_a - numeric_b).abs()
        rows.append(
            {
                "feature": column,
                "feature_type": (
                    "clinical_rule_output"
                    if column.startswith("clinical_")
                    else
                    "quality_metadata"
                    if column.startswith("spo2_") and column
                    in {"spo2_valid_count", "spo2_zero_count", "spo2_missing_count", "spo2_quality_flag"}
                    else "threshold_flag"
                    if column.startswith("flag_")
                    else "numeric_spo2_feature"
                ),
                "matched_patient_hours": len(common),
                "original_nonmissing": int(a.notna().sum()),
                "corrected_nonmissing": int(b.notna().sum()),
                "original_mean": float(numeric_a.mean()) if numeric_a.notna().any() else np.nan,
                "corrected_mean": float(numeric_b.mean()) if numeric_b.notna().any() else np.nan,
                "original_min": float(numeric_a.min()) if numeric_a.notna().any() else np.nan,
                "corrected_min": float(numeric_b.min()) if numeric_b.notna().any() else np.nan,
                "original_max": float(numeric_a.max()) if numeric_a.notna().any() else np.nan,
                "corrected_max": float(numeric_b.max()) if numeric_b.notna().any() else np.nan,
                "changed_values": int(changed.sum()),
                "changed_nonmissing_values": int(
                    (changed & a.notna() & b.notna()).sum()
                ),
                "newly_missing": int((a.notna() & b.isna()).sum()),
                "newly_observed": int((a.isna() & b.notna()).sum()),
                "changed_flag_count": int(changed.sum()) if column.startswith("flag_") else 0,
                "mean_absolute_difference": (
                    float(numeric_change.dropna().mean())
                    if numeric_change.notna().any()
                    else np.nan
                ),
                "status": "EVALUATED",
                "explanation": (
                    "Computed with the existing pipeline's exact-hour deltas and "
                    "gap-aware rolling-window rules."
                ),
            }
        )
    corrected_indexed = corrected.set_index(KEYS)
    for column in [
        "spo2_valid_count",
        "spo2_zero_count",
        "spo2_missing_count",
        "spo2_quality_flag",
    ]:
        if column not in corrected_indexed.columns:
            continue
        values = corrected_indexed.loc[common, column]
        numeric = pd.to_numeric(values, errors="coerce")
        rows.append(
            {
                "feature": column,
                "feature_type": "quality_metadata",
                "matched_patient_hours": len(common),
                "original_nonmissing": 0,
                "corrected_nonmissing": int(values.notna().sum()),
                "original_mean": np.nan,
                "corrected_mean": (
                    float(numeric.mean()) if numeric.notna().any() else np.nan
                ),
                "original_min": np.nan,
                "corrected_min": (
                    float(numeric.min()) if numeric.notna().any() else np.nan
                ),
                "original_max": np.nan,
                "corrected_max": (
                    float(numeric.max()) if numeric.notna().any() else np.nan
                ),
                "changed_values": int(values.notna().sum()),
                "changed_nonmissing_values": 0,
                "newly_missing": 0,
                "newly_observed": int(values.notna().sum()),
                "changed_flag_count": 0,
                "mean_absolute_difference": np.nan,
                "status": "ADDED_METADATA",
                "explanation": (
                    "This quality column is available only in the corrected experiment; "
                    "it is reported as metadata, not as a legacy model feature."
                ),
            }
        )
    return pd.DataFrame(rows)


def protected_production_paths() -> list[Path]:
    names = [
        "hourly_patient_state.csv",
        "hourly_patient_state_trends.csv",
        "hourly_patient_state_rules_v2.csv",
        "hourly_patient_state_anomalies_v2.csv",
        "hourly_patient_state_final_alerts.csv",
        "hourly_patient_state_notifications.csv",
        "deterioration_train.csv",
        "deterioration_test.csv",
        "deterioration_probability_predictions.csv",
        "anomaly_train_v2.csv",
        "anomaly_test_v2.csv",
    ]
    paths = [PROCESSED / name for name in names]
    paths.extend(
        [
            ROOT / "ml" / "preprocessing.py",
            ROOT / "ml" / "cleaning.py",
            ROOT / "ml" / "merge_patient_state.py",
            ROOT / "ml" / "trend_features.py",
            ROOT / "ml" / "clinical_rules.py",
            ROOT / "ml" / "final_alert_fusion_v2.py",
            ROOT / "ml" / "build_alert_notification_layer.py",
        ]
    )
    model_dir = ROOT / "ml" / "models"
    if model_dir.exists():
        paths.extend(path for path in model_dir.iterdir() if path.is_file())
    return [path for path in paths if path.is_file()]


def make_report(
    summary: pd.DataFrame,
    row_differences: pd.DataFrame,
    features: pd.DataFrame,
    model_status: str,
    alert_status: str,
    no_unexpected_non_spo2_changes: bool,
) -> str:
    def metric(version: str, name: str) -> Any:
        selected = summary.loc[
            summary["dataset_version"].eq(version) & summary["metric"].eq(name),
            "value",
        ]
        return selected.iloc[0] if len(selected) else "not available"

    changed = int(row_differences["spo2_changed"].sum())
    newly_missing = int(row_differences["newly_missing"].sum())
    raw_changed = int(metric("paired_raw", "changed_spo2_rows"))
    raw_newly_missing = int(metric("paired_raw", "newly_missing_spo2_rows"))
    changed_model_features = features.loc[
        features["changed_values"].gt(0)
        & features["feature_type"].ne("quality_metadata")
    ]
    feature_change_table = "\n".join(
        [
            "| Feature | Changed values | Newly missing | Mean absolute difference |",
            "|---|---:|---:|---:|",
        ]
        + [
            "| "
            + str(row.feature)
            + f" | {int(row.changed_values):,} | {int(row.newly_missing):,} | "
            + (
                f"{float(row.mean_absolute_difference):.4f}"
                if pd.notna(row.mean_absolute_difference)
                else "n/a"
            )
            + " |"
            for row in changed_model_features.itertuples(index=False)
        ]
    )
    metadata_features = features.loc[
        features["feature_type"].eq("quality_metadata"), "feature"
    ].tolist()
    flag_changes = int(features["changed_flag_count"].sum())
    abnormality_flag_changes = int(
        features.loc[
            features["feature"].isin(
                ["flag_hypoxemia_lt_92", "flag_severe_hypoxemia_lt_88"]
            ),
            "changed_flag_count",
        ].sum()
    )
    trend_flag_changes = int(
        features.loc[
            features["feature"].isin(
                ["flag_falling_spo2_gt_3", "flag_rapid_falling_spo2_gt_5"]
            ),
            "changed_flag_count",
        ].sum()
    )
    state_changes = int(
        (
            row_differences["spo2_changed"]
            | row_differences["changed_spo2_feature_or_flag"]
        ).sum()
    )
    rule = (
        "No non-SpO2 changes were found on matched cleaned patient-hours."
        if no_unexpected_non_spo2_changes
        else "Non-SpO2 changes were found; see the per-column and row-level change fields."
    )
    threshold_versions = [
        "original_raw",
        "corrected_raw",
        "original_clean",
        "corrected_clean",
    ]
    threshold_metrics = [
        "spo2_le_20_count",
        "spo2_le_30_count",
        "spo2_le_40_count",
        "spo2_le_50_count",
        "spo2_le_70_count",
        "spo2_le_80_count",
        "spo2_gt_100_count",
    ]
    threshold_table = "\n".join(
        [
            "| Dataset | "
            + " | ".join(
                ["≤20", "≤30", "≤40", "≤50", "≤70", "≤80", ">100"]
            )
            + " |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        + [
            "| "
            + version.replace("_", " ")
            + " | "
            + " | ".join(
                f"{int(metric(version, name)):,}" for name in threshold_metrics
            )
            + " |"
            for version in threshold_versions
        ]
    )
    return f"""# Original vs corrected SpO2 dataset impact

## Verdict

**PASS WITH LIMITATIONS** — the paired dataset comparison is independently
checkable; frozen-model scoring and complete alert replay are not evaluated
because no frozen model artifacts are present.

## Verified findings

- Primary inputs: original cleaned rows {int(metric('original_clean', 'rows')):,};
  corrected cleaned rows {int(metric('corrected_clean', 'rows')):,}.
- On matched valid raw patient-hour keys, {raw_changed:,} SpO2 values changed:
  {raw_newly_missing:,} zero-only hours became missing and
  {raw_changed - raw_newly_missing:,} numeric means changed.
- On the matched cleaned cohort, {changed:,} SpO2 values changed; {newly_missing:,}
  became newly missing. The original cleaner had already converted zero-only
  hours to missing, so no additional cleaned rows became missing.
- Of the {raw_changed - raw_newly_missing:,} mixed-zero hours with a changed
  mean, {int(metric('paired_raw', 'hours_with_increased_mean_after_zero_exclusion')):,}
  had a higher candidate mean after zero exclusion. This describes the arithmetic
  change, not clinical plausibility.
- Corrected-clean SpO2 missing count: {int(metric('corrected_clean', 'spo2_missing_count')):,};
  exact-zero values: {int(metric('corrected_clean', 'spo2_zero_count')):,}.
- Corrected raw quality flags distinguish mixed-zero, zero-only, valid-nonzero,
  and missing-only hours. These labels describe source composition; they do not
  assert clinical validity.
- {rule}
- Changed SpO2-derived features: {', '.join(changed_model_features['feature'].tolist()) if not changed_model_features.empty else 'none'}.
- Newly available corrected-data quality metadata (not legacy model inputs):
  {', '.join(metadata_features) if metadata_features else 'none'}.
- Changed SpO2 abnormality flags: {abnormality_flag_changes:,}; changed trend
  flags: {trend_flag_changes:,} ({flag_changes:,} combined). Patient-hours
  with a SpO2 value, feature, or flag change: {state_changes:,}.
- Model evaluation: **{model_status}**.
- Alert/notification evaluation: **{alert_status}**.

## Dataset-level SpO2 snapshot

| Dataset | Rows | Stays | Missing | Zero | Nonzero | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original raw | {int(metric('original_raw', 'rows')):,} | {int(metric('original_raw', 'unique_icu_stays')):,} | {int(metric('original_raw', 'spo2_missing_count')):,} | {int(metric('original_raw', 'spo2_zero_count')):,} | {int(metric('original_raw', 'spo2_nonzero_count')):,} | {float(metric('original_raw', 'spo2_mean')):.3f} | {float(metric('original_raw', 'spo2_median')):.3f} | {float(metric('original_raw', 'spo2_min')):.3f} | {float(metric('original_raw', 'spo2_max')):.3f} |
| Corrected raw | {int(metric('corrected_raw', 'rows')):,} | {int(metric('corrected_raw', 'unique_icu_stays')):,} | {int(metric('corrected_raw', 'spo2_missing_count')):,} | {int(metric('corrected_raw', 'spo2_zero_count')):,} | {int(metric('corrected_raw', 'spo2_nonzero_count')):,} | {float(metric('corrected_raw', 'spo2_mean')):.3f} | {float(metric('corrected_raw', 'spo2_median')):.3f} | {float(metric('corrected_raw', 'spo2_min')):.3f} | {float(metric('corrected_raw', 'spo2_max')):.3f} |
| Original clean | {int(metric('original_clean', 'rows')):,} | {int(metric('original_clean', 'unique_icu_stays')):,} | {int(metric('original_clean', 'spo2_missing_count')):,} | {int(metric('original_clean', 'spo2_zero_count')):,} | {int(metric('original_clean', 'spo2_nonzero_count')):,} | {float(metric('original_clean', 'spo2_mean')):.3f} | {float(metric('original_clean', 'spo2_median')):.3f} | {float(metric('original_clean', 'spo2_min')):.3f} | {float(metric('original_clean', 'spo2_max')):.3f} |
| Corrected clean | {int(metric('corrected_clean', 'rows')):,} | {int(metric('corrected_clean', 'unique_icu_stays')):,} | {int(metric('corrected_clean', 'spo2_missing_count')):,} | {int(metric('corrected_clean', 'spo2_zero_count')):,} | {int(metric('corrected_clean', 'spo2_nonzero_count')):,} | {float(metric('corrected_clean', 'spo2_mean')):.3f} | {float(metric('corrected_clean', 'spo2_median')):.3f} | {float(metric('corrected_clean', 'spo2_min')):.3f} | {float(metric('corrected_clean', 'spo2_max')):.3f} |

Cleaning removed {int(metric('original_clean', 'rows_removed_during_cleaning')):,}
rows in the original workflow and
{int(metric('corrected_clean', 'rows_removed_during_cleaning')):,} in the
experimental workflow. Threshold counts, duplicate counts, full missingness,
and row-level differences are in the CSV deliverables.

### SpO2 threshold counts

{threshold_table}

The raw files each contain
{int(metric('original_raw', 'invalid_patient_hour_key_rows')):,} rows without a
valid patient-hour key; these are excluded from keyed raw matching. Each
cleaning workflow removes 136 total rows.

## Feature impact

Temporal SpO2 features reproduce `trend_features.py`: exact 1/3/6-hour
differences and rolling windows of 3/6 observations with a minimum of two
observations, restarted after a gap greater than one hour. Feature comparisons
use matched patient-hour keys only. Direct SpO2 rule flags mirror the existing
thresholds `<92`, `<88`, one-hour drop `<-3`, and rapid drop `<-5`.

{feature_change_table}

SpO2 level corrections propagate to exact-hour differences and rolling
statistics; threshold and trend flags change when those recalculated values
cross the existing fixed rule boundaries. Counts and distributions are in
`feature_impact_summary.csv`.

## Model, alert, and notification impact

No frozen files are present in `ml/models` (including the paths expected by the
training scripts). Retraining is prohibited for this milestone, so model metrics
and model-dependent alert/notification differences are **NOT EVALUATED**.
Existing train/test and score CSVs are not substitutes for a frozen estimator:
they cannot rescore the corrected feature matrix. The alert outputs explicitly
record this blocker; no alert counts or safety effects are inferred.

The candidate deterioration model is a median-imputation, standard-scaling
LogisticRegression pipeline using 22 current, trend, rolling, fluid, treatment,
and clinical-count features; its scripted evaluation reports AUROC, AUPRC,
baseline PR, and Brier score. Its training/test files are
`deterioration_train.csv` and `deterioration_test.csv`. The anomaly candidate is
a median-imputation IsolationForest (300 trees, 2% contamination), using the
17 features listed by `train_anomaly_model_v2.py`; the saved per-row evaluation
contains 30,913 scores but no reusable estimator. Neither candidate has a
frozen `.joblib` artifact in `ml/models`, so no paired performance delta can be
computed.

The label-generation code defines `deterioration_label_24h` as an ICU stay
marked expired whose ICU discharge offset is within the next 24 hours of the
prediction hour. It is a death-near-ICU-discharge target, not a general
deterioration outcome. The existing split selects `patientunitstayid` ICU-stay
IDs; person-level separation across multiple stays cannot be confirmed from
the saved modeling artifacts. No split or model was regenerated here.

## Observations, hypotheses, limitations, and recommendations

- **Data-quality impact:** exact-zero contamination is removed from corrected
  hourly means, 47 zero-only raw candidates are missing instead of zero, and
  quality metadata is retained. The corrected statistics are consistent with
  the intended arithmetic policy, but this is not evidence that the data are
  clinically more accurate.
- **Observation:** removing exact zeros from candidate hourly averages changes
  selected means and turns zero-only hours into missing candidates while
  preserving low nonzero values and quality metadata.
- **Hypothesis:** removing mixed-zero downward contamination may reduce
  spurious low-SpO2 signals. This comparison alone does not establish clinical
  validity or improved patient outcomes.
- **Limitations:** retrospective eICU demo data; nonzero values are not
  automatically clinically valid; cleaned cohorts may omit empty measurement
  rows; no frozen estimators exist for paired scoring; full alert and
  notification replay was therefore not performed.
- **Recommendation:** retain this as an experimental candidate for further
  data-quality and cohort review. Do not promote it or claim clinical
  validation. Restore frozen model artifacts and document the intended split
  before a separate, approved model/alert evaluation.
- **Further experimental development:** the corrected dataset may be considered
  as an experimental input after clinical/data-governance review; it is not
  validated for model performance or clinical use.
"""


def main() -> None:
    missing_paths = [str(path) for path in DATASETS.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Required comparison inputs are missing: {missing_paths}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_metadata_before = {
        name: file_metadata(path) for name, path in DATASETS.items()
    }
    production_fingerprints_before = {
        str(path.resolve()): sha256(path)
        for path in protected_production_paths()
    }

    frames = {
        name: pd.read_csv(path, low_memory=False)
        for name, path in DATASETS.items()
    }
    for name, frame in frames.items():
        require_schema(frame, name)
    rules_path = PROCESSED / "hourly_patient_state_rules_v2.csv"
    if not rules_path.is_file():
        raise FileNotFoundError(
            f"Cannot reproduce downstream rule features; missing {rules_path}"
        )
    saved_rules = pd.read_csv(rules_path, low_memory=False)

    cleaning_removed = {
        "original_raw": None,
        "original_clean": len(frames["original_raw"]) - len(frames["original_clean"]),
        "corrected_raw": None,
        "corrected_clean": len(frames["corrected_raw"]) - len(frames["corrected_clean"]),
    }
    metrics = [
        row
        for name, frame in frames.items()
        for row in dataset_metrics(name, frame, DATASETS[name], cleaning_removed[name])
    ]
    summary = pd.DataFrame(metrics)

    raw_pair = pair_data(
        frames["original_raw"],
        frames["corrected_raw"],
        "original_raw",
        "corrected_raw",
    )
    clean_pair = pair_data(
        frames["original_clean"],
        frames["corrected_clean"],
        "original_clean",
        "corrected_clean",
    )

    raw_matched = raw_pair.loc[raw_pair["_merge"].eq("both")].copy()
    original_raw_spo2 = pd.to_numeric(
        raw_matched["spo2_original"], errors="coerce"
    ).reset_index(drop=True)
    corrected_raw_spo2 = pd.to_numeric(
        raw_matched["spo2_corrected"], errors="coerce"
    ).reset_index(drop=True)
    raw_spo2_changed = different(original_raw_spo2, corrected_raw_spo2)
    raw_newly_missing = original_raw_spo2.notna() & corrected_raw_spo2.isna()
    raw_quality = frames["corrected_raw"].set_index(KEYS)
    raw_quality_flag = raw_quality.get(
        "spo2_quality_flag", pd.Series(index=raw_quality.index, dtype="object")
    )
    zero_counts = pd.to_numeric(
        raw_quality.get("spo2_zero_count", pd.Series(index=raw_quality.index)),
        errors="coerce",
    )
    valid_counts = pd.to_numeric(
        raw_quality.get("spo2_valid_count", pd.Series(index=raw_quality.index)),
        errors="coerce",
    )
    corrected_keys = pd.MultiIndex.from_frame(raw_matched[KEYS])
    original_raw_spo2.index = corrected_keys
    corrected_raw_spo2.index = corrected_keys
    raw_spo2_changed.index = corrected_keys
    shared_zero_counts = zero_counts.reindex(corrected_keys)
    shared_valid_counts = valid_counts.reindex(corrected_keys)
    raw_mean_increased = (
        shared_zero_counts.gt(0)
        & shared_valid_counts.gt(0)
        & original_raw_spo2.lt(corrected_raw_spo2)
        & corrected_raw_spo2.notna()
    )

    clean_pair["original_spo2"] = pd.to_numeric(
        clean_pair["spo2_original"], errors="coerce"
    )
    clean_pair["corrected_spo2"] = pd.to_numeric(
        clean_pair["spo2_corrected"], errors="coerce"
    )
    clean_pair["absolute_spo2_difference"] = (
        clean_pair["original_spo2"] - clean_pair["corrected_spo2"]
    ).abs()
    clean_pair["original_spo2_missing"] = clean_pair["original_spo2"].isna()
    clean_pair["corrected_spo2_missing"] = clean_pair["corrected_spo2"].isna()
    clean_pair["spo2_changed"] = different(
        clean_pair["original_spo2"], clean_pair["corrected_spo2"]
    )
    clean_pair["newly_missing"] = (
        clean_pair["original_spo2"].notna()
        & clean_pair["corrected_spo2"].isna()
    )
    clean_pair["newly_observed"] = (
        clean_pair["original_spo2"].isna()
        & clean_pair["corrected_spo2"].notna()
    )
    quality_source = frames["corrected_clean"].set_index(KEYS)
    for column in [
        "spo2_raw_mean",
        "spo2_valid_count",
        "spo2_zero_count",
        "spo2_missing_count",
        "spo2_quality_flag",
    ]:
        if column in quality_source.columns:
            clean_pair[f"corrected_{column}"] = pd.MultiIndex.from_frame(
                clean_pair[KEYS]
            ).map(
                quality_source[column]
            )
        else:
            clean_pair[f"corrected_{column}"] = np.nan
    for column in [
        "spo2_valid_count",
        "spo2_zero_count",
        "spo2_missing_count",
        "spo2_quality_flag",
    ]:
        clean_pair[f"original_{column}"] = np.nan

    original_raw_by_key = frames["original_raw"].loc[
        frames["original_raw"][KEYS].notna().all(axis=1)
    ].set_index(KEYS)
    corrected_raw_by_key = frames["corrected_raw"].loc[
        frames["corrected_raw"][KEYS].notna().all(axis=1)
    ].set_index(KEYS)
    clean_key_index = pd.MultiIndex.from_frame(clean_pair[KEYS])
    clean_pair["original_raw_spo2"] = clean_key_index.map(
        original_raw_by_key["spo2"]
    )
    clean_pair["corrected_raw_spo2"] = clean_key_index.map(
        corrected_raw_by_key["spo2"]
    )
    clean_pair["original_spo2_raw_mean"] = clean_pair["original_raw_spo2"]
    clean_pair["corrected_spo2_raw_mean"] = clean_key_index.map(
        quality_source["spo2_raw_mean"]
    )
    clean_pair["raw_absolute_spo2_difference"] = (
        pd.to_numeric(clean_pair["original_raw_spo2"], errors="coerce")
        - pd.to_numeric(clean_pair["corrected_raw_spo2"], errors="coerce")
    ).abs()
    clean_pair["raw_spo2_changed"] = different(
        clean_pair["original_raw_spo2"], clean_pair["corrected_raw_spo2"]
    ) & clean_pair["_merge"].eq("both")
    clean_pair["raw_newly_missing"] = (
        pd.to_numeric(clean_pair["original_raw_spo2"], errors="coerce").notna()
        & pd.to_numeric(clean_pair["corrected_raw_spo2"], errors="coerce").isna()
        & clean_pair["_merge"].eq("both")
    )

    shared_clean = clean_pair.loc[clean_pair["_merge"].eq("both")].copy()
    other_columns = [
        column
        for column in MEASUREMENT_COLUMNS
        if column in frames["original_clean"].columns
        and column in frames["corrected_clean"].columns
    ]
    change_by_column: dict[str, pd.Series] = {}
    for column in other_columns:
        left_name = f"{column}_original"
        right_name = f"{column}_corrected"
        changed_column = different(
            shared_clean[left_name], shared_clean[right_name]
        )
        change_by_column[column] = changed_column
        clean_pair[f"original_{column}"] = clean_pair[left_name]
        clean_pair[f"corrected_{column}"] = clean_pair[right_name]
    clean_pair = clean_pair.copy()
    shared_clean = clean_pair.loc[clean_pair["_merge"].eq("both")].copy()
    shared_changed_names = pd.DataFrame(change_by_column, index=shared_clean.index)
    clean_pair["changed_non_spo2_columns"] = shared_changed_names.apply(
        lambda row: "|".join(row.index[row.to_numpy(dtype=bool)]), axis=1
    ).reindex(clean_pair.index, fill_value="")
    clean_pair["any_non_spo2_measurement_changed"] = (
        clean_pair["changed_non_spo2_columns"].ne("")
    )
    clean_pair["changed_spo2_feature_or_flag"] = False

    original_features = add_rule_impact_features(
        make_spo2_features(frames["original_clean"]), saved_rules
    ).set_index(KEYS)
    corrected_features = add_rule_impact_features(
        make_spo2_features(frames["corrected_clean"]), saved_rules
    ).set_index(KEYS)
    baseline_rule_output = saved_rules.set_index(KEYS)
    baseline_columns = [
        "clinical_warning_count",
        "clinical_severe_count",
        "clinical_rule_severity",
        "clinical_rule_explanation",
        "clinical_warning_count_v2",
        "clinical_severe_count_v2",
        "clinical_rule_severity_v2",
        "clinical_rule_explanation_v2",
    ]
    baseline_differences = {
        column: int(
            different(
                original_features[column],
                baseline_rule_output[column].reindex(original_features.index),
            ).sum()
        )
        for column in baseline_columns
    }
    if any(baseline_differences.values()):
        raise ValueError(
            "Rebuilt original rule outputs do not match the saved production rules: "
            f"{baseline_differences}"
        )
    feature_summary = compare_features(
        frames["original_clean"], frames["corrected_clean"], saved_rules
    )
    common_feature_keys = original_features.index.intersection(corrected_features.index)
    feature_names = [name for name in original_features.columns if name not in KEYS]
    feature_change_mask = pd.Series(False, index=common_feature_keys)
    for name in feature_names:
        feature_change_mask |= different(
            original_features.loc[common_feature_keys, name],
            corrected_features.loc[common_feature_keys, name],
        )
    feature_changed_keys = set(feature_change_mask.index[feature_change_mask])
    clean_pair["changed_spo2_feature_or_flag"] = [
        (patient, hour) in feature_changed_keys
        for patient, hour in zip(clean_pair["patientunitstayid"], clean_pair["hour"])
    ]
    clean_pair["comparison_status"] = clean_pair["_merge"].astype(str)
    row_columns = (
        KEYS
        + [
            "comparison_status",
            "original_spo2",
            "corrected_spo2",
            "absolute_spo2_difference",
            "original_spo2_missing",
            "corrected_spo2_missing",
            "newly_missing",
            "newly_observed",
            "spo2_changed",
            "original_raw_spo2",
            "corrected_raw_spo2",
            "raw_absolute_spo2_difference",
            "raw_spo2_changed",
            "raw_newly_missing",
            "original_spo2_raw_mean",
            "corrected_spo2_raw_mean",
            "original_spo2_valid_count",
            "corrected_spo2_valid_count",
            "original_spo2_zero_count",
            "corrected_spo2_zero_count",
            "original_spo2_missing_count",
            "corrected_spo2_missing_count",
            "original_spo2_quality_flag",
            "corrected_spo2_quality_flag",
        ]
        + [
            item
            for column in other_columns
            for item in (f"original_{column}", f"corrected_{column}")
        ]
        + [
            "changed_non_spo2_columns",
            "any_non_spo2_measurement_changed",
            "changed_spo2_feature_or_flag",
        ]
    )
    row_differences = clean_pair[row_columns].copy()
    row_differences.to_csv(
        OUTPUT_DIR / "row_level_spo2_differences.csv", index=False
    )

    missingness_rows = []
    for version, frame in frames.items():
        for column in frame.columns:
            missing_count = int(frame[column].isna().sum())
            missingness_rows.append(
                {
                    "dataset_version": version,
                    "column": column,
                    "row_count": len(frame),
                    "missing_count": missing_count,
                    "missing_percent": 100 * missing_count / len(frame) if len(frame) else 0,
                }
            )
    missingness = pd.DataFrame(missingness_rows)
    missingness.to_csv(OUTPUT_DIR / "missingness_comparison.csv", index=False)

    summary_extra = [
        ("paired_raw", "matched_patient_hours", int(raw_pair["_merge"].eq("both").sum())),
        ("paired_raw", "unmatched_original_rows", int(raw_pair["_merge"].eq("left_only").sum())),
        ("paired_raw", "unmatched_corrected_rows", int(raw_pair["_merge"].eq("right_only").sum())),
        ("paired_raw", "original_rows_excluded_for_null_join_keys", int(frames["original_raw"][KEYS].isna().any(axis=1).sum())),
        ("paired_raw", "corrected_rows_excluded_for_null_join_keys", int(frames["corrected_raw"][KEYS].isna().any(axis=1).sum())),
        ("paired_raw", "changed_spo2_rows", int(raw_spo2_changed.sum())),
        ("paired_raw", "newly_missing_spo2_rows", int(raw_newly_missing.sum())),
        ("paired_raw", "mixed_zero_hours", int(raw_quality_flag.eq("MIXED_ZERO").sum())),
        ("paired_raw", "zero_only_hours", int(raw_quality_flag.eq("ZERO_ONLY").sum())),
        ("paired_raw", "valid_nonzero_no_zero_hours", int(raw_quality_flag.eq("VALID").sum())),
        ("paired_raw", "missing_only_hours", int(raw_quality_flag.eq("MISSING_ONLY").sum())),
        ("paired_raw", "hours_with_increased_mean_after_zero_exclusion", int(raw_mean_increased.sum())),
        ("paired_clean", "matched_patient_hours", int(clean_pair["_merge"].eq("both").sum())),
        ("paired_clean", "unmatched_original_rows", int(clean_pair["_merge"].eq("left_only").sum())),
        ("paired_clean", "unmatched_corrected_rows", int(clean_pair["_merge"].eq("right_only").sum())),
        ("paired_clean", "changed_spo2_rows", int(clean_pair["spo2_changed"].sum())),
        ("paired_clean", "unchanged_spo2_rows", int((~clean_pair["spo2_changed"]).sum())),
        ("paired_clean", "newly_missing_spo2_rows", int(clean_pair["newly_missing"].sum())),
        ("paired_clean", "newly_observed_spo2_rows", int(clean_pair["newly_observed"].sum())),
        ("paired_clean", "patient_hours_with_spo2_feature_or_flag_change", len(feature_changed_keys)),
        ("paired_clean", "patient_hours_with_any_non_spo2_measurement_change", int(clean_pair["any_non_spo2_measurement_changed"].sum())),
    ]
    for column, changed in change_by_column.items():
        summary_extra.append(
            ("paired_clean", f"changed_non_spo2_measurement_rows:{column}", int(changed.sum()))
        )
    for version, metric_name, value in summary_extra:
        summary.loc[len(summary)] = {
            "dataset_version": version,
            "metric": metric_name,
            "value": value,
        }
    summary.to_csv(OUTPUT_DIR / "dataset_comparison_summary.csv", index=False)
    feature_summary.to_csv(OUTPUT_DIR / "feature_impact_summary.csv", index=False)

    model_artifacts = [
        ROOT / "ml" / "models" / "deterioration_logistic_v1.joblib",
        ROOT / "ml" / "models" / "anomaly_isolation_forest_v2.joblib",
    ]
    missing_models = [str(path) for path in model_artifacts if not path.is_file()]
    model_status = (
        "EVALUATED"
        if not missing_models
        else "NOT EVALUATED — frozen model artifacts are absent; retraining is out of scope."
    )
    model_rows = [
        {
            "model": "deterioration_logistic_v1",
            "status": "NOT EVALUATED",
            "model_type": "LogisticRegression pipeline (imputer + scaler + classifier)",
            "input_features": json.dumps(
                [
                    "heart_rate", "spo2", "respiration", "systolic_bp",
                    "diastolic_bp", "mean_bp", "heart_rate_change_1h",
                    "spo2_change_1h", "respiration_change_1h",
                    "systolic_bp_change_1h", "diastolic_bp_change_1h",
                    "mean_bp_change_1h", "heart_rate_rolling_mean_3h",
                    "spo2_rolling_mean_3h", "respiration_rolling_mean_3h",
                    "mean_bp_rolling_mean_3h", "net_fluid_balance_ml",
                    "urine_output_ml", "active_infusion_count",
                    "vasoactive_support", "clinical_warning_count",
                    "clinical_severe_count",
                ]
            ),
            "target": "deterioration_label_24h",
            "target_definition": (
                "expired ICU stay and discharge within the following 24 hours"
            ),
            "training_data": "data/processed/deterioration_train.csv",
            "test_data": "data/processed/deterioration_test.csv",
            "split_and_preprocessing": (
                "80/20 split of patientunitstayid with random_state=42; "
                "median imputation, standard scaling, LogisticRegression "
                "(class_weight=None, max_iter=2000, random_state=42)"
            ),
            "evaluation_script_and_metrics": (
                "ml/train_deterioration_probability_model.py: AUROC, AUPRC, "
                "baseline PR, Brier; ml/evaluate_deterioration_patient_level.py: "
                "patient/stay-level AUROC, AUPRC, precision, recall, F1"
            ),
            "evaluation_cohort": (
                "Existing patientunitstayid split; person-level separation across "
                "multiple stays is unverified"
            ),
            "metrics": np.nan,
            "blocker": "" if model_artifacts[0].is_file() else str(model_artifacts[0]),
            "explanation": (
                "Frozen artifact was not found. The training script is not run because retraining is prohibited."
                if not model_artifacts[0].is_file()
                else "Artifact exists; paired evaluation still requires explicit compatibility checks."
            ),
        },
        {
            "model": "anomaly_isolation_forest_v2",
            "status": "NOT EVALUATED",
            "model_type": "IsolationForest pipeline",
            "input_features": json.dumps(
                [
                    "heart_rate", "spo2", "respiration", "systolic_bp",
                    "diastolic_bp", "mean_bp", "heart_rate_change_1h",
                    "spo2_change_1h", "mean_bp_change_1h",
                    "heart_rate_rolling_mean_3h", "spo2_rolling_mean_3h",
                    "respiration_rolling_mean_3h", "mean_bp_rolling_mean_3h",
                    "net_fluid_balance_ml", "urine_output_ml",
                    "active_infusion_count", "vasoactive_support",
                ]
            ),
            "target": "unsupervised anomaly score",
            "target_definition": "No supervised target",
            "training_data": "data/processed/anomaly_train_v2.csv",
            "test_data": "data/processed/anomaly_test_v2.csv",
            "split_and_preprocessing": (
                "80/20 split of patientunitstayid with random_state=42; "
                "median imputation; IsolationForest(n_estimators=300, "
                "contamination=0.02, random_state=42)"
            ),
            "evaluation_script_and_metrics": (
                "ml/train_anomaly_model_v2.py reports anomaly rate and score "
                "distribution; ml/evaluate_anomaly_v2.py and "
                "ml/evaluate_anomaly_v2_with_rules_v2.py compare anomaly/rule "
                "overlap, without a clinical ground-truth target"
            ),
            "evaluation_cohort": (
                "Existing patientunitstayid split; person-level separation across "
                "multiple stays is unverified"
            ),
            "metrics": np.nan,
            "blocker": "" if model_artifacts[1].is_file() else str(model_artifacts[1]),
            "explanation": (
                "Frozen artifact was not found. The training script is not run because retraining is prohibited."
                if not model_artifacts[1].is_file()
                else "Artifact exists; paired evaluation still requires explicit compatibility checks."
            ),
        },
    ]
    pd.DataFrame(model_rows).to_csv(
        OUTPUT_DIR / "model_performance_comparison.csv", index=False
    )

    alert_reason = (
        "NOT EVALUATED: the alert and notification pipeline consumes model outputs; "
        "the frozen anomaly and deterioration estimators are absent. Existing prediction "
        "CSVs cannot rescore the corrected features, and training/replacing scores is prohibited."
    )
    alert_metrics = [
        "total_alert_observations",
        "unique_stays_with_alerts",
        "severe_alerts",
        "moderate_alerts",
        "low_priority_alerts",
        "spo2_related_alerts",
        "alerts_changed_due_to_spo2_features",
        "new_alerts",
        "removed_alerts",
        "changed_severity",
        "changed_explanations",
        "changed_notification_actions",
        "changed_suppression_or_escalation",
    ]
    pd.DataFrame(
        [
            {
                "metric": name,
                "original_value": np.nan,
                "corrected_value": np.nan,
                "difference": np.nan,
                "status": "NOT EVALUATED",
                "explanation": alert_reason,
            }
            for name in alert_metrics
        ]
    ).to_csv(OUTPUT_DIR / "alert_impact_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "evaluation_status": "NOT EVALUATED",
                "patientunitstayid": np.nan,
                "hour": np.nan,
                "original_alert": np.nan,
                "corrected_alert": np.nan,
                "change_origin": np.nan,
                "blocker": alert_reason,
            }
        ]
    ).to_csv(OUTPUT_DIR / "alert_level_differences.csv", index=False)

    production_fingerprints_after = {
        str(path.resolve()): sha256(path)
        for path in protected_production_paths()
    }
    if production_fingerprints_before != production_fingerprints_after:
        raise RuntimeError("A protected production file changed during the comparison.")
    for name, path in DATASETS.items():
        if input_metadata_before[name]["sha256"] != sha256(path):
            raise RuntimeError(f"An input dataset changed during the comparison: {path}")
    config = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "script_version": "1.0",
        "project_root": str(ROOT),
        "output_directory": str(OUTPUT_DIR.resolve()),
        "input_datasets": {
            name: {
                **input_metadata_before[name],
                "rows": len(frames[name]),
                "columns": list(frames[name].columns),
                "duplicate_patient_hour_keys": int(frames[name].duplicated(KEYS).sum()),
            }
            for name, path in DATASETS.items()
        },
        "primary_comparison": {
            "original": str(DATASETS["original_clean"].resolve()),
            "corrected": str(DATASETS["corrected_clean"].resolve()),
            "join_keys": KEYS,
            "matched_cleaned_patient_hours": int(clean_pair["_merge"].eq("both").sum()),
            "full_data_and_matched_cohort_reported": True,
            "numeric_comparison_tolerance": {"rtol": 1e-9, "atol": 1e-9},
        },
        "configuration": {
            "zero_policy": "exclude exact zero only from corrected hourly mean",
            "low_nonzero_values_retained": True,
            "episode_or_clinical_validity_claim": False,
            "feature_rules": {
                "exact_hour_deltas": [1, 3, 6],
                "rolling_hours": [3, 6],
                "rolling_min_periods": 2,
                "restart_after_gap_hours": 1,
                "hypoxemia_lt": 92,
                "severe_hypoxemia_lt": 88,
                "falling_spo2_change_lt": -3,
                "rapid_falling_spo2_change_lt": -5,
            },
            "model_retraining": False,
            "alert_threshold_changes": False,
        },
        "model_artifact_paths": [str(path.resolve()) for path in model_artifacts],
        "missing_model_artifacts": missing_models,
        "model_evaluation_status": model_status,
        "alert_evaluation_status": "NOT EVALUATED",
        "production_files_sha256_before": production_fingerprints_before,
        "production_files_sha256_after": production_fingerprints_after,
        "outputs": [
            "dataset_comparison_summary.csv",
            "row_level_spo2_differences.csv",
            "missingness_comparison.csv",
            "feature_impact_summary.csv",
            "model_performance_comparison.csv",
            "alert_impact_summary.csv",
            "alert_level_differences.csv",
            "experiment_configuration.json",
            "comparison_report.md",
        ],
    }
    (OUTPUT_DIR / "experiment_configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "comparison_report.md").write_text(
        make_report(
            summary,
            row_differences,
            feature_summary,
            model_status,
            "NOT EVALUATED",
            not bool(clean_pair["any_non_spo2_measurement_changed"].any()),
        ),
        encoding="utf-8",
    )
    print(f"SpO2 comparison completed: {OUTPUT_DIR}")
    print(
        "Clean pair: "
        f"{int(clean_pair['_merge'].eq('both').sum()):,} matched, "
        f"{int(clean_pair['spo2_changed'].sum()):,} changed SpO2, "
        f"{int(clean_pair['newly_missing'].sum()):,} newly missing."
    )
    print(
        "Changed SpO2 model/rule features: "
        f"{int((feature_summary['changed_values'].gt(0) & feature_summary['feature_type'].ne('quality_metadata')).sum())}; "
        "new quality metadata fields: "
        f"{int(feature_summary['feature_type'].eq('quality_metadata').sum())}; "
        "models and alerts: NOT EVALUATED."
    )


if __name__ == "__main__":
    main()
