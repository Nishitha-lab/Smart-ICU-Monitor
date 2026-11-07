from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED / "spo2_comparison"
VALIDATION_DIR = OUTPUT_DIR / "validation"
KEYS = ["patientunitstayid", "hour"]
DATASETS = {
    "original_raw": PROCESSED / "hourly_vitals_labs.csv",
    "original_clean": PROCESSED / "hourly_vitals_labs_clean.csv",
    "corrected_raw": PROCESSED / "hourly_vitals_labs_spo2_experiment.csv",
    "corrected_clean": PROCESSED / "hourly_vitals_labs_spo2_experiment_clean.csv",
}
RULES_PATH = PROCESSED / "hourly_patient_state_rules_v2.csv"
NON_SPO2_MEASUREMENTS = [
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value_diff(left: pd.Series, right: pd.Series) -> pd.Series:
    both_missing = left.isna() & right.isna()
    numeric_left = pd.to_numeric(left, errors="coerce")
    numeric_right = pd.to_numeric(right, errors="coerce")
    numeric_comparable = (
        both_missing
        | (left.notna() & numeric_left.notna())
        | (right.notna() & numeric_right.notna())
    )
    equal = pd.Series(False, index=left.index)
    equal.loc[both_missing] = True
    numeric_mask = numeric_comparable & ~both_missing
    if numeric_mask.any():
        equal.loc[numeric_mask] = np.isclose(
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
    return ~equal.where(numeric_comparable, string_equal)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def independent_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[KEYS + ["spo2"]].copy()
    data["hour"] = pd.to_numeric(data["hour"], errors="coerce")
    data["spo2"] = pd.to_numeric(data["spo2"], errors="coerce")
    data = data.sort_values(KEYS).reset_index(drop=True)
    features = data[KEYS + ["spo2"]].copy()
    groups = data.groupby("patientunitstayid", sort=False)

    lag_value = groups["spo2"].shift(1)
    lag_hour = groups["hour"].shift(1)
    features["spo2_change_1h"] = (data["spo2"] - lag_value).where(
        (data["hour"] - lag_hour).eq(1)
    )

    lookup = data.set_index(KEYS)["spo2"]
    for hours in [3, 6]:
        target = pd.MultiIndex.from_arrays(
            [data["patientunitstayid"], data["hour"] - hours],
            names=KEYS,
        )
        prior = lookup.reindex(target).to_numpy()
        features[f"spo2_change_{hours}h"] = data["spo2"].to_numpy() - prior

    gap = groups["hour"].diff()
    segment_break = gap.isna() | gap.gt(1)
    segment_seq = segment_break.groupby(
        data["patientunitstayid"], sort=False
    ).cumsum()
    segment_id = (
        data["patientunitstayid"].astype(str)
        + "_"
        + segment_seq.astype(str)
    )
    segment_groups = data["spo2"].groupby(segment_id, sort=False)
    for hours in [3, 6]:
        features[f"spo2_rolling_mean_{hours}h"] = segment_groups.transform(
            lambda series: series.rolling(hours, min_periods=2).mean()
        )
        features[f"spo2_rolling_std_{hours}h"] = segment_groups.transform(
            lambda series: series.rolling(hours, min_periods=2).std()
        )
    features["flag_hypoxemia_lt_92"] = features["spo2"].lt(92).astype("Int64")
    features["flag_severe_hypoxemia_lt_88"] = features["spo2"].lt(88).astype(
        "Int64"
    )
    features["flag_falling_spo2_gt_3"] = (
        features["spo2_change_1h"].lt(-3).astype("Int64")
    )
    features["flag_rapid_falling_spo2_gt_5"] = (
        features["spo2_change_1h"].lt(-5).astype("Int64")
    )
    return features


def independent_rule_features(
    frame: pd.DataFrame, saved_rules: pd.DataFrame
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
    rule_columns = [
        *warning_columns,
        *severe_columns,
        "rule_severe_hypertension",
        "hypertension_explanation",
    ]
    required = KEYS + rule_columns
    missing = [column for column in required if column not in saved_rules]
    if missing:
        raise ValueError(f"Saved clinical-rule output is missing columns: {missing}")

    rebuilt = independent_features(frame).merge(
        saved_rules[required],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    rebuilt["rule_hypoxemia"] = rebuilt["flag_hypoxemia_lt_92"]
    rebuilt["rule_severe_hypoxemia"] = rebuilt["flag_severe_hypoxemia_lt_88"]
    rebuilt["rule_falling_spo2"] = rebuilt["flag_falling_spo2_gt_3"]
    rebuilt["rule_rapid_falling_spo2"] = rebuilt["flag_rapid_falling_spo2_gt_5"]
    count_columns = warning_columns + severe_columns + ["rule_severe_hypertension"]
    rebuilt[count_columns] = (
        rebuilt[count_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    warnings = rebuilt[warning_columns]
    severe = rebuilt[severe_columns]
    hypertension = rebuilt["rule_severe_hypertension"]
    rebuilt["clinical_warning_count"] = warnings.sum(axis=1)
    rebuilt["clinical_severe_count"] = severe.sum(axis=1)
    rebuilt["clinical_warning_count_v2"] = (
        rebuilt["clinical_warning_count"] + hypertension
    )
    rebuilt["clinical_severe_count_v2"] = (
        rebuilt["clinical_severe_count"] + hypertension
    )

    def severity(warning_count: pd.Series, severe_count: pd.Series) -> pd.Series:
        return pd.Series(
            np.select(
                [
                    severe_count.ge(2),
                    severe_count.eq(1),
                    warning_count.ge(2),
                    warning_count.eq(1),
                ],
                ["critical", "high", "moderate", "low"],
                default="normal",
            ),
            index=rebuilt.index,
        )

    base_severity = severity(
        rebuilt["clinical_warning_count"], rebuilt["clinical_severe_count"]
    )
    severity_rank = base_severity.map(
        {"normal": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
    )
    severity_rank = severity_rank.where(
        ~hypertension.eq(1), severity_rank.clip(lower=3)
    )
    rebuilt["clinical_rule_severity"] = base_severity
    rebuilt["clinical_rule_severity_v2"] = severity_rank.map(
        {0: "normal", 1: "low", 2: "moderate", 3: "high", 4: "critical"}
    )

    labels = [
        "tachycardia",
        "bradycardia",
        "low SpO2",
        "abnormal respiration",
        "low systolic BP",
        "low MAP",
        "rapid HR rise",
        "rapid HR fall",
        "falling SpO2",
        "falling MAP",
        "vasoactive support",
        "multiple active infusions",
        "large positive fluid balance",
        "large negative fluid balance",
    ]
    rebuilt["clinical_rule_explanation"] = [
        "; ".join(
            label for label, active in zip(labels, values) if active == 1
        )
        or "No rule-based abnormality"
        for values in warnings.itertuples(index=False, name=None)
    ]
    hypertension_text = rebuilt["hypertension_explanation"].fillna("").astype(str)
    rebuilt["clinical_rule_explanation_v2"] = [
        (
            high
            if high and text in {"", "No rule-based abnormality", "nan"}
            else text + "; " + high
            if high
            else text
        )
        for text, high in zip(
            rebuilt["clinical_rule_explanation"], hypertension_text
        )
    ]
    return rebuilt.set_index(KEYS)[
        [
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


def add_check(
    checks: list[dict[str, Any]],
    discrepancies: list[dict[str, Any]],
    name: str,
    expected: Any,
    actual: Any,
    passed: bool | None,
    explanation: str,
    critical: bool = True,
) -> None:
    status = "NOT VERIFIED" if passed is None else "PASS" if passed else "FAIL"
    checks.append(
        {
            "check_name": name,
            "expected_result": expected,
            "independently_calculated_result": actual,
            "status": status,
            "critical": critical,
            "explanation": explanation,
        }
    )
    if passed is False:
        discrepancies.append(
            {
                "check_name": name,
                "expected": expected,
                "actual": actual,
                "explanation": explanation,
            }
        )


def metric_value(summary: pd.DataFrame, version: str, metric: str) -> Any:
    values = summary.loc[
        summary["dataset_version"].eq(version) & summary["metric"].eq(metric),
        "value",
    ]
    return values.iloc[0] if len(values) else None


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    required_outputs = [
        "experiment_configuration.json",
        "dataset_comparison_summary.csv",
        "row_level_spo2_differences.csv",
        "missingness_comparison.csv",
        "feature_impact_summary.csv",
        "model_performance_comparison.csv",
        "alert_impact_summary.csv",
        "alert_level_differences.csv",
    ]
    missing_inputs = [
        str(path)
        for path in [*DATASETS.values(), RULES_PATH]
        if not path.is_file()
    ]
    missing_outputs = [
        name for name in required_outputs if not (OUTPUT_DIR / name).is_file()
    ]
    if missing_inputs or missing_outputs:
        raise FileNotFoundError(
            f"Missing inputs={missing_inputs}; missing comparison outputs={missing_outputs}"
        )

    config = json.loads(
        (OUTPUT_DIR / "experiment_configuration.json").read_text(
            encoding="utf-8"
        )
    )
    summary = pd.read_csv(OUTPUT_DIR / "dataset_comparison_summary.csv")
    row_output = pd.read_csv(
        OUTPUT_DIR / "row_level_spo2_differences.csv", low_memory=False
    )
    missingness_output = pd.read_csv(
        OUTPUT_DIR / "missingness_comparison.csv"
    )
    feature_output = pd.read_csv(OUTPUT_DIR / "feature_impact_summary.csv")
    model_output = pd.read_csv(
        OUTPUT_DIR / "model_performance_comparison.csv"
    )
    alert_output = pd.read_csv(OUTPUT_DIR / "alert_impact_summary.csv")

    frames = {
        name: pd.read_csv(path, low_memory=False)
        for name, path in DATASETS.items()
    }
    saved_rules = pd.read_csv(RULES_PATH, low_memory=False)
    checks: list[dict[str, Any]] = []
    discrepancies: list[dict[str, Any]] = []
    add_check(
        checks,
        discrepancies,
        "Both source dataset versions exist",
        "All four original/corrected raw and cleaned files exist",
        [str(path) for path in DATASETS.values() if path.is_file()],
        all(path.is_file() for path in DATASETS.values()),
        "Checked the actual files used by the prior preprocessing and cleaning workflows.",
    )

    config_input_hashes = {
        name: config.get("input_datasets", {}).get(name, {}).get("sha256")
        for name in DATASETS
    }
    actual_input_hashes = {
        name: sha256(path) for name, path in DATASETS.items()
    }
    input_hashes_match = config_input_hashes == actual_input_hashes
    add_check(
        checks,
        discrepancies,
        "Input files unchanged since comparison",
        config_input_hashes,
        actual_input_hashes,
        input_hashes_match,
        "Recomputed SHA-256 hashes independently for all four input datasets.",
    )

    key_checks = {}
    for name, frame in frames.items():
        valid_key_rows = frame[KEYS].notna().all(axis=1)
        key_checks[name] = {
            "null_keys": int(frame[KEYS].isna().any(axis=1).sum()),
            "duplicate_keys": int(frame.loc[valid_key_rows].duplicated(KEYS).sum()),
        }
    keys_valid = all(values["duplicate_keys"] == 0 for values in key_checks.values())
    keys_valid &= all(
        key_checks[name]["null_keys"] == 0
        for name in ("original_clean", "corrected_clean")
    )
    add_check(
        checks,
        discrepancies,
        "Composite keys are valid and unique",
        "Cleaned join keys complete and unique; raw null-key rows excluded and reported",
        key_checks,
        keys_valid,
        "Independently checked key quality before reconstructing joins.",
    )

    original_clean = frames["original_clean"]
    corrected_clean = frames["corrected_clean"]
    original_raw = frames["original_raw"]
    corrected_raw = frames["corrected_raw"]
    clean_join = original_clean.loc[
        original_clean[KEYS].notna().all(axis=1)
    ].merge(
        corrected_clean.loc[corrected_clean[KEYS].notna().all(axis=1)],
        on=KEYS,
        how="outer",
        suffixes=("_original", "_corrected"),
        indicator=True,
        validate="one_to_one",
    )
    raw_join = original_raw.loc[
        original_raw[KEYS].notna().all(axis=1)
    ].merge(
        corrected_raw.loc[corrected_raw[KEYS].notna().all(axis=1)],
        on=KEYS,
        how="outer",
        suffixes=("_original", "_corrected"),
        indicator=True,
        validate="one_to_one",
    )
    clean_both = clean_join.loc[clean_join["_merge"].eq("both")].copy()
    raw_both = raw_join.loc[raw_join["_merge"].eq("both")].copy()
    expected_row_output_count = len(clean_join)
    row_keys_match = (
        len(row_output) == expected_row_output_count
        and not row_output.duplicated(KEYS).any()
    )
    add_check(
        checks,
        discrepancies,
        "Matched and unmatched row counts reconcile",
        {
            "rows_in_outer_join": len(clean_join),
            "matched": len(clean_both),
            "unmatched_original": int(clean_join["_merge"].eq("left_only").sum()),
            "unmatched_corrected": int(clean_join["_merge"].eq("right_only").sum()),
        },
        {
            "row_level_output_rows": len(row_output),
            "matched": int(row_output["comparison_status"].eq("both").sum()),
            "unmatched_original": int(row_output["comparison_status"].eq("left_only").sum()),
            "unmatched_corrected": int(row_output["comparison_status"].eq("right_only").sum()),
        },
        row_keys_match
        and len(clean_both) == int(row_output["comparison_status"].eq("both").sum())
        and int(clean_join["_merge"].eq("left_only").sum())
        == int(row_output["comparison_status"].eq("left_only").sum())
        and int(clean_join["_merge"].eq("right_only").sum())
        == int(row_output["comparison_status"].eq("right_only").sum()),
        "Rebuilt the outer patient-hour join from source CSVs and compared its row counts with the deliverable.",
    )

    original_spo2 = pd.to_numeric(clean_both["spo2_original"], errors="coerce")
    corrected_spo2 = pd.to_numeric(clean_both["spo2_corrected"], errors="coerce")
    changed_mask = value_diff(original_spo2, corrected_spo2)
    newly_missing_mask = original_spo2.notna() & corrected_spo2.isna()
    newly_observed_mask = original_spo2.isna() & corrected_spo2.notna()
    row_changed = row_output.loc[row_output["comparison_status"].eq("both")]
    changed_reconciles = (
        int(changed_mask.sum()) == int(row_changed["spo2_changed"].sum())
        and int(newly_missing_mask.sum()) == int(row_changed["newly_missing"].sum())
        and int(newly_observed_mask.sum()) == int(row_changed["newly_observed"].sum())
    )
    add_check(
        checks,
        discrepancies,
        "Changed SpO2 and missingness are independently identified",
        {
            "changed": int(changed_mask.sum()),
            "newly_missing": int(newly_missing_mask.sum()),
            "newly_observed": int(newly_observed_mask.sum()),
        },
        {
            "changed": int(row_changed["spo2_changed"].sum()),
            "newly_missing": int(row_changed["newly_missing"].sum()),
            "newly_observed": int(row_changed["newly_observed"].sum()),
        },
        changed_reconciles,
        "Values and missingness were recalculated from source data, not inferred from the comparison output.",
    )

    raw_original_spo2 = pd.to_numeric(raw_both["spo2_original"], errors="coerce")
    raw_corrected_spo2 = pd.to_numeric(raw_both["spo2_corrected"], errors="coerce")
    raw_changed_count = int(value_diff(raw_original_spo2, raw_corrected_spo2).sum())
    raw_newly_missing_count = int(
        (raw_original_spo2.notna() & raw_corrected_spo2.isna()).sum()
    )
    raw_summary_matches = (
        metric_value(summary, "paired_raw", "changed_spo2_rows") is not None
        and int(float(metric_value(summary, "paired_raw", "changed_spo2_rows")))
        == raw_changed_count
        and int(
            float(metric_value(summary, "paired_raw", "newly_missing_spo2_rows"))
        )
        == raw_newly_missing_count
    )
    add_check(
        checks,
        discrepancies,
        "Raw SpO2 changes and zero-only missingness reconcile",
        {
            "changed_spo2_rows": raw_changed_count,
            "newly_missing_after_zero_exclusion": raw_newly_missing_count,
        },
        {
            "changed_spo2_rows": metric_value(
                summary, "paired_raw", "changed_spo2_rows"
            ),
            "newly_missing_after_zero_exclusion": metric_value(
                summary, "paired_raw", "newly_missing_spo2_rows"
            ),
        },
        raw_summary_matches,
        "Independently compared the original and corrected raw hourly tables after excluding null composite keys.",
    )

    input_missing_rows = []
    for version, frame in frames.items():
        for column in frame.columns:
            input_missing_rows.append(
                {
                    "dataset_version": version,
                    "column": column,
                    "missing_count": int(frame[column].isna().sum()),
                }
            )
    missing_rebuilt = pd.DataFrame(input_missing_rows)
    missing_join = missing_rebuilt.merge(
        missingness_output[
            ["dataset_version", "column", "missing_count"]
        ],
        on=["dataset_version", "column"],
        how="outer",
        suffixes=("_independent", "_reported"),
        indicator=True,
        validate="one_to_one",
    )
    missing_ok = (
        missing_join["_merge"].eq("both").all()
        and missing_join["missing_count_independent"].eq(
            missing_join["missing_count_reported"]
        ).all()
    )
    add_check(
        checks,
        discrepancies,
        "Per-column missingness reconciles",
        "All source missing counts",
        {
            "rows_checked": len(missing_join),
            "mismatches": int(
                (
                    ~missing_join["missing_count_independent"].eq(
                        missing_join["missing_count_reported"]
                    )
                ).sum()
            ),
        },
        missing_ok,
        "Recomputed missing counts for every column in all four source datasets.",
    )

    spo2_counts = {}
    for version, frame in frames.items():
        values = pd.to_numeric(frame["spo2"], errors="coerce")
        spo2_counts[version] = {
            "rows": len(frame),
            "unique_icu_stays": int(frame["patientunitstayid"].nunique()),
            "spo2_missing_count": int(values.isna().sum()),
            "spo2_zero_count": int(values.eq(0).sum()),
            "spo2_nonzero_count": int((values.notna() & values.ne(0)).sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_patient_hour_keys": int(frame.duplicated(KEYS).sum()),
            "invalid_patient_hour_key_rows": int(frame[KEYS].isna().any(axis=1).sum()),
            "spo2_min": float(values.min()) if values.notna().any() else np.nan,
            "spo2_max": float(values.max()) if values.notna().any() else np.nan,
            "spo2_mean": float(values.mean()),
            "spo2_median": float(values.median()),
            "spo2_std": float(values.std()),
        }
        for threshold in [20, 30, 40, 50, 70, 80]:
            spo2_counts[version][f"spo2_le_{threshold}_count"] = int(
                values.le(threshold).sum()
            )
        spo2_counts[version]["spo2_gt_100_count"] = int(values.gt(100).sum())
        if version.endswith("_clean"):
            spo2_counts[version]["rows_removed_during_cleaning"] = len(
                frames[version.replace("_clean", "_raw")]
            ) - len(frame)
    summary_ok = True
    for version, values in spo2_counts.items():
        for metric, actual in values.items():
            reported = metric_value(summary, version, metric)
            if (
                reported is None
                or pd.isna(reported)
                or not np.isclose(float(reported), float(actual), rtol=1e-9, atol=1e-9)
            ):
                summary_ok = False
    add_check(
        checks,
        discrepancies,
        "Dataset totals reconcile with raw inputs",
        "Rows, stays, SpO2 missing/zero/nonzero, duplicates, and cleaning removals",
        spo2_counts,
        summary_ok,
        "Independently recalculated the principal dataset-wide totals.",
    )

    clean_rows = row_output.loc[row_output["comparison_status"].eq("both")]
    column_change_counts: dict[str, int] = {}
    for column in NON_SPO2_MEASUREMENTS:
        if column in clean_both.columns:
            diff = value_diff(
                clean_both[f"{column}_original"],
                clean_both[f"{column}_corrected"],
            )
            column_change_counts[column] = int(diff.sum())
    non_spo2_unchanged = all(count == 0 for count in column_change_counts.values())
    row_output_non_spo2_changed = int(
        clean_rows["any_non_spo2_measurement_changed"].sum()
    )
    add_check(
        checks,
        discrepancies,
        "Non-SpO2 measurements remain unchanged on matched rows",
        "Zero changed non-SpO2 measurement values",
        {
            "independent_per_column_changes": column_change_counts,
            "row_output_changed_rows": row_output_non_spo2_changed,
        },
        non_spo2_unchanged and row_output_non_spo2_changed == 0,
        "Only measurement columns are assessed here; row removals and SpO2 quality metadata are handled separately.",
    )

    metadata_source = corrected_clean.set_index(KEYS)
    row_metadata = row_output.loc[row_output["comparison_status"].eq("both")]
    metadata_keys = pd.MultiIndex.from_frame(row_metadata[KEYS])
    metadata_mismatches = {}
    for column in [
        "spo2_valid_count",
        "spo2_zero_count",
        "spo2_missing_count",
        "spo2_quality_flag",
    ]:
        original_column = f"original_{column}"
        corrected_column = f"corrected_{column}"
        expected = metadata_source[column].reindex(metadata_keys).reset_index(drop=True)
        actual = row_metadata[corrected_column].reset_index(drop=True)
        metadata_mismatches[column] = int(value_diff(expected, actual).sum())
        metadata_mismatches[f"{column}_original_unavailable"] = int(
            row_metadata[original_column].notna().sum()
        )
        reported_feature = feature_output.loc[
            feature_output["feature"].eq(column)
        ]
        if len(reported_feature) != 1:
            metadata_mismatches[f"{column}_summary_row_missing"] = 1
        else:
            metadata_mismatches[f"{column}_summary_row_missing"] = 0
            metadata_mismatches[f"{column}_reported_as_metadata"] = int(
                reported_feature["feature_type"].iloc[0] != "quality_metadata"
            )
    metadata_ok = all(value == 0 for value in metadata_mismatches.values())
    add_check(
        checks,
        discrepancies,
        "SpO2 quality metadata is row-aligned and documented",
        "Corrected counts/flags match the corrected dataset; legacy fields are unavailable rather than fabricated",
        metadata_mismatches,
        metadata_ok,
        "Valid count, zero count, missing count, and quality flag are independently aligned by patient-hour key.",
    )

    independently_rebuilt_features = {
        "original": independent_features(original_clean).set_index(KEYS),
        "corrected": independent_features(corrected_clean).set_index(KEYS),
    }
    left = independently_rebuilt_features["original"]
    right = independently_rebuilt_features["corrected"]
    feature_keys = left.index.intersection(right.index)
    feature_reconciliation = []
    features_ok = True
    for feature in [column for column in left.columns if column not in KEYS]:
        a = left.loc[feature_keys, feature]
        b = right.loc[feature_keys, feature]
        changed = value_diff(a.reset_index(drop=True), b.reset_index(drop=True))
        newly_missing = int((a.notna() & b.isna()).sum())
        reported_rows = feature_output.loc[
            feature_output["feature"].eq(feature)
        ]
        reported_changed = (
            int(reported_rows["changed_values"].iloc[0])
            if len(reported_rows)
            else None
        )
        reported_new_missing = (
            int(reported_rows["newly_missing"].iloc[0])
            if len(reported_rows)
            else None
        )
        feature_reconciliation.append(
            {
                "feature": feature,
                "independent_changed": int(changed.sum()),
                "reported_changed": reported_changed,
                "independent_newly_missing": newly_missing,
                "reported_newly_missing": reported_new_missing,
            }
        )
        features_ok &= (
            reported_changed == int(changed.sum())
            and reported_new_missing == newly_missing
        )
    add_check(
        checks,
        discrepancies,
        "SpO2 feature comparisons use same keys and reconcile",
        f"{len(feature_keys)} matched patient-hour keys; exact-hour deltas and gap-aware rolling windows",
        {
            "matched_keys": len(feature_keys),
            "feature_comparison": feature_reconciliation,
        },
        features_ok,
        "Features were rebuilt independently using sorted within-stay timelines; rolling segments restart after gaps.",
    )

    original_rules = independent_rule_features(original_clean, saved_rules)
    corrected_rules = independent_rule_features(corrected_clean, saved_rules)
    baseline_columns = list(original_rules.columns)
    saved_rule_columns_missing = [
        column for column in baseline_columns if column not in saved_rules
    ]
    saved_rules_unique = not saved_rules.duplicated(KEYS).any()
    baseline = saved_rules.set_index(KEYS)
    baseline_differences = {}
    if saved_rules_unique and not saved_rule_columns_missing:
        aligned_baseline = baseline.reindex(original_rules.index)
        baseline_differences = {
            column: int(
                value_diff(
                    original_rules[column].reset_index(drop=True),
                    aligned_baseline[column].reset_index(drop=True),
                ).sum()
            )
            for column in baseline_columns
        }

    clinical_feature_reconciliation = []
    clinical_features_ok = (
        saved_rules_unique
        and not saved_rule_columns_missing
        and all(value == 0 for value in baseline_differences.values())
    )
    clinical_keys = original_rules.index.intersection(corrected_rules.index)
    clinical_summary_columns = [
        "matched_patient_hours",
        "original_nonmissing",
        "corrected_nonmissing",
        "original_mean",
        "corrected_mean",
        "original_min",
        "corrected_min",
        "original_max",
        "corrected_max",
        "changed_values",
        "changed_nonmissing_values",
        "newly_missing",
        "newly_observed",
    ]

    def metric_matches(reported: Any, actual: Any) -> bool:
        if pd.isna(reported) or pd.isna(actual):
            return bool(pd.isna(reported) and pd.isna(actual))
        try:
            return bool(np.isclose(float(reported), float(actual), rtol=1e-9, atol=1e-9))
        except (TypeError, ValueError):
            return reported == actual

    for feature in baseline_columns:
        original_values = original_rules.loc[
            clinical_keys, feature
        ].reset_index(drop=True)
        corrected_values = corrected_rules.loc[
            clinical_keys, feature
        ].reset_index(drop=True)
        changed = value_diff(
            original_values,
            corrected_values,
        )
        original_numeric = pd.to_numeric(original_values, errors="coerce")
        corrected_numeric = pd.to_numeric(corrected_values, errors="coerce")
        absolute_difference = (original_numeric - corrected_numeric).abs()
        independent_metrics = {
            "matched_patient_hours": len(clinical_keys),
            "original_nonmissing": int(original_values.notna().sum()),
            "corrected_nonmissing": int(corrected_values.notna().sum()),
            "original_mean": (
                float(original_numeric.mean())
                if original_numeric.notna().any()
                else np.nan
            ),
            "corrected_mean": (
                float(corrected_numeric.mean())
                if corrected_numeric.notna().any()
                else np.nan
            ),
            "original_min": (
                float(original_numeric.min())
                if original_numeric.notna().any()
                else np.nan
            ),
            "corrected_min": (
                float(corrected_numeric.min())
                if corrected_numeric.notna().any()
                else np.nan
            ),
            "original_max": (
                float(original_numeric.max())
                if original_numeric.notna().any()
                else np.nan
            ),
            "corrected_max": (
                float(corrected_numeric.max())
                if corrected_numeric.notna().any()
                else np.nan
            ),
            "changed_values": int(changed.sum()),
            "changed_nonmissing_values": int(
                (changed & original_values.notna() & corrected_values.notna()).sum()
            ),
            "newly_missing": int(
                (original_values.notna() & corrected_values.isna()).sum()
            ),
            "newly_observed": int(
                (original_values.isna() & corrected_values.notna()).sum()
            ),
        }
        reported_rows = feature_output.loc[feature_output["feature"].eq(feature)]
        if len(reported_rows) != 1:
            mismatches = ["summary_row_count"]
        else:
            reported = reported_rows.iloc[0]
            mismatches = [
                column
                for column in clinical_summary_columns
                if not metric_matches(
                    reported[column], independent_metrics[column]
                )
            ]
        clinical_features_ok &= not mismatches
        clinical_feature_reconciliation.append(
            {
                "feature": feature,
                "independently_changed": independent_metrics["changed_values"],
                "reported_changed": (
                    int(reported_rows["changed_values"].iloc[0])
                    if len(reported_rows) == 1
                    else None
                ),
                "summary_mismatches": mismatches,
            }
        )
    add_check(
        checks,
        discrepancies,
        "Clinical-rule impact independently reconciles",
        {
            "original rule outputs match saved baseline": True,
            "all eight clinical rule output summaries reconcile": True,
        },
        {
            "saved_rule_keys_unique": saved_rules_unique,
            "missing_saved_rule_columns": saved_rule_columns_missing,
            "baseline_differences": baseline_differences,
            "feature_reconciliation": clinical_feature_reconciliation,
        },
        clinical_features_ok,
        "Rebuilt SpO2-dependent clinical-rule counts, severities, and explanations independently; checked the original baseline and each reported impact statistic.",
    )

    expected_model_paths = config.get("model_artifact_paths", [])
    missing_model_paths = [
        path for path in expected_model_paths if not Path(path).is_file()
    ]
    models_not_evaluated = (
        bool(missing_model_paths)
        and model_output["status"].eq("NOT EVALUATED").all()
        and model_output["metrics"].isna().all()
    )
    add_check(
        checks,
        discrepancies,
        "Frozen model comparison status is explicit",
        "PASS only with the same frozen model/cohort; otherwise NOT VERIFIED",
        {
            "missing_frozen_models": missing_model_paths,
            "comparison_rows": model_output["status"].tolist(),
        },
        None if models_not_evaluated else False,
        "No model was retrained. Because the required saved estimators are absent, paired model metrics cannot be verified.",
        critical=False,
    )
    alerts_not_evaluated = (
        config.get("alert_evaluation_status") == "NOT EVALUATED"
        and alert_output["status"].eq("NOT EVALUATED").all()
    )
    add_check(
        checks,
        discrepancies,
        "Alert and notification comparison status is explicit",
        "PASS only if identical rules/settings and model scores were replayed; otherwise NOT VERIFIED",
        {
            "alert_status": config.get("alert_evaluation_status"),
            "summary_statuses": sorted(alert_output["status"].dropna().unique().tolist()),
        },
        None if alerts_not_evaluated else False,
        "Full alert replay is not claimed because corrected data cannot be rescored with missing frozen model artifacts.",
        critical=False,
    )

    output_paths = [
        OUTPUT_DIR / name for name in config.get("outputs", [])
    ]
    paths_are_isolated = all(
        path.resolve().parent == OUTPUT_DIR.resolve() for path in output_paths
    )
    before_hashes = config.get("production_files_sha256_before", {})
    after_hashes = config.get("production_files_sha256_after", {})
    protected_unchanged = before_hashes == after_hashes and all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in before_hashes.items()
    )
    add_check(
        checks,
        discrepancies,
        "No production artifact was overwritten",
        "Outputs confined to spo2_comparison; protected production hashes unchanged",
        {
            "outputs_confined": paths_are_isolated,
            "protected_files_unchanged": protected_unchanged,
        },
        paths_are_isolated and protected_unchanged,
        "Checked configured output paths and independently hashed all captured production files.",
    )

    clean_rows = row_output.loc[row_output["comparison_status"].eq("both")]
    summary_rows = {
        ("paired_clean", "changed_spo2_rows"): int(changed_mask.sum()),
        ("paired_clean", "newly_missing_spo2_rows"): int(newly_missing_mask.sum()),
        ("paired_clean", "newly_observed_spo2_rows"): int(newly_observed_mask.sum()),
        ("paired_clean", "patient_hours_with_any_non_spo2_measurement_change"): row_output_non_spo2_changed,
        ("paired_raw", "changed_spo2_rows"): raw_changed_count,
        ("paired_raw", "newly_missing_spo2_rows"): raw_newly_missing_count,
    }
    row_reconciles = True
    for (version, metric), actual in summary_rows.items():
        reported = metric_value(summary, version, metric)
        row_reconciles &= (
            reported is not None
            and not pd.isna(reported)
            and int(float(reported)) == actual
        )
    add_check(
        checks,
        discrepancies,
        "Reported impact totals reconcile with row-level output",
        summary_rows,
        {
            key[1]: metric_value(summary, key[0], key[1])
            for key in summary_rows
        },
        row_reconciles,
        "Totals were compared to both independent calculations and the full row-level CSV.",
    )

    failures = [row for row in checks if row["status"] == "FAIL"]
    not_verified = [row for row in checks if row["status"] == "NOT VERIFIED"]
    critical_failures = [
        row for row in failures if row["critical"]
    ]
    if critical_failures:
        verdict = "FAIL — CORRECTIONS REQUIRED"
    elif failures or not_verified:
        verdict = "PASS WITH LIMITATIONS"
    else:
        verdict = "PASS — COMPARISON VALIDATED"

    checks_frame = pd.DataFrame(checks)
    discrepancies_frame = pd.DataFrame(
        discrepancies,
        columns=["check_name", "expected", "actual", "explanation"],
    )
    checks_frame.to_csv(VALIDATION_DIR / "validation_checks.csv", index=False)
    discrepancies_frame.to_csv(
        VALIDATION_DIR / "validation_discrepancies.csv", index=False
    )
    report = f"""# Independent validation — SpO2 dataset impact

## Verdict

**{verdict}**

Checks: {int(checks_frame.status.eq('PASS').sum())} PASS,
{int(checks_frame.status.eq('FAIL').sum())} FAIL,
{int(checks_frame.status.eq('NOT VERIFIED').sum())} NOT VERIFIED.
Critical failures: {len(critical_failures)}.

## Independently verified counts

- Original raw / clean rows: {len(original_raw):,} / {len(original_clean):,}.
- Corrected raw / clean rows: {len(corrected_raw):,} / {len(corrected_clean):,}.
- Matched cleaned patient-hours: {len(clean_both):,}; unmatched original:
  {int(clean_join['_merge'].eq('left_only').sum()):,}; unmatched corrected:
  {int(clean_join['_merge'].eq('right_only').sum()):,}.
- SpO2 changes on matched cleaned rows: {int(changed_mask.sum()):,}.
- Newly missing / newly observed: {int(newly_missing_mask.sum()):,} /
  {int(newly_observed_mask.sum()):,}.
- Raw SpO2 values changed on valid keyed rows: {raw_changed_count:,};
  newly missing after zero exclusion: {raw_newly_missing_count:,}.
- Raw rows with invalid patient-hour keys: original
  {key_checks['original_raw']['null_keys']:,}, corrected
  {key_checks['corrected_raw']['null_keys']:,}; these are not joined across datasets.
- Non-SpO2 measurement changes on matched cleaned rows:
  {sum(column_change_counts.values()):,}.

## Statuses

{markdown_table(checks_frame[['check_name', 'status', 'critical', 'explanation']])}

## Discrepancies

{len(discrepancies_frame)} independent validation discrepancy records.
{"No discrepancies were found." if discrepancies_frame.empty else markdown_table(discrepancies_frame)}

## Limitations

Frozen model artifacts are absent, so model metrics are NOT VERIFIED and no
retraining was attempted. Full alert and notification replay is likewise NOT
VERIFIED. The comparison is retrospective and does not establish clinical
validity or safety.
"""
    (VALIDATION_DIR / "validation_report.md").write_text(
        report, encoding="utf-8"
    )
    print(verdict)
    print(
        f"Checks: PASS={int(checks_frame.status.eq('PASS').sum())}, "
        f"FAIL={int(checks_frame.status.eq('FAIL').sum())}, "
        f"NOT VERIFIED={int(checks_frame.status.eq('NOT VERIFIED').sum())}"
    )
    print(f"Validation outputs: {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
