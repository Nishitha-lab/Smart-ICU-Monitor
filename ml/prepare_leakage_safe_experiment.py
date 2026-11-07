from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = PROCESSED / "experiments" / "leakage_safe"
KEYS = ["patientunitstayid", "hour"]
SEED = 20261107
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
PATHS = {
    "patient": ROOT
    / "data"
    / "real"
    / "eicu_demo"
    / "eicu-collaborative-research-database-demo-2.0.1"
    / "patient.csv.gz",
    "state": PROCESSED / "hourly_patient_state_alerts.csv",
    "rules": PROCESSED / "hourly_patient_state_rules_v2.csv",
    "labels": PROCESSED / "deterioration_labels.csv",
    "baseline_spo2": PROCESSED / "hourly_vitals_labs_clean.csv",
    "corrected_spo2": PROCESSED
    / "hourly_vitals_labs_spo2_experiment_clean.csv",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_keyed(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    if frame[KEYS].isna().any(axis=None):
        frame = frame.dropna(subset=KEYS)
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path.name} has duplicate patient-hour keys.")
    return frame


def gap_aware_spo2_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values(KEYS).copy()
    groups = data.groupby("patientunitstayid", sort=False)
    previous_value = groups["spo2"].shift(1)
    previous_hour = groups["hour"].shift(1)
    data["spo2_change_1h"] = (data["spo2"] - previous_value).where(
        (data["hour"] - previous_hour).eq(1)
    )
    gap = groups["hour"].diff()
    segment_break = gap.isna() | gap.gt(1)
    segment_number = segment_break.groupby(
        data["patientunitstayid"], sort=False
    ).cumsum()
    segment_id = (
        data["patientunitstayid"].astype(str) + "_" + segment_number.astype(str)
    )
    segmented = data["spo2"].groupby(segment_id, sort=False)
    data["spo2_rolling_mean_3h"] = segmented.transform(
        lambda values: values.rolling(window=3, min_periods=2).mean()
    )
    return data


def create_version(
    state: pd.DataFrame,
    rules: pd.DataFrame,
    labels: pd.DataFrame,
    spo2: pd.DataFrame,
    version: str,
) -> pd.DataFrame:
    features = state[
        KEYS + [column for column in FEATURES if column in state]
    ].copy()
    features = features.drop(columns=["spo2_change_1h", "spo2_rolling_mean_3h"])
    features = features.merge(
        spo2[KEYS + ["spo2"]].rename(columns={"spo2": "_version_spo2"}),
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    features["spo2"] = pd.to_numeric(features.pop("_version_spo2"), errors="coerce")
    features = gap_aware_spo2_features(features)
    rule_columns = KEYS + WARNINGS + SEVERE
    missing_rules = [column for column in rule_columns if column not in rules]
    if missing_rules:
        raise ValueError(f"Rule table missing required columns: {missing_rules}")
    rule_values = features[KEYS].merge(
        rules[rule_columns],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    for column in WARNINGS + SEVERE:
        rule_values[column] = pd.to_numeric(
            rule_values[column], errors="coerce"
        ).fillna(0)
    rule_values["rule_hypoxemia"] = features["spo2"].lt(92).astype(int).to_numpy()
    rule_values["rule_falling_spo2"] = (
        features["spo2_change_1h"].lt(-3).astype(int).to_numpy()
    )
    rule_values["rule_severe_hypoxemia"] = (
        features["spo2"].lt(88).astype(int).to_numpy()
    )
    rule_values["rule_rapid_falling_spo2"] = (
        features["spo2_change_1h"].lt(-5).astype(int).to_numpy()
    )
    features["clinical_warning_count"] = rule_values[WARNINGS].sum(axis=1)
    features["clinical_severe_count"] = rule_values[SEVERE].sum(axis=1)
    features = features.merge(
        labels[KEYS + ["deterioration_label_24h"]],
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    features.insert(
        0,
        "dataset_version",
        version,
    )
    if features["deterioration_label_24h"].isna().any():
        raise ValueError(f"{version} contains missing target values.")
    return features.sort_values(KEYS).reset_index(drop=True)


def split_patients(
    patient_summary: pd.DataFrame,
) -> pd.DataFrame:
    patients = patient_summary["uniquepid"].to_numpy()
    labels = patient_summary["patient_positive"].to_numpy()
    train_ids, holdout_ids = train_test_split(
        patients,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )
    holdout_labels = (
        patient_summary.set_index("uniquepid")
        .loc[holdout_ids, "patient_positive"]
        .to_numpy()
    )
    validation_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=0.50,
        random_state=SEED + 1,
        stratify=holdout_labels,
    )
    assignment = pd.DataFrame(
        {
            "uniquepid": patients,
            "patient_positive": labels,
            "split": "",
        }
    )
    assignment.loc[assignment["uniquepid"].isin(train_ids), "split"] = "train"
    assignment.loc[
        assignment["uniquepid"].isin(validation_ids), "split"
    ] = "validation"
    assignment.loc[assignment["uniquepid"].isin(test_ids), "split"] = "test"
    if assignment["split"].eq("").any():
        raise ValueError("Some patients were not assigned to a partition.")
    return assignment.sort_values("uniquepid").reset_index(drop=True)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required experiment inputs missing: {missing}")

    patients = pd.read_csv(
        PATHS["patient"], usecols=["patientunitstayid", "uniquepid"], low_memory=False
    ).drop_duplicates("patientunitstayid")
    if patients["uniquepid"].isna().any():
        raise ValueError("Patient table has missing uniquepid for ICU stays.")

    state_columns = list(dict.fromkeys(KEYS + FEATURES))
    state = load_keyed(PATHS["state"], state_columns)
    rules = load_keyed(PATHS["rules"], list(dict.fromkeys(KEYS + WARNINGS + SEVERE)))
    labels = load_keyed(
        PATHS["labels"], KEYS + ["deterioration_label_24h"]
    )
    baseline_spo2 = load_keyed(
        PATHS["baseline_spo2"], KEYS + ["spo2"]
    )
    corrected_spo2 = load_keyed(
        PATHS["corrected_spo2"], KEYS + ["spo2"]
    )

    baseline = create_version(
        state, rules, labels, baseline_spo2, "baseline"
    )
    corrected = create_version(
        state, rules, labels, corrected_spo2, "corrected"
    )
    baseline_keys = pd.MultiIndex.from_frame(baseline[KEYS])
    corrected_keys = pd.MultiIndex.from_frame(corrected[KEYS])
    if not baseline_keys.equals(corrected_keys):
        raise ValueError("Baseline and corrected patient-hour cohorts differ.")
    if not baseline["deterioration_label_24h"].equals(
        corrected["deterioration_label_24h"]
    ):
        raise ValueError("Baseline and corrected targets are not identical.")

    combined = baseline[KEYS + ["deterioration_label_24h"]].drop_duplicates()
    combined = combined.merge(
        patients, on="patientunitstayid", how="inner", validate="many_to_one"
    )
    patient_summary = (
        combined.groupby("uniquepid", as_index=False)
        .agg(
            stay_count=("patientunitstayid", "nunique"),
            patient_hours=("patientunitstayid", "size"),
            patient_positive=("deterioration_label_24h", "max"),
        )
    )
    manifest = split_patients(patient_summary)
    assignment = manifest.set_index("uniquepid")["split"]
    stay_split = combined[KEYS + ["uniquepid"]].drop_duplicates(KEYS)
    stay_split["split"] = stay_split["uniquepid"].map(assignment)
    if stay_split["split"].isna().any():
        raise ValueError("Some ICU stays have no patient-level split.")
    row_split = stay_split[KEYS + ["uniquepid", "split"]]

    baseline = baseline.merge(
        row_split, on=KEYS, how="left", validate="one_to_one"
    )
    corrected = corrected.merge(
        row_split, on=KEYS, how="left", validate="one_to_one"
    )
    for name, frame in [("baseline", baseline), ("corrected", corrected)]:
        if frame["split"].isna().any():
            missing_keys = frame.loc[frame["split"].isna(), KEYS].head(5)
            raise ValueError(
                f"{name} has {int(frame['split'].isna().sum())} patient-hours "
                f"outside the common cohort; sample={missing_keys.to_dict('records')}"
            )
        frame[["patientunitstayid", "hour", "uniquepid", "split", *FEATURES,
               "deterioration_label_24h"]].to_csv(
            OUTPUT / f"model_inputs_{name}.csv", index=False
        )

    manifest = manifest.merge(
        patient_summary[["uniquepid", "stay_count", "patient_hours"]],
        on="uniquepid",
        how="left",
        validate="one_to_one",
    )
    manifest.to_csv(OUTPUT / "patient_split_manifest.csv", index=False)
    cohort_summary = pd.DataFrame(
        [
            {
                "dataset_version": version,
                "patients": int(frame["uniquepid"].nunique()),
                "icu_stays": int(frame["patientunitstayid"].nunique()),
                "patient_hours": int(len(frame)),
                "positive_hours": int(frame["deterioration_label_24h"].sum()),
                "negative_hours": int(frame["deterioration_label_24h"].eq(0).sum()),
                "positive_rate": float(frame["deterioration_label_24h"].mean()),
                "spo2_missing_hours": int(frame["spo2"].isna().sum()),
                "spo2_zero_hours": int(frame["spo2"].eq(0).sum()),
            }
            for version, frame in [("baseline", baseline), ("corrected", corrected)]
        ]
    )
    cohort_summary.to_csv(OUTPUT / "cohort_summary.csv", index=False)
    distribution_rows = []
    for version, frame in [("baseline", baseline), ("corrected", corrected)]:
        for split, subset in frame.groupby("split", sort=False):
            distribution_rows.append(
                {
                    "dataset_version": version,
                    "split": split,
                    "patients": int(subset["uniquepid"].nunique()),
                    "icu_stays": int(subset["patientunitstayid"].nunique()),
                    "patient_hours": int(len(subset)),
                    "positive_hours": int(subset["deterioration_label_24h"].sum()),
                    "negative_hours": int(subset["deterioration_label_24h"].eq(0).sum()),
                    "positive_rate": float(subset["deterioration_label_24h"].mean()),
                    "positive_patients": int(
                        subset.loc[
                            subset["deterioration_label_24h"].eq(1), "uniquepid"
                        ].nunique()
                    ),
                }
            )
    pd.DataFrame(distribution_rows).to_csv(
        OUTPUT / "target_distribution.csv", index=False
    )

    patient_sets = {
        split: set(manifest.loc[manifest["split"].eq(split), "uniquepid"])
        for split in ("train", "validation", "test")
    }
    overlap = {
        f"{left}_{right}": len(patient_sets[left] & patient_sets[right])
        for left, right in [
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ]
    }
    if any(overlap.values()):
        raise ValueError(f"Patient leakage in generated split: {overlap}")
    if any(
        not (pd.DataFrame(distribution_rows).query("split == @split")[
            "positive_hours"
        ] > 0).all()
        for split in ("train", "validation", "test")
    ):
        raise ValueError("At least one partition has no positive labels.")

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "split": {
            "group_identifier": "uniquepid",
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "stratification": "patient-level ever-positive target indicator",
        },
        "target": {
            "name": "deterioration_label_24h",
            "definition": (
                "unitdischargestatus == expired and "
                "0 <= unitdischargeoffset / 60 - hour <= 24"
            ),
            "warning": "This target is not generic clinical deterioration.",
        },
        "features": FEATURES,
        "cohort": {
            "matched_patient_hours": len(baseline),
            "patients": int(manifest["uniquepid"].nunique()),
            "icu_stays": int(baseline["patientunitstayid"].nunique()),
            "split_patient_overlap": overlap,
            "baseline_corrected_keys_match": True,
            "targets_match": True,
        },
        "input_sha256": {
            name: file_hash(path) for name, path in PATHS.items()
        },
        "output_directory": str(OUTPUT),
        "model_scope": (
            "Supervised deterioration model input preparation; unsupervised "
            "IsolationForest evaluation remains descriptive."
        ),
    }
    (OUTPUT / "experiment_configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    report = f"""# Leakage-safe split report

## Status

Patient-level split generated with `uniquepid` groups, seed {SEED}, and
stratification by whether each person has any positive hourly target.

- Patients: {manifest['uniquepid'].nunique():,}
- ICU stays: {baseline['patientunitstayid'].nunique():,}
- Matched patient-hours per dataset version: {len(baseline):,}
- Train / validation / test patient counts: {len(patient_sets['train']):,} /
  {len(patient_sets['validation']):,} / {len(patient_sets['test']):,}
- Patient overlap by partition: {overlap}
- Baseline/corrected key sets and target labels match exactly.

## Target

`deterioration_label_24h` marks an hourly row positive when `unitdischargestatus`
is `expired` and ICU discharge occurs 0–24 hours after the prediction hour.
Multiple hourly rows per stay can be positive. This target is not general
clinical deterioration.

## Constraints

- The dataset represents the supplied eICU demo extract and is retrospective.
- Splitting is grouped by natural patient ID, so all of a person's ICU stays
  remain together.
- The target label and prediction hours are taken from the previously generated
  label file and are identical across dataset versions.
- The experiment feature table reuses non-SpO2 state variables and regenerates
  SpO2 current value, one-hour change, three-hour rolling mean, and dependent
  rule counts with the same gap-aware temporal rules.
- No production files or prior experiment outputs were overwritten.
"""
    (OUTPUT / "split_validation_report.md").write_text(report, encoding="utf-8")
    print(
        f"Leakage-safe cohort: {len(manifest):,} patients, "
        f"{baseline['patientunitstayid'].nunique():,} stays, "
        f"{len(baseline):,} hours; patient overlap={overlap}"
    )


if __name__ == "__main__":
    main()
