from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = PROCESSED / "experiments" / "leakage_safe"
PATIENTS = (
    ROOT
    / "data"
    / "real"
    / "eicu_demo"
    / "eicu-collaborative-research-database-demo-2.0.1"
    / "patient.csv.gz"
)
KEYS = ["patientunitstayid", "hour"]
LABEL = "deterioration_label_24h"
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
INPUTS = {
    "patient": PATIENTS,
    "state": PROCESSED / "hourly_patient_state_alerts.csv",
    "rules": PROCESSED / "hourly_patient_state_rules_v2.csv",
    "labels": PROCESSED / "deterioration_labels.csv",
    "baseline_spo2": PROCESSED / "hourly_vitals_labs_clean.csv",
    "corrected_spo2": PROCESSED / "hourly_vitals_labs_spo2_experiment_clean.csv",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_keyed(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame = frame.dropna(subset=KEYS)
    require(not frame.duplicated(KEYS).any(), f"Duplicate keys in {path.name}.")
    return frame


def derive_spo2_features(frame: pd.DataFrame) -> pd.DataFrame:
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
    data["spo2_rolling_mean_3h"] = data["spo2"].groupby(
        segment_id, sort=False
    ).transform(lambda values: values.rolling(3, min_periods=2).mean())
    return data


def compare_numeric(
    actual: pd.Series, expected: pd.Series, name: str, tolerance: float = 1e-10
) -> None:
    left = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=float)
    require(
        np.allclose(left, right, rtol=tolerance, atol=tolerance, equal_nan=True),
        f"Feature mismatch: {name}.",
    )


def main() -> None:
    checks: list[dict[str, str]] = []

    def passed(name: str, detail: str) -> None:
        checks.append({"check": name, "status": "PASS", "detail": detail})

    config_path = OUTPUT / "experiment_configuration.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["random_seed"] == SEED, "Unexpected split seed.")
    require(set(config["input_sha256"]) == set(INPUTS), "Input hash list differs.")
    for name, path in INPUTS.items():
        require(path.is_file(), f"Missing input: {path}")
        require(
            sha256(path) == config["input_sha256"][name],
            f"Input hash changed since cohort generation: {name}.",
        )
    passed("input_hashes", "All source inputs match recorded SHA-256 hashes.")

    manifest = pd.read_csv(OUTPUT / "patient_split_manifest.csv", low_memory=False)
    require(not manifest["uniquepid"].duplicated().any(), "Duplicate patient manifest.")
    require(not manifest["split"].isna().any(), "Missing patient split.")
    allowed_splits = {"train", "validation", "test"}
    require(set(manifest["split"]) == allowed_splits, "Unexpected split labels.")
    partition_sets = {
        split: set(manifest.loc[manifest["split"].eq(split), "uniquepid"])
        for split in allowed_splits
    }
    for left, right in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        require(not partition_sets[left] & partition_sets[right], f"Patient leakage: {left}/{right}.")

    source_labels = load_keyed(INPUTS["labels"], KEYS + [LABEL])
    patients = pd.read_csv(INPUTS["patient"], usecols=["patientunitstayid", "uniquepid"])
    patients = patients.drop_duplicates("patientunitstayid")
    summary = source_labels.merge(
        patients, on="patientunitstayid", how="inner", validate="many_to_one"
    )
    patient_summary = (
        summary.groupby("uniquepid", as_index=False)
        .agg(patient_positive=(LABEL, "max"))
    )
    expected_patients = patient_summary["uniquepid"].to_numpy()
    patient_outcomes = patient_summary["patient_positive"].to_numpy()
    train_ids, holdout_ids = train_test_split(
        expected_patients,
        test_size=0.30,
        random_state=SEED,
        stratify=patient_outcomes,
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
    expected_assignment = {
        **dict.fromkeys(train_ids, "train"),
        **dict.fromkeys(validation_ids, "validation"),
        **dict.fromkeys(test_ids, "test"),
    }
    actual_assignment = manifest.set_index("uniquepid")["split"].to_dict()
    require(actual_assignment == expected_assignment, "Patient split is not reproducible.")
    require(set(actual_assignment) == set(expected_assignment), "Patient set differs.")
    passed(
        "patient_splits",
        f"{len(manifest)} unique patients; deterministic 70/15/15 split; no overlap.",
    )

    state_columns = list(dict.fromkeys(KEYS + FEATURES))
    state = load_keyed(INPUTS["state"], state_columns)
    rules = load_keyed(INPUTS["rules"], list(dict.fromkeys(KEYS + WARNINGS + SEVERE)))
    output_frames: dict[str, pd.DataFrame] = {}
    source_name = {
        "baseline": "baseline_spo2",
        "corrected": "corrected_spo2",
    }
    for version, source_key in source_name.items():
        path = OUTPUT / f"model_inputs_{version}.csv"
        frame = pd.read_csv(path, low_memory=False)
        required_columns = ["patientunitstayid", "hour", "uniquepid", "split", *FEATURES, LABEL]
        require(frame.columns.tolist() == required_columns, f"Unexpected columns in {path.name}.")
        require(not frame.duplicated(KEYS).any(), f"Duplicate output keys in {path.name}.")
        require(
            not frame[KEYS + ["uniquepid", "split", LABEL]].isna().any(axis=None),
            f"Missing required identifiers, split, or target in {path.name}.",
        )
        expected = state[KEYS + [column for column in FEATURES if column in state]]
        expected = expected.drop(
            columns=["spo2", "spo2_change_1h", "spo2_rolling_mean_3h"]
        )
        source_spo2 = load_keyed(INPUTS[source_key], KEYS + ["spo2"])
        expected = expected.merge(source_spo2, on=KEYS, how="inner", validate="one_to_one")
        expected = derive_spo2_features(expected)
        rule_values = expected[KEYS].merge(
            rules[KEYS + WARNINGS + SEVERE],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
        for column in WARNINGS + SEVERE:
            rule_values[column] = pd.to_numeric(
                rule_values[column], errors="coerce"
            ).fillna(0)
        rule_values["rule_hypoxemia"] = expected["spo2"].lt(92).astype(int).to_numpy()
        rule_values["rule_falling_spo2"] = (
            expected["spo2_change_1h"].lt(-3).astype(int).to_numpy()
        )
        rule_values["rule_severe_hypoxemia"] = expected["spo2"].lt(88).astype(int).to_numpy()
        rule_values["rule_rapid_falling_spo2"] = (
            expected["spo2_change_1h"].lt(-5).astype(int).to_numpy()
        )
        expected["clinical_warning_count"] = rule_values[WARNINGS].sum(axis=1)
        expected["clinical_severe_count"] = rule_values[SEVERE].sum(axis=1)
        expected = expected.merge(source_labels, on=KEYS, how="inner", validate="one_to_one")
        stay_patients = patients.drop_duplicates("patientunitstayid")
        expected = expected.merge(
            stay_patients,
            on="patientunitstayid",
            how="left",
            validate="many_to_one",
        )
        expected = expected.merge(
            manifest[["uniquepid", "split"]], on="uniquepid", how="left", validate="many_to_one"
        )
        require(len(frame) == len(expected), f"Row count mismatch for {version}.")
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        expected = expected.sort_values(KEYS).reset_index(drop=True)
        require(frame[KEYS].equals(expected[KEYS]), f"Source keys mismatch for {version}.")
        require(frame["uniquepid"].equals(expected["uniquepid"]), f"Patient IDs mismatch for {version}.")
        require(frame["split"].equals(expected["split"]), f"Split assignments mismatch for {version}.")
        require(frame[LABEL].equals(expected[LABEL]), f"Target mismatch for {version}.")
        for feature in FEATURES:
            compare_numeric(frame[feature], expected[feature], f"{version}.{feature}")
        require(
            all(
                not set(frame.loc[frame["split"].eq(left), "uniquepid"])
                & set(frame.loc[frame["split"].eq(right), "uniquepid"])
                for left, right in [
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                ]
            ),
            f"Patient leakage in {version} output.",
        )
        require(
            all(frame.loc[frame["split"].eq(split), LABEL].sum() > 0 for split in allowed_splits),
            f"Missing positive labels in {version} partition.",
        )
        output_frames[version] = frame
        passed(
            f"{version}_reconstruction",
            "Keys, patient mapping, targets, all features, and split assignments independently reconstructed.",
        )

    baseline = output_frames["baseline"].sort_values(KEYS).reset_index(drop=True)
    corrected = output_frames["corrected"].sort_values(KEYS).reset_index(drop=True)
    require(baseline[KEYS].equals(corrected[KEYS]), "Baseline/corrected keys differ.")
    require(baseline[LABEL].equals(corrected[LABEL]), "Baseline/corrected targets differ.")
    require(baseline["uniquepid"].equals(corrected["uniquepid"]), "Patient IDs differ by version.")
    require(baseline["split"].equals(corrected["split"]), "Split assignments differ by version.")
    passed("paired_cohort", "Baseline and corrected versions have identical keys, labels, and splits.")

    results = pd.DataFrame(checks)
    results.to_csv(OUTPUT / "validation_checks.csv", index=False)
    report_lines = [
        "# Independent leakage-safe experiment validation",
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
            "Validation independently reconstructs cohort source joins, the patient-level",
            "split, the row-to-person mapping, SpO2 temporal features, rule counts, targets,",
            "and all reported model-input feature values. It does not evaluate model",
            "performance or clinical validity.",
            "",
        ]
    )
    (OUTPUT / "validation_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(f"Leakage-safe experiment validation: PASS ({len(results)} checks).")


if __name__ == "__main__":
    main()
