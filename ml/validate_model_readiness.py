from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed" / "model_readiness"
VALIDATION_DIR = OUTPUT_DIR / "validation"
INVENTORY_PATH = OUTPUT_DIR / "model_artifact_inventory.csv"
RECOVERY_PATH = OUTPUT_DIR / "model_recovery_status.csv"
READINESS_PATH = OUTPUT_DIR / "evaluation_readiness_matrix.csv"
SUPPORTED_EXTENSIONS = {
    ".pkl",
    ".pickle",
    ".joblib",
    ".pt",
    ".pth",
    ".onnx",
    ".h5",
    ".keras",
    ".sav",
    ".model",
    ".json",
    ".yaml",
    ".yml",
    ".config",
    ".ipynb",
}
SERIALIZED_MODEL_EXTENSIONS = SUPPORTED_EXTENSIONS - {
    ".json",
    ".yaml",
    ".yml",
    ".config",
    ".ipynb",
}
EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
}


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(cell(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def scan_project_artifact_paths() -> set[str]:
    matches: set[str] = set()
    for base, directories, filenames in os.walk(ROOT):
        directories[:] = [
            name for name in directories if name not in EXCLUDED_DIRECTORIES
        ]
        for filename in filenames:
            path = Path(base) / filename
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                matches.add(relative(path))
    return matches


def archive_model_artifacts() -> list[str]:
    archive = ROOT / "Smart-ICU.zip"
    if not archive.is_file():
        return ["archive missing"]

    found: list[str] = []
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            path = item.filename.replace("\\", "/").lower()
            parts = set(path.split("/"))
            if parts.intersection(EXCLUDED_DIRECTORIES) or "site-packages" in parts:
                continue
            if Path(path).suffix in SERIALIZED_MODEL_EXTENSIONS:
                found.append(item.filename)
    return found


def literal_assignment(script: Path, name: str) -> list[Any] | None:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                return None
            return value if isinstance(value, list) else None
    return None


def csv_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


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
            "expected_result": json.dumps(expected, ensure_ascii=False, default=str),
            "actual_result": json.dumps(actual, ensure_ascii=False, default=str),
            "status": status,
            "critical": critical,
            "explanation": explanation,
        }
    )
    if passed is False:
        discrepancies.append(
            {
                "check_name": name,
                "expected": json.dumps(expected, ensure_ascii=False, default=str),
                "actual": json.dumps(actual, ensure_ascii=False, default=str),
                "explanation": explanation,
            }
        )


def audit_saved_splits() -> dict[str, Any]:
    patient_path = ROOT / "data" / "real" / "eicu_demo" / (
        "eicu-collaborative-research-database-demo-2.0.1"
    ) / "patient.csv.gz"
    patients = pd.read_csv(
        patient_path,
        usecols=["patientunitstayid", "uniquepid"],
        low_memory=False,
    ).drop_duplicates("patientunitstayid")
    results: dict[str, Any] = {}
    for label, train_name, test_name in [
        (
            "deterioration",
            "deterioration_train.csv",
            "deterioration_test.csv",
        ),
        ("anomaly_v1", "anomaly_train.csv", "anomaly_test.csv"),
        ("anomaly_v2", "anomaly_train_v2.csv", "anomaly_test_v2.csv"),
    ]:
        train_ids = set(
            pd.read_csv(
                ROOT / "data" / "processed" / train_name,
                usecols=["patientunitstayid"],
                low_memory=False,
            )["patientunitstayid"].dropna()
        )
        test_ids = set(
            pd.read_csv(
                ROOT / "data" / "processed" / test_name,
                usecols=["patientunitstayid"],
                low_memory=False,
            )["patientunitstayid"].dropna()
        )
        train_people = set(
            patients.loc[
                patients["patientunitstayid"].isin(train_ids), "uniquepid"
            ].dropna()
        )
        test_people = set(
            patients.loc[
                patients["patientunitstayid"].isin(test_ids), "uniquepid"
            ].dropna()
        )
        results[label] = {
            "train_stays": len(train_ids),
            "test_stays": len(test_ids),
            "stay_overlap": len(train_ids & test_ids),
            "train_unique_people": len(train_people),
            "test_unique_people": len(test_people),
            "unique_person_overlap": len(train_people & test_people),
        }
    deterioration_train = pd.read_csv(
        ROOT / "data" / "processed" / "deterioration_train.csv",
        usecols=["patientunitstayid"],
        low_memory=False,
    )
    train_stay_order = deterioration_train["patientunitstayid"].unique()
    model_stays, calibration_stays = train_test_split(
        train_stay_order,
        test_size=0.20,
        random_state=123,
    )
    model_people = set(
        patients.loc[
            patients["patientunitstayid"].isin(model_stays), "uniquepid"
        ].dropna()
    )
    calibration_people = set(
        patients.loc[
            patients["patientunitstayid"].isin(calibration_stays), "uniquepid"
        ].dropna()
    )
    outer_test_people = set(
        patients.loc[
            patients["patientunitstayid"].isin(
                pd.read_csv(
                    ROOT / "data" / "processed" / "deterioration_test.csv",
                    usecols=["patientunitstayid"],
                    low_memory=False,
                )["patientunitstayid"].unique()
            ),
            "uniquepid",
        ].dropna()
    )
    results["deterioration_calibration"] = {
        "train_stays": len(model_stays),
        "test_stays": len(calibration_stays),
        "stay_overlap": len(set(model_stays) & set(calibration_stays)),
        "train_unique_people": len(model_people),
        "test_unique_people": len(calibration_people),
        "unique_person_overlap": len(model_people & calibration_people),
        "outer_test_overlap_with_model_train": len(model_people & outer_test_people),
        "outer_test_overlap_with_calibration": len(
            calibration_people & outer_test_people
        ),
    }
    return results


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    required_outputs = [
        INVENTORY_PATH,
        RECOVERY_PATH,
        READINESS_PATH,
        OUTPUT_DIR / "training_pipeline_reconstruction.md",
        OUTPUT_DIR / "model_input_contracts.md",
        OUTPUT_DIR / "future_evaluation_protocol.md",
    ]
    missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
    if missing_outputs:
        raise FileNotFoundError(f"Missing readiness outputs: {missing_outputs}")

    inventory = read_csv_rows(INVENTORY_PATH)
    recovery = read_csv_rows(RECOVERY_PATH)
    readiness = read_csv_rows(READINESS_PATH)
    checks: list[dict[str, Any]] = []
    discrepancies: list[dict[str, Any]] = []

    scanned_paths = scan_project_artifact_paths()
    inventoried_paths = {
        row["relative_path"]
        for row in inventory
        if row.get("scan_scope") == "filesystem"
    }
    add_check(
        checks,
        discrepancies,
        "Inventory completeness for requested file types",
        sorted(scanned_paths),
        sorted(inventoried_paths),
        scanned_paths == inventoried_paths,
        "Independently enumerated requested model/configuration extensions across the project, excluding .git, virtual environments, and dependency directories.",
    )

    inventory_paths = [row["relative_path"] for row in inventory]
    duplicates = sorted(
        {
            path
            for path in inventory_paths
            if inventory_paths.count(path) > 1
        }
    )
    add_check(
        checks,
        discrepancies,
        "No duplicate inventory paths",
        [],
        duplicates,
        not duplicates,
        "Each artifact or expected-but-missing artifact must have one inventory record.",
    )

    path_errors = []
    hash_mismatches = []
    references_missing = []
    for row in inventory:
        relative_path = row["relative_path"]
        path = ROOT / Path(relative_path)
        expected_exists = row.get("expected_exists", "").lower() == "true"
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            path_errors.append(relative_path)
            continue
        exists = path.is_file()
        if exists != expected_exists:
            path_errors.append(relative_path)
        if exists and row.get("sha256"):
            if sha256(path) != row["sha256"]:
                hash_mismatches.append(relative_path)
        for reference in filter(None, row.get("referencing_scripts", "").split("|")):
            if not (ROOT / Path(reference)).is_file():
                references_missing.append(reference)
    add_check(
        checks,
        discrepancies,
        "Artifact path correctness and existence states",
        "Existing inventory files exist; expected missing models are explicitly absent; all paths remain inside the project",
        {"path_errors": path_errors, "missing_referencing_scripts": references_missing},
        not path_errors and not references_missing,
        "Checked every inventory path and every referenced script without loading serialized artifacts.",
    )

    add_check(
        checks,
        discrepancies,
        "Protected source and data hashes unchanged",
        "All protected inventory hashes match the audit-time snapshot",
        hash_mismatches,
        not hash_mismatches,
        "The inventory stores hashes for the pre-existing scripts, datasets, predictions, and processed artifacts reviewed during this milestone.",
    )

    model_rows = [
        row
        for row in inventory
        if row.get("artifact_category") == "serialized_model"
    ]
    missing_model_rows = [
        row["relative_path"]
        for row in model_rows
        if row.get("expected_exists", "").lower() == "false"
        and not (ROOT / Path(row["relative_path"])).exists()
    ]
    unexpected_model_files = [
        relative(path)
        for path in (ROOT / "ml" / "models").rglob("*")
        if path.is_file()
        and path.suffix.lower() in SERIALIZED_MODEL_EXTENSIONS
    ]
    archive_models = archive_model_artifacts()
    model_inventory_ok = (
        len(missing_model_rows) == 2
        and not unexpected_model_files
        and not archive_models
    )
    add_check(
        checks,
        discrepancies,
        "Serialized model recovery state and archive scan",
        {
            "expected_missing_model_paths": 2,
            "unexpected_live_model_files": [],
            "archive_model_files_outside_excluded_dependencies": [],
        },
        {
            "missing_expected_model_paths": missing_model_rows,
            "unexpected_live_model_files": unexpected_model_files,
            "archive_model_files_outside_excluded_dependencies": archive_models,
        },
        model_inventory_ok,
        "Checked model directories and ZIP central-directory names only; no pickle/joblib/model payload was deserialized or executed.",
    )

    recovery_errors = []
    allowed_statuses = {
        "RECOVERED_AND_TRACEABLE",
        "RECOVERED_BUT_INCOMPLETE",
        "REFERENCED_BUT_MISSING",
        "TRAINING_CODE_ONLY",
        "EVALUATION_RESULTS_ONLY",
        "UNKNOWN_PROVENANCE",
    }
    for row in recovery:
        if row.get("status") not in allowed_statuses:
            recovery_errors.append(
                f"{row.get('model_id')}: invalid status {row.get('status')}"
            )
        if row.get("evaluation_ready", "").lower() != "false":
            recovery_errors.append(
                f"{row.get('model_id')}: incorrectly marked evaluation-ready"
            )
        for field in ("training_script", "inference_script"):
            reference = row.get(field, "")
            if reference and not (ROOT / Path(reference)).is_file():
                recovery_errors.append(
                    f"{row.get('model_id')}: missing {field} {reference}"
                )
    add_check(
        checks,
        discrepancies,
        "Recovery classifications are traceable and conservative",
        "All seven identified fit workflows use an allowed status; no missing artifact is marked ready",
        {"model_rows": len(recovery), "errors": recovery_errors},
        len(recovery) == 7 and not recovery_errors,
        "Checked model rows against the requested status vocabulary and referenced scripts.",
    )

    model_script_errors = []
    for row in recovery:
        model_id = row["model_id"]
        training_script = ROOT / Path(row["training_script"])
        inference_script = ROOT / Path(row["inference_script"])
        training_text = training_script.read_text(encoding="utf-8")
        inference_text = inference_script.read_text(encoding="utf-8")
        expected_artifact = row.get("artifact_path", "")
        if expected_artifact:
            artifact_name = Path(expected_artifact).name
            if artifact_name not in (training_text + inference_text):
                model_script_errors.append(
                    f"{model_id}: expected artifact name is not referenced by training/inference code"
                )
            if '"feature_columns": feature_columns' not in training_text:
                model_script_errors.append(
                    f"{model_id}: training does not serialize its ordered feature_columns"
                )
            if 'artifact["feature_columns"]' not in inference_text:
                model_script_errors.append(
                    f"{model_id}: inference does not read the serialized feature order"
                )
        if "feature_columns" not in inference_text:
            model_script_errors.append(
                f"{model_id}: inference does not consume recorded feature order"
            )
    add_check(
        checks,
        discrepancies,
        "Model-to-training and inference references",
        "Training and inference scripts exist and expected artifact/feature metadata references agree",
        model_script_errors,
        not model_script_errors,
        "Compared recorded model paths and inference feature-selection behavior with source code.",
    )
    add_check(
        checks,
        discrepancies,
        "Serialized fitted feature order and preprocessing state are inspectable",
        "Frozen bundles are present and provenance-checked before any use",
        {
            "expected_artifacts": [
                row["relative_path"]
                for row in model_rows
                if row.get("expected_exists", "").lower() == "false"
            ],
            "recovered_artifacts": [
                row["relative_path"]
                for row in model_rows
                if (ROOT / Path(row["relative_path"])).is_file()
            ],
        },
        None,
        "The expected bundles are absent. Their fitted coefficients/trees, actual serialized feature order, and fitted imputer/scaler state cannot be inspected; no artifact was loaded.",
        critical=False,
    )

    deterioration_scripts = [
        "train_deterioration_model.py",
        "train_deterioration_probability_model.py",
        "calibrate_deterioration_model.py",
    ]
    deterioration_features = [
        literal_assignment(ROOT / "ml" / script, "feature_columns")
        for script in deterioration_scripts
    ]
    anomaly_features = literal_assignment(
        ROOT / "ml" / "train_anomaly_model_v2.py", "feature_columns"
    )
    feature_lists_match = (
        all(deterioration_features)
        and deterioration_features[0]
        == deterioration_features[1]
        == deterioration_features[2]
        and bool(anomaly_features)
    )
    add_check(
        checks,
        discrepancies,
        "Training feature declarations are consistent",
        {
            "deterioration_feature_count": 22,
            "anomaly_v2_feature_count": 17,
            "deterioration_training_variants_match": True,
        },
        {
            "deterioration_feature_counts": [
                len(features) if features else None
                for features in deterioration_features
            ],
            "anomaly_v2_feature_count": (
                len(anomaly_features) if anomaly_features else None
            ),
            "deterioration_training_variants_match": (
                bool(deterioration_features[0])
                and deterioration_features[0]
                == deterioration_features[1]
                == deterioration_features[2]
            ),
        },
        feature_lists_match
        and len(deterioration_features[0]) == 22
        and len(anomaly_features) == 17,
        "Parsed literal feature lists from model scripts; no training code was executed.",
    )

    deterioration_columns = [
        "patientunitstayid",
        "hour",
        *(deterioration_features[0] or []),
        "deterioration_label_24h",
    ]
    anomaly_v2_columns = [
        "patientunitstayid",
        "hour",
        *(anomaly_features or []),
    ]
    schema_mismatches = {}
    for name in ("deterioration_train.csv", "deterioration_test.csv"):
        actual = csv_columns(ROOT / "data" / "processed" / name)
        if actual != deterioration_columns:
            schema_mismatches[name] = {
                "expected_count": len(deterioration_columns),
                "actual_count": len(actual),
            }
    for name in ("anomaly_train_v2.csv", "anomaly_test_v2.csv"):
        actual = csv_columns(ROOT / "data" / "processed" / name)
        if actual != anomaly_v2_columns:
            schema_mismatches[name] = {
                "expected_count": len(anomaly_v2_columns),
                "actual_count": len(actual),
            }
    anomaly_v1_train = csv_columns(
        ROOT / "data" / "processed" / "anomaly_train.csv"
    )
    anomaly_v1_test = csv_columns(
        ROOT / "data" / "processed" / "anomaly_test.csv"
    )
    if (
        anomaly_v1_train != anomaly_v1_test
        or anomaly_v1_train[:2] != ["patientunitstayid", "hour"]
        or len(anomaly_v1_train) != 62
    ):
        schema_mismatches["anomaly_train.csv/anomaly_test.csv"] = {
            "train_columns": len(anomaly_v1_train),
            "test_columns": len(anomaly_v1_test),
        }
    add_check(
        checks,
        discrepancies,
        "Feature schema matches saved training and test files",
        {
            "deterioration_columns": len(deterioration_columns),
            "anomaly_v2_columns": len(anomaly_v2_columns),
            "anomaly_v1_columns": 62,
        },
        schema_mismatches,
        not schema_mismatches,
        "Compared ordered columns in existing train/test CSV headers with the code-defined numeric feature contracts.",
    )

    label_source = (
        ROOT / "ml" / "prepare_deterioration_labels.py"
    ).read_text(encoding="utf-8")
    label_traceable = (
        'HORIZON = 24' in label_source
        and 'unitdischargestatus' in label_source
        and 'str.lower()' in label_source
        and '"expired"' in label_source
        and 'df["hours_until_discharge"] <= HORIZON' in label_source
        and 'df["hours_until_discharge"] >= 0' in label_source
        and '"deterioration_label_24h"' in label_source
    )
    add_check(
        checks,
        discrepancies,
        "Target label definition is traceable",
        "Expired ICU discharge within the following 0–24 hours; target is not generic deterioration",
        {
            "source": "ml/prepare_deterioration_labels.py",
            "horizon_hours": 24,
            "positive_event": "unitdischargestatus == expired",
            "prediction_window": "0 <= hours_until_discharge <= 24",
        },
        label_traceable,
        "Verified the target-generation expressions directly in source.",
    )

    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in ("pandas", "numpy", "sklearn", "joblib")
    }
    add_check(
        checks,
        discrepancies,
        "Runtime preprocessing libraries are available",
        "pandas, numpy, scikit-learn, and joblib are import-discoverable",
        dependencies,
        all(dependencies.values()),
        "Checked package availability without loading a model artifact.",
    )

    split_results = audit_saved_splits()
    stay_leakage = {
        name: values["stay_overlap"]
        for name, values in split_results.items()
    }
    add_check(
        checks,
        discrepancies,
        "Saved train/test sets have no ICU-stay overlap",
        "Zero patientunitstayid overlap in all saved splits",
        stay_leakage,
        all(value == 0 for value in stay_leakage.values()),
        "Independently compared stay identifiers in the saved train/test CSVs.",
    )
    person_overlap = {
        name: {
            "test_unique_people": values["test_unique_people"],
            "unique_person_overlap": values["unique_person_overlap"],
        }
        for name, values in split_results.items()
    }
    person_disjoint = all(
        values["unique_person_overlap"] == 0
        for values in split_results.values()
    )
    add_check(
        checks,
        discrepancies,
        "Saved splits are disjoint by natural-person identifier",
        "Zero `uniquepid` overlap across train and test",
        person_overlap,
        person_disjoint,
        "The existing split groups by ICU stay ID, not the raw eICU `uniquepid`; any overlap blocks a patient-independent evaluation claim.",
        critical=False,
    )
    calibration_leakage = split_results["deterioration_calibration"]
    add_check(
        checks,
        discrepancies,
        "Calibration partitions are independent by natural person",
        "Zero uniquepid overlap between model-fit, calibration, and outer test partitions",
        {
            "model_fit_vs_calibration": calibration_leakage[
                "unique_person_overlap"
            ],
            "model_fit_vs_outer_test": calibration_leakage[
                "outer_test_overlap_with_model_train"
            ],
            "calibration_vs_outer_test": calibration_leakage[
                "outer_test_overlap_with_calibration"
            ],
        },
        all(
            calibration_leakage[key] == 0
            for key in (
                "unique_person_overlap",
                "outer_test_overlap_with_model_train",
                "outer_test_overlap_with_calibration",
            )
        ),
        "Recreated the recorded random_state=123 stay split from saved training stays and joined to raw uniquepid; overlap is reported as an evaluation limitation.",
        critical=False,
    )

    declared_ready = [
        row["model_id"]
        for row in recovery
        if row.get("evaluation_ready", "").lower() == "true"
    ]
    artifact_ready = any(
        row.get("status") == "RECOVERED_AND_TRACEABLE"
        and row.get("evaluation_ready", "").lower() == "true"
        for row in recovery
    )
    add_check(
        checks,
        discrepancies,
        "Evaluation readiness classifications reflect available artifacts",
        "No model is ready without frozen weights, preprocessing state, and a compatible cohort",
        {
            "models_marked_ready": declared_ready,
            "recovered_evaluation_ready_model_exists": artifact_ready,
            "all_models_not_ready": all(
                row.get("evaluation_ready", "").lower() == "false"
                for row in recovery
            ),
        },
        not declared_ready
        and not artifact_ready
        and all(
            row.get("evaluation_ready", "").lower() == "false"
            for row in recovery
        ),
        "Checked readiness decisions against the recovered-artifact inventory and input contract.",
    )

    training_scripts = [
        ROOT / "ml" / name
        for name in [
            "train_deterioration_model.py",
            "train_deterioration_probability_model.py",
            "calibrate_deterioration_model.py",
            "train_anomaly_model.py",
            "train_anomaly_model_v2.py",
            "build_deployment_deterioration.py",
            "build_deployment_anomaly.py",
        ]
    ]
    training_source_hashes = {
        relative(path): sha256(path) for path in training_scripts
    }
    model_outputs = [
        relative(path)
        for path in (ROOT / "ml" / "models").rglob("*")
        if path.is_file()
        and path.suffix.lower() in SERIALIZED_MODEL_EXTENSIONS
    ]
    no_new_training_artifact = not model_outputs and not hash_mismatches
    add_check(
        checks,
        discrepancies,
        "No model training was run as part of this audit",
        "No model artifact exists in ml/models and protected training sources match their recorded hashes",
        {
            "serialized_model_files_in_ml_models": model_outputs,
            "training_script_hashes_recomputed": len(training_source_hashes),
            "protected_hash_mismatches": hash_mismatches,
        },
        no_new_training_artifact,
        "The validator only parses source and reads file metadata/CSV headers; it never calls fit, predict, joblib.load, or pickle.load.",
    )

    critical_failures = [
        row
        for row in checks
        if row["status"] == "FAIL" and row["critical"]
    ]
    failures = [row for row in checks if row["status"] == "FAIL"]
    not_verified = [row for row in checks if row["status"] == "NOT VERIFIED"]
    missing_models = [
        row for row in recovery if row.get("status") == "REFERENCED_BUT_MISSING"
    ]
    if critical_failures:
        verdict = "FAIL — INVENTORY OR TRACEABILITY ERRORS"
    elif missing_models or failures or not_verified:
        verdict = "PASS WITH LIMITATIONS — RECOVERY INCOMPLETE"
    else:
        verdict = "PASS — EVALUATION READY"

    checks_path = VALIDATION_DIR / "validation_checks.csv"
    discrepancies_path = VALIDATION_DIR / "validation_discrepancies.csv"
    pd.DataFrame(checks).to_csv(checks_path, index=False)
    pd.DataFrame(
        discrepancies,
        columns=["check_name", "expected", "actual", "explanation"],
    ).to_csv(discrepancies_path, index=False)

    model_readiness_report = f"""# Model readiness report

## Verdict

**{verdict}**

## Findings

- Five experimental model configurations and two additional deployment-time fit workflows are identified.
- Frozen serialized model artifacts recovered: 0.
- Models referenced by training/inference code but missing: {len(missing_models)}.
- The other fit workflows have source code and historical CSV predictions/results, but no recoverable fitted estimator/calibrator.
- Standalone fitted scalers, imputers, encoders, feature selectors, and threshold objects were not found. The primary models define imputation/scaling inside sklearn pipelines; their fitted state would be bundled in the missing model artifact.
- Saved train/test partitions are disjoint by ICU stay identifier, but not by natural-person identifier: {person_overlap}.
- The saved deterioration test population therefore cannot be treated as an independent patient-level holdout without excluding natural persons represented in the corresponding training partition.
- The calibrated deterioration workflow also has natural-person overlap: model-fit versus calibration {calibration_leakage['unique_person_overlap']}, model-fit versus outer test {calibration_leakage['outer_test_overlap_with_model_train']}, and calibration versus outer test {calibration_leakage['outer_test_overlap_with_calibration']}.
- The corrected SpO2 comparison changes model-input SpO2 values and SpO2-derived trends/flags. It does not add legacy model features for the quality metadata.
- Safest next step: recover the exact original bundles and provenance from the training environment or backup, then verify an `uniquepid`-disjoint cohort and the full inference preprocessing contract before considering any evaluation.

## Model recovery and readiness

{markdown_table(recovery, ['model_id', 'status', 'artifact_path', 'preprocessing_available', 'feature_schema_available', 'evaluation_ready', 'evidence'])}

## Evaluation readiness matrix

{markdown_table(readiness, ['model_id', 'feature_compatibility', 'missingness_compatibility', 'target_availability', 'split_reproducibility', 'same_frozen_model_possible', 'evaluation_ready', 'blockers'])}

## Validation

Independent checks: {sum(row['status'] == 'PASS' for row in checks)} PASS,
{sum(row['status'] == 'FAIL' for row in checks)} FAIL,
{sum(row['status'] == 'NOT VERIFIED' for row in checks)} NOT VERIFIED.
Critical failures: {len(critical_failures)}.

No model was loaded, trained, retrained, scored, or evaluated. The saved prediction CSVs are reported only as historical outputs; their model/code/data provenance cannot be established from the available files.
"""
    (OUTPUT_DIR / "model_readiness_report.md").write_text(
        model_readiness_report, encoding="utf-8"
    )

    validation_report = f"""# Independent validation — model readiness

## Verdict

**{verdict}**

Checks: {sum(row['status'] == 'PASS' for row in checks)} PASS,
{sum(row['status'] == 'FAIL' for row in checks)} FAIL,
{sum(row['status'] == 'NOT VERIFIED' for row in checks)} NOT VERIFIED.
Critical failures: {len(critical_failures)}.

## Check results

{markdown_table(checks, ['check_name', 'status', 'critical', 'explanation'])}

## Discrepancies

{len(discrepancies)} discrepancy records.
{"No discrepancies were found." if not discrepancies else markdown_table(discrepancies, ['check_name', 'expected', 'actual', 'explanation'])}

## Scope and limitations

The validator checked filesystem inventory, artifact references, code-defined feature order, saved dataset headers, label-generation logic, split IDs, and protected hashes. It did not deserialize or execute pickle/joblib/model artifacts and did not run training or model evaluation.

There are no frozen serialized estimators in the project or archive outside excluded virtual-environment/dependency content. The existing splits separate ICU stays, not natural persons; the independently verified person overlap is reported in the check results. Model metrics and model-dependent alert replay remain unavailable.
"""
    (VALIDATION_DIR / "validation_report.md").write_text(
        validation_report, encoding="utf-8"
    )

    print(verdict)
    print(
        f"Checks: PASS={sum(row['status'] == 'PASS' for row in checks)}, "
        f"FAIL={sum(row['status'] == 'FAIL' for row in checks)}, "
        f"NOT VERIFIED={sum(row['status'] == 'NOT VERIFIED' for row in checks)}"
    )
    print(f"Critical failures: {len(critical_failures)}")
    print(f"Validation outputs: {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
