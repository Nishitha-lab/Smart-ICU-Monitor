from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    ROOT
    / "data"
    / "real"
    / "eicu_demo"
    / "eicu-collaborative-research-database-demo-2.0.1"
)
AUDIT_DIR = ROOT / "data" / "processed" / "spo2_audit"
VALIDATION_DIR = AUDIT_DIR / "validation"

MATCH_WINDOW = 15
EPISODE_GAP = 10
LOW_LIMIT = 50
SUPPORT_LIMIT = 50
CONFLICT_LIMIT = 70
THRESHOLDS = (20, 30, 40, 50)
KNOWN_STAYS = (3157219, 3204052, 3067309, 758327, 1229669, 939222)
SOURCES = ("abg_pao2", "abg_o2_sat", "resp_sao2", "map_context")

OUTPUTS = {
    "readings": AUDIT_DIR / "spo2_cross_validation_readings.csv",
    "matches": AUDIT_DIR / "spo2_cross_validation_matches.csv",
    "episodes": AUDIT_DIR / "spo2_cross_validation_episodes.csv",
    "summary": AUDIT_DIR / "spo2_cross_validation_summary.csv",
    "policy": AUDIT_DIR / "spo2_quality_policy.md",
}

PROTECTED_PATHS = [
    ROOT / "ml" / "preprocessing.py",
    ROOT / "ml" / "cleaning.py",
    ROOT / "ml" / "preprocessing_spo2_experiment.py",
    ROOT / "ml" / "cleaning_spo2_experiment.py",
    ROOT / "ml" / "merge_patient_state.py",
    ROOT / "ml" / "trend_features.py",
]
RAW_PATHS = [
    DATA_DIR / filename
    for filename in [
        "patient.csv.gz",
        "vitalPeriodic.csv.gz",
        "vitalAperiodic.csv.gz",
        "lab.csv.gz",
        "respiratoryCharting.csv.gz",
    ]
]


def canonical_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def parse_value(value: object) -> tuple[float, str]:
    if pd.isna(value):
        return np.nan, "MISSING"
    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan, "MISSING"
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
        return float(text), "NUMERIC"
    return np.nan, "MALFORMED"


def label_kind(label: object) -> str | None:
    if pd.isna(label):
        return None
    normalized = re.sub(r"[^a-z0-9]", "", str(label).casefold())
    if normalized == "pao2":
        return "abg_pao2"
    if normalized in {
        "o2sat",
        "o2satpercent",
        "o2saturation",
        "o2saturationpercent",
        "sao2",
        "sao2percent",
        "oxygensaturation",
        "oxygensaturationpercent",
    }:
        return "abg_o2_sat"
    return None


def fingerprint(paths: list[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            result[str(path)] = "MISSING"
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[str(path)] = digest.hexdigest()
    return result


def load_stays() -> pd.DataFrame:
    stays = pd.read_csv(
        DATA_DIR / "patient.csv.gz",
        usecols=["patientunitstayid", "unitdischargeoffset"],
        low_memory=False,
    )
    stays["unitdischargeoffset"] = pd.to_numeric(
        stays["unitdischargeoffset"],
        errors="coerce",
    )
    stays = stays.dropna(
        subset=["patientunitstayid", "unitdischargeoffset"]
    )
    if stays["patientunitstayid"].duplicated().any():
        raise ValueError("patient.csv.gz has duplicate stay IDs.")
    return stays


def load_source(
    filename: str,
    columns: list[str],
    offset_name: str,
    stays: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    source = pd.read_csv(
        DATA_DIR / filename,
        usecols=columns,
        low_memory=False,
    )
    source[offset_name] = pd.to_numeric(source[offset_name], errors="coerce")
    eligible_stay = source["patientunitstayid"].isin(
        stays["patientunitstayid"]
    )
    valid_offset = source[offset_name].notna()
    joined = source.merge(
        stays,
        on="patientunitstayid",
        how="inner",
        validate="many_to_one",
    )
    in_stay = joined[offset_name].ge(0) & joined[offset_name].le(
        joined["unitdischargeoffset"]
    )
    included = joined.loc[in_stay].copy()
    return included, {
        "raw_rows": len(source),
        "unknown_stay_rows": int((~eligible_stay).sum()),
        "malformed_offset_rows": int((~valid_offset).sum()),
        "outside_interval_rows": int(
            (eligible_stay & valid_offset).sum() - len(included)
        ),
    }


def load_raw_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, int]],
]:
    stays = load_stays()
    periodic, periodic_stats = load_source(
        "vitalPeriodic.csv.gz",
        [
            "vitalperiodicid",
            "patientunitstayid",
            "observationoffset",
            "sao2",
        ],
        "observationoffset",
        stays,
    )
    periodic["raw_sao2"] = periodic["sao2"].astype("string")
    parsed = periodic["sao2"].map(parse_value)
    periodic["sao2"] = parsed.map(lambda item: item[0])
    periodic["sao2_status"] = parsed.map(lambda item: item[1])
    periodic["hour"] = (periodic["observationoffset"] // 60).astype(int)

    lab, lab_stats = load_source(
        "lab.csv.gz",
        [
            "labid",
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "labresult",
            "labresulttext",
            "labmeasurenamesystem",
            "labmeasurenameinterface",
        ],
        "labresultoffset",
        stays,
    )
    lab["source"] = lab["labname"].map(label_kind)
    lab = lab.loc[lab["source"].notna()].copy()
    primary_parse = lab["labresult"].map(parse_value)
    text_parse = lab["labresulttext"].map(parse_value)
    use_text = primary_parse.map(lambda item: pd.isna(item[0])) & text_parse.map(
        lambda item: pd.notna(item[0])
    )
    lab["value"] = [
        text_item[0] if use else raw_item[0]
        for raw_item, text_item, use in zip(primary_parse, text_parse, use_text)
    ]
    lab["parse_status"] = [
        (
            "NUMERIC"
            if raw_item[1] == "NUMERIC"
            else (
                "TEXT_FALLBACK"
                if use and text_item[1] == "NUMERIC"
                else (
                    "MISSING"
                    if raw_item[1] == "MISSING"
                    and text_item[1] == "MISSING"
                    else "MALFORMED"
                )
            )
        )
        for raw_item, text_item, use in zip(primary_parse, text_parse, use_text)
    ]
    lab["display_raw_value"] = [
        (
            str(raw)
            if raw_item[1] != "MISSING"
            else (str(text) if text_item[1] != "MISSING" else pd.NA)
        )
        for raw, text, raw_item, text_item in zip(
            lab["labresult"],
            lab["labresulttext"],
            primary_parse,
            text_parse,
        )
    ]
    lab["measurement_id"] = lab["labid"].map(canonical_id)
    lab["measurement_offset"] = lab["labresultoffset"]
    lab["measurement_label"] = lab["labname"]
    lab["units"] = lab["labmeasurenamesystem"]
    lab["units_interface"] = lab["labmeasurenameinterface"]

    respiratory, respiratory_stats = load_source(
        "respiratoryCharting.csv.gz",
        [
            "respchartid",
            "patientunitstayid",
            "respchartoffset",
            "respchartvaluelabel",
            "respchartvalue",
        ],
        "respchartoffset",
        stays,
    )
    resp_label = (
        respiratory["respchartvaluelabel"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    respiratory = respiratory.loc[resp_label.eq("sao2")].copy()
    respiratory["source"] = "resp_sao2"
    respiratory["measurement_id"] = respiratory["respchartid"].map(canonical_id)
    respiratory["measurement_offset"] = respiratory["respchartoffset"]
    respiratory["measurement_label"] = respiratory["respchartvaluelabel"]
    respiratory["value"], respiratory["parse_status"] = zip(
        *respiratory["respchartvalue"].map(parse_value)
    )
    respiratory["display_raw_value"] = respiratory[
        "respchartvalue"
    ].astype("string")
    respiratory["units"] = "%"
    respiratory["units_interface"] = "%"

    map_data, map_stats = load_source(
        "vitalAperiodic.csv.gz",
        [
            "vitalaperiodicid",
            "patientunitstayid",
            "observationoffset",
            "noninvasivemean",
        ],
        "observationoffset",
        stays,
    )
    map_data["source"] = "map_context"
    map_data["measurement_id"] = map_data[
        "vitalaperiodicid"
    ].map(canonical_id)
    map_data["measurement_offset"] = map_data["observationoffset"]
    map_data["measurement_label"] = "noninvasivemean"
    map_data["value"], map_data["parse_status"] = zip(
        *map_data["noninvasivemean"].map(parse_value)
    )
    map_data["display_raw_value"] = map_data[
        "noninvasivemean"
    ].astype("string")
    map_data["units"] = "mmHg"
    map_data["units_interface"] = "mmHg"

    references = pd.concat(
        [
            lab[
                [
                    "patientunitstayid",
                    "source",
                    "measurement_id",
                    "measurement_offset",
                    "measurement_label",
                    "value",
                    "display_raw_value",
                    "parse_status",
                    "units",
                    "units_interface",
                ]
            ],
            respiratory[
                [
                    "patientunitstayid",
                    "source",
                    "measurement_id",
                    "measurement_offset",
                    "measurement_label",
                    "value",
                    "display_raw_value",
                    "parse_status",
                    "units",
                    "units_interface",
                ]
            ],
            map_data[
                [
                    "patientunitstayid",
                    "source",
                    "measurement_id",
                    "measurement_offset",
                    "measurement_label",
                    "value",
                    "display_raw_value",
                    "parse_status",
                    "units",
                    "units_interface",
                ]
            ],
        ],
        ignore_index=True,
    )
    source_stats = {
        "periodic": periodic_stats,
        "lab": lab_stats,
        "respiratory": respiratory_stats,
        "map": map_stats,
    }
    return stays, periodic, references, source_stats


def rebuild_hourly(periodic: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        periodic.groupby(["patientunitstayid", "hour"], as_index=False)
        .agg(
            raw_mean=("sao2", "mean"),
            valid_nonzero_count=("sao2", lambda values: values.gt(0).sum()),
            zero_count=("sao2", lambda values: values.eq(0).sum()),
            numeric_count=("sao2", "count"),
            periodic_rows=("sao2", "size"),
            zero_excluded_mean=(
                "sao2",
                lambda values: values.loc[values.gt(0)].mean(),
            ),
        )
    )
    hourly["missing_count"] = hourly["periodic_rows"] - hourly["numeric_count"]
    hourly["quality_flag"] = np.select(
        [
            hourly["valid_nonzero_count"].gt(0) & hourly["zero_count"].gt(0),
            hourly["valid_nonzero_count"].gt(0) & hourly["zero_count"].eq(0),
            hourly["valid_nonzero_count"].eq(0) & hourly["zero_count"].gt(0),
        ],
        ["MIXED_ZERO", "VALID", "ZERO_ONLY"],
        default="MISSING_ONLY",
    )
    return hourly


def rebuild_episodes(readings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = readings.sort_values(
        ["patientunitstayid", "observationoffset", "vitalperiodicid"],
        kind="stable",
    ).copy()
    gaps = ordered.groupby("patientunitstayid")["observationoffset"].diff()
    starts = gaps.isna() | gaps.gt(EPISODE_GAP)
    numbers = starts.groupby(ordered["patientunitstayid"]).cumsum()
    ordered["episode_id"] = (
        ordered["patientunitstayid"].astype(str)
        + "_"
        + numbers.astype(int).astype(str).str.zfill(5)
    )
    episode_rows: list[dict[str, Any]] = []
    for episode_id, episode in ordered.groupby("episode_id", sort=False):
        offsets = episode["observationoffset"]
        values = episode["sao2"]
        episode_rows.append(
            {
                "episode_id": episode_id,
                "patientunitstayid": int(episode["patientunitstayid"].iloc[0]),
                "start_offset": int(offsets.min()),
                "end_offset": int(offsets.max()),
                "duration_minutes": int(offsets.max() - offsets.min()),
                "low_reading_count": len(episode),
                "minimum_spo2": float(values.min()),
                "mean_spo2": float(values.mean()),
                "median_spo2": float(values.median()),
                "constituent_reading_ids": "|".join(
                    episode["vitalperiodicid"].map(canonical_id)
                ),
                "gap_split_count": int(
                    episode["observationoffset"].diff().gt(EPISODE_GAP).sum()
                ),
                **{
                    f"readings_lt_{threshold}": int(values.lt(threshold).sum())
                    for threshold in THRESHOLDS
                },
                **{
                    f"readings_le_{threshold}": int(values.le(threshold).sum())
                    for threshold in THRESHOLDS
                },
            }
        )
    return ordered, pd.DataFrame(episode_rows)


def independently_match(
    readings: pd.DataFrame,
    references: pd.DataFrame,
) -> pd.DataFrame:
    groups = {
        stay: frame.sort_values("measurement_offset", kind="stable")
        for stay, frame in references.groupby("patientunitstayid")
    }
    results: list[dict[str, Any]] = []
    for reading in readings.itertuples(index=False):
        stay_refs = groups.get(reading.patientunitstayid)
        if stay_refs is None:
            continue
        low = stay_refs["measurement_offset"].ge(
            reading.observationoffset - MATCH_WINDOW
        )
        high = stay_refs["measurement_offset"].le(
            reading.observationoffset + MATCH_WINDOW
        )
        nearby = stay_refs.loc[low & high]
        for match in nearby.itertuples(index=False):
            results.append(
                {
                    "reading_id": canonical_id(reading.vitalperiodicid),
                    "episode_id": reading.episode_id,
                    "patientunitstayid": int(reading.patientunitstayid),
                    "spo2_offset": float(reading.observationoffset),
                    "spo2": float(reading.sao2),
                    "measurement_source": match.source,
                    "measurement_id": canonical_id(match.measurement_id),
                    "measurement_label": match.measurement_label,
                    "measurement_offset": float(match.measurement_offset),
                    "measurement_value": match.value,
                    "raw_measurement_value": match.display_raw_value,
                    "time_difference_minutes": float(
                        match.measurement_offset - reading.observationoffset
                    ),
                    "units": match.units,
                    "units_interface": match.units_interface,
                    "measurement_parse_status": match.parse_status,
                }
            )
    return pd.DataFrame(results)


def classify_independently(matches: pd.DataFrame) -> tuple[str, str]:
    direct = matches.loc[
        matches["measurement_source"].isin(["abg_o2_sat", "resp_sao2"])
    ]
    numeric = direct.loc[direct["measurement_value"].notna()]
    values = numeric["measurement_value"]
    usable = values.loc[values.gt(0) & values.le(100)]
    supports = usable.le(SUPPORT_LIMIT).any()
    conflicts = usable.ge(CONFLICT_LIMIT).any()
    if supports and conflicts:
        return "CONFLICTING", "mixed direct saturation evidence"
    if conflicts:
        return "CONFLICTING", "direct saturation in disagreement band"
    if supports:
        return "SUPPORTED", "direct low saturation reference"
    if not numeric.empty:
        return "INSUFFICIENT_DATA", "direct saturation in indeterminate or invalid range"
    if matches["measurement_source"].isin(["abg_o2_sat", "abg_pao2"]).any():
        return "INSUFFICIENT_DATA", "ABG record exists but direct comparison is incomplete"
    if matches["measurement_parse_status"].isin(["MALFORMED", "MISSING"]).any():
        return "INSUFFICIENT_DATA", "nearby oxygenation measurement is incomplete"
    return "NO_INDEPENDENT_EVIDENCE", "no direct saturation reference in window"


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    expected: object,
    actual: object,
    passed: bool | None,
    explanation: str,
    critical: bool = True,
) -> None:
    checks.append(
        {
            "check_name": name,
            "expected_result": str(expected),
            "independently_calculated_result": str(actual),
            "status": (
                "NOT VERIFIED"
                if passed is None
                else ("PASS" if passed else "FAIL")
            ),
            "critical": critical,
            "explanation": explanation,
        }
    )


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def discrepancy(
    rows: list[dict[str, Any]],
    category: str,
    record_key: object,
    expected: object,
    actual: object,
    explanation: str,
) -> None:
    rows.append(
        {
            "category": category,
            "record_key": record_key,
            "expected": expected,
            "actual": actual,
            "explanation": explanation,
        }
    )


def compare_readings(
    independent: pd.DataFrame,
    audit: pd.DataFrame,
    discrepancies: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    source_ids = independent["vitalperiodicid"].map(canonical_id)
    audit_ids = audit["reading_id"].map(canonical_id)
    source_count = source_ids.value_counts()
    audit_count = audit_ids.value_counts()
    duplicate_source = set(source_count[source_count.gt(1)].index)
    duplicate_audit = set(audit_count[audit_count.gt(1)].index)
    source_set = set(source_ids)
    audit_set = set(audit_ids)
    for reading_id in sorted(source_set - audit_set):
        discrepancy(
            discrepancies,
            "MISSING_READING",
            reading_id,
            "present in detailed output",
            "missing",
            "Eligible raw source reading is omitted.",
        )
    for reading_id in sorted(audit_set - source_set):
        discrepancy(
            discrepancies,
            "EXTRA_READING",
            reading_id,
            "not present",
            "present in detailed output",
            "Detailed output row does not map to an eligible raw observation.",
        )
    for reading_id in sorted(duplicate_source | duplicate_audit):
        discrepancy(
            discrepancies,
            "DUPLICATE_READING",
            reading_id,
            "one occurrence",
            f"raw={source_count.get(reading_id, 0)}, output={audit_count.get(reading_id, 0)}",
            "Source or output observation ID is duplicated.",
        )

    raw_by_id = independent.assign(
        reading_id=source_ids
    ).set_index("reading_id")
    out_by_id = audit.assign(reading_id=audit_ids).set_index("reading_id")
    matched_ids = source_set & audit_set
    value_mismatches = 0
    stay_mismatches = 0
    offset_mismatches = 0
    for reading_id in matched_ids:
        raw = raw_by_id.loc[reading_id]
        out = out_by_id.loc[reading_id]
        if isinstance(raw, pd.DataFrame) or isinstance(out, pd.DataFrame):
            continue
        if int(raw["patientunitstayid"]) != int(out["patientunitstayid"]):
            stay_mismatches += 1
            discrepancy(
                discrepancies,
                "READING_STAY_MISMATCH",
                reading_id,
                int(raw["patientunitstayid"]),
                int(out["patientunitstayid"]),
                "Output ICU stay differs from raw source.",
            )
        if not np.isclose(
            raw["observationoffset"],
            out["observationoffset"],
            rtol=0,
            atol=0,
        ):
            offset_mismatches += 1
            discrepancy(
                discrepancies,
                "READING_OFFSET_MISMATCH",
                reading_id,
                raw["observationoffset"],
                out["observationoffset"],
                "Output observation offset differs from raw source.",
            )
        if not np.isclose(raw["sao2"], out["spo2"], rtol=0, atol=1e-12):
            value_mismatches += 1
            discrepancy(
                discrepancies,
                "READING_VALUE_MISMATCH",
                reading_id,
                raw["sao2"],
                out["spo2"],
                "Output SpO2 differs from raw numeric value.",
            )
    return [
        {
            "check_name": "Raw-to-output strict-low reading reconciliation",
            "expected_result": f"{len(independent)} unique eligible raw readings, exact ID-set equality",
            "independently_calculated_result": (
                f"raw={len(independent)}; output={len(audit)}; "
                f"missing={len(source_set - audit_set)}; extra={len(audit_set - source_set)}"
            ),
            "status": (
                "PASS"
                if source_set == audit_set
                and not duplicate_source
                and not duplicate_audit
                and not value_mismatches
                and not stay_mismatches
                and not offset_mismatches
                else "FAIL"
            ),
            "critical": True,
            "explanation": (
                f"value mismatches={value_mismatches}; stay mismatches={stay_mismatches}; "
                f"offset mismatches={offset_mismatches}"
            ),
        }
    ], source_set & audit_set


def validate_episode_output(
    independent_episodes: pd.DataFrame,
    audit_episodes: pd.DataFrame,
    discrepancies: list[dict[str, Any]],
) -> tuple[bool, int]:
    expected = independent_episodes.set_index("episode_id")
    actual = audit_episodes.set_index("episode_id")
    expected_ids = set(expected.index)
    actual_ids = set(actual.index)
    for episode_id in sorted(expected_ids - actual_ids):
        discrepancy(
            discrepancies,
            "MISSING_EPISODE",
            episode_id,
            "present",
            "missing",
            "Raw-reconstructed episode is missing from audit output.",
        )
    for episode_id in sorted(actual_ids - expected_ids):
        discrepancy(
            discrepancies,
            "EXTRA_EPISODE",
            episode_id,
            "missing",
            "present",
            "Audit episode does not match reconstructed raw population.",
        )
    mismatch_count = 0
    columns = [
        "patientunitstayid",
        "start_offset",
        "end_offset",
        "duration_minutes",
        "low_reading_count",
        "minimum_spo2",
        "mean_spo2",
        "median_spo2",
        "constituent_reading_ids",
        *[f"readings_lt_{t}" for t in THRESHOLDS],
        *[f"readings_le_{t}" for t in THRESHOLDS],
    ]
    for episode_id in expected_ids & actual_ids:
        for column in columns:
            left = expected.at[episode_id, column]
            right = actual.at[episode_id, column]
            if column in {
                "minimum_spo2",
                "mean_spo2",
                "median_spo2",
            }:
                equal = np.isclose(
                    float(left),
                    float(right),
                    rtol=0,
                    atol=1e-10,
                    equal_nan=True,
                )
            elif column == "constituent_reading_ids":
                equal = str(left) == str(right)
            else:
                equal = int(left) == int(right)
            if not equal:
                mismatch_count += 1
                discrepancy(
                    discrepancies,
                    "EPISODE_METRIC_MISMATCH",
                    f"{episode_id}:{column}",
                    left,
                    right,
                    "Audit episode metric differs from independent reconstruction.",
                )
    return (
        expected_ids == actual_ids and mismatch_count == 0,
        mismatch_count,
    )


def validate_match_output(
    expected: pd.DataFrame,
    audit: pd.DataFrame,
    discrepancies: list[dict[str, Any]],
) -> tuple[bool, int, int]:
    keys = ["reading_id", "measurement_source", "measurement_id"]
    expected = expected.copy()
    audit = audit.copy()
    for frame in (expected, audit):
        frame["reading_id"] = frame["reading_id"].map(canonical_id)
        frame["measurement_id"] = frame["measurement_id"].map(canonical_id)
    expected_keys = set(map(tuple, expected[keys].itertuples(index=False, name=None)))
    audit_keys = set(map(tuple, audit[keys].itertuples(index=False, name=None)))
    for key in sorted(expected_keys - audit_keys):
        discrepancy(
            discrepancies,
            "OMITTED_REFERENCE_MATCH",
            "|".join(key),
            "present",
            "missing",
            "Raw source contains a matching measurement within ±15 minutes.",
        )
    for key in sorted(audit_keys - expected_keys):
        discrepancy(
            discrepancies,
            "EXTRA_REFERENCE_MATCH",
            "|".join(key),
            "missing",
            "present",
            "Audit output match is not present in the independent raw match set.",
        )
    expected_index = expected.set_index(keys)
    audit_index = audit.set_index(keys)
    field_mismatches = 0
    for key in expected_keys & audit_keys:
        left = expected_index.loc[key]
        right = audit_index.loc[key]
        fields = [
            "patientunitstayid",
            "spo2_offset",
            "spo2",
            "measurement_label",
            "measurement_offset",
            "measurement_value",
            "time_difference_minutes",
            "units",
            "units_interface",
        ]
        for field in fields:
            x = left[field]
            y = right[field]
            if field in {
                "spo2_offset",
                "spo2",
                "measurement_offset",
                "measurement_value",
                "time_difference_minutes",
            }:
                equal = np.isclose(
                    float(x) if pd.notna(x) else np.nan,
                    float(y) if pd.notna(y) else np.nan,
                    rtol=0,
                    atol=1e-9,
                    equal_nan=True,
                )
            else:
                equal = (
                    canonical_id(x) == canonical_id(y)
                    if field == "patientunitstayid"
                    else (str(x) == str(y) or (pd.isna(x) and pd.isna(y)))
                )
            if not equal:
                field_mismatches += 1
                discrepancy(
                    discrepancies,
                    "REFERENCE_FIELD_MISMATCH",
                    f"{'|'.join(key)}:{field}",
                    x,
                    y,
                    "Reference metadata or time calculation differs.",
                )
    expected_count = len(expected_keys)
    actual_count = len(audit_keys)
    boundary_ok = bool(
        audit["time_difference_minutes"].abs().le(MATCH_WINDOW).all()
    )
    passed = (
        expected_keys == audit_keys
        and field_mismatches == 0
        and boundary_ok
        and not audit.duplicated(keys).any()
    )
    return passed, expected_count, actual_count


def known_case_review(
    readings: pd.DataFrame,
    matches: pd.DataFrame,
    episodes: pd.DataFrame,
    raw_case_readings: pd.DataFrame,
    stays: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    episode_lookup = episodes.set_index("episode_id")
    for stay_id in KNOWN_STAYS:
        stay_readings = readings.loc[
            readings["patientunitstayid"].eq(stay_id)
        ].sort_values("observationoffset")
        raw_stay = raw_case_readings.loc[
            raw_case_readings["patientunitstayid"].eq(stay_id)
        ].copy()
        raw_stay["sao2"] = pd.to_numeric(raw_stay["sao2"], errors="coerce")
        raw_low = raw_stay.loc[raw_stay["sao2"].gt(0) & raw_stay["sao2"].lt(50)]
        discharge = stays.loc[
            stays["patientunitstayid"].eq(stay_id),
            "unitdischargeoffset",
        ]
        raw_low_in_stay = raw_low.loc[
            raw_low["observationoffset"].ge(0)
            & raw_low["observationoffset"].le(
                discharge.iloc[0] if not discharge.empty else -1
            )
        ]
        raw_outside_offsets = raw_low.loc[
            ~raw_low.index.isin(raw_low_in_stay.index)
        ]
        stay_matches = matches.loc[
            matches["patientunitstayid"].eq(stay_id)
        ]
        stay_episodes = episodes.loc[
            episodes["patientunitstayid"].eq(stay_id)
        ]
        classification = (
            "|".join(sorted(stay_episodes["evidence_classification"].unique()))
            if not stay_episodes.empty
            else "NO_LOW_EPISODE"
        )
        low_values = ";".join(
            f"{int(row.observationoffset)}:{row.sao2:g}"
            for row in stay_readings.itertuples(index=False)
        )
        source_summaries: dict[str, str] = {}
        for source in SOURCES:
            subset = stay_matches.loc[
                stay_matches["measurement_source"].eq(source)
            ]
            source_summaries[source] = ";".join(
                (
                    f"{int(row.measurement_offset)}:{row.measurement_value}"
                    f"(d={row.time_difference_minutes:+g})"
                )
                for row in subset.itertuples(index=False)
            )
        if not stay_episodes.empty:
            classification = "|".join(
                sorted(
                    {
                        classify_independently(
                            stay_matches.loc[
                                stay_matches["episode_id"].eq(episode_id)
                            ]
                        )[0]
                        for episode_id in stay_episodes["episode_id"]
                    }
                )
            )
        rows.append(
            {
                "patientunitstayid": stay_id,
                "raw_strict_low_reading_count_without_stay_filter": len(raw_low),
                "raw_strict_low_reading_count_inside_icu_stay": len(
                    raw_low_in_stay
                ),
                "raw_strict_low_reading_offsets_excluded_by_stay_filter": ";".join(
                    f"{int(row.observationoffset)}:{row.sao2:g}"
                    for row in raw_outside_offsets.itertuples(index=False)
                ),
                "strict_low_reading_count": len(stay_readings),
                "strict_low_reading_offsets_values": low_values,
                "pao2_matches_offset_value_delta": source_summaries["abg_pao2"],
                "abg_o2_saturation_matches_offset_value_delta": source_summaries[
                    "abg_o2_sat"
                ],
                "respiratory_sao2_matches_offset_value_delta": source_summaries[
                    "resp_sao2"
                ],
                "map_context_matches_offset_value_delta": source_summaries[
                    "map_context"
                ],
                "episode_count": len(stay_episodes),
                "independent_evidence_conclusion": classification,
            }
        )
    return pd.DataFrame(rows)


def make_threshold_report(periodic: pd.DataFrame) -> pd.DataFrame:
    nonzero = periodic.loc[periodic["sao2"].gt(0)]
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        selected = nonzero.loc[nonzero["sao2"].le(threshold)]
        rows.append(
            {
                "threshold": threshold,
                "comparator": "<=",
                "eligible_raw_readings": len(selected),
                "distinct_stays": selected["patientunitstayid"].nunique(),
                "percentage_of_all_eligible_nonzero_spo2": (
                    100 * len(selected) / len(nonzero) if len(nonzero) else 0
                ),
                "equal_to_threshold_count": int(
                    nonzero["sao2"].eq(threshold).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    protected_files = (
        PROTECTED_PATHS
        + RAW_PATHS
        + list(OUTPUTS.values())
    )
    before_hashes = fingerprint(protected_files)

    readings_out = pd.read_csv(OUTPUTS["readings"], low_memory=False)
    matches_out = pd.read_csv(OUTPUTS["matches"], low_memory=False)
    episodes_out = pd.read_csv(OUTPUTS["episodes"], low_memory=False)
    summary_out = pd.read_csv(OUTPUTS["summary"], low_memory=False)
    policy_text = OUTPUTS["policy"].read_text(encoding="utf-8")

    stays, periodic, references, source_stats = load_raw_data()
    raw_known_periodic = pd.read_csv(
        DATA_DIR / "vitalPeriodic.csv.gz",
        usecols=[
            "vitalperiodicid",
            "patientunitstayid",
            "observationoffset",
            "sao2",
        ],
        low_memory=False,
    )
    raw_known_periodic = raw_known_periodic.loc[
        raw_known_periodic["patientunitstayid"].isin(KNOWN_STAYS)
    ].copy()
    eligible = periodic.loc[
        periodic["sao2"].gt(0) & periodic["sao2"].lt(LOW_LIMIT)
    ].copy()
    eligible, independent_episodes = rebuild_episodes(eligible)
    eligible["reading_id"] = eligible["vitalperiodicid"].map(canonical_id)
    readings_out["reading_id"] = readings_out["reading_id"].map(canonical_id)
    independent_matches = independently_match(eligible, references)
    if independent_matches.empty:
        independent_matches = pd.DataFrame(
            columns=[
                "reading_id",
                "episode_id",
                "patientunitstayid",
                "spo2_offset",
                "spo2",
                "measurement_source",
                "measurement_id",
                "measurement_label",
                "measurement_offset",
                "measurement_value",
                "raw_measurement_value",
                "time_difference_minutes",
                "units",
                "units_interface",
                "measurement_parse_status",
            ]
        )

    checks: list[dict[str, Any]] = []
    discrepancies: list[dict[str, Any]] = []

    reading_checks, matched_reading_ids = compare_readings(
        eligible,
        readings_out,
        discrepancies,
    )
    checks.extend(reading_checks)
    check_read = readings_out.loc[
        readings_out["spo2"].eq(50)
        | readings_out["spo2"].eq(0)
        | readings_out["spo2"].lt(0)
        | readings_out["spo2"].isna()
    ]
    add_check(
        checks,
        "Strict <50 boundary and positive-only population",
        "Every detailed row has 0 < SpO2 < 50; no SpO2 == 50",
        f"invalid rows={len(check_read)}; raw eligible={len(eligible)}",
        check_read.empty,
        "Independent raw population is reconstructed from in-stay observations.",
    )
    add_check(
        checks,
        "Distinct valid ICU stay identities",
        "All raw and output stay IDs exist in patient.csv.gz",
        f"unknown output stays={readings_out.loc[~readings_out.patientunitstayid.isin(stays.patientunitstayid), 'patientunitstayid'].nunique()}",
        readings_out["patientunitstayid"].isin(
            stays["patientunitstayid"]
        ).all(),
        "Stay membership was checked against patient.csv.gz.",
    )
    add_check(
        checks,
        "Duplicate raw and output observation IDs",
        "No duplicate IDs in source or detailed output",
        (
            f"raw duplicates={int(eligible['vitalperiodicid'].duplicated().sum())}; "
            f"output duplicates={int(readings_out['reading_id'].duplicated().sum())}"
        ),
        not eligible["vitalperiodicid"].duplicated().any()
        and not readings_out["reading_id"].duplicated().any(),
        "Unique source vitalperiodicid values were compared, not just row counts.",
    )

    threshold_report = make_threshold_report(periodic)
    threshold_audit = summary_out.set_index("metric")["value"]
    threshold_passes = True
    threshold_episode_counts: dict[int, int] = {}
    exact_50 = int((periodic["sao2"].gt(0) & periodic["sao2"].eq(50)).sum())
    strict_50 = int((periodic["sao2"].gt(0) & periodic["sao2"].lt(50)).sum())
    inclusive_50 = int((periodic["sao2"].gt(0) & periodic["sao2"].le(50)).sum())
    for threshold in THRESHOLDS:
        independent_count = int(
            (periodic["sao2"].gt(0) & periodic["sao2"].le(threshold)).sum()
        )
        audit_count = int(
            float(threshold_audit.get(f"low_readings_le_{threshold}", -1))
        )
        exact_at_threshold = int(
            (periodic["sao2"].gt(0) & periodic["sao2"].eq(threshold)).sum()
        )
        threshold_readings = periodic.loc[
            periodic["sao2"].gt(0) & periodic["sao2"].le(threshold)
        ].copy()
        _, threshold_episodes = rebuild_episodes(threshold_readings)
        independent_episode_count = len(threshold_episodes)
        threshold_episode_counts[threshold] = independent_episode_count
        audit_episode_count = int(
            float(
                threshold_audit.get(
                    f"episodes_with_readings_le_{threshold}",
                    -1,
                )
            )
        )
        passed = (
            independent_count == audit_count
            and independent_episode_count == audit_episode_count
        )
        threshold_passes &= passed
        if not passed:
            discrepancy(
                discrepancies,
                "INCLUSIVE_THRESHOLD_MISMATCH",
                f"<= {threshold}",
                independent_count,
                audit_count,
                "Audit summary threshold count differs from raw nonzero in-stay data.",
            )
        if independent_episode_count != audit_episode_count:
            discrepancy(
                discrepancies,
                "INCLUSIVE_THRESHOLD_EPISODE_COUNT",
                f"episodes_with_readings_le_{threshold}",
                independent_episode_count,
                audit_episode_count,
                "Threshold episode count must group that threshold's raw qualifying readings using the configured 10-minute gap.",
            )
        threshold_report.loc[
            threshold_report["threshold"].eq(threshold),
            "audit_summary_count",
        ] = audit_count
        threshold_report.loc[
            threshold_report["threshold"].eq(threshold),
            "count_matches_audit",
        ] = passed
        threshold_report.loc[
            threshold_report["threshold"].eq(threshold),
            "equal_to_threshold_count",
        ] = exact_at_threshold
        threshold_report.loc[
            threshold_report["threshold"].eq(threshold),
            "independent_episode_count",
        ] = independent_episode_count
        threshold_report.loc[
            threshold_report["threshold"].eq(threshold),
            "audit_episode_count",
        ] = audit_episode_count
        threshold_report.loc[
            threshold_report["threshold"].eq(threshold),
            "episode_count_matches_audit",
        ] = independent_episode_count == audit_episode_count
    add_check(
        checks,
        "Inclusive thresholds <=20/30/40/50",
        "Independent counts equal audit summary using positive in-stay raw readings",
        threshold_report[
            ["threshold", "eligible_raw_readings", "audit_summary_count"]
        ].to_dict("records"),
        threshold_passes,
        "Reading totals and threshold-specific episode groups are independently recomputed; zeros are excluded.",
    )
    strict_audit_readings = int(
        float(threshold_audit.get("eligible_low_readings_lt_50", -1))
    )
    strict_audit_stays = int(
        float(threshold_audit.get("unique_stays_with_eligible_low_readings", -1))
    )
    strict_audit_episodes = int(
        float(threshold_audit.get("episodes_with_readings_lt_50", -1))
    )
    add_check(
        checks,
        "Strict <50 population and episode count",
        "374 readings, 103 stays, 181 episodes",
        (
            f"independent={len(eligible)} readings/{eligible['patientunitstayid'].nunique()} stays/"
            f"{len(independent_episodes)} episodes; audit={strict_audit_readings} readings/"
            f"{strict_audit_stays} stays/{strict_audit_episodes} episodes"
        ),
        (
            len(eligible) == 374
            and eligible["patientunitstayid"].nunique() == 103
            and len(independent_episodes) == 181
            and strict_audit_readings == len(eligible)
            and strict_audit_stays == eligible["patientunitstayid"].nunique()
            and strict_audit_episodes == len(independent_episodes)
        ),
        "Strict detail population stays unchanged; episode grouping uses the configured 10-minute gap.",
    )
    inclusive_audit_episodes = int(
        float(threshold_audit.get("episodes_with_readings_le_50", -1))
    )
    add_check(
        checks,
        "Inclusive <=50 population and episode count",
        "408 readings and 192 threshold-specific episodes",
        (
            f"independent={inclusive_50} readings/{threshold_episode_counts[50]} episodes; "
            f"audit={int(float(threshold_audit.get('low_readings_le_50', -1)))} readings/"
            f"{inclusive_audit_episodes} episodes"
        ),
        (
            inclusive_50 == 408
            and threshold_episode_counts[50] == 192
            and int(float(threshold_audit.get("low_readings_le_50", -1)))
            == inclusive_50
            and inclusive_audit_episodes == threshold_episode_counts[50]
        ),
        "Inclusive threshold episodes are grouped from all positive in-stay readings <=50, separately from strict detail episodes.",
    )
    add_check(
        checks,
        "Exactly-50 and inclusive/strict reconciliation",
        "34 readings exactly 50; 408 inclusive = 374 strict + 34 exact-50",
        f"exact-50={exact_50}; inclusive={inclusive_50}; strict={strict_50}; difference={inclusive_50 - strict_50}",
        (
            exact_50 == 34
            and strict_50 == 374
            and inclusive_50 == 408
            and inclusive_50 - strict_50 == exact_50
        ),
        "All counts use the same ICU stay and in-stay offset population, excluding zero.",
    )
    add_check(
        checks,
        "Difference between strict <50 and inclusive <=50",
        f"{inclusive_50} - {strict_50} = {exact_50} exact-50 readings",
        f"{inclusive_50} - {strict_50} = {inclusive_50 - strict_50}; raw exact 50={exact_50}",
        inclusive_50 - strict_50 == exact_50 == 34,
        "All counts use the same ICU stay and in-stay offset population, excluding zero.",
    )

    hourly = rebuild_hourly(periodic)
    zero_count = int(periodic["sao2"].eq(0).sum())
    mixed_hours = int(hourly["quality_flag"].eq("MIXED_ZERO").sum())
    zero_only_hours = int(hourly["quality_flag"].eq("ZERO_ONLY").sum())
    zero_only_with_mean = int(
        (
            hourly["quality_flag"].eq("ZERO_ONLY")
            & hourly["zero_excluded_mean"].notna()
        ).sum()
    )
    summary_map = summary_out.set_index("metric")["value"]
    hourly_out_path = AUDIT_DIR / "spo2_hourly_zero_quality.csv"
    hourly_out = pd.read_csv(hourly_out_path, low_memory=False)
    hourly_merge = hourly.merge(
        hourly_out,
        on=["patientunitstayid", "hour"],
        how="outer",
        suffixes=("_independent", "_audit"),
        indicator=True,
        validate="one_to_one",
    )
    hourly_keys_equal = hourly_merge["_merge"].eq("both").all()
    hourly_numeric_fields = [
        ("raw_mean_independent", "raw_mean_audit"),
        ("zero_excluded_mean_independent", "zero_excluded_mean_audit"),
        ("valid_nonzero_count_independent", "valid_nonzero_count_audit"),
        ("zero_count_independent", "zero_count_audit"),
        ("missing_count_independent", "missing_count_audit"),
    ]
    hourly_value_matches = True
    for left, right in hourly_numeric_fields:
        hourly_value_matches &= bool(
            np.isclose(
                pd.to_numeric(hourly_merge[left], errors="coerce"),
                pd.to_numeric(hourly_merge[right], errors="coerce"),
                rtol=0,
                atol=1e-9,
                equal_nan=True,
            ).all()
        )
    quality_matches = hourly_merge["quality_flag_independent"].eq(
        hourly_merge["quality_flag_audit"]
    ).all()
    checks.append(
        {
            "check_name": "Zero handling and patient-hour recomputation",
            "expected_result": (
                "Exact raw counts 1,554 zeros, 240 mixed-zero hours, "
                "47 zero-only hours; candidate means exclude zero"
            ),
            "independently_calculated_result": (
                f"zeros={zero_count}; mixed={mixed_hours}; zero_only={zero_only_hours}; "
                f"zero_only_numeric_mean={zero_only_with_mean}"
            ),
            "status": (
                "PASS"
                if zero_count == 1554
                and mixed_hours == 240
                and zero_only_hours == 47
                and zero_only_with_mean == 0
                and hourly_keys_equal
                and hourly_value_matches
                and quality_matches
                else "FAIL"
            ),
            "critical": True,
            "explanation": (
                f"hourly keys complete={hourly_keys_equal}; values/counts match={hourly_value_matches}; "
                f"quality flags match={quality_matches}"
            ),
        }
    )
    for metric, actual, expected in [
        ("raw_zero_spo2_rows", zero_count, 1554),
        ("mixed_zero_patient_hours", mixed_hours, 240),
        ("zero_only_patient_hours", zero_only_hours, 47),
    ]:
        audit_value = int(float(summary_map.get(metric, -1)))
        add_check(
            checks,
            f"Audit summary {metric}",
            expected,
            f"independent={actual}; audit={audit_value}",
            actual == audit_value == expected,
            "Recomputed from eligible in-stay raw periodic records.",
        )

    match_ok, expected_match_count, actual_match_count = validate_match_output(
        independent_matches,
        matches_out,
        discrepancies,
    )
    add_check(
        checks,
        "Cross-source primary-window match completeness and fidelity",
        "Exact independently reconstructed source-match set and source metadata",
        (
            f"independent={expected_match_count}; audit={actual_match_count}; "
            f"omitted={len(set(map(tuple, independent_matches[['reading_id','measurement_source','measurement_id']].itertuples(index=False,name=None))) - set(map(tuple, matches_out[['reading_id','measurement_source','measurement_id']].itertuples(index=False,name=None))))}"
        ),
        match_ok,
        "Raw labels, same-stay key, in-stay eligibility, offsets, values, units, and ±15 minute deltas were independently reconstructed.",
    )
    add_check(
        checks,
        "No match outside inclusive ±15 minutes",
        "Every retained match has absolute time difference <=15 minutes",
        (
            f"independent outside={int(independent_matches['time_difference_minutes'].abs().gt(MATCH_WINDOW).sum())}; "
            f"audit outside={int(matches_out['time_difference_minutes'].abs().gt(MATCH_WINDOW).sum())}"
        ),
        independent_matches["time_difference_minutes"].abs().le(
            MATCH_WINDOW
        ).all()
        and matches_out["time_difference_minutes"].abs().le(
            MATCH_WINDOW
        ).all(),
        "Offsets use minutes and signed delta = reference offset - SpO2 offset.",
    )
    source_counts_expected = independent_matches["measurement_source"].value_counts()
    source_counts_audit = matches_out["measurement_source"].value_counts()
    for source in SOURCES:
        left = int(source_counts_expected.get(source, 0))
        right = int(source_counts_audit.get(source, 0))
        add_check(
            checks,
            f"Source match count {source}",
            left,
            right,
            left == right,
            (
                "MAP is context only."
                if source == "map_context"
                else "PaO2, lab O2 saturation, respiratory SaO2 remain distinct sources."
            ),
        )

    episodes_ok, episode_metric_mismatches = validate_episode_output(
        independent_episodes,
        episodes_out,
        discrepancies,
    )
    add_check(
        checks,
        "Episode boundaries, metrics, and traceability",
        f"{len(independent_episodes)} episodes reconstructed from raw readings at max gap {EPISODE_GAP} minutes",
        (
            f"output episodes={len(episodes_out)}; metric mismatches={episode_metric_mismatches}; "
            f"all source IDs trace={episodes_ok}"
        ),
        episodes_ok,
        "Rebuilt independently by sorting raw stay/offset/source ID and splitting only when gap >10 minutes.",
    )
    audit_classifications = episodes_out.set_index("episode_id")[
        "evidence_classification"
    ]
    independent_classifications: dict[str, str] = {}
    for episode_id in independent_episodes["episode_id"]:
        episode_matches = independent_matches.loc[
            independent_matches["episode_id"].eq(episode_id)
        ]
        independent_classifications[episode_id] = classify_independently(
            episode_matches
        )[0]
    classification_differences = []
    for episode_id, independent_class in independent_classifications.items():
        if episode_id not in audit_classifications.index:
            continue
        audit_class = audit_classifications.loc[episode_id]
        if independent_class != audit_class:
            classification_differences.append(
                (episode_id, independent_class, audit_class)
            )
            discrepancy(
                discrepancies,
                "CLASSIFICATION_MISMATCH",
                episode_id,
                independent_class,
                audit_class,
                "Independent category differs using the documented provisional rules.",
            )
    add_check(
        checks,
        "Evidence classification recomputation",
        "Classifications match independent direct-saturation/reference rules",
        (
            f"independent={pd.Series(independent_classifications).value_counts().to_dict()}; "
            f"audit={audit_classifications.value_counts().to_dict()}; "
            f"mismatches={len(classification_differences)}"
        ),
        len(classification_differences) == 0,
        "MAP alone is context; PaO2 is not converted to saturation; missing evidence is not invalid.",
    )
    map_supported = episodes_out.loc[
        episodes_out["evidence_classification"].eq("SUPPORTED")
        & ~episodes_out["direct_evidence_sources"].fillna("").str.contains(
            "abg_o2_sat|resp_sao2",
            regex=True,
        )
    ]
    add_check(
        checks,
        "MAP alone does not produce SUPPORTED",
        "Zero SUPPORTED episodes without direct saturation evidence",
        f"violating episodes={len(map_supported)}",
        map_supported.empty,
        "A PaO2 reference is contextual and cannot be translated to a pulse-oximeter value.",
    )

    policy_checks = {
        "zero policy": "exclude exact-zero" in policy_text.casefold()
        or "exclude exact-zero" in policy_text.casefold(),
        "retain nonzero lows": "preserve all nonzero" in policy_text.casefold(),
        "do not delete lows": "do not" in policy_text.casefold()
        and "delete or cap" in policy_text.casefold(),
        "missing not invalid": "missing corroboration is not evidence" in policy_text.casefold(),
        "MAP contextual": "map as context" in policy_text.casefold(),
        "demo limitation": "demo" in policy_text.casefold(),
        "not clinical truth": "clinical" in policy_text.casefold(),
        "classification categories": all(
            category in policy_text
            for category in [
                "SUPPORTED",
                "CONFLICTING",
                "NO_INDEPENDENT_EVIDENCE",
                "INSUFFICIENT_DATA",
            ]
        ),
    }
    add_check(
        checks,
        "Documented quality policy covers required limitations",
        "All policy terms and categories present",
        policy_checks,
        all(policy_checks.values()),
        "Documentation review is a content check, not clinical adjudication.",
        critical=False,
    )

    # Validate that the project's protected scripts, raw sources, and original
    # audit outputs were not altered while this validator ran.
    after_hashes = fingerprint(protected_files)
    changed = [
        path
        for path, before in before_hashes.items()
        if after_hashes.get(path) != before
    ]
    add_check(
        checks,
        "Protected project files unchanged during validation",
        "No changes to raw sources, production scripts, or original audit outputs",
        changed,
        not changed,
        "SHA-256 fingerprints were compared before and after independent validation.",
    )

    threshold_report["inclusive_threshold_episode_count_independent"] = [
        threshold_episode_counts[threshold] for threshold in THRESHOLDS
    ]
    threshold_report.to_csv(
        VALIDATION_DIR / "spo2_validation_thresholds.csv",
        index=False,
    )

    # Independent episode metrics plus source-coverage counts from independent
    # match rows make this file a side-by-side comparison artifact.
    for source, output_name in [
        ("abg_pao2", "pao2_measurement_count"),
        ("abg_o2_sat", "abg_o2_sat_measurement_count"),
        ("resp_sao2", "resp_sao2_measurement_count"),
        ("map_context", "map_measurement_count"),
    ]:
        counts = independent_matches.loc[
            independent_matches["measurement_source"].eq(source)
        ].drop_duplicates(["episode_id", "measurement_id"]).groupby(
            "episode_id"
        ).size()
        independent_episodes[output_name] = (
            independent_episodes["episode_id"].map(counts).fillna(0).astype(int)
        )
    independent_episodes["independent_classification"] = [
        classify_independently(
            independent_matches.loc[
                independent_matches["episode_id"].eq(episode_id)
            ]
        )[0]
        for episode_id in independent_episodes["episode_id"]
    ]
    independent_episodes.to_csv(
        VALIDATION_DIR / "spo2_validation_episodes.csv",
        index=False,
    )
    known_cases = known_case_review(
        eligible,
        independent_matches,
        independent_episodes.assign(
            evidence_classification=independent_episodes[
                "independent_classification"
            ]
        ),
        raw_known_periodic,
        stays,
    )
    known_cases.to_csv(
        VALIDATION_DIR / "spo2_validation_known_cases.csv",
        index=False,
    )

    # Specific checks for known examples, using raw-source evidence only.
    known_expectations = {
        3157219: (1, 1),
        3204052: (1, 1),
        3067309: (1, 1),
        758327: (1, 0),
        1229669: (1, 0),
        939222: (0, 0),
    }
    for stay_id, (expects_low, expects_direct) in known_expectations.items():
        case_readings = eligible.loc[
            eligible["patientunitstayid"].eq(stay_id)
        ]
        case_matches = independent_matches.loc[
            independent_matches["patientunitstayid"].eq(stay_id)
        ]
        direct_count = int(
            case_matches["measurement_source"].isin(
                ["abg_o2_sat", "resp_sao2"]
            ).sum()
        )
        if stay_id == 939222:
            case_raw = raw_known_periodic.loc[
                raw_known_periodic["patientunitstayid"].eq(stay_id)
            ].copy()
            case_raw["sao2"] = pd.to_numeric(case_raw["sao2"], errors="coerce")
            raw_low = case_raw.loc[
                case_raw["sao2"].gt(0) & case_raw["sao2"].lt(50)
            ]
            discharge = stays.loc[
                stays["patientunitstayid"].eq(stay_id),
                "unitdischargeoffset",
            ].iloc[0]
            all_after_discharge = raw_low["observationoffset"].gt(
                discharge
            ).all()
            passed = (
                len(raw_low) > 0
                and len(case_readings) == 0
                and bool(all_after_discharge)
            )
            actual_case = (
                f"raw low={len(raw_low)}; eligible={len(case_readings)}; "
                f"discharge={discharge}; all raw lows after discharge={all_after_discharge}"
            )
        else:
            passed = (
                (bool(len(case_readings)) == bool(expects_low))
                and (
                    direct_count > 0
                    if expects_direct
                    else direct_count == 0
                )
            )
            actual_case = (
                f"low readings={len(case_readings)}; direct match rows={direct_count}; "
                f"MAP matches={int(case_matches['measurement_source'].eq('map_context').sum())}"
            )
        add_check(
            checks,
            f"Known stay {stay_id} independent evidence review",
            (
                "raw low readings are outside ICU stay and correctly excluded"
                if stay_id == 939222
                else f"low readings present={bool(expects_low)}; direct matches expected={expects_direct}"
            ),
            actual_case,
            passed,
            "See spo2_validation_known_cases.csv for raw offsets, values, and deltas.",
            critical=False,
        )

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(
        VALIDATION_DIR / "spo2_validation_checks.csv",
        index=False,
    )
    if discrepancies:
        discrepancy_frame = pd.DataFrame(discrepancies)
    else:
        discrepancy_frame = pd.DataFrame(
            columns=[
                "category",
                "record_key",
                "expected",
                "actual",
                "explanation",
            ]
        )
    discrepancy_frame.to_csv(
        VALIDATION_DIR / "spo2_validation_discrepancies.csv",
        index=False,
    )

    failures = checks_frame.loc[checks_frame["status"].eq("FAIL")]
    critical_failures = failures.loc[failures["critical"]]
    not_verified = checks_frame.loc[
        checks_frame["status"].eq("NOT VERIFIED")
    ]
    if len(critical_failures):
        verdict = "FAIL — CORRECTIONS REQUIRED"
    elif len(failures) or len(not_verified):
        verdict = "PASS WITH LIMITATIONS"
    else:
        verdict = "PASS — INDEPENDENTLY VALIDATED"
    report = make_report(
        verdict,
        checks_frame,
        discrepancy_frame,
        threshold_report,
        hourly,
        periodic,
        eligible,
        independent_episodes,
        known_cases,
        source_stats,
        expected_match_count=len(independent_matches),
        audit_match_count=len(matches_out),
        exact_50=exact_50,
        inclusive_episodes=threshold_episode_counts[50],
        audit_inclusive_episodes=inclusive_audit_episodes,
    )
    (VALIDATION_DIR / "spo2_validation_report.md").write_text(
        report,
        encoding="utf-8",
    )

    print(verdict)
    print(
        "Checks: "
        f"PASS={int(checks_frame.status.eq('PASS').sum())}, "
        f"FAIL={int(checks_frame.status.eq('FAIL').sum())}, "
        f"NOT VERIFIED={int(checks_frame.status.eq('NOT VERIFIED').sum())}"
    )
    print(
        f"Strict low population: {len(eligible):,} readings, "
        f"{eligible['patientunitstayid'].nunique():,} stays, "
        f"{len(independent_episodes):,} episodes"
    )
    print(
        f"Primary matches: independent={len(independent_matches):,}, "
        f"audit={len(matches_out):,}; discrepancies={len(discrepancy_frame):,}"
    )
    print(f"Validation outputs: {VALIDATION_DIR}")


def make_report(
    verdict: str,
    checks: pd.DataFrame,
    discrepancies: pd.DataFrame,
    thresholds: pd.DataFrame,
    hourly: pd.DataFrame,
    periodic: pd.DataFrame,
    readings: pd.DataFrame,
    episodes: pd.DataFrame,
    known_cases: pd.DataFrame,
    source_stats: dict[str, dict[str, int]],
    expected_match_count: int,
    audit_match_count: int,
    exact_50: int,
    inclusive_episodes: int,
    audit_inclusive_episodes: int,
) -> str:
    n_pass = int(checks["status"].eq("PASS").sum())
    n_fail = int(checks["status"].eq("FAIL").sum())
    n_unknown = int(checks["status"].eq("NOT VERIFIED").sum())
    class_counts = episodes["independent_classification"].value_counts()
    checks_table = checks[
        ["check_name", "status", "critical", "explanation"]
    ]
    checks_table = markdown_table(checks_table)
    thresholds_table = markdown_table(thresholds)
    cases_table = markdown_table(known_cases)
    discrepancy_summary = (
        discrepancies["category"].value_counts().to_dict()
        if not discrepancies.empty
        else {}
    )
    critical_fails = checks.loc[
        checks["status"].eq("FAIL") & checks["critical"],
        "check_name",
    ].tolist()
    return f"""# Independent SpO2 audit validation

## Executive verdict

**{verdict}**

Independent raw-data checks: {n_pass} PASS, {n_fail} FAIL, {n_unknown} NOT
VERIFIED. Critical failed checks: {critical_fails or "none"}.

## Methodology

This validator reads the raw eICU demo data directly and does not import the
audit script or use its calculated tables as a source of truth. ICU stay
eligibility is established from `patient.csv.gz`; each source observation must
have a numeric offset from zero through the stay's discharge offset. The
detailed population is independently rebuilt as numeric, nonzero
`vitalPeriodic.sao2 < 50`. Raw source identifiers are compared against the
detailed output.

For each detailed reading, the validator independently filters same-stay,
in-stay ABG PaO2, lab O2 saturation, respiratory SaO2, and non-invasive MAP
records within inclusive ±15 minutes. All matches are retained and compared by
source ID, measurement type, value, units, offset, and signed minute delta.
Episodes are independently rebuilt after sorting by stay, minute offset, and
periodic observation ID, splitting only when consecutive low readings differ
by more than 10 minutes.

## Independent results

- In-stay periodic observations: {len(periodic):,}; numeric SpO2 observations:
  {int(periodic["sao2"].notna().sum()):,}.
- Detailed strict-low population: {len(readings):,} readings across
  {readings["patientunitstayid"].nunique():,} stays and {len(episodes):,}
  episodes.
- Inclusive threshold counts:

{thresholds_table}

- Exactly-50 observations explain the strict/inclusive difference:
  independent `<50` = {int((periodic["sao2"].gt(0) & periodic["sao2"].lt(50)).sum()):,};
  independent `<=50` = {int((periodic["sao2"].gt(0) & periodic["sao2"].le(50)).sum()):,};
  exactly 50 = {exact_50:,}.
- Zero handling: {int(periodic["sao2"].eq(0).sum()):,} exact-zero observations,
  {int(hourly["quality_flag"].eq("MIXED_ZERO").sum()):,} mixed-zero hours,
  {int(hourly["quality_flag"].eq("ZERO_ONLY").sum()):,} zero-only hours.
  Zero-only candidate means remain missing.
- Matching: independently found {expected_match_count:,} source matches; audit
  output contains {audit_match_count:,}. Details are in the discrepancy file.
- Independent episode evidence classifications:
  {class_counts.to_dict()}.

## PASS/FAIL/NOT VERIFIED checks

{checks_table}

## Discrepancies

Total discrepancy records: {len(discrepancies)}. Categories:
`{discrepancy_summary}`.
{markdown_table(discrepancies) if not discrepancies.empty else "No record-level discrepancies were found."}

Inclusive-threshold episodes were independently grouped from all positive
in-stay readings `<=50` using the documented 10-minute gap: this yields
{inclusive_episodes} episodes, and the audit threshold summary reports
{audit_inclusive_episodes}. This is separate from the detailed strict `<50`
population, which remains at 181 episodes.

## Known-case review

{cases_table}

Stay 939222 is additionally checked because it is mentioned in the validation
request. Conclusions are limited to the raw in-window evidence available in
this demo extract.

## Remaining limitations

The validator verifies data lineage, arithmetic, joins, offsets, grouping,
and report consistency. It cannot establish clinical truth. The eICU demo
subset is limited; charted measurements may be incomplete, temporally offset,
or not independent. The support/conflict bands are provisional data-quality
comparison rules, not clinical criteria. PaO2 is not interchangeable with
SpO2, and MAP is contextual only. A no-reference episode means no qualifying
measurement was found in the window, not that the SpO2 is invalid.

## Final recommendation

{verdict}. Preserve exact zeros as quality metadata while excluding them from
the candidate mean, retain low nonzero observations, and do not promote
retrospective labels or preprocessing changes without review and independent
clinical/device validation. No later project milestone was performed.
"""


if __name__ == "__main__":
    main()
