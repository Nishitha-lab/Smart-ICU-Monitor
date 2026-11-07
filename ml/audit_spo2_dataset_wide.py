from pathlib import Path
import re
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
OUTPUT_DIR = ROOT / "data" / "processed" / "spo2_audit"

MATCH_WINDOW_MINUTES = 15
EPISODE_GAP_MINUTES = 10
LOW_SPO2_MAX = 50
LOW_SPO2_MAX_INCLUSIVE = False
SUPPORT_SATURATION_MAX = 50
CONFLICT_SATURATION_MIN = 70
THRESHOLDS = (20, 30, 40, 50)

KNOWN_STAYS = (3157219, 3204052, 3067309, 758327, 1229669)

LAB_NAME_PATTERNS = {
    "abg_pao2": re.compile(r"\bpa\s*o\s*2\b", flags=re.IGNORECASE),
    "abg_o2_sat": re.compile(
        r"\bo\s*2\s*sat(?:uration)?(?:\s*\(%\))?(?![a-z])|"
        r"\bsao\s*2\b|\boxygen\s*saturation\b",
        flags=re.IGNORECASE,
    ),
}


def parse_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan
    match = re.fullmatch(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        text,
    )
    return float(match.group(0)) if match else np.nan


def load_patients() -> pd.DataFrame:
    patients = pd.read_csv(
        DATA_DIR / "patient.csv.gz",
        usecols=["patientunitstayid", "unitdischargeoffset"],
        low_memory=False,
    )
    patients["unitdischargeoffset"] = pd.to_numeric(
        patients["unitdischargeoffset"],
        errors="coerce",
    )
    patients = patients.dropna(
        subset=["patientunitstayid", "unitdischargeoffset"]
    )
    if patients["patientunitstayid"].duplicated().any():
        raise ValueError("patient.csv.gz has duplicate ICU stay identifiers.")
    return patients


def load_in_stay(
    filename: str,
    columns: list[str],
    offset_column: str,
    patients: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    data = pd.read_csv(
        DATA_DIR / filename,
        usecols=columns,
        low_memory=False,
    )
    data[offset_column] = pd.to_numeric(
        data[offset_column],
        errors="coerce",
    )
    patient_ids = set(patients["patientunitstayid"])
    unknown_stays = int((~data["patientunitstayid"].isin(patient_ids)).sum())
    valid_offset = data[offset_column].notna()
    known_patient = data["patientunitstayid"].isin(patient_ids)
    in_stay = data.merge(
        patients,
        on="patientunitstayid",
        how="inner",
        validate="many_to_one",
    )
    within_interval = in_stay[offset_column].between(
        0,
        in_stay["unitdischargeoffset"],
        inclusive="both",
    )
    retained = in_stay.loc[within_interval].copy()
    stats = {
        "raw_rows": len(data),
        "unknown_stay_rows": unknown_stays,
        "malformed_offset_rows": int((~valid_offset).sum()),
        "known_stay_rows": int(known_patient.sum()),
        "outside_icu_interval_rows": int(
            (known_patient & valid_offset).sum() - len(retained)
        ),
    }
    return retained, stats


def parse_lab_results(labs: pd.DataFrame) -> pd.DataFrame:
    labs = labs.copy()
    numeric_result = labs["labresult"].map(parse_number)
    numeric_text = labs["labresulttext"].map(parse_number)
    use_text = numeric_result.isna() & numeric_text.notna()
    labs["measurement_value"] = numeric_result.where(
        numeric_result.notna(),
        numeric_text,
    )
    labs["parsed_from"] = np.select(
        [numeric_result.notna(), use_text],
        ["labresult", "labresulttext"],
        default="unparsed",
    )
    raw_result_present = (
        labs["labresult"].notna()
        & labs["labresult"].astype(str).str.strip().ne("")
    )
    raw_text_present = (
        labs["labresulttext"].notna()
        & labs["labresulttext"].astype(str).str.strip().ne("")
    )
    labs["measurement_parse_status"] = np.select(
        [
            numeric_result.notna(),
            use_text,
            ~raw_result_present & ~raw_text_present,
        ],
        ["NUMERIC", "TEXT_FALLBACK", "MISSING"],
        default="MALFORMED",
    )
    labs["raw_measurement_value"] = labs["labresult"].astype("string")
    labs.loc[
        labs["labresult"].isna() | labs["labresult"].astype(str).str.strip().eq(""),
        "raw_measurement_value",
    ] = labs["labresulttext"].astype("string")
    labs["measurement_source"] = labs["lab_evidence_type"]
    labs["measurement_offset"] = labs["labresultoffset"]
    labs["measurement_label"] = labs["labname"]
    labs["measurement_id"] = labs["labid"]
    labs["units"] = labs["labmeasurenamesystem"].astype("string")
    labs["units_interface"] = labs["labmeasurenameinterface"].astype("string")
    labs["raw_measurement_text"] = labs["labresulttext"].astype("string")
    return labs


def prepare_references(
    patients: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    lab_columns = [
        "labid",
        "patientunitstayid",
        "labresultoffset",
        "labname",
        "labresult",
        "labresulttext",
        "labmeasurenamesystem",
        "labmeasurenameinterface",
    ]
    labs, lab_stats = load_in_stay(
        "lab.csv.gz",
        lab_columns,
        "labresultoffset",
        patients,
    )
    names = labs["labname"].astype("string").fillna("")
    pao2_mask = names.str.contains(
        LAB_NAME_PATTERNS["abg_pao2"],
        na=False,
    )
    saturation_mask = names.str.contains(
        LAB_NAME_PATTERNS["abg_o2_sat"],
        na=False,
    )
    labs["lab_evidence_type"] = np.select(
        [pao2_mask, saturation_mask],
        ["abg_pao2", "abg_o2_sat"],
        default="",
    )
    labs = parse_lab_results(
        labs.loc[pao2_mask | saturation_mask].copy()
    )
    labs["offset_valid"] = labs["measurement_offset"].notna()

    respiratory, respiratory_stats = load_in_stay(
        "respiratoryCharting.csv.gz",
        [
            "respchartid",
            "patientunitstayid",
            "respchartoffset",
            "respchartvaluelabel",
            "respchartvalue",
        ],
        "respchartoffset",
        patients,
    )
    resp_label = (
        respiratory["respchartvaluelabel"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    respiratory = respiratory.loc[resp_label.eq("sao2")].copy()
    respiratory["measurement_source"] = "resp_sao2"
    respiratory["measurement_offset"] = respiratory["respchartoffset"]
    respiratory["measurement_label"] = respiratory["respchartvaluelabel"]
    respiratory["measurement_value"] = respiratory[
        "respchartvalue"
    ].map(parse_number)
    respiratory["raw_measurement_value"] = respiratory[
        "respchartvalue"
    ].astype("string")
    respiratory["measurement_id"] = respiratory["respchartid"]
    respiratory["parsed_from"] = np.where(
        respiratory["measurement_value"].notna(),
        "respchartvalue",
        "unparsed",
    )
    resp_raw_present = (
        respiratory["respchartvalue"].notna()
        & respiratory["respchartvalue"].astype(str).str.strip().ne("")
    )
    respiratory["measurement_parse_status"] = np.where(
        respiratory["measurement_value"].notna(),
        "NUMERIC",
        np.where(resp_raw_present, "MALFORMED", "MISSING"),
    )
    respiratory["units"] = "%"
    respiratory["units_interface"] = "%"
    respiratory["raw_measurement_text"] = pd.NA
    respiratory["offset_valid"] = respiratory[
        "measurement_offset"
    ].notna()

    map_data, map_stats = load_in_stay(
        "vitalAperiodic.csv.gz",
        [
            "vitalaperiodicid",
            "patientunitstayid",
            "observationoffset",
            "noninvasivemean",
        ],
        "observationoffset",
        patients,
    )
    map_data["measurement_source"] = "map_context"
    map_data["measurement_offset"] = map_data["observationoffset"]
    map_data["measurement_label"] = "noninvasivemean"
    map_data["measurement_value"] = map_data[
        "noninvasivemean"
    ].map(parse_number)
    map_data["raw_measurement_value"] = map_data[
        "noninvasivemean"
    ].astype("string")
    map_data["measurement_id"] = map_data["vitalaperiodicid"]
    map_data["parsed_from"] = np.where(
        map_data["measurement_value"].notna(),
        "noninvasivemean",
        "unparsed",
    )
    map_raw_present = (
        map_data["noninvasivemean"].notna()
        & map_data["noninvasivemean"].astype(str).str.strip().ne("")
    )
    map_data["measurement_parse_status"] = np.where(
        map_data["measurement_value"].notna(),
        "NUMERIC",
        np.where(map_raw_present, "MALFORMED", "MISSING"),
    )
    map_data["units"] = "mmHg"
    map_data["units_interface"] = "mmHg"
    map_data["raw_measurement_text"] = pd.NA
    map_data["offset_valid"] = map_data["measurement_offset"].notna()

    refs = {
        "abg_pao2": labs.loc[labs["measurement_source"].eq("abg_pao2")].copy(),
        "abg_o2_sat": labs.loc[
            labs["measurement_source"].eq("abg_o2_sat")
        ].copy(),
        "resp_sao2": respiratory,
        "map_context": map_data,
    }
    stats = {
        "lab": lab_stats,
        "respiratory": respiratory_stats,
        "map": map_stats,
    }
    for frame in refs.values():
        if frame["measurement_id"].duplicated().any():
            raise ValueError(
                "A reference source contains duplicate measurement IDs."
            )
    return refs, stats


def assign_episodes(readings: pd.DataFrame) -> pd.DataFrame:
    readings = readings.sort_values(
        ["patientunitstayid", "observationoffset", "vitalperiodicid"],
        kind="stable",
    ).copy()
    gaps = readings.groupby("patientunitstayid")[
        "observationoffset"
    ].diff()
    starts_episode = gaps.isna() | gaps.gt(EPISODE_GAP_MINUTES)
    episode_numbers = starts_episode.groupby(
        readings["patientunitstayid"]
    ).cumsum()
    readings["episode_id"] = (
        readings["patientunitstayid"].astype(str)
        + "_"
        + episode_numbers.astype(int).astype(str).str.zfill(5)
    )
    return readings


def classify_episode(
    matches: pd.DataFrame,
) -> tuple[str, str, bool]:
    direct = matches.loc[
        matches["measurement_source"].isin(["abg_o2_sat", "resp_sao2"])
    ]
    numeric = direct.loc[direct["measurement_value"].notna()]
    values = numeric["measurement_value"]
    valid_values = values.loc[values.gt(0) & values.le(100)]
    supporting = valid_values.le(SUPPORT_SATURATION_MAX).any()
    conflicting = valid_values.ge(CONFLICT_SATURATION_MIN).any()

    if supporting and conflicting:
        return "CONFLICTING", "Direct saturation evidence is mixed.", True
    if conflicting:
        return (
            "CONFLICTING",
            "Direct saturation at or above the configurable disagreement band.",
            True,
        )
    if supporting:
        sources = sorted(numeric.loc[
            numeric["measurement_value"].gt(0)
            & numeric["measurement_value"].le(SUPPORT_SATURATION_MAX),
            "measurement_source",
        ].unique())
        return "SUPPORTED", "Low direct saturation from: " + ", ".join(sources), True
    if not numeric.empty:
        if not valid_values.empty:
            return (
                "INSUFFICIENT_DATA",
                "Direct saturation is present but within the provisional indeterminate band.",
                True,
            )
        return (
            "INSUFFICIENT_DATA",
            "Direct saturation has a numeric value outside 0–100.",
            True,
        )
    abg = matches.loc[
        matches["measurement_source"].isin(["abg_o2_sat", "abg_pao2"])
    ]
    malformed_reference = matches["measurement_parse_status"].isin(
        ["MALFORMED", "MISSING"]
    ).any()
    if not abg.empty:
        return (
            "INSUFFICIENT_DATA",
            "PaO2 is available but cannot be translated to SpO2 without additional context.",
            True,
        )
    if malformed_reference:
        return (
            "INSUFFICIENT_DATA",
            "A nearby oxygenation reference is missing or has a nonnumeric value.",
            True,
        )
    return (
        "NO_INDEPENDENT_EVIDENCE",
        "No numeric ABG O2 saturation or respiratory SaO2 in the primary window.",
        True,
    )


def nearby_rows(
    stay_refs: dict[object, tuple[pd.DataFrame, np.ndarray]],
    stay_id: object,
    offset: float,
) -> pd.DataFrame:
    stay = stay_refs.get(stay_id)
    if stay is None:
        return pd.DataFrame()
    frame, offsets = stay
    left = np.searchsorted(
        offsets,
        offset - MATCH_WINDOW_MINUTES,
        side="left",
    )
    right = np.searchsorted(
        offsets,
        offset + MATCH_WINDOW_MINUTES,
        side="right",
    )
    return frame.iloc[left:right]


def build_matches(
    readings: pd.DataFrame,
    references: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_by_source_stay: dict[
        str, dict[object, tuple[pd.DataFrame, np.ndarray]]
    ] = {}
    for source, frame in references.items():
        ref_by_source_stay[source] = {}
        valid = frame.loc[frame["measurement_offset"].notna()]
        for stay_id, stay_frame in valid.groupby("patientunitstayid"):
            stay_frame = stay_frame.sort_values(
                ["measurement_offset", "measurement_id"],
                kind="stable",
            ).reset_index(drop=True)
            ref_by_source_stay[source][stay_id] = (
                stay_frame,
                stay_frame["measurement_offset"].to_numpy(dtype=float),
            )

    match_rows: list[dict[str, Any]] = []
    reading_rows: list[dict[str, Any]] = []
    for reading in readings.itertuples(index=False):
        per_source: dict[str, pd.DataFrame] = {}
        for source in references:
            found = nearby_rows(
                ref_by_source_stay[source],
                reading.patientunitstayid,
                reading.observationoffset,
            )
            if "measurement_value" not in found.columns:
                found = references[source].iloc[0:0]
            per_source[source] = found
        for source, found in per_source.items():
            for record in found.itertuples(index=False):
                match_rows.append(
                    {
                        "reading_id": reading.vitalperiodicid,
                        "episode_id": reading.episode_id,
                        "patientunitstayid": reading.patientunitstayid,
                        "spo2_offset": reading.observationoffset,
                        "spo2": reading.sao2,
                        "measurement_source": source,
                        "measurement_id": record.measurement_id,
                        "measurement_label": record.measurement_label,
                        "measurement_offset": record.measurement_offset,
                        "measurement_value": record.measurement_value,
                        "raw_measurement_value": record.raw_measurement_value,
                        "raw_measurement_text": record.raw_measurement_text,
                        "time_difference_minutes": (
                            record.measurement_offset
                            - reading.observationoffset
                        ),
                        "units": record.units,
                        "units_interface": record.units_interface,
                        "parsed_from": record.parsed_from,
                        "measurement_parse_status": (
                            record.measurement_parse_status
                        ),
                        "evidence_role": (
                            "CONTEXT_ONLY"
                            if source in {"abg_pao2", "map_context"}
                            else "DIRECT_OXYGENATION"
                        ),
                    }
                )

        summary: dict[str, Any] = {
            "reading_id": reading.vitalperiodicid,
            "episode_id": reading.episode_id,
            "patientunitstayid": reading.patientunitstayid,
            "observationoffset": reading.observationoffset,
            "spo2": reading.sao2,
            "periodic_raw_value": reading.raw_sao2,
            "periodic_value_parse_status": reading.sao2_parse_status,
            "zero_contamination_hour_flag": reading.zero_contamination_hour_flag,
            "zero_only_hour_flag": reading.zero_only_hour_flag,
            "spo2_hour_quality_flag": reading.spo2_hour_quality_flag,
            "hourly_valid_nonzero_count": reading.hourly_valid_nonzero_count,
            "hourly_zero_count": reading.hourly_zero_count,
            "hourly_missing_count": reading.hourly_missing_count,
        }
        source_to_prefix = {
            "abg_pao2": "pao2",
            "abg_o2_sat": "abg_o2_sat",
            "resp_sao2": "resp_sao2",
            "map_context": "map",
        }
        for source, prefix in source_to_prefix.items():
            found = per_source[source]
            summary[f"{prefix}_match_count"] = len(found)
            summary[f"{prefix}_numeric_match_count"] = int(
                found["measurement_value"].notna().sum()
            )
            summary[f"{prefix}_malformed_match_count"] = int(
                found["measurement_parse_status"].eq("MALFORMED").sum()
            )
            summary[f"{prefix}_missing_value_match_count"] = int(
                found["measurement_parse_status"].eq("MISSING").sum()
            )
            if found.empty:
                nearest = None
                nearest_numeric = None
                nearest_tie_count = 0
                nearest_numeric_tie_count = 0
            else:
                delta = (
                    found["measurement_offset"]
                    - reading.observationoffset
                ).abs()
                nearest = found.loc[delta.idxmin()]
                nearest_tie_count = int(delta.eq(delta.min()).sum())
                numeric_found = found.loc[
                    found["measurement_value"].notna()
                ]
                if numeric_found.empty:
                    nearest_numeric = None
                    nearest_numeric_tie_count = 0
                else:
                    numeric_delta = (
                        numeric_found["measurement_offset"]
                        - reading.observationoffset
                    ).abs()
                    nearest_numeric = numeric_found.loc[
                        numeric_delta.idxmin()
                    ]
                    nearest_numeric_tie_count = int(
                        numeric_delta.eq(numeric_delta.min()).sum()
                    )
            summary[f"nearest_{prefix}_match_offset"] = (
                nearest["measurement_offset"] if nearest is not None else np.nan
            )
            summary[f"nearest_{prefix}_match_delta_minutes"] = (
                nearest["measurement_offset"] - reading.observationoffset
                if nearest is not None
                else np.nan
            )
            summary[f"nearest_{prefix}_tie_count"] = nearest_tie_count
            summary[f"nearest_{prefix}_numeric_tie_count"] = (
                nearest_numeric_tie_count
            )
            summary[f"nearest_{prefix}_numeric_value"] = (
                nearest_numeric["measurement_value"]
                if nearest_numeric is not None
                else np.nan
            )
            summary[f"nearest_{prefix}_numeric_offset"] = (
                nearest_numeric["measurement_offset"]
                if nearest_numeric is not None
                else np.nan
            )
            summary[f"nearest_{prefix}_numeric_delta_minutes"] = (
                nearest_numeric["measurement_offset"]
                - reading.observationoffset
                if nearest_numeric is not None
                else np.nan
            )
        reading_rows.append(summary)

    matches = pd.DataFrame(match_rows)
    if matches.empty:
        matches = pd.DataFrame(
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
                "parsed_from",
                "measurement_parse_status",
                "raw_measurement_text",
                "units_interface",
                "evidence_role",
            ]
        )
    reading_summary = pd.DataFrame(reading_rows)
    if matches.duplicated(
        ["reading_id", "measurement_source", "measurement_id"]
    ).any():
        raise ValueError("Duplicate observation/reference match rows found.")
    if (
        pd.to_numeric(matches["time_difference_minutes"], errors="coerce")
        .abs()
        .gt(MATCH_WINDOW_MINUTES)
        .any()
    ):
        raise ValueError("A primary match lies outside the configured window.")
    return reading_summary, matches


def summarize_episodes(
    readings: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for episode_id, episode_readings in readings.groupby(
        "episode_id",
        sort=False,
    ):
        episode_matches = matches.loc[
            matches["episode_id"].eq(episode_id)
        ]
        unique_refs = episode_matches.drop_duplicates(
            ["measurement_source", "measurement_id"]
        )
        record: dict[str, Any] = {
            "episode_id": episode_id,
            "patientunitstayid": episode_readings[
                "patientunitstayid"
            ].iloc[0],
            "start_offset": episode_readings["observationoffset"].min(),
            "end_offset": episode_readings["observationoffset"].max(),
            "duration_minutes": (
                episode_readings["observationoffset"].max()
                - episode_readings["observationoffset"].min()
            ),
            "low_reading_count": len(episode_readings),
            "minimum_spo2": episode_readings["sao2"].min(),
            "mean_spo2": episode_readings["sao2"].mean(),
            "median_spo2": episode_readings["sao2"].median(),
            "episode_gap_rule_minutes": EPISODE_GAP_MINUTES,
        }
        for threshold in THRESHOLDS:
            record[f"readings_lt_{threshold}"] = int(
                episode_readings["sao2"].lt(threshold).sum()
            )
            record[f"readings_le_{threshold}"] = int(
                episode_readings["sao2"].le(threshold).sum()
            )

        for source, prefix in [
            ("abg_pao2", "pao2"),
            ("abg_o2_sat", "abg_o2_sat"),
            ("resp_sao2", "resp_sao2"),
            ("map_context", "map"),
        ]:
            source_refs = unique_refs.loc[
                unique_refs["measurement_source"].eq(source)
            ]
            record[f"{prefix}_measurement_count"] = len(source_refs)
            if source_refs.empty:
                record[f"closest_{prefix}_delta_minutes"] = np.nan
                record[f"closest_{prefix}_offset"] = np.nan
            else:
                idx = pd.to_numeric(
                    source_refs["time_difference_minutes"],
                    errors="coerce",
                ).abs().idxmin()
                closest = source_refs.loc[idx]
                record[f"closest_{prefix}_delta_minutes"] = closest[
                    "time_difference_minutes"
                ]
                record[f"closest_{prefix}_offset"] = closest[
                    "measurement_offset"
                ]

        classification, explanation, provisional = classify_episode(
            episode_matches
        )
        record["evidence_classification"] = classification
        record["evidence_explanation"] = explanation
        record["classification_is_provisional"] = provisional
        record["constituent_reading_ids"] = "|".join(
            episode_readings["vitalperiodicid"].astype(str)
        )
        if episode_matches.empty:
            record["direct_evidence_sources"] = ""
        else:
            direct_sources = episode_matches.loc[
                episode_matches["measurement_source"].isin(
                    ["abg_o2_sat", "resp_sao2"]
                ),
                "measurement_source",
            ].unique()
            record["direct_evidence_sources"] = "|".join(
                sorted(direct_sources)
            )
        rows.append(record)
    episodes = pd.DataFrame(rows)
    if episodes["episode_id"].duplicated().any():
        raise ValueError("Episode summary contains duplicate episode IDs.")
    return episodes


def build_hourly_zero_summary(periodic: pd.DataFrame) -> pd.DataFrame:
    data = periodic.copy()
    data["hour"] = (data["observationoffset"] // 60).astype(int)
    grouped = data.groupby(
        ["patientunitstayid", "hour"],
        as_index=False,
    )
    hourly = grouped.agg(
        raw_mean=("sao2", "mean"),
        zero_excluded_mean=(
            "sao2",
            lambda values: values.loc[values.gt(0)].mean(),
        ),
        valid_nonzero_count=("sao2", lambda values: values.gt(0).sum()),
        zero_count=("sao2", lambda values: values.eq(0).sum()),
        numeric_count=("sao2", "count"),
        periodic_row_count=("sao2", "size"),
    )
    hourly["missing_count"] = (
        hourly["periodic_row_count"] - hourly["numeric_count"]
    )
    hourly["quality_flag"] = np.select(
        [
            hourly["valid_nonzero_count"].gt(0)
            & hourly["zero_count"].gt(0),
            hourly["valid_nonzero_count"].gt(0)
            & hourly["zero_count"].eq(0),
            hourly["valid_nonzero_count"].eq(0)
            & hourly["zero_count"].gt(0),
        ],
        ["MIXED_ZERO", "VALID", "ZERO_ONLY"],
        default="MISSING_ONLY",
    )
    if (
        hourly.loc[hourly["quality_flag"].eq("ZERO_ONLY"), "zero_excluded_mean"]
        .notna()
        .any()
    ):
        raise ValueError("Zero-only hours must have a missing corrected mean.")
    return hourly


def add_threshold_summaries(
    summary: dict[str, Any],
    periodic: pd.DataFrame,
    readings: pd.DataFrame,
    episodes: pd.DataFrame,
) -> None:
    nonzero = periodic.loc[periodic["sao2"].gt(0)]
    threshold_episode_counts: dict[int, int] = {}
    for threshold in THRESHOLDS:
        reading_mask = nonzero["sao2"].le(threshold)
        threshold_readings = nonzero.loc[reading_mask].copy()
        threshold_episodes = assign_episodes(threshold_readings)
        threshold_episode_count = int(
            threshold_episodes["episode_id"].nunique()
        )
        threshold_episode_counts[threshold] = threshold_episode_count
        summary[f"low_readings_le_{threshold}"] = int(reading_mask.sum())
        summary[f"stays_with_readings_le_{threshold}"] = int(
            nonzero.loc[reading_mask, "patientunitstayid"].nunique()
        )
        summary[f"nonzero_spo2_percent_le_{threshold}"] = round(
            float(reading_mask.mean() * 100) if len(nonzero) else 0.0,
            3,
        )
        summary[f"episodes_with_readings_le_{threshold}"] = int(
            threshold_episode_count
        )
    inclusive_episode_denominator = threshold_episode_counts[LOW_SPO2_MAX]
    for threshold, count in threshold_episode_counts.items():
        summary[f"episode_percent_with_readings_le_{threshold}"] = round(
            100 * count / inclusive_episode_denominator
            if inclusive_episode_denominator
            else 0.0,
            3,
        )
    below_50 = nonzero["sao2"].lt(50)
    summary["low_readings_lt_50"] = int(below_50.sum())
    summary["stays_with_readings_lt_50"] = int(
        nonzero.loc[below_50, "patientunitstayid"].nunique()
    )
    summary["episodes_with_readings_lt_50"] = int(
        episodes["readings_lt_50"].gt(0).sum()
    )


def coverage(
    source: str,
    prefix: str,
    matches: pd.DataFrame,
    readings: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, Any]:
    source_matches = matches.loc[
        matches["measurement_source"].eq(source)
    ]
    reading_ids = set(source_matches["reading_id"])
    episode_ids = set(source_matches["episode_id"])
    return {
        f"{prefix}_match_rows": len(source_matches),
        f"{prefix}_numeric_match_rows": int(
            source_matches["measurement_value"].notna().sum()
        ),
        f"{prefix}_malformed_match_rows": int(
            source_matches["measurement_parse_status"].eq("MALFORMED").sum()
        ),
        f"{prefix}_missing_value_match_rows": int(
            source_matches["measurement_parse_status"].eq("MISSING").sum()
        ),
        f"{prefix}_text_fallback_match_rows": int(
            source_matches["measurement_parse_status"].eq("TEXT_FALLBACK").sum()
        ),
        f"{prefix}_readings_with_match": len(reading_ids),
        f"{prefix}_reading_coverage_percent": round(
            100 * len(reading_ids) / len(readings) if len(readings) else 0,
            3,
        ),
        f"{prefix}_episodes_with_match": len(episode_ids),
        f"{prefix}_episode_coverage_percent": round(
            100 * len(episode_ids) / len(episodes) if len(episodes) else 0,
            3,
        ),
    }


def make_summary(
    patients: pd.DataFrame,
    periodic: pd.DataFrame,
    readings: pd.DataFrame,
    matches: pd.DataFrame,
    episodes: pd.DataFrame,
    hourly: pd.DataFrame,
    input_stats: dict[str, dict[str, int]],
    references: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "eligible_low_readings_lt_50": len(readings),
        "nonzero_spo2_readings_le_50": int(
            (periodic["sao2"].gt(0) & periodic["sao2"].le(50)).sum()
        ),
        "unique_stays_with_eligible_low_readings": int(
            readings["patientunitstayid"].nunique()
        ),
        "episode_count": len(episodes),
        "raw_patient_stays": len(patients),
        "stays_with_in_stay_periodic_data": int(
            periodic["patientunitstayid"].nunique()
        ),
        "in_stay_periodic_rows": len(periodic),
        "raw_nonmissing_numeric_spo2_rows": int(periodic["sao2"].notna().sum()),
        "raw_zero_spo2_rows": int(periodic["sao2"].eq(0).sum()),
        "raw_nonzero_spo2_rows": int(periodic["sao2"].gt(0).sum()),
        "raw_negative_spo2_rows": int(periodic["sao2"].lt(0).sum()),
        "raw_spo2_above_100_rows": int(periodic["sao2"].gt(100).sum()),
        "raw_malformed_spo2_rows": int(
            periodic["sao2_parse_status"].eq("MALFORMED").sum()
        ),
        "raw_missing_spo2_rows": int(
            periodic["sao2_parse_status"].eq("MISSING").sum()
        ),
        "patient_hours_with_periodic_rows": len(hourly),
        "mixed_zero_patient_hours": int(hourly["quality_flag"].eq("MIXED_ZERO").sum()),
        "zero_only_patient_hours": int(hourly["quality_flag"].eq("ZERO_ONLY").sum()),
        "zero_only_hours_with_numeric_corrected_mean": int(
            (
                hourly["quality_flag"].eq("ZERO_ONLY")
                & hourly["zero_excluded_mean"].notna()
            ).sum()
        ),
        "mixed_zero_corrected_mean_gt_80_hours": int(
            (
                hourly["quality_flag"].eq("MIXED_ZERO")
                & hourly["zero_excluded_mean"].gt(80)
            ).sum()
        ),
        "mixed_zero_corrected_mean_gt_50_le_80_hours": int(
            (
                hourly["quality_flag"].eq("MIXED_ZERO")
                & hourly["zero_excluded_mean"].gt(50)
                & hourly["zero_excluded_mean"].le(80)
            ).sum()
        ),
        "mixed_zero_corrected_mean_le_50_hours": int(
            (
                hourly["quality_flag"].eq("MIXED_ZERO")
                & hourly["zero_excluded_mean"].le(50)
            ).sum()
        ),
        "primary_match_window_minutes_each_side": MATCH_WINDOW_MINUTES,
        "episode_gap_minutes": EPISODE_GAP_MINUTES,
        "support_saturation_max_inclusive": SUPPORT_SATURATION_MAX,
        "conflict_saturation_min_inclusive": CONFLICT_SATURATION_MIN,
        "classification_thresholds_are_provisional": True,
    }
    add_threshold_summaries(summary, periodic, readings, episodes)
    for source, prefix in [
        ("abg_pao2", "pao2"),
        ("abg_o2_sat", "abg_o2_sat"),
        ("resp_sao2", "resp_sao2"),
        ("map_context", "map"),
    ]:
        summary.update(
            coverage(source, prefix, matches, readings, episodes)
        )
    class_counts = episodes["evidence_classification"].value_counts()
    for category in [
        "SUPPORTED",
        "CONFLICTING",
        "NO_INDEPENDENT_EVIDENCE",
        "INSUFFICIENT_DATA",
    ]:
        summary[f"episodes_classified_{category.lower()}"] = int(
            class_counts.get(category, 0)
        )
    summary["episodes_with_any_supporting_direct_saturation"] = int(
        episodes["episode_id"].isin(
            matches.loc[
                matches["measurement_source"].isin(
                    ["abg_o2_sat", "resp_sao2"]
                )
                & matches["measurement_value"].le(SUPPORT_SATURATION_MAX),
                "episode_id",
            ]
        ).sum()
    )
    summary["episodes_with_any_conflicting_direct_saturation"] = int(
        episodes["episode_id"].isin(
            matches.loc[
                matches["measurement_source"].isin(
                    ["abg_o2_sat", "resp_sao2"]
                )
                & matches["measurement_value"].ge(CONFLICT_SATURATION_MIN),
                "episode_id",
            ]
        ).sum()
    )
    durations = episodes["duration_minutes"]
    for name, value in {
        "min": durations.min(),
        "median": durations.median(),
        "mean": durations.mean(),
        "p90": durations.quantile(0.9),
        "max": durations.max(),
    }.items():
        summary[f"episode_duration_{name}_minutes"] = (
            float(value) if pd.notna(value) else np.nan
        )
    for source, stats in input_stats.items():
        for key, value in stats.items():
            summary[f"{source}_{key}"] = value
    for source, frame in [
        ("abg_pao2", references["abg_pao2"]),
        ("abg_o2_sat", references["abg_o2_sat"]),
        ("resp_sao2", references["resp_sao2"]),
        ("map_context", references["map_context"]),
    ]:
        summary[f"{source}_in_stay_records"] = len(frame)
        summary[f"{source}_numeric_records"] = int(
            frame["measurement_parse_status"].isin(
                ["NUMERIC", "TEXT_FALLBACK"]
            ).sum()
        )
        summary[f"{source}_malformed_value_records"] = int(
            frame["measurement_parse_status"].eq("MALFORMED").sum()
        )
        summary[f"{source}_missing_value_records"] = int(
            frame["measurement_parse_status"].eq("MISSING").sum()
        )
    return summary


def validate_known_cases(
    readings: pd.DataFrame,
    matches: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(
        stay: int,
        case: str,
        passed: bool,
        details: str,
    ) -> None:
        checks.append(
            {
                "patientunitstayid": stay,
                "case": case,
                "passed": bool(passed),
                "details": details,
            }
        )

    def r(stay: int, offset: int) -> pd.DataFrame:
        return readings.loc[
            readings["patientunitstayid"].eq(stay)
            & readings["observationoffset"].eq(offset)
        ]

    def m(stay: int, offset: int, source: str) -> pd.DataFrame:
        return matches.loc[
            matches["patientunitstayid"].eq(stay)
            & matches["spo2_offset"].eq(offset)
            & matches["measurement_source"].eq(source)
        ]

    low121 = r(3157219, 121)
    abg121 = m(3157219, 121, "abg_o2_sat")
    add(
        3157219,
        "SpO2 6 at 121; ABG O2 saturation 38.7 near 134",
        bool(
            not low121.empty
            and low121["sao2"].eq(6).any()
            and abg121["measurement_value"].sub(38.7).abs().lt(0.1).any()
            and abg121["measurement_offset"].eq(134).any()
        ),
        f"periodic={len(low121)}, ABG saturation matches={len(abg121)}",
    )

    low1455 = r(3204052, 1455)
    resp1455 = m(3204052, 1455, "resp_sao2")
    add(
        3204052,
        "SpO2 21 at 1455; respiratory SaO2 92 at 1451",
        bool(
            not low1455.empty
            and low1455["sao2"].eq(21).any()
            and resp1455["measurement_value"].eq(92).any()
            and resp1455["measurement_offset"].eq(1451).any()
        ),
        f"periodic={len(low1455)}, respiratory matches={len(resp1455)}",
    )
    for offset, expected_spo2 in [(1725, 47), (1730, 43)]:
        low = r(3204052, offset)
        resp_match = m(3204052, offset, "resp_sao2")
        add(
            3204052,
            f"SpO2 {expected_spo2} at {offset}; respiratory SaO2 47 at 1730",
            bool(
                not low.empty
                and low["sao2"].eq(expected_spo2).any()
                and resp_match["measurement_value"].eq(47).any()
                and resp_match["measurement_offset"].eq(1730).any()
            ),
            f"periodic={len(low)}, respiratory matches={len(resp_match)}",
        )

    low306 = r(3067309, 345)
    o2sat306 = m(3067309, 345, "abg_o2_sat")
    pao2_306 = m(3067309, 345, "abg_pao2")
    add(
        3067309,
        "SpO2 48; ABG O2 saturation 86 and PaO2 78 at 345",
        bool(
            not low306.empty
            and low306["sao2"].eq(48).any()
            and o2sat306["measurement_value"].eq(86).any()
            and o2sat306["measurement_offset"].eq(345).any()
            and pao2_306["measurement_value"].eq(78).any()
            and pao2_306["measurement_offset"].eq(345).any()
        ),
        f"periodic={len(low306)}, O2 saturation matches={len(o2sat306)}, PaO2 matches={len(pao2_306)}",
    )

    stay758 = readings.loc[readings["patientunitstayid"].eq(758327)]
    ep758 = episodes.loc[episodes["patientunitstayid"].eq(758327)]
    direct_758 = matches.loc[
        matches["patientunitstayid"].eq(758327)
        & matches["measurement_source"].isin(["abg_o2_sat", "resp_sao2"])
    ]
    add(
        758327,
        "Low readings retained; no direct saturation reference in matched windows",
        bool(
            not stay758.empty
            and not ep758.empty
            and direct_758.empty
            and ep758["evidence_classification"]
            .eq("NO_INDEPENDENT_EVIDENCE")
            .all()
        ),
        f"low readings={len(stay758)}, episodes={len(ep758)}, direct matches={len(direct_758)}",
    )

    low122 = readings.loc[readings["patientunitstayid"].eq(1229669)]
    map122 = matches.loc[
        matches["patientunitstayid"].eq(1229669)
        & matches["measurement_source"].eq("map_context")
        & matches["measurement_value"].eq(59)
    ]
    map_support = episodes.loc[
        episodes["patientunitstayid"].eq(1229669)
        & episodes["evidence_classification"].eq("SUPPORTED")
    ]
    add(
        1229669,
        "Low readings and MAP 59 retained as context, not support",
        bool(not low122.empty and not map122.empty and map_support.empty),
        f"low readings={len(low122)}, MAP 59 matches={len(map122)}, MAP-only supported episodes={len(map_support)}",
    )

    result = pd.DataFrame(checks)
    if not result["passed"].all():
        failed = result.loc[~result["passed"]].to_dict("records")
        raise AssertionError(f"Known-case validation failed: {failed}")
    return result


def make_policy_report(
    summary: dict[str, Any],
    known_cases: pd.DataFrame,
) -> str:
    classifications = {
        category: summary[f"episodes_classified_{category.lower()}"]
        for category in [
            "SUPPORTED",
            "CONFLICTING",
            "NO_INDEPENDENT_EVIDENCE",
            "INSUFFICIENT_DATA",
        ]
    }
    failed = int((~known_cases["passed"]).sum())
    threshold_lines = "\n".join(
        (
            f"| ≤{threshold} | {summary[f'low_readings_le_{threshold}']:,} | "
            f"{summary[f'nonzero_spo2_percent_le_{threshold}']:.3f}% | "
            f"{summary[f'stays_with_readings_le_{threshold}']:,} | "
            f"{summary[f'episodes_with_readings_le_{threshold}']:,} | "
            f"{summary[f'episode_percent_with_readings_le_{threshold}']:.3f}% |"
        )
        for threshold in THRESHOLDS
    )
    coverage_lines = "\n".join(
        (
            f"| {source} | {summary[f'{prefix}_match_rows']:,} | "
            f"{summary[f'{prefix}_numeric_match_rows']:,} | "
            f"{summary[f'{prefix}_readings_with_match']:,} "
            f"({summary[f'{prefix}_reading_coverage_percent']:.3f}%) | "
            f"{summary[f'{prefix}_episodes_with_match']:,} "
            f"({summary[f'{prefix}_episode_coverage_percent']:.3f}%) |"
        )
        for source, prefix in [
            ("ABG PaO2 (context)", "pao2"),
            ("ABG O2 saturation", "abg_o2_sat"),
            ("Respiratory SaO2", "resp_sao2"),
            ("MAP (context only)", "map"),
        ]
    )
    case_lines = "\n".join(
        f"| {row.patientunitstayid} | {row.case} | "
        f"{'PASS' if row.passed else 'FAIL'} | {row.details} |"
        for row in known_cases.itertuples(index=False)
    )
    return f"""# Dataset-wide SpO2 cross-validation and research quality policy

## Data-quality problem and scope

The existing hourly mean can be contaminated by exact-zero periodic monitor
values. An exact zero is excluded from a candidate mean but is retained as a
count and quality flag. This audit uses the local eICU demo extract only; it
does not establish clinical validity or change the preprocessing or production
pipeline.

## Method

Raw periodic values (`vitalPeriodic.sao2`) and the independent patient stay
discharge offsets define the in-stay observation set. Offsets in all sources
are minutes. The primary evidence window is inclusive ±{MATCH_WINDOW_MINUTES}
minutes. All records within the window are kept in the match-detail CSV; the
nearest numeric match is also summarized per reading without discarding the
other matches.

The observation-level and episode-level audit set is nonzero numeric periodic
SpO2 strictly below {LOW_SPO2_MAX}. The separate threshold table counts all
nonzero periodic observations at or below 20, 30, 40, and 50; its percentage
denominator is all nonzero periodic observations. Each threshold's episode
count is independently grouped from all qualifying readings at or below that
threshold. The detailed episode output remains restricted to readings strictly
below 50. Episodes group low readings in the same stay whenever consecutive
readings are no more than {EPISODE_GAP_MINUTES} minutes apart. This grouping
is configurable and is an operational audit rule, not a clinical episode
definition.

PaO2 (mm Hg) is retained as oxygenation context but is not converted into an
SpO2 estimate. Lab O2 saturation and respiratory-charted SaO2 (percent) are
the direct saturation references. FiO2, PEEP, and oxygen-flow chart rows are
not selected. Non-invasive mean blood pressure is contextual cardiovascular
evidence only and never qualifies as saturation support.

## Provisional evidence classification

The classifier uses explicit, configurable comparison bands: direct saturation
at or below {SUPPORT_SATURATION_MAX} is labelled support; direct saturation at
or above {CONFLICT_SATURATION_MIN} is labelled conflict; values between these
bands are insufficient for a binary comparison. If both supporting and
conflicting direct values occur in an episode, it is classified
`CONFLICTING`, preserving the mixed evidence in the detailed matches.
Malformed-only direct reference records are `INSUFFICIENT_DATA`. If no direct
saturation measurement is available, the episode is
`NO_INDEPENDENT_EVIDENCE`, regardless of nearby MAP or PaO2.

These cutoffs are provisional audit bands anchored to the low-reading ceiling
and separated comparison band; they are not validated clinical thresholds.
Classifications are retrospective data-quality evidence labels, not diagnoses.

## Dataset-wide findings

- In-stay periodic observations: {summary['in_stay_periodic_rows']:,}; numeric
  SpO2 observations: {summary['raw_nonmissing_numeric_spo2_rows']:,}.
- Eligible nonzero low readings (<{LOW_SPO2_MAX}): {summary['eligible_low_readings_lt_50']:,}
  across {summary['unique_stays_with_eligible_low_readings']:,} stays and
  {summary['episode_count']:,} episodes.
- Nonzero readings ≤50 across the full periodic dataset:
  {summary['low_readings_le_50']:,} readings across
  {summary['stays_with_readings_le_50']:,} stays.
- Episode classifications: {classifications}.
- Mixed-zero hours: {summary['mixed_zero_patient_hours']:,}; zero-only hours:
  {summary['zero_only_patient_hours']:,}; zero-only hours with a numerical
  corrected mean: {summary['zero_only_hours_with_numeric_corrected_mean']:,}.
- Mixed-zero corrected means: {summary['mixed_zero_corrected_mean_gt_80_hours']:,}
  above 80, {summary['mixed_zero_corrected_mean_gt_50_le_80_hours']:,} above 50
  through 80, and {summary['mixed_zero_corrected_mean_le_50_hours']:,} at or
  below 50.
- Zero-only hours with a numeric zero-excluded mean:
  {summary['zero_only_hours_with_numeric_corrected_mean']:,}.
- Episode duration in minutes (min/median/mean/p90/max):
  {summary['episode_duration_min_minutes']:.1f}/
  {summary['episode_duration_median_minutes']:.1f}/
  {summary['episode_duration_mean_minutes']:.2f}/
  {summary['episode_duration_p90_minutes']:.1f}/
  {summary['episode_duration_max_minutes']:.1f}.

### Reading and episode threshold totals

| Threshold | Readings | % of nonzero SpO2 | ICU stays | Threshold-specific episodes | % of ≤50 episodes |
|---|---:|---:|---:|---:|---:|
{threshold_lines}

### Primary-window source coverage

| Source | Match rows | Numeric rows | Readings with ≥1 match | Episodes with ≥1 match |
|---|---:|---:|---:|---:|
{coverage_lines}

MAP is included only to report contextual availability. PaO2 coverage is
reported separately and is not direct saturation confirmation.

## Known-case validation

The audit rechecked the five supplied stays and their expected examples.
Validation checks passed: {len(known_cases) - failed}/{len(known_cases)}.
| Stay | Expected example | Check | Audit result |
|---:|---|---|---|
{case_lines}

## Proposed research preprocessing policy

1. Exclude exact-zero SpO2 only from the candidate hourly numerical mean;
   preserve raw mean, zero count, valid nonzero count, missing count, and
   quality flags.
2. Leave zero-only hourly candidate means missing; do not convert them to 0%
   or impute them.
3. Preserve all nonzero SpO2 observations, including extreme lows. Do not
   delete or cap values based on an arbitrary lower cutoff.
4. Distinguish observation availability from independent evidence availability.
   Missing corroboration is not evidence that a reading is invalid.
5. Treat MAP as context and do not use it as SpO2 confirmation.
6. Do not promote the experiment to production or use these provisional
   retrospective labels as clinical diagnoses or training truth without
   review.

## Limitations and remaining uncertainty

This is a demo database subset. Recorded offsets and chart labels can be
incomplete or inaccurate, and measurements from distinct sources may not be
independent. ABG O2 saturation and respiratory SaO2 may have their own
measurement/documentation limitations. A ±15-minute match is temporal
proximity, not proof of simultaneity. Episode boundaries and comparison bands
are configurable research conventions. PaO2 cannot be interpreted as a
direct pulse-oximeter reference without additional clinical context.

## Recommended next validation

Review all conflicting and supported source records, verify original units
and result fields, and assess sensitivity to the configurable episode gap and
matching window. If feasible, obtain clinician-adjudicated or device-level
reference data before treating the labels as ground truth. Reassess the
zero-exclusion policy on the intended target dataset before any downstream
model or alert changes.
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    patients = load_patients()

    periodic, periodic_stats = load_in_stay(
        "vitalPeriodic.csv.gz",
        [
            "vitalperiodicid",
            "patientunitstayid",
            "observationoffset",
            "sao2",
        ],
        "observationoffset",
        patients,
    )
    periodic["raw_sao2"] = periodic["sao2"].astype("string")
    periodic["sao2"] = periodic["sao2"].map(parse_number)
    periodic["sao2_parse_status"] = np.select(
        [
            periodic["sao2"].notna(),
            periodic["raw_sao2"].isna()
            | periodic["raw_sao2"].str.strip().eq("").fillna(False),
        ],
        ["NUMERIC", "MISSING"],
        default="MALFORMED",
    )
    if periodic["vitalperiodicid"].duplicated().any():
        raise ValueError("Periodic observations have duplicate source IDs.")

    periodic["hour"] = (periodic["observationoffset"] // 60).astype(int)
    hourly = build_hourly_zero_summary(periodic)
    periodic = periodic.merge(
        hourly[
            [
                "patientunitstayid",
                "hour",
                "quality_flag",
                "zero_excluded_mean",
                "valid_nonzero_count",
                "zero_count",
                "missing_count",
            ]
        ].rename(
            columns={
                "quality_flag": "spo2_hour_quality_flag",
                "zero_excluded_mean": "hourly_candidate_spo2",
                "valid_nonzero_count": "hourly_valid_nonzero_count",
                "zero_count": "hourly_zero_count",
                "missing_count": "hourly_missing_count",
            }
        ),
        on=["patientunitstayid", "hour"],
        how="left",
        validate="many_to_one",
    )
    periodic["zero_contamination_hour_flag"] = (
        periodic["spo2_hour_quality_flag"].eq("MIXED_ZERO")
    )
    periodic["zero_only_hour_flag"] = (
        periodic["spo2_hour_quality_flag"].eq("ZERO_ONLY")
    )

    readings = periodic.loc[
        periodic["sao2"].gt(0)
        & (
            periodic["sao2"].le(LOW_SPO2_MAX)
            if LOW_SPO2_MAX_INCLUSIVE
            else periodic["sao2"].lt(LOW_SPO2_MAX)
        )
    ].copy()
    readings = assign_episodes(readings)
    references, input_stats = prepare_references(patients)
    reading_summary, matches = build_matches(readings, references)
    episodes = summarize_episodes(readings, matches)

    if reading_summary["reading_id"].duplicated().any():
        raise ValueError("Cross-validation output duplicates source readings.")
    if set(readings["patientunitstayid"]) - set(patients["patientunitstayid"]):
        raise ValueError("A low reading references an unknown ICU stay.")
    if len(episodes) != readings["episode_id"].nunique():
        raise ValueError("Episode summary does not reconcile to readings.")
    if int(episodes["low_reading_count"].sum()) != len(readings):
        raise ValueError("Episode reading counts do not reconcile.")
    if readings["vitalperiodicid"].duplicated().any():
        raise ValueError("A periodic reading was assigned more than once.")
    source_reading_ids = set(
        readings["vitalperiodicid"].astype(str)
    )
    episode_reading_ids = [
        reading_id
        for value in episodes["constituent_reading_ids"]
        for reading_id in str(value).split("|")
    ]
    if (
        len(episode_reading_ids) != len(readings)
        or len(set(episode_reading_ids)) != len(episode_reading_ids)
        or set(episode_reading_ids) != source_reading_ids
    ):
        raise ValueError("Episode reading membership is not uniquely traceable.")
    if not readings["sao2"].gt(0).all():
        raise ValueError("The low-reading set must exclude exact-zero values.")
    if LOW_SPO2_MAX_INCLUSIVE and not readings["sao2"].le(LOW_SPO2_MAX).all():
        raise ValueError("A low reading exceeds the inclusive configured limit.")
    if not LOW_SPO2_MAX_INCLUSIVE and not readings["sao2"].lt(LOW_SPO2_MAX).all():
        raise ValueError("A low reading is not strictly below the configured limit.")
    for threshold in THRESHOLDS:
        reading_count = int(readings["sao2"].le(threshold).sum())
        episode_reading_count = int(
            episodes[f"readings_le_{threshold}"].sum()
        )
        episode_count = int(
            episodes[f"readings_le_{threshold}"].gt(0).sum()
        )
        if reading_count != episode_reading_count:
            raise ValueError(
                f"Reading counts at threshold {threshold} do not reconcile."
            )
        if episode_count != int(
            readings.loc[
                readings["sao2"].le(threshold),
                "episode_id",
            ].nunique()
        ):
            raise ValueError(
                f"Episode counts at threshold {threshold} do not reconcile."
            )
    if int(reading_summary["pao2_match_count"].sum()) != int(
        matches["measurement_source"].eq("abg_pao2").sum()
    ):
        raise ValueError("PaO2 match counts do not reconcile.")
    for source, prefix in [
        ("abg_o2_sat", "abg_o2_sat"),
        ("resp_sao2", "resp_sao2"),
        ("map_context", "map"),
    ]:
        if int(reading_summary[f"{prefix}_match_count"].sum()) != int(
            matches["measurement_source"].eq(source).sum()
        ):
            raise ValueError(f"{prefix} match counts do not reconcile.")

    reading_summary["evidence_classification"] = reading_summary[
        "episode_id"
    ].map(episodes.set_index("episode_id")["evidence_classification"])
    reading_summary.to_csv(
        OUTPUT_DIR / "spo2_cross_validation_readings.csv",
        index=False,
    )
    matches.to_csv(
        OUTPUT_DIR / "spo2_cross_validation_matches.csv",
        index=False,
    )
    episodes.to_csv(
        OUTPUT_DIR / "spo2_cross_validation_episodes.csv",
        index=False,
    )
    hourly.to_csv(
        OUTPUT_DIR / "spo2_hourly_zero_quality.csv",
        index=False,
    )

    known_cases = validate_known_cases(readings, matches, episodes)
    known_cases.to_csv(
        OUTPUT_DIR / "spo2_known_case_validation.csv",
        index=False,
    )
    summary = make_summary(
        patients,
        periodic,
        readings,
        matches,
        episodes,
        hourly,
        {
            "periodic": periodic_stats,
            "lab": input_stats["lab"],
            "respiratory": input_stats["respiratory"],
            "map": input_stats["map"],
        },
        references,
    )
    for threshold in THRESHOLDS:
        expected = int(
            (periodic["sao2"].gt(0) & periodic["sao2"].le(threshold)).sum()
        )
        if summary[f"low_readings_le_{threshold}"] != expected:
            raise ValueError(
                f"Full-dataset threshold count at {threshold} does not reconcile."
            )
    summary["known_case_checks"] = len(known_cases)
    summary["known_case_checks_passed"] = int(
        known_cases["passed"].sum()
    )
    summary["known_case_checks_failed"] = int(
        (~known_cases["passed"]).sum()
    )
    summary_frame = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    )
    summary_frame.to_csv(
        OUTPUT_DIR / "spo2_cross_validation_summary.csv",
        index=False,
    )
    (OUTPUT_DIR / "spo2_quality_policy.md").write_text(
        make_policy_report(summary, known_cases),
        encoding="utf-8",
    )

    print("Dataset-wide SpO2 audit completed.")
    comparator = "<=" if LOW_SPO2_MAX_INCLUSIVE else "<"
    print(
        f"Low nonzero readings {comparator} {LOW_SPO2_MAX}: {len(readings):,} "
        f"across {readings['patientunitstayid'].nunique():,} stays"
    )
    print(f"Episodes (gap <= {EPISODE_GAP_MINUTES} min): {len(episodes):,}")
    print("Evidence classification:")
    print(
        episodes["evidence_classification"]
        .value_counts()
        .reindex(
            [
                "SUPPORTED",
                "CONFLICTING",
                "NO_INDEPENDENT_EVIDENCE",
                "INSUFFICIENT_DATA",
            ],
            fill_value=0,
        )
        .to_string()
    )
    print(
        "Known-case validation: "
        f"{int(known_cases['passed'].sum())}/{len(known_cases)} passed"
    )
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
