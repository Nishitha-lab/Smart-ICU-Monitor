from pathlib import Path

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
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "spo2_audit"

WINDOW_MINUTES = 15
LOW_SPO2_THRESHOLD = 50
EPISODE_GAP_MINUTES = 15


def load_in_stay_data(
    filename: str,
    columns: list[str],
    offset_column: str,
    patients: pd.DataFrame,
) -> pd.DataFrame:
    data = pd.read_csv(
        DATA_DIR / filename,
        usecols=columns,
        low_memory=False,
    )
    data[offset_column] = pd.to_numeric(
        data[offset_column],
        errors="coerce",
    )
    data = data.merge(
        patients[["patientunitstayid", "unitdischargeoffset"]],
        on="patientunitstayid",
        how="inner",
        validate="many_to_one",
    )
    return data.loc[
        data[offset_column].between(
            0,
            data["unitdischargeoffset"],
            inclusive="both",
        )
    ].copy()


def make_hourly_audit(periodic: pd.DataFrame) -> pd.DataFrame:
    def nonzero_mean(values: pd.Series) -> float:
        return values.loc[values.gt(0)].mean()

    hourly = (
        periodic.groupby(
            ["patientunitstayid", "hour"],
            as_index=False,
        )
        .agg(
            spo2_raw_mean=("sao2", "mean"),
            spo2_corrected_mean=("sao2", nonzero_mean),
            spo2_valid_count=("sao2", lambda values: values.gt(0).sum()),
            spo2_zero_count=("sao2", lambda values: values.eq(0).sum()),
            spo2_missing_count=("sao2", "count"),
            periodic_rows=("sao2", "size"),
        )
    )
    hourly["spo2_missing_count"] = (
        hourly["periodic_rows"] - hourly["spo2_missing_count"]
    )
    hourly["spo2_quality_flag"] = np.select(
        [
            hourly["spo2_valid_count"].gt(0)
            & hourly["spo2_zero_count"].gt(0),
            hourly["spo2_valid_count"].gt(0)
            & hourly["spo2_zero_count"].eq(0),
            hourly["spo2_valid_count"].eq(0)
            & hourly["spo2_zero_count"].gt(0),
        ],
        ["MIXED_ZERO", "VALID", "ZERO_ONLY"],
        default="MISSING_ONLY",
    )
    return hourly


def prepare_reference_data(
    filename: str,
    columns: list[str],
    offset_column: str,
    value_column: str,
    patients: pd.DataFrame,
    label_column: str | None,
) -> pd.DataFrame:
    data = load_in_stay_data(
        filename,
        columns,
        offset_column,
        patients,
    )
    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce",
    )
    data = data.dropna(subset=[offset_column, value_column])
    rename_columns = {
        offset_column: "reference_offset",
        value_column: "reference_value",
    }
    if label_column is not None:
        rename_columns[label_column] = "reference_label"
    data = data.rename(columns=rename_columns)
    if label_column is None:
        data["reference_label"] = ""
    return data


def find_nearby(
    frame: pd.DataFrame,
    offsets: np.ndarray,
    reading_offset: float,
) -> pd.DataFrame:
    start = np.searchsorted(
        offsets,
        reading_offset - WINDOW_MINUTES,
        side="left",
    )
    end = np.searchsorted(
        offsets,
        reading_offset + WINDOW_MINUTES,
        side="right",
    )
    return frame.iloc[start:end]


def assign_episode_ids(readings: pd.DataFrame) -> pd.DataFrame:
    readings = readings.sort_values(
        ["patientunitstayid", "observationoffset", "vitalperiodicid"]
    ).copy()
    gap = readings.groupby("patientunitstayid")[
        "observationoffset"
    ].diff()
    new_episode = gap.isna() | gap.gt(EPISODE_GAP_MINUTES)
    episode_number = new_episode.groupby(
        readings["patientunitstayid"]
    ).cumsum()
    readings["episode_id"] = (
        readings["patientunitstayid"].astype(str)
        + "_"
        + episode_number.astype(int).astype(str).str.zfill(4)
    )
    return readings


def classify_direct_saturation(value: float) -> str:
    if not 0 < value <= 100:
        return "OUT_OF_RANGE"
    if value <= LOW_SPO2_THRESHOLD:
        return "DIRECT_LOW_SUPPORT"
    if value >= 70:
        return "DIRECT_HIGH_DISAGREEMENT"
    return "DIRECT_INTERMEDIATE"


def build_evidence(
    readings: pd.DataFrame,
    reference_sets: dict[str, tuple[pd.DataFrame, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_by_stay: dict[
        str, dict[object, tuple[pd.DataFrame, np.ndarray]]
    ] = {}
    for source, (frame, _) in reference_sets.items():
        reference_by_stay[source] = {
            stay_id: (
                stay.sort_values("reference_offset").reset_index(drop=True),
                stay.sort_values("reference_offset")[
                    "reference_offset"
                ].to_numpy(),
            )
            for stay_id, stay in frame.groupby("patientunitstayid")
        }

    evidence_rows: list[dict[str, object]] = []
    match_rows: list[dict[str, object]] = []
    direct_saturation_sources = {"lab_o2_saturation", "resp_sao2"}

    for reading in readings.itertuples(index=False):
        found: dict[str, pd.DataFrame] = {}
        for source, (all_rows, _) in reference_sets.items():
            stay_data = reference_by_stay[source].get(
                reading.patientunitstayid
            )
            if stay_data is None:
                found[source] = all_rows.iloc[0:0]
                continue
            stay_rows, offsets = stay_data
            found[source] = find_nearby(
                stay_rows,
                offsets,
                reading.observationoffset,
            )

        per_reading_matches: dict[str, int] = {}
        for source, frame in found.items():
            id_column = reference_sets[source][1]
            for reference in frame.itertuples(index=False):
                reference_id = getattr(reference, id_column)
                reference_offset = reference.reference_offset
                reference_value = reference.reference_value
                label = reference.reference_label
                role = (
                    "CONTEXT_ONLY"
                    if source in {"lab_pao2", "map_context"}
                    else "DIRECT_SATURATION"
                )
                direct_class = (
                    classify_direct_saturation(reference_value)
                    if source in direct_saturation_sources
                    else "CONTEXT_ONLY"
                )
                match_rows.append(
                    {
                        "reading_id": reading.vitalperiodicid,
                        "episode_id": reading.episode_id,
                        "patientunitstayid": reading.patientunitstayid,
                        "spo2_offset": reading.observationoffset,
                        "spo2": reading.sao2,
                        "reference_source": source,
                        "reference_id": reference_id,
                        "reference_offset": reference_offset,
                        "offset_delta_minutes": (
                            reference_offset - reading.observationoffset
                        ),
                        "reference_label": label,
                        "reference_value": reference_value,
                        "evidence_role": role,
                        "direct_saturation_interpretation": direct_class,
                    }
                )
                per_reading_matches[source] = (
                    per_reading_matches.get(source, 0) + 1
                )

        direct_values = [
            float(value)
            for source in direct_saturation_sources
            for value in found[source]["reference_value"]
            if 0 < float(value) <= 100
        ]
        evidence_rows.append(
            {
                "reading_id": reading.vitalperiodicid,
                "episode_id": reading.episode_id,
                "patientunitstayid": reading.patientunitstayid,
                "observationoffset": reading.observationoffset,
                "sao2": reading.sao2,
                "nearby_lab_pao2_count": per_reading_matches.get(
                    "lab_pao2", 0
                ),
                "nearby_lab_o2_saturation_count": per_reading_matches.get(
                    "lab_o2_saturation", 0
                ),
                "nearby_resp_sao2_count": per_reading_matches.get(
                    "resp_sao2", 0
                ),
                "nearby_map_context_count": per_reading_matches.get(
                    "map_context", 0
                ),
                "direct_saturation_values": ";".join(
                    f"{value:g}" for value in direct_values
                ),
                "has_direct_low_support": any(
                    value <= LOW_SPO2_THRESHOLD
                    for value in direct_values
                ),
                "has_direct_high_disagreement": any(
                    value >= 70 for value in direct_values
                ),
            }
        )

    evidence = pd.DataFrame(evidence_rows)
    matches = pd.DataFrame(match_rows)
    episodes = summarize_episodes(readings, matches)
    return evidence, matches, episodes


def summarize_episodes(
    readings: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    episode_rows: list[dict[str, object]] = []
    if matches.empty:
        matches_by_episode: dict[str, pd.DataFrame] = {}
    else:
        matches_by_episode = {
            episode_id: group.drop_duplicates(
                ["reference_source", "reference_id"]
            )
            for episode_id, group in matches.groupby("episode_id")
        }

    for episode_id, group in readings.groupby("episode_id", sort=False):
        episode_matches = matches_by_episode.get(
            episode_id,
            matches.iloc[0:0],
        )
        direct = episode_matches.loc[
            episode_matches["evidence_role"].eq("DIRECT_SATURATION")
        ]
        direct_values = pd.to_numeric(
            direct["reference_value"],
            errors="coerce",
        )
        direct_values = direct_values.loc[
            direct_values.gt(0) & direct_values.le(100)
        ]
        has_low_support = direct_values.le(LOW_SPO2_THRESHOLD).any()
        has_high_disagreement = direct_values.ge(70).any()
        if has_low_support and has_high_disagreement:
            classification = "MIXED_DIRECT_SATURATION_EVIDENCE"
        elif has_low_support:
            classification = "CORROBORATING_DIRECT_SATURATION"
        elif has_high_disagreement:
            classification = "CONFLICTING_DIRECT_SATURATION"
        elif not direct_values.empty:
            classification = "INTERMEDIATE_DIRECT_SATURATION"
        else:
            classification = "NO_DIRECT_SATURATION_REFERENCE"

        episode_rows.append(
            {
                "episode_id": episode_id,
                "patientunitstayid": group["patientunitstayid"].iloc[0],
                "start_offset": group["observationoffset"].min(),
                "end_offset": group["observationoffset"].max(),
                "duration_minutes": (
                    group["observationoffset"].max()
                    - group["observationoffset"].min()
                ),
                "low_reading_count": len(group),
                "minimum_spo2": group["sao2"].min(),
                "maximum_spo2": group["sao2"].max(),
                "mean_spo2": group["sao2"].mean(),
                "direct_saturation_reference_count": len(direct),
                "direct_low_support_count": int(
                    direct_values.le(LOW_SPO2_THRESHOLD).sum()
                ),
                "direct_high_disagreement_count": int(
                    direct_values.ge(70).sum()
                ),
                "nearby_lab_pao2_count": int(
                    episode_matches["reference_source"].eq("lab_pao2").sum()
                ),
                "nearby_map_context_count": int(
                    episode_matches["reference_source"].eq("map_context").sum()
                ),
                "evidence_classification": classification,
            }
        )
    return pd.DataFrame(episode_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    patient_file = DATA_DIR / "patient.csv.gz"
    patients = pd.read_csv(
        patient_file,
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

    periodic = load_in_stay_data(
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
    periodic["sao2"] = pd.to_numeric(
        periodic["sao2"],
        errors="coerce",
    )
    periodic["hour"] = (
        periodic["observationoffset"] // 60
    ).astype(int)

    hourly = make_hourly_audit(periodic)
    raw_hourly_keys = hourly[["patientunitstayid", "hour"]].copy()
    experiment_file = PROCESSED_DIR / (
        "hourly_vitals_labs_spo2_experiment.csv"
    )
    experiment = pd.read_csv(
        experiment_file,
        usecols=[
            "patientunitstayid",
            "hour",
            "spo2_raw_mean",
            "spo2",
            "spo2_valid_count",
            "spo2_zero_count",
            "spo2_missing_count",
            "spo2_quality_flag",
        ],
        low_memory=False,
    )
    experiment_only_hours = experiment.merge(
        raw_hourly_keys,
        on=["patientunitstayid", "hour"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if experiment.duplicated(
        ["patientunitstayid", "hour"]
    ).any():
        raise ValueError(
            "The experimental hourly dataset has duplicate stay-hour keys."
        )
    hourly = hourly.merge(
        experiment,
        on=["patientunitstayid", "hour"],
        how="left",
        suffixes=("_raw_recomputed", "_experiment"),
        validate="one_to_one",
        indicator=True,
    )
    for column in [
        "spo2_raw_mean",
        "spo2_valid_count",
        "spo2_zero_count",
        "spo2_missing_count",
    ]:
        recomputed = (
            "spo2_raw_mean_raw_recomputed"
            if column == "spo2_raw_mean"
            else f"{column}_raw_recomputed"
        )
        candidate = f"{column}_experiment"
        matches = np.isclose(
            pd.to_numeric(hourly[recomputed], errors="coerce"),
            pd.to_numeric(hourly[candidate], errors="coerce"),
            rtol=0,
            atol=1e-8,
            equal_nan=True,
        )
        hourly[f"{column}_matches_experiment"] = matches
    hourly["corrected_mean_matches_experiment"] = np.isclose(
        pd.to_numeric(hourly["spo2_corrected_mean"], errors="coerce"),
        pd.to_numeric(hourly["spo2"], errors="coerce"),
        rtol=0,
        atol=1e-8,
        equal_nan=True,
    )
    hourly["quality_flag_matches_experiment"] = (
        hourly["spo2_quality_flag_raw_recomputed"]
        .eq(hourly["spo2_quality_flag_experiment"])
    )
    hourly.to_csv(
        OUTPUT_DIR / "spo2_hourly_raw_validation.csv",
        index=False,
    )

    raw_lab = prepare_reference_data(
        "lab.csv.gz",
        [
            "labid",
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "labresult",
        ],
        "labresultoffset",
        "labresult",
        patients,
        "labname",
    )
    lab_names = raw_lab["reference_label"].astype(str)
    is_pao2 = lab_names.str.contains(
        r"pa\s*o2|pao2",
        case=False,
        regex=True,
        na=False,
    )
    is_o2_saturation = lab_names.str.contains(
        r"o2\s*sat|oxygen\s*saturation|sao2",
        case=False,
        regex=True,
        na=False,
    )
    raw_lab = raw_lab.loc[is_pao2 | is_o2_saturation].copy()
    raw_lab["lab_evidence_type"] = np.where(
        is_pao2.loc[raw_lab.index],
        "lab_pao2",
        "lab_o2_saturation",
    )
    lab_pao2 = raw_lab.loc[
        raw_lab["lab_evidence_type"].eq("lab_pao2")
    ]
    lab_o2_saturation = raw_lab.loc[
        raw_lab["lab_evidence_type"].eq("lab_o2_saturation")
    ]

    respiratory = prepare_reference_data(
        "respiratoryCharting.csv.gz",
        [
            "respchartid",
            "patientunitstayid",
            "respchartoffset",
            "respchartvaluelabel",
            "respchartvalue",
        ],
        "respchartoffset",
        "respchartvalue",
        patients,
        "respchartvaluelabel",
    )
    respiratory = respiratory.loc[
        respiratory["reference_label"].astype(str).str.contains(
            r"sao2",
            case=False,
            regex=True,
            na=False,
        )
    ]

    blood_pressure = prepare_reference_data(
        "vitalAperiodic.csv.gz",
        [
            "vitalaperiodicid",
            "patientunitstayid",
            "observationoffset",
            "noninvasivemean",
        ],
        "observationoffset",
        "noninvasivemean",
        patients,
        None,
    )
    blood_pressure["reference_label"] = "noninvasivemean"

    low_readings = periodic.loc[
        periodic["sao2"].gt(0)
        & periodic["sao2"].le(LOW_SPO2_THRESHOLD)
    ].copy()
    low_readings = assign_episode_ids(low_readings)
    reference_sets = {
        "lab_pao2": (lab_pao2, "labid"),
        "lab_o2_saturation": (lab_o2_saturation, "labid"),
        "resp_sao2": (respiratory, "respchartid"),
        "map_context": (blood_pressure, "vitalaperiodicid"),
    }
    reading_evidence, matches, episodes = build_evidence(
        low_readings,
        reference_sets,
    )
    reading_evidence.to_csv(
        OUTPUT_DIR / "spo2_low_reading_evidence.csv",
        index=False,
    )
    matches.to_csv(
        OUTPUT_DIR / "spo2_reference_matches.csv",
        index=False,
    )
    episodes.to_csv(
        OUTPUT_DIR / "spo2_low_reading_clusters.csv",
        index=False,
    )

    valid_sao2 = periodic["sao2"].dropna()
    mixed_zero = hourly["spo2_quality_flag_raw_recomputed"].eq(
        "MIXED_ZERO"
    )
    corrected_mixed = hourly.loc[mixed_zero, "spo2_corrected_mean"]
    summary: dict[str, object] = {
        "raw_patient_rows": len(patients),
        "raw_periodic_rows_during_icu_stay": len(periodic),
        "icu_stays_with_periodic_data": periodic[
            "patientunitstayid"
        ].nunique(),
        "raw_nonmissing_sao2_rows": int(valid_sao2.notna().sum()),
        "raw_exact_zero_sao2_rows": int(valid_sao2.eq(0).sum()),
        "raw_nonzero_sao2_rows_le_50": int(
            (valid_sao2.gt(0) & valid_sao2.le(50)).sum()
        ),
        "raw_nonzero_sao2_rows_le_70": int(
            (valid_sao2.gt(0) & valid_sao2.le(70)).sum()
        ),
        "raw_nonzero_sao2_rows_le_80": int(
            (valid_sao2.gt(0) & valid_sao2.le(80)).sum()
        ),
        "raw_sao2_outside_0_100_rows": int(
            (valid_sao2.lt(0) | valid_sao2.gt(100)).sum()
        ),
        "patient_hours_with_periodic_data": len(hourly),
        "experimental_hourly_rows": len(experiment),
        "experimental_hours_without_periodic_rows": int(
            experiment_only_hours["_merge"].eq("left_only").sum()
        ),
        "periodic_hours_missing_from_experiment": int(
            hourly["_merge"].eq("left_only").sum()
        ),
        "mixed_zero_patient_hours": int(mixed_zero.sum()),
        "zero_only_patient_hours": int(
            hourly["spo2_quality_flag_raw_recomputed"]
            .eq("ZERO_ONLY")
            .sum()
        ),
        "missing_only_patient_hours": int(
            hourly["spo2_quality_flag_raw_recomputed"]
            .eq("MISSING_ONLY")
            .sum()
        ),
        "mixed_zero_corrected_mean_gt_80_hours": int(
            corrected_mixed.gt(80).sum()
        ),
        "mixed_zero_corrected_mean_gt_50_le_80_hours": int(
            (corrected_mixed.gt(50) & corrected_mixed.le(80)).sum()
        ),
        "mixed_zero_corrected_mean_le_50_hours": int(
            corrected_mixed.le(50).sum()
        ),
        "hourly_raw_mean_mismatches_vs_experiment": int(
            (~hourly["spo2_raw_mean_matches_experiment"]).sum()
        ),
        "hourly_corrected_mean_mismatches_vs_experiment": int(
            (~hourly["corrected_mean_matches_experiment"]).sum()
        ),
        "hourly_quality_flag_mismatches_vs_experiment": int(
            (~hourly["quality_flag_matches_experiment"]).sum()
        ),
        "low_nonzero_readings_le_50": len(low_readings),
        "stays_with_low_nonzero_readings_le_50": low_readings[
            "patientunitstayid"
        ].nunique(),
        "low_reading_clusters": len(episodes),
        "stays_with_low_reading_clusters": episodes[
            "patientunitstayid"
        ].nunique(),
        "clusters_with_corroborating_direct_saturation": int(
            episodes["evidence_classification"].isin(
                [
                    "CORROBORATING_DIRECT_SATURATION",
                    "MIXED_DIRECT_SATURATION_EVIDENCE",
                ]
            ).sum()
        ),
        "clusters_with_conflicting_direct_saturation": int(
            episodes["evidence_classification"].isin(
                [
                    "CONFLICTING_DIRECT_SATURATION",
                    "MIXED_DIRECT_SATURATION_EVIDENCE",
                ]
            ).sum()
        ),
        "clusters_without_direct_saturation_reference": int(
            episodes["evidence_classification"]
            .eq("NO_DIRECT_SATURATION_REFERENCE")
            .sum()
        ),
        "clusters_with_intermediate_direct_saturation": int(
            episodes["evidence_classification"]
            .eq("INTERMEDIATE_DIRECT_SATURATION")
            .sum()
        ),
        "direct_saturation_reference_matches": int(
            matches["evidence_role"].eq("DIRECT_SATURATION").sum()
        ),
        "map_context_matches_not_confirmation": int(
            matches["reference_source"].eq("map_context").sum()
        ),
        "window_minutes_inclusive": WINDOW_MINUTES,
        "cluster_gap_minutes": EPISODE_GAP_MINUTES,
    }
    corrected_hourly = pd.to_numeric(
        hourly["spo2_corrected_mean"],
        errors="coerce",
    )
    for threshold in [50, 70, 80]:
        under_threshold = corrected_hourly.le(threshold)
        summary[f"corrected_hourly_spo2_le_{threshold}_hours"] = int(
            under_threshold.sum()
        )
        summary[f"corrected_hourly_spo2_le_{threshold}_stays"] = int(
            hourly.loc[
                under_threshold,
                "patientunitstayid",
            ].nunique()
        )
    summary_frame = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    )
    summary_frame.to_csv(
        OUTPUT_DIR / "spo2_audit_summary.csv",
        index=False,
    )

    print("SpO2 cross-validation complete.")
    print(f"In-stay periodic observations: {len(periodic):,}")
    print(f"Mixed-zero hours: {int(mixed_zero.sum()):,}")
    print(f"Zero-only hours: {summary['zero_only_patient_hours']:,}")
    print(
        "Low nonzero readings (<=50): "
        f"{len(low_readings):,} across {len(episodes):,} clusters"
    )
    print(
        "Hourly mismatches versus experiment: "
        f"raw={summary['hourly_raw_mean_mismatches_vs_experiment']}, "
        "corrected="
        f"{summary['hourly_corrected_mean_mismatches_vs_experiment']}, "
        "flags="
        f"{summary['hourly_quality_flag_mismatches_vs_experiment']}"
    )
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
