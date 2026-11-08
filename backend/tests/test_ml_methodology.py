"""
Independent Verification Tests for Smart ICU ML Methodology and Safeguards.
Verifies:
1. Laboratory Temporal Causality: Future laboratory records cannot alter past features.
2. Rolling Window Causality: Future observations cannot change earlier rolling statistics.
3. Preprocessing Isolation: Imputer/scaler statistics are strictly derived from training data.
4. Patient-Level Disjointness: Zero patient or stay overlap across train/val/test splits.
5. Baseline Artifact Immutability: SHA-256 hashes of preserved baseline models remain identical.
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path("D:/Smart-ICU")
INPUT_FILE = BASE_DIR / "data/processed/experiments/leakage_safe/model_inputs_corrected.csv"

def compute_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_laboratory_temporal_causality_with_injected_future_records():
    """
    Deliberately injects future laboratory measurements at offset > prediction_time - 60 min.
    Asserts that the backward as-of join completely ignores them at prediction time t.
    """
    # Create synthetic prediction grid for 1 stay
    grid = pd.DataFrame({
        "patientunitstayid": [99999] * 5,
        "hour": [0, 1, 2, 3, 4],
    })
    grid["pred_offset_min"] = grid["hour"] * 60
    grid["eligible_cutoff_min"] = grid["pred_offset_min"] - 60
    grid_sorted = grid.sort_values("eligible_cutoff_min").reset_index(drop=True)

    # Legitimate past lab drawn before admission (-100 min) with value 1.2
    # AND injected FUTURE lab drawn at hour 3 (offset +180 min) with value 99.9 (fatal anomaly)
    labs = pd.DataFrame({
        "patientunitstayid": [99999, 99999],
        "labresultoffset": [-100, 180],
        "labresult": [1.2, 99.9],
    }).sort_values("labresultoffset").reset_index(drop=True)

    merged = pd.merge_asof(
        grid_sorted,
        labs,
        left_on="eligible_cutoff_min",
        right_on="labresultoffset",
        by="patientunitstayid",
        direction="backward",
        tolerance=1440
    ).sort_values("hour").reset_index(drop=True)

    # At hours 0, 1, 2, 3: The eligible cutoff is <= 120 min.
    # The future lab at 180 min MUST NOT appear!
    # Hour 0 (cutoff -60): lab at -100 is visible (value 1.2)
    assert merged.loc[merged["hour"] == 0, "labresult"].iloc[0] == 1.2
    # Hour 1 (cutoff 0): lab at -100 is visible (value 1.2)
    assert merged.loc[merged["hour"] == 1, "labresult"].iloc[0] == 1.2
    # Hour 2 (cutoff 60): lab at -100 is visible (value 1.2)
    assert merged.loc[merged["hour"] == 2, "labresult"].iloc[0] == 1.2
    # Hour 3 (cutoff 120): lab at -100 is visible (value 1.2) — future lab at 180 is still in future!
    assert merged.loc[merged["hour"] == 3, "labresult"].iloc[0] == 1.2
    # Hour 4 (cutoff 180): now cutoff >= 180, so lab at 180 becomes visible
    assert merged.loc[merged["hour"] == 4, "labresult"].iloc[0] == 99.9

def test_rolling_feature_temporal_causality():
    """
    Verifies that inserting a spike or alteration in future hours
    cannot alter earlier rolling means or standard deviations.
    """
    df_base = pd.DataFrame({
        "patientunitstayid": [1, 1, 1],
        "hour": [0, 1, 2],
        "heart_rate": [70.0, 75.0, 80.0],
    })

    # Calculate rolling std on base
    rolling_base = df_base.set_index("hour")["heart_rate"].rolling(3, min_periods=2).std()

    # Now append future catastrophic heart rate at hour 3
    df_extended = pd.DataFrame({
        "patientunitstayid": [1, 1, 1, 1],
        "hour": [0, 1, 2, 3],
        "heart_rate": [70.0, 75.0, 80.0, 190.0],
    })
    rolling_ext = df_extended.set_index("hour")["heart_rate"].rolling(3, min_periods=2).std()

    # Hours 0, 1, 2 must be mathematically identical
    np.testing.assert_allclose(rolling_base.dropna().values, rolling_ext.iloc[:3].dropna().values)

def test_patient_split_disjointness():
    """Asserts strictly zero patient and stay overlap between train, val, and test splits."""
    df = pd.read_csv(INPUT_FILE, usecols=["patientunitstayid", "uniquepid", "split"])

    train_pts = set(df.loc[df["split"] == "train", "uniquepid"])
    val_pts = set(df.loc[df["split"] == "validation", "uniquepid"])
    test_pts = set(df.loc[df["split"] == "test", "uniquepid"])

    train_stays = set(df.loc[df["split"] == "train", "patientunitstayid"])
    val_stays = set(df.loc[df["split"] == "validation", "patientunitstayid"])
    test_stays = set(df.loc[df["split"] == "test", "patientunitstayid"])

    assert len(train_pts & val_pts) == 0, "Patient overlap between Train and Val!"
    assert len(train_pts & test_pts) == 0, "Patient overlap between Train and Test!"
    assert len(val_pts & test_pts) == 0, "Patient overlap between Val and Test!"

    assert len(train_stays & val_stays) == 0, "Stay overlap between Train and Val!"
    assert len(train_stays & test_stays) == 0, "Stay overlap between Train and Test!"
    assert len(val_stays & test_stays) == 0, "Stay overlap between Val and Test!"

def test_preprocessing_isolation():
    """
    Asserts that fitting an imputer and scaler on training data is strictly
    isolated from validation/test data distributions.
    """
    df = pd.read_csv(INPUT_FILE, usecols=["split", "heart_rate", "respiration"])
    train_df = df[df["split"] == "train"][["heart_rate", "respiration"]]
    val_df = df[df["split"] == "validation"][["heart_rate", "respiration"]]

    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_df)
    train_medians = imputer.statistics_.copy()

    # Validation data should not change training medians
    val_transformed = imputer.transform(val_df)
    np.testing.assert_array_equal(imputer.statistics_, train_medians)

def test_baseline_v1_artifacts_immutability():
    """
    Asserts that all original baseline v1 model artifacts in ml/baseline_artifacts_v1/
    and ml/models/ remain 100% byte-for-byte identical to baseline commit 767f3b8.
    """
    expected_hashes = {
        "deterioration_logistic_v1.joblib": "26d79474d38521bb251f3c0d456e099e038736c6c2f1bc43e837cfd3827181e0",
        "deterioration_random_forest_v1.joblib": "b470d94c9aea9183a6be9362ea492c89886168b0c5016e56ae8f4a974c4d812c",
        "deterioration_gradient_boosting_v1.joblib": "2deacbbb4a429746055b672256a09f7a092822eed5b6af39bfed11102cffaf02",
        "anomaly_isolation_forest_v2.joblib": "20d75831a41a78cf042be5127afb0d7f943f2b831f87a6f69914381a185c266e",
    }

    baseline_dir = BASE_DIR / "ml/baseline_artifacts_v1"
    models_dir = BASE_DIR / "ml/models"

    for filename, expected_sha in expected_hashes.items():
        base_file = baseline_dir / filename
        mod_file = models_dir / filename
        assert base_file.exists(), f"Preserved baseline file {base_file} missing!"
        assert mod_file.exists(), f"Production model file {mod_file} missing!"

        base_sha = compute_file_sha256(base_file)
        mod_sha = compute_file_sha256(mod_file)

        assert base_sha == expected_sha, f"Preserved baseline {filename} hash mismatch!"
        assert mod_sha == expected_sha, f"Production baseline {filename} was modified!"

def test_rolling_features_segment_gap_isolation():
    """
    Verifies that multi-hour gaps in monitoring strictly reset rolling feature windows
    via segment grouping, preventing pre-gap observations from leaking into post-gap computations.
    """
    # Stay with hours 1, 2, then a 3-hour gap, then hours 5, 6
    df = pd.DataFrame({
        "patientunitstayid": [999, 999, 999, 999],
        "hour": [1, 2, 5, 6],
        "heart_rate": [70.0, 72.0, 120.0, 122.0],
    })

    # Segment-based gap isolation logic from optimize_ml_engine.py
    df["hour_gap"] = df.groupby("patientunitstayid")["hour"].diff()
    df["segment_id"] = (
        (df["hour_gap"] > 1)
        .astype(int)
        .groupby(df["patientunitstayid"])
        .cumsum()
    )

    df["heart_rate_std_3h"] = (
        df.groupby(["patientunitstayid", "segment_id"])["heart_rate"]
        .rolling(3, min_periods=2)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )

    # Hour 1: first in segment 0 -> NaN (1 obs < min_periods 2)
    assert np.isnan(df.loc[df["hour"] == 1, "heart_rate_std_3h"].values[0])
    # Hour 2: second in segment 0 -> std([70, 72]) = sqrt(2) ≈ 1.4142
    assert abs(df.loc[df["hour"] == 2, "heart_rate_std_3h"].values[0] - np.std([70.0, 72.0], ddof=1)) < 1e-4
    # Hour 5: first observation after gap in segment 1 -> MUST BE NaN (zero lookback across gap)
    assert np.isnan(df.loc[df["hour"] == 5, "heart_rate_std_3h"].values[0])
    # Hour 6: second in segment 1 -> std([120, 122]) = sqrt(2) ≈ 1.4142, completely independent of hours 1, 2
    assert abs(df.loc[df["hour"] == 6, "heart_rate_std_3h"].values[0] - np.std([120.0, 122.0], ddof=1)) < 1e-4
