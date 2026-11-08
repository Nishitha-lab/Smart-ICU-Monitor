"""
Regression test suite for Phase 3 Candidate ML Inference Engine.

Verifies:
1. Candidate artifact deserialization and metadata integrity.
2. Exact 52-feature schema validation against optimization metrics.
3. Strict data source boundaries (historical eICU allowed; live telemetry/manual entry rejected).
4. Safe handling of missing laboratory features (NaN values).
5. Numerical validity of output probabilities (bounded in [0, 1], non-NaN, non-inf).
6. Transparent fallback to baseline v1 with provenance tracking when allow_fallback=True.
7. Baseline v1 immutability and preservation as default production model.
8. Deterministic scoring across repeated invocations.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from services.ml_service import ml_service, CANDIDATE_ARTIFACT, CANDIDATE_METRICS_PATH, DETERIORATION_ARTIFACT


def is_candidate_deserializable() -> bool:
    """Check if candidate artifact can be deserialized in the current runtime environment."""
    return ml_service._get_candidate_bundle() is not None


CANDIDATE_INCOMPATIBLE_REASON = (
    f"Candidate v2 artifact was serialized under scikit-learn 1.8.0 and cannot be deserialized "
    f"under runtime scikit-learn {sklearn.__version__} (ModuleNotFoundError: No module named '_loss'). "
    f"Execute in dedicated candidate evaluation environment (ml/.venv_candidate) with scikit-learn 1.8.0."
)


@pytest.fixture
def sample_52_features() -> list[str]:
    """Load ordered 52 features from candidate bundle if available, or optimization metrics."""
    bundle = ml_service._get_candidate_bundle()
    if bundle is not None:
        return list(bundle["features"])
    metrics = ml_service._load_candidate_metrics()
    if metrics and "features" in metrics:
        return list(metrics["features"])
    raise RuntimeError("Candidate features could not be loaded from bundle or metrics")


@pytest.fixture
def valid_candidate_dataframe(sample_52_features: list[str]) -> pd.DataFrame:
    """Creates a synthetic single-stay 5-hour trajectory with all 52 features."""
    np.random.seed(42)
    n_rows = 5
    data = {}
    for col in sample_52_features:
        if "is_missing" in col:
            data[col] = 0.0
        elif "change" in col:
            data[col] = np.random.uniform(-5.0, 5.0, size=n_rows)
        elif "rolling" in col or "std" in col:
            data[col] = np.random.uniform(1.0, 10.0, size=n_rows)
        elif "lab" in col:
            data[col] = np.random.uniform(0.5, 5.0, size=n_rows)
        elif "bp" in col:
            data[col] = np.random.uniform(70.0, 130.0, size=n_rows)
        elif "shock" in col:
            data[col] = np.random.uniform(0.5, 1.2, size=n_rows)
        elif "pulse" in col:
            data[col] = np.random.uniform(30.0, 60.0, size=n_rows)
        else:
            data[col] = np.random.uniform(60.0, 100.0, size=n_rows)
    return pd.DataFrame(data)


def test_candidate_bundle_structure_and_deserialization():
    """Verify deserialization of candidate artifact bundle and all required top-level keys."""
    if not is_candidate_deserializable():
        pytest.skip(CANDIDATE_INCOMPATIBLE_REASON)

    bundle = ml_service._get_candidate_bundle()
    assert bundle is not None
    required_keys = [
        "pipeline", "calibrator", "features", "feature_count",
        "decision_threshold", "threshold_status", "calibration_method",
        "episode_policy", "git_commit_sha", "dataset_hashes",
        "model_name", "target", "feature_set_name"
    ]
    for key in required_keys:
        assert key in bundle, f"Missing key '{key}' in candidate bundle"

    assert bundle["feature_count"] == 52
    assert len(bundle["features"]) == 52
    assert bundle["decision_threshold"] > 0.0
    assert bundle["threshold_status"] == "FEASIBLE THRESHOLD IDENTIFIED"
    assert "hourly_patient_state" in bundle["dataset_hashes"]
    assert "model_inputs_corrected" in bundle["dataset_hashes"]


def test_candidate_features_match_optimization_metrics(sample_52_features: list[str]):
    """Verify that serialized features match optimization_metrics.json 100% in count, names, and order."""
    if not is_candidate_deserializable():
        pytest.skip(CANDIDATE_INCOMPATIBLE_REASON)

    metrics = ml_service._load_candidate_metrics()
    assert metrics is not None
    metrics_features = metrics["features"]
    assert len(sample_52_features) == len(metrics_features) == 52
    assert sample_52_features == metrics_features



def test_candidate_rejects_live_telemetry_source(valid_candidate_dataframe: pd.DataFrame):
    """Verify that live telemetry is strictly rejected with UNSUPPORTED_DATA_SOURCE."""
    result = ml_service.predict_deterioration_candidate(
        valid_candidate_dataframe,
        data_source="live_telemetry",
        allow_fallback=False,
    )
    assert result["available"] is False
    assert result["status"] == "UNSUPPORTED_DATA_SOURCE"
    assert "historical eICU" in result["error"]
    assert result["fallback_applied"] is False
    assert result["probabilities"] is None


def test_candidate_rejects_manual_entry_source(valid_candidate_dataframe: pd.DataFrame):
    """Verify that manual entry data source is strictly rejected."""
    result = ml_service.predict_deterioration_candidate(
        valid_candidate_dataframe,
        data_source="manual_entry",
        allow_fallback=False,
    )
    assert result["available"] is False
    assert result["status"] == "UNSUPPORTED_DATA_SOURCE"
    assert result["fallback_applied"] is False


def test_candidate_rejects_unknown_data_source(valid_candidate_dataframe: pd.DataFrame):
    """Verify that unknown/arbitrary data sources are strictly rejected."""
    result = ml_service.predict_deterioration_candidate(
        valid_candidate_dataframe,
        data_source="unknown_ehr_feed",
        allow_fallback=False,
    )
    assert result["available"] is False
    assert result["status"] == "UNSUPPORTED_DATA_SOURCE"


def test_candidate_accepts_historical_eicu_source(valid_candidate_dataframe: pd.DataFrame):
    """Verify that historical eICU data source is accepted and produces valid predictions."""
    if not is_candidate_deserializable():
        pytest.skip(CANDIDATE_INCOMPATIBLE_REASON)

    result = ml_service.predict_deterioration_candidate(
        valid_candidate_dataframe,
        data_source="historical_eicu",
        allow_fallback=False,
    )
    assert result["available"] is True
    assert result["status"] == "SUCCESS"
    assert result["fallback_applied"] is False
    assert result["fallback_reason"] is None
    assert result["feature_count"] == 52
    assert len(result["probabilities"]) == len(valid_candidate_dataframe)
    assert len(result["risk_bands"]) == len(valid_candidate_dataframe)
    assert result["latest_prediction"] is not None


def test_candidate_rejects_missing_features(valid_candidate_dataframe: pd.DataFrame):
    """Verify that omitting required features returns INCOMPATIBLE_FEATURES when allow_fallback=False."""
    if not is_candidate_deserializable():
        pytest.skip(CANDIDATE_INCOMPATIBLE_REASON)

    incomplete_df = valid_candidate_dataframe.drop(columns=["lab_creatinine_lagged", "shock_index"])
    result = ml_service.predict_deterioration_candidate(
        incomplete_df,
        data_source="historical_eicu",
        allow_fallback=False,
    )
    assert result["available"] is False
    assert result["status"] == "INCOMPATIBLE_FEATURES"
    assert "Missing required candidate v2 features" in result["error"]
    assert result["fallback_applied"] is False


def test_candidate_handles_nan_lab_features_safely(valid_candidate_dataframe: pd.DataFrame):
    """Verify that NaN laboratory values are handled safely without raising unhandled exceptions."""
    if not is_candidate_deserializable():
        pytest.skip(CANDIDATE_INCOMPATIBLE_REASON)

    df_with_nans = valid_candidate_dataframe.copy()
    lab_cols = [c for c in df_with_nans.columns if "lab" in c]
    for col in lab_cols:
        df_with_nans[col] = np.nan
        if f"{col}_is_missing" in df_with_nans.columns:
            df_with_nans[f"{col}_is_missing"] = 1.0

    result = ml_service.predict_deterioration_candidate(
        df_with_nans,
        data_source="historical_eicu",
        allow_fallback=False,
    )
    assert result["available"] is True
    assert result["status"] == "SUCCESS"
    for p in result["probabilities"]:
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0
        assert not np.isnan(p)
        assert not np.isinf(p)


def test_candidate_output_validity_bounds(valid_candidate_dataframe: pd.DataFrame):
    """Verify that candidate probabilities are strictly in [0.0, 1.0] and risk bands match thresholds."""
    if not is_candidate_deserializable():
        pytest.skip(CANDIDATE_INCOMPATIBLE_REASON)

    result = ml_service.predict_deterioration_candidate(
        valid_candidate_dataframe,
        data_source="historical_eicu",
    )
    assert result["status"] == "SUCCESS"
    for p, band in zip(result["probabilities"], result["risk_bands"]):
        assert 0.0 <= p <= 1.0
        if p < 0.05:
            assert band == "Low"
        elif p < 0.15:
            assert band == "Moderate"
        elif p < 0.30:
            assert band == "High"
        else:
            assert band == "Extreme"


def test_candidate_fallback_to_v1_when_requested(valid_candidate_dataframe: pd.DataFrame):
    """Verify that when allow_fallback=True and data source is unsupported, fallback to v1 succeeds."""
    result = ml_service.predict_deterioration_candidate(
        valid_candidate_dataframe,
        data_source="live_telemetry",
        allow_fallback=True,
    )
    assert result["available"] is True
    assert result["status"] == "SUCCESS"
    assert result["fallback_applied"] is True
    assert "Fell back to production baseline v1" in result["fallback_reason"]
    assert result["model_name"] == "deterioration_logistic_v1"
    assert result["model_version"] == "1.0.0"


def test_candidate_fallback_fails_if_baseline_incompatible():
    """Verify that if fallback is allowed but records cannot satisfy baseline v1, fails safely."""
    minimal_df = pd.DataFrame({"random_col": [1.0, 2.0]})
    result = ml_service.predict_deterioration_candidate(
        minimal_df,
        data_source="live_telemetry",
        allow_fallback=True,
    )
    assert result["available"] is False
    assert result["fallback_applied"] is False


def test_candidate_determinism(valid_candidate_dataframe: pd.DataFrame):
    """Verify that repeated candidate predictions on identical inputs yield identical outputs."""
    if not is_candidate_deserializable():
        pytest.skip(CANDIDATE_INCOMPATIBLE_REASON)

    res1 = ml_service.predict_deterioration_candidate(valid_candidate_dataframe, data_source="historical_eicu")
    res2 = ml_service.predict_deterioration_candidate(valid_candidate_dataframe, data_source="historical_eicu")
    assert res1["probabilities"] == res2["probabilities"]
    assert res1["risk_bands"] == res2["risk_bands"]


def test_get_status_preserves_baseline_as_default():
    """Verify that ml_service.get_status() preserves baseline v1 as default and candidate as unpromoted."""
    status = ml_service.get_status()
    assert status["status"] == "ONLINE"
    assert status["deterioration_model"]["model_name"] == "deterioration_logistic_v1"
    assert status["deterioration_model"]["version"] == "1.0.0"

    cand = status["candidate_v2"]
    assert cand["promotion_status"] == "UNPROMOTED_CANDIDATE"
    assert cand["default_production_model"] is False
    assert cand["supported_data_sources"] == ["historical_eicu"]
    assert cand["feature_count"] == 52

    if is_candidate_deserializable():
        assert cand["available"] is True
        assert cand["status"] == "AVAILABLE_FOR_EVALUATION"
        assert cand["runtime_compatible"] is True
    else:
        assert cand["available"] is False
        assert cand["status"] == "UNAVAILABLE_INCOMPATIBLE_RUNTIME"
        assert cand["runtime_compatible"] is False
        assert cand["incompatibility_reason"] is not None


def test_candidate_runtime_incompatibility_reported_when_fallback_disabled(valid_candidate_dataframe: pd.DataFrame):
    """Verify that when runtime is incompatible and fallback is disabled, UNAVAILABLE_INCOMPATIBLE_RUNTIME is returned."""
    if is_candidate_deserializable():
        pytest.skip("Test applies only to runtimes incompatible with Candidate v2 (e.g. scikit-learn 1.9.0).")

    res = ml_service.predict_deterioration_candidate(
        valid_candidate_dataframe,
        data_source="historical_eicu",
        allow_fallback=False,
    )
    assert res["available"] is False
    assert res["status"] == "UNAVAILABLE_INCOMPATIBLE_RUNTIME"
    assert res["fallback_applied"] is False
    assert "scikit-learn 1.8.0" in res["error"] or "incompatible" in res["error"]


def test_candidate_runtime_incompatibility_falls_back_when_enabled(valid_candidate_dataframe: pd.DataFrame):
    """Verify that when runtime is incompatible and fallback is enabled on historical eICU data, falls back to v1."""
    if is_candidate_deserializable():
        pytest.skip("Test applies only to runtimes incompatible with Candidate v2 (e.g. scikit-learn 1.9.0).")

    res = ml_service.predict_deterioration_candidate(
        valid_candidate_dataframe,
        data_source="historical_eicu",
        allow_fallback=True,
    )
    assert res["available"] is True
    assert res["status"] == "SUCCESS"
    assert res["fallback_applied"] is True
    assert res["candidate_requested"] is True
    assert res["model_name"] == "deterioration_logistic_v1"
    assert "incompatible with runtime" in res["fallback_reason"]
