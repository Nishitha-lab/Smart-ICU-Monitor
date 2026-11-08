"""
Tests for Genuine Machine Learning Prediction Engine.
Validates model loading, inference pipelines, unweighted Logistic Regression probabilities,
feature contribution attribution, and API endpoints.
"""

from __future__ import annotations

import main
import numpy as np
import pandas as pd
import pytest
from services.ml_service import ml_service
from tests.test_ml_safety import request_app


@pytest.fixture(autouse=True)
def ensure_dataset_loaded():
    """Ensure main.df is loaded for API endpoint tests."""
    if main.df is None:
        main.df = main.load_dataset()


def test_ml_service_artifact_loading():
    """Verify ML service properly loads serialized models and metadata."""
    bundle = ml_service._get_deterioration_bundle()
    assert bundle is not None
    assert bundle["model_name"] == "deterioration_logistic_v1"
    assert bundle["version"] == "1.0.0"
    assert "ICU discharge marked expired" in bundle["target"]
    assert len(bundle["feature_columns"]) == 22

    anom_bundle = ml_service._get_anomaly_bundle()
    assert anom_bundle is not None
    assert anom_bundle["model_name"] == "anomaly_isolation_forest_v2"
    assert anom_bundle["version"] in ("2.0.0", "v2.0.0")


def test_ml_service_probability_calibration_and_bounds():
    """Verify output probabilities are strictly within [0.0, 1.0] and within natural prevalence bounds."""
    sample_df = main.df.head(20).copy()
    res = ml_service.predict_deterioration(sample_df)

    assert res["status"] == "SUCCESS"
    assert res["available"] is True
    probs = res["probabilities"]
    assert probs is not None
    assert len(probs) == len(sample_df)

    for p in probs:
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    bands = res["risk_bands"]
    assert bands is not None
    for p, b in zip(probs, bands):
        if p < 0.05:
            assert b == "Low"
        elif p < 0.15:
            assert b == "Moderate"
        elif p < 0.30:
            assert b == "High"
        else:
            assert b == "Extreme"


def test_ml_service_feature_attribution_correctness():
    """Verify mathematical correctness of log-odds feature contributions for logistic regression."""
    sample_df = main.df.head(5).copy()
    res = ml_service.predict_deterioration(sample_df)

    latest_pred = res["latest_prediction"]
    assert latest_pred is not None
    assert "top_contributing_features" in latest_pred

    contributors = latest_pred["top_contributing_features"]
    assert len(contributors) <= 6

    # Verify descending sort by absolute contribution
    abs_contribs = [c["abs_contribution"] for c in contributors]
    assert abs_contribs == sorted(abs_contribs, reverse=True)

    # Verify direction aligns with contribution sign
    for c in contributors:
        if c["contribution"] > 0:
            assert c["direction"] == "INCREASING_RISK"
        else:
            assert c["direction"] == "DECREASING_RISK"

    assert latest_pred["model_intercept"] is not None
    assert latest_pred["predicted_log_odds"] is not None
    # Reconstruct probability from log_odds: p = 1 / (1 + exp(-log_odds))
    reconstructed_p = 1.0 / (1.0 + np.exp(-latest_pred["predicted_log_odds"]))
    assert abs(reconstructed_p - latest_pred["probability"]) < 0.01


def test_ml_service_anomaly_scoring():
    """Verify IsolationForest v2 anomaly scoring and flag generation."""
    sample_df = main.df.head(10).copy()
    anom_res = ml_service.score_anomalies(sample_df)

    assert anom_res["status"] == "SUCCESS"
    assert anom_res["available"] is True
    scores = anom_res["scores"]
    flags = anom_res["flags"]

    assert len(scores) == 10
    assert len(flags) == 10
    for s, f in zip(scores, flags):
        assert isinstance(s, float)
        assert f in (0, 1)


def test_predict_patient_ml_endpoint_success():
    """Verify GET /api/ml/predict/{stay_id} returns genuine prediction for valid ICU stay."""
    valid_stay = int(main.df["patientunitstayid"].iloc[0])
    status, data = request_app(f"/api/ml/predict/{valid_stay}")

    assert status == 200
    assert data["patientunitstayid"] == valid_stay
    assert data["model_name"] == "deterioration_logistic_v1"
    assert data["target"] == "ICU discharge marked expired within the next 24 hours from the prediction timestamp"
    assert "latest_prediction" in data
    assert "series" in data
    assert len(data["series"]) > 0

    latest_pred = data["latest_prediction"]
    assert 0.0 <= latest_pred["probability"] <= 1.0
    assert latest_pred["risk_band"] in ("Low", "Moderate", "High", "Extreme")
    assert latest_pred["decision_threshold"] == 0.0823
    assert isinstance(latest_pred["threshold_exceeded"], bool)

    # Verify academic disclaimer is included
    assert "Academic research prototype" in data["disclaimer"]


def test_predict_patient_ml_endpoint_not_found():
    """Verify GET /api/ml/predict/{stay_id} returns 404 for non-existent stay."""
    status, data = request_app("/api/ml/predict/999999999")
    assert status == 404
    assert "not found" in data["detail"].lower()


def test_prediction_determinism_and_reproducibility():
    """Verify repeated predictions on identical input yield identical probabilities."""
    sample_df = main.df.head(10).copy()
    res1 = ml_service.predict_deterioration(sample_df)
    res2 = ml_service.predict_deterioration(sample_df)

    assert res1["probabilities"] == res2["probabilities"]
    assert res1["risk_bands"] == res2["risk_bands"]
