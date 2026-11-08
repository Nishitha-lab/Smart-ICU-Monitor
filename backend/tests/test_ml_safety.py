"""Tests for ML safety guarantees, missing artifact detection, and no fabricated predictions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import main
import pandas as pd
from services.ml_service import ml_service


def request_app(
    path: str,
    method: str = "GET",
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    body = json.dumps(payload).encode() if payload is not None else b""

    async def call_app() -> tuple[int, dict[str, object]]:
        messages: list[dict[str, object]] = []
        request_sent = False

        async def receive() -> dict[str, object]:
            nonlocal request_sent
            if not request_sent:
                request_sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        raw_path, _, query_string = path.partition("?")
        await main.app(
            {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": method,
                "scheme": "http",
                "path": raw_path,
                "raw_path": raw_path.encode(),
                "query_string": query_string.encode(),
                "root_path": "",
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8000),
            },
            receive,
            send,
        )
        response = next(message for message in messages if message["type"] == "http.response.start")
        response_body = next(message for message in messages if message["type"] == "http.response.body")
        return int(response["status"]), json.loads(response_body["body"])

    return asyncio.run(call_app())


def test_ml_status_reports_online_when_artifacts_present():
    """Verify that when real model artifacts exist, status reports ONLINE with genuine metrics."""
    status, data = request_app("/api/ml/status")
    assert status == 200
    assert data["status"] == "ONLINE"

    det = data["deterioration_model"]
    assert det["available"] is True
    assert det["status"] == "AVAILABLE"
    assert det["target_definition"] == "ICU discharge marked expired within the next 24 hours from the prediction timestamp"
    assert det["version"] == "1.0.0"

    metrics = det["validation_metrics"]
    assert metrics is not None
    assert metrics["roc_auc"] == 0.6875
    assert metrics["brier_score"] == 0.0154
    assert metrics["decision_threshold"] == 0.0823

    anom = data["anomaly_model"]
    assert anom["available"] is True
    assert anom["status"] == "AVAILABLE"
    assert "Statistical multi-variable outlier detection" in anom["statistical_interpretation"]

    rules = data["clinical_rules_status"]
    assert rules["status"] == "ACTIVE"


def test_ml_status_reports_unavailable_when_artifacts_missing():
    """Verify that if artifacts are missing, status strictly reports UNAVAILABLE without guessing."""
    with patch("services.ml_service.DETERIORATION_ARTIFACT", Path("non_existent_det.joblib")), \
         patch("services.ml_service.ANOMALY_ARTIFACT", Path("non_existent_anom.joblib")):
        status, data = request_app("/api/ml/status")
        assert status == 200
        assert data["status"] == "UNAVAILABLE"

        det = data["deterioration_model"]
        assert det["available"] is False
        assert det["status"] == "UNAVAILABLE"
        assert det["target_definition"] == "ICU discharge marked expired within the next 24 hours from the prediction timestamp"
        assert "not found" in det["notes"].lower()

        anom = data["anomaly_model"]
        assert anom["available"] is False
        assert anom["status"] == "UNAVAILABLE"


def test_predict_deterioration_returns_unavailable_when_artifact_missing():
    """Verify predict_deterioration returns UNAVAILABLE without fabricating probabilities."""
    with patch("services.ml_service.DETERIORATION_ARTIFACT", Path("non_existent_det.joblib")):
        res = ml_service.predict_deterioration(pd.DataFrame([{"heart_rate": 80}]))
        assert res["available"] is False
        assert res["status"] == "UNAVAILABLE"
        assert res["probabilities"] is None
        assert res["target"] == "ICU discharge marked expired within the next 24 hours from the prediction timestamp"


def test_score_anomalies_returns_unavailable_when_artifact_missing():
    """Verify score_anomalies returns UNAVAILABLE without fabricating anomaly scores."""
    with patch("services.ml_service.ANOMALY_ARTIFACT", Path("non_existent_anom.joblib")):
        res = ml_service.score_anomalies(pd.DataFrame([{"heart_rate": 80}]))
        assert res["available"] is False
        assert res["status"] == "UNAVAILABLE"
        assert res["scores"] is None
        assert res["flags"] is None


def test_predict_deterioration_rejects_incompatible_features():
    """Verify feature validation: missing required features cannot produce fake predictions."""
    incomplete_data = pd.DataFrame([{"heart_rate": 80, "spo2": 98}])
    res = ml_service.predict_deterioration(incomplete_data)
    assert res["available"] is False
    assert res["status"] == "INCOMPATIBLE_FEATURES"
    assert res["probabilities"] is None
    assert "Missing required features" in res["error"]
