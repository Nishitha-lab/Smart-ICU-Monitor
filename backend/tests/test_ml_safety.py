"""Tests for ML safety guarantees, missing artifact detection, and no fabricated predictions."""

import asyncio
import json
import main
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


def test_ml_status_reports_unavailable_accurately():
    status, data = request_app("/api/ml/status")
    assert status == 200

    det = data["deterioration_model"]
    assert det["available"] is False
    assert det["status"] == "UNAVAILABLE"
    assert det["target_definition"] == "ICU discharge marked expired within 24 hours"
    assert "not found" in det["notes"].lower()

    anom = data["anomaly_model"]
    assert anom["available"] is False
    assert anom["status"] == "UNAVAILABLE"
    assert "Statistical multi-variable outlier detection" in anom["statistical_interpretation"]

    rules = data["clinical_rules_status"]
    assert rules["status"] == "ACTIVE"


def test_predict_deterioration_returns_unavailable_without_fabrication():
    res = ml_service.predict_deterioration(None)
    assert res["available"] is False
    assert res["status"] == "UNAVAILABLE"
    assert res["probabilities"] is None
    assert res["target"] == "ICU discharge marked expired within 24 hours"


def test_score_anomalies_returns_unavailable_without_fabrication():
    res = ml_service.score_anomalies(None)
    assert res["available"] is False
    assert res["status"] == "UNAVAILABLE"
    assert res["scores"] is None
    assert res["flags"] is None
