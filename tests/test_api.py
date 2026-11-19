import pytest
from fastapi.testclient import TestClient

from hybrid_risk import app as app_module

VALID = {"review_text": "The zipper broke after one wear and support never replied.", "age": 35,
         "positive_feedback_count": 2, "division_name": "General", "department_name": "Dresses",
         "class_name": "Dresses"}

FAKE_RESULT = {
    "model_outputs": {"final_score": 0.45,
                      "probabilities": {"low": 0.3, "medium": 0.3, "high": 0.4}, "risk_level": "high",
                      "recommended_action": "escalate_immediately", "human_review_required": True,
                      "review_reasons": []},
    "llm_explanation": {"summary": "s", "recommended_action": "escalate_immediately", "reason": "r",
                        "human_review_required": True, "risk_notes": ["n"]},
    "_meta": {"explanation_source": "fallback"},
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.hybrid_inference, "predict_hybrid", lambda d: FAKE_RESULT)
    monkeypatch.delenv("API_KEY", raising=False)
    return TestClient(app_module.app)


def test_health(client):
    assert client.get("/").json()["status"] == "ok"


def test_predict_ok_and_alias(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["_meta"]["explanation_source"] == "fallback"
    assert body["model_outputs"]["risk_level"] == "high"


def test_legacy_image_fields_are_ignored(client):
    r = client.post("/predict", json={**VALID, "image": [[0.0]], "image_available": True, "recommended_ind": 1})
    assert r.status_code == 200


@pytest.mark.parametrize("patch", [{"review_text": ""}, {"review_text": "x" * 5001}, {"age": -1}, {"age": 500},
                                   {"positive_feedback_count": -3}])
def test_validation_rejects_bad_input(client, patch):
    assert client.post("/predict", json={**VALID, **patch}).status_code == 422


def test_only_review_text_is_required(client):
    assert client.post("/predict", json={"review_text": "The zipper broke and nobody answered."}).status_code == 200
    bad = dict(VALID)
    bad.pop("review_text")
    assert client.post("/predict", json=bad).status_code == 422


def test_internal_errors_do_not_leak_details(monkeypatch):
    def boom(d):
        raise RuntimeError("secret path /etc/passwd")
    monkeypatch.setattr(app_module.hybrid_inference, "predict_hybrid", boom)
    monkeypatch.delenv("API_KEY", raising=False)
    r = TestClient(app_module.app).post("/predict", json=VALID)
    assert r.status_code == 500
    assert "secret" not in r.text


def test_missing_model_artifact_is_503(monkeypatch):
    def missing(d):
        raise FileNotFoundError("/app/models/transformer_assets/weights_int8.npz")
    monkeypatch.setattr(app_module.hybrid_inference, "predict_hybrid", missing)
    monkeypatch.delenv("API_KEY", raising=False)
    r = TestClient(app_module.app).post("/predict", json=VALID)
    assert r.status_code == 503
    assert "/app" not in r.text


def test_api_key_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "s3cret")
    assert client.post("/predict", json=VALID).status_code == 401
    assert client.post("/predict", json=VALID, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/predict", json=VALID, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_stats_endpoint_records_predictions(client):
    app_module.monitor.__init__(baseline={})
    client.post("/predict", json=VALID)
    snap = client.get("/stats").json()
    assert snap["n"] == 1
    assert snap["current"]["level_share"]["high"] == 1.0
