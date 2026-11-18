"""End-to-end smoke test against the committed model artifacts (needs TensorFlow)."""
import os

import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("sklearn")

import hybrid_inference as hi  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (os.path.exists(hi.CLASSICAL_MODEL_PATH) and os.path.exists(hi.RNN_MODEL_PATH)),
    reason="model artifacts not present",
)

BASE = {"age": 34, "positive_feedback_count": 1, "division_name": "General",
        "department_name": "Tops", "class_name": "Knits"}


def run(text):
    return hi.predict_hybrid({**BASE, "review_text": text}, explain=False)


def test_output_contract_and_probabilities_sum_to_one():
    out = run("The seam tore on the first wash and customer service never answered my emails.")
    mo = out["model_outputs"]
    assert mo["risk_level"] in ("low", "medium", "high")
    assert sum(mo["probabilities"].values()) == pytest.approx(1.0, abs=0.01)
    assert out["_meta"]["cnn_used"] is False
    assert out["llm_explanation"] is None


def test_clearly_positive_review_can_be_auto_replied():
    mo = run("Love this sweater! The color is gorgeous, it fits perfectly and the fabric is soft.")["model_outputs"]
    assert mo["risk_level"] == "low"
    assert mo["recommended_action"] == "auto_reply"
    assert mo["human_review_required"] is False


def test_very_short_text_is_never_auto_replied():
    mo = run("ok")["model_outputs"]
    assert "short_text" in mo["review_reasons"]
    assert mo["recommended_action"] != "auto_reply"


def test_single_sample_inference_does_not_crash_on_keras_dtype():
    # regression: Keras 3 rejects numpy unicode arrays for batch size 1
    assert run("Fine.")["model_outputs"]["risk_level"]
