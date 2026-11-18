"""End-to-end smoke test against the committed model artifacts (needs torch + transformers)."""
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sklearn")

from hybrid_risk import hybrid_inference as hi  # noqa: E402
from hybrid_risk.paths import MODELS  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (os.path.exists(hi.CLASSICAL_MODEL_PATH) and os.path.exists(str(MODELS / "transformer_assets" / "weights_int8.npz"))),
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
    assert "cnn_used" not in out["_meta"]
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


def test_single_sample_and_batch_agree():
    one = hi.transformer_proba(["Terrible quality, it ripped on the first day."])
    both = hi.transformer_proba(["Terrible quality, it ripped on the first day.", "Lovely dress, fits well."])
    assert one.shape == (1, 3) and both.shape == (2, 3)
    assert abs(one[0] - both[0]).max() < 1e-3
