"""
Hybrid scoring and end-to-end inference.

Components that make the decision (all deterministic once trained):
  * classical ML  - TF-IDF + tabular features -> Logistic Regression
  * RNN / LSTM    - review text only
Their calibrated class probabilities are blended and mapped to a risk level by
decision.py using parameters tuned on the validation split (decision_config.json).
The LLM (llm_explainer.py) only EXPLAINS the result; it cannot change it.

The CNN is NOT part of the decision. It was trained on Fashion-MNIST, so it
cannot judge a complaint photo, and feeding it a constant when no image exists
only added noise (see README, "Why the CNN is not in the decision"). Any
`image` / `image_available` fields in the payload are ignored.

--------------------------------------------------------------------------
Input schema (mirrors the Women's E-Commerce Clothing Reviews columns)
--------------------------------------------------------------------------
    {
        "customer_id": "C001",            # optional, echoed nowhere sensitive
        "review_text": "The zipper is broken ...",
        "age": 35,
        "positive_feedback_count": 2,
        "division_name": "General",
        "department_name": "Dresses",
        "class_name": "Dresses"
    }
`recommended_ind` is accepted but ignored (label-leakage feature, see risk_common.py).

Smoke test:  python hybrid_inference.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from decision import action_for, assess, final_score, load_config, review_reasons
from llm_explainer import explain_decision
from risk_common import CLASSES, FEATURE_COLUMNS, proba_to_score, reorder_proba

_HERE = os.path.dirname(os.path.abspath(__file__))
CLASSICAL_MODEL_PATH = os.path.join(_HERE, "classical_model.pkl")
RNN_MODEL_PATH = os.path.join(_HERE, "rnn_model.keras")

MIN_WORDS = 4  # shorter texts give the text models too little to go on -> force review

_classical_model = None
_rnn_model = None


def _load_classical():
    global _classical_model
    if _classical_model is None:
        _classical_model = joblib.load(CLASSICAL_MODEL_PATH)
    return _classical_model


def _load_rnn():
    global _rnn_model
    if _rnn_model is None:
        import tensorflow as tf  # imported lazily so pure-logic tests need no TF
        _rnn_model = tf.keras.models.load_model(RNN_MODEL_PATH)
    return _rnn_model


def classical_proba(df):
    """(N, 3) probabilities in CLASSES order from a DataFrame with FEATURE_COLUMNS."""
    model = _load_classical()
    return reorder_proba(model.predict_proba(df[FEATURE_COLUMNS]), model.classes_)


def rnn_proba(texts):
    """(N, 3) probabilities in CLASSES order from a sequence of review texts.

    Passed as a tf.string tensor: Keras 3 rejects numpy unicode arrays for
    batch size 1 ("Invalid dtype: str<N>")."""
    import tensorflow as tf
    model = _load_rnn()
    batch = tf.constant([str(t) for t in texts], dtype=tf.string)
    return np.asarray(model.predict(batch, verbose=0))


def _input_frame(input_data):
    return pd.DataFrame([{
        "Review Text": input_data["review_text"],
        "Age": input_data["age"],
        "Positive Feedback Count": input_data["positive_feedback_count"],
        "Division Name": input_data["division_name"],
        "Department Name": input_data["department_name"],
        "Class Name": input_data["class_name"],
    }])


def predict_hybrid(input_data, explain=True):
    """Run the two text models, decide deterministically, then (optionally) explain."""
    cfg = load_config()
    df = _input_frame(input_data)
    text = str(input_data["review_text"])

    p_cl = classical_proba(df)
    p_rn = rnn_proba([text])

    short = np.array([len(text.split()) < MIN_WORDS])
    result = assess(p_cl, p_rn, cfg, extra_flags={"short_text": short})
    level = str(result["level"][0])
    review = bool(result["review"][0])
    reasons = review_reasons(result["flags"])
    p_final = result["p_final"]

    model_outputs = {
        "classical_score": round(float(proba_to_score(result["p_classical"], CLASSES)[0]), 3),
        "rnn_score": round(float(proba_to_score(result["p_rnn"], CLASSES)[0]), 3),
        "final_score": round(float(final_score(p_final)[0]), 3),
        "probabilities": {c: round(float(p), 3) for c, p in zip(CLASSES, p_final[0])},
        "risk_level": level,
        "recommended_action": action_for(level, review),
        "human_review_required": review,
        "review_reasons": reasons,
    }

    if explain:
        explanation = explain_decision({"complaint_text": text, "model_outputs": model_outputs})
    else:
        explanation = None

    source = (explanation or {}).pop("_source", None) if explanation else None
    return {
        "model_outputs": model_outputs,
        "llm_explanation": explanation,
        "_meta": {"cnn_used": False, "explanation_source": source},
    }


if __name__ == "__main__":
    with open(os.path.join(_HERE, "sample_inputs.json")) as f:
        samples = json.load(f)

    for sample in samples:
        result = predict_hybrid(sample)
        print(f"\n=== {sample.get('customer_id', '?')} ===")
        print(json.dumps(result, indent=2))
