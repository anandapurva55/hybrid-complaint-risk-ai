"""
Hybrid scoring logic and end-to-end inference.

Combines the three trained components (classical ML, CNN, RNN) into a single
weighted risk score, maps it to a risk_level, and, through predict_hybrid,
hands the result to the LLM explanation layer (llm_explainer.py) for a
structured business explanation.

--------------------------------------------------------------------------
Input schema
--------------------------------------------------------------------------
A generic complaint payload would typically use generic fields
(customer_tenure, previous_complaints, ...). This system is built on the
Women's E-Commerce Clothing Reviews dataset instead (see data_exploration.ipynb
for why), and its trained classical and RNN models expect that dataset's
actual columns. So `input_data` here mirrors those fields: still "customer
profile data plus complaint text plus an image flag", just with the field
names the trained models actually understand:

    {
        "customer_id": "C001",
        "review_text": "The product arrived damaged and the zipper is broken.",
        "age": 35,
        "positive_feedback_count": 2,
        "recommended_ind": 0,
        "division_name": "General",
        "department_name": "Dresses",
        "class_name": "Dresses",
        "image_available": True,
        "image": None,   # optional 28x28 grayscale array; see get_cnn_score
    }

Run as a script for a quick smoke test against sample_inputs.json:
    python hybrid_inference.py
"""

import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from llm_explainer import explain_with_gemini
from risk_common import CLASSES, proba_to_score
from train_cnn_model import cnn_score_from_proba

CLASSICAL_MODEL_PATH = "classical_model.pkl"
CNN_MODEL_PATH = "cnn_model.keras"
RNN_MODEL_PATH = "rnn_model.keras"

# Hybrid weighting used to combine the three component scores
WEIGHTS = {"classical": 0.4, "cnn": 0.3, "rnn": 0.3}

# Lazily-loaded, module-level singletons; each model is loaded once and reused.
_classical_model = None
_cnn_model = None
_rnn_model = None


def _load_classical():
    global _classical_model
    if _classical_model is None:
        _classical_model = joblib.load(CLASSICAL_MODEL_PATH)
    return _classical_model


def _load_cnn():
    global _cnn_model
    if _cnn_model is None:
        _cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    return _cnn_model


def _load_rnn():
    global _rnn_model
    if _rnn_model is None:
        _rnn_model = tf.keras.models.load_model(RNN_MODEL_PATH)
    return _rnn_model


def _classical_input_row(input_data):
    return pd.DataFrame([{
        "Review Text": input_data["review_text"],
        "Age": input_data["age"],
        "Positive Feedback Count": input_data["positive_feedback_count"],
        "Recommended IND": input_data["recommended_ind"],
        "Division Name": input_data["division_name"],
        "Department Name": input_data["department_name"],
        "Class Name": input_data["class_name"],
    }])


def get_classical_score(input_data):
    """Classical ML risk score in [0, 1] (TF-IDF + tabular Logistic Regression)."""
    model = _load_classical()
    proba = model.predict_proba(_classical_input_row(input_data))
    return float(proba_to_score(proba, model.classes_)[0])


def get_rnn_score(input_data):
    """RNN/LSTM risk score in [0, 1], from the review text sequence alone.

    Single-sample batches must be passed as a tf.string tensor rather than a
    raw numpy unicode array. Keras 3 raises "Invalid dtype: str<N>" for the
    latter when the batch size is 1 (it works fine for the larger batches used
    during training and evaluation, where Keras builds an internal tf.data pipeline)."""
    model = _load_rnn()
    text = tf.constant([input_data["review_text"]], dtype=tf.string)
    proba = model.predict(text, verbose=0)
    return float(proba_to_score(proba, CLASSES)[0])


def get_cnn_score(input_data):
    """CNN risk score in [0, 1].

    Real path: if a 28x28 grayscale image array is supplied, run it through
    the trained Fashion-MNIST CNN and convert via cnn_score_from_proba
    (see train_cnn_model.py).

    Simulated path: this dataset and scenario has no complaint images, so the
    common case has no `image`. We then fall back to a small, clearly-labelled
    simulated signal driven by the `image_available` flag, as a stand-in for
    "a photo exists and would need scoring" versus "no visual evidence at all".
    train_cnn_model.py's docstring documents why this is simulated and how a
    real system would replace it with a learned score.
    """
    image = input_data.get("image")
    if image is not None:
        model = _load_cnn()
        x = np.asarray(image, dtype="float32")
        if x.max() > 1.0:
            x = x / 255.0
        x = x.reshape(1, 28, 28, 1)
        proba = model.predict(x, verbose=0)
        return float(cnn_score_from_proba(proba)[0]), "real"

    simulated_score = 0.6 if input_data.get("image_available") else 0.4
    return float(simulated_score), "simulated"


def combine_scores(classical_score, cnn_score, rnn_score):
    """Weighted combination of the three component scores: 0.4*classical + 0.3*cnn + 0.3*rnn."""
    return (
        WEIGHTS["classical"] * classical_score
        + WEIGHTS["cnn"] * cnn_score
        + WEIGHTS["rnn"] * rnn_score
    )


def map_risk(score):
    """Threshold mapping from a final score to a risk level (see decision_policy.md)."""
    if score >= 0.75:
        return "high"
    elif score >= 0.45:
        return "medium"
    else:
        return "low"


def predict_hybrid(input_data):
    """
    Runs classical ML, CNN, RNN, combines scores,
    then calls the LLM explanation layer.
    """
    classical_score = get_classical_score(input_data)
    cnn_score, cnn_mode = get_cnn_score(input_data)
    rnn_score = get_rnn_score(input_data)

    final_score = combine_scores(classical_score, cnn_score, rnn_score)
    risk_level = map_risk(final_score)

    model_outputs = {
        "classical_score": round(classical_score, 3),
        "cnn_score": round(cnn_score, 3),
        "rnn_score": round(rnn_score, 3),
        "final_score": round(final_score, 3),
        "risk_level": risk_level,
    }

    llm_explanation = explain_with_gemini({
        "complaint_text": input_data.get("review_text", ""),
        "model_outputs": model_outputs,
    })

    return {
        "model_outputs": model_outputs,
        "llm_explanation": llm_explanation,
        "_meta": {"cnn_mode": cnn_mode},
    }


if __name__ == "__main__":
    with open("sample_inputs.json") as f:
        samples = json.load(f)

    for sample in samples:
        result = predict_hybrid(sample)
        print(f"\n=== {sample.get('customer_id', '?')} ===")
        print(json.dumps(result, indent=2))
