"""
Baseline models, NOT used in the decision. They exist so the reported comparisons stay reproducible:

  * the classical TF-IDF + tabular Logistic Regression model (train_classical_model.py). It used to be
    blended into the decision at 10% weight; evaluate_hybrid.py still reports that blend as an
    ablation, and with each system tuned on its own thresholds the blend was not distinguishable
    from DistilBERT alone, so it left the served system;
  * the LSTM text classifier (train_rnn_model.py), which needs tensorflow (pip install -e ".[baselines]").
"""
import os

import joblib
import numpy as np

from hybrid_risk.paths import MODELS
from hybrid_risk.risk_common import FEATURE_COLUMNS, reorder_proba

RNN_MODEL_PATH = str(MODELS / "rnn_model.keras")
CLASSICAL_MODEL_PATH = str(MODELS / "classical_model.pkl")
_classical = None


def classical_proba(df):
    """(N, 3) probabilities in CLASSES order from the classical baseline."""
    global _classical
    if _classical is None:
        _classical = joblib.load(CLASSICAL_MODEL_PATH)
    return reorder_proba(_classical.predict_proba(df[FEATURE_COLUMNS]), _classical.classes_)


def rnn_available():
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        return False
    return os.path.exists(RNN_MODEL_PATH)


def rnn_proba(texts):
    import tensorflow as tf
    model = tf.keras.models.load_model(RNN_MODEL_PATH)
    return np.asarray(model.predict(tf.constant([str(t) for t in texts], dtype=tf.string), verbose=0))
