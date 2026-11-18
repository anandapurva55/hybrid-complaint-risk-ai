"""
Baseline-only models, NOT used in the decision: the LSTM text classifier
(train_rnn_model.py). It is reported in evaluate_hybrid.py for comparison. Its
blend weight was 0 once the fine-tuned transformer existed, so it (and
TensorFlow) were removed from serving. Needs tensorflow (requirements-dev.txt).
"""
import os

import numpy as np

from hybrid_risk.paths import MODELS

RNN_MODEL_PATH = str(MODELS / "rnn_model.keras")


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
