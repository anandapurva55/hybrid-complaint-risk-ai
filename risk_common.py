"""
Shared constants and conventions for the hybrid complaint-risk system.

Every component (classical ML, CNN, RNN) predicts the same 3-class target
(risk_level: low/medium/high) but the hybrid scorer in hybrid_inference.py
needs a single continuous score per component on a [0, 1] scale.

Convention: a component's score is the expected value of the ordinal class
encoding low=0.0, medium=0.5, high=1.0:

    score = 0.0 * P(low) + 0.5 * P(medium) + 1.0 * P(high)

This keeps every component's output on the same scale that the hybrid
weighted sum and the low/medium/high thresholds operate on, and is shared
here so training and inference code can never drift out of sync.
"""

import numpy as np

RANDOM_STATE = 42

CLASSES = ["low", "medium", "high"]
ORDINAL_WEIGHTS = np.array([0.0, 0.5, 1.0])

TEXT_FEATURE = "Review Text"
NUMERIC_FEATURES = ["Age", "Positive Feedback Count", "Recommended IND"]
CATEGORICAL_FEATURES = ["Division Name", "Department Name", "Class Name"]
TARGET = "risk_level"

FEATURE_COLUMNS = [TEXT_FEATURE] + NUMERIC_FEATURES + CATEGORICAL_FEATURES


def proba_to_score(proba, model_classes):
    """Convert an array of per-class probabilities into ordinal risk scores in [0, 1].

    `model_classes` is the class order the model itself reports (e.g.
    `pipeline.classes_` / a label encoder's `.classes_`), since sklearn /
    keras don't guarantee alphabetical order matches CLASSES.
    """
    proba = np.asarray(proba)
    order = [list(model_classes).index(c) for c in CLASSES]
    ordered = proba[:, order]
    return ordered @ ORDINAL_WEIGHTS
