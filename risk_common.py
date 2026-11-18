"""
Shared constants and conventions for the hybrid dissatisfaction-risk system.

IMPORTANT FRAMING: the dataset has no escalation outcomes. `risk_level` is a
PROXY derived from the customer's star rating (see rating_to_risk), so the
system really predicts "how dissatisfied is this customer's feedback", which
is used as a stand-in for escalation risk. See README "What this is, and is not".

Every trained text component (classical ML, RNN) predicts the same 3-class
target (low / medium / high) and returns a probability vector in CLASSES order.
Probabilities (not a collapsed scalar) are what the decision layer in
decision.py works on, because an expected-value score cannot tell a confident
"medium" from a 50/50 split between "low" and "high".

`proba_to_score` still provides a single 0-1 number for display and for the
legacy CNN demo: score = 0.0*P(low) + 0.5*P(medium) + 1.0*P(high).
"""

import numpy as np

RANDOM_STATE = 42

CLASSES = ["low", "medium", "high"]
ORDINAL_WEIGHTS = np.array([0.0, 0.5, 1.0])

TEXT_FEATURE = "Review Text"
# "Recommended IND" is deliberately NOT a feature: it is a near-direct proxy of
# the star rating the label is derived from (label leakage), and a real
# complaint would not carry it. evaluate_hybrid.py reports the ablation.
NUMERIC_FEATURES = ["Age", "Positive Feedback Count"]
CATEGORICAL_FEATURES = ["Division Name", "Department Name", "Class Name"]
TARGET = "risk_level"

FEATURE_COLUMNS = [TEXT_FEATURE] + NUMERIC_FEATURES + CATEGORICAL_FEATURES


def rating_to_risk(rating):
    """Proxy label: 1-2 stars -> high, 3 -> medium, 4-5 -> low."""
    if rating <= 2:
        return "high"
    if rating == 3:
        return "medium"
    return "low"


def reorder_proba(proba, model_classes):
    """Return `proba` with columns in CLASSES order.

    `model_classes` is the class order the model itself reports (sklearn's
    `classes_` is alphabetical: high, low, medium), so never assume it matches.
    """
    proba = np.asarray(proba)
    order = [list(model_classes).index(c) for c in CLASSES]
    return proba[:, order]


def proba_to_score(proba, model_classes):
    """Ordinal expected-value score in [0, 1] (display / legacy use only)."""
    return reorder_proba(proba, model_classes) @ ORDINAL_WEIGHTS
