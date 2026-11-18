import numpy as np
import pytest

from risk_common import CLASSES, proba_to_score, rating_to_risk, reorder_proba


@pytest.mark.parametrize("rating,expected", [(1, "high"), (2, "high"), (3, "medium"), (4, "low"), (5, "low")])
def test_rating_to_risk(rating, expected):
    assert rating_to_risk(rating) == expected


def test_reorder_proba_handles_sklearn_alphabetical_order():
    # sklearn reports classes alphabetically: high, low, medium
    proba = np.array([[0.7, 0.2, 0.1]])  # P(high)=.7, P(low)=.2, P(medium)=.1
    out = reorder_proba(proba, ["high", "low", "medium"])
    assert out.tolist() == [[0.2, 0.1, 0.7]]
    assert CLASSES == ["low", "medium", "high"]


def test_proba_to_score_bounds():
    assert proba_to_score(np.array([[1, 0, 0]]), CLASSES)[0] == 0.0
    assert proba_to_score(np.array([[0, 0, 1]]), CLASSES)[0] == 1.0
    assert proba_to_score(np.array([[0, 1, 0]]), CLASSES)[0] == 0.5


def test_expected_value_cannot_distinguish_conflict_from_medium():
    """Documents WHY decisions use probabilities, not this scalar."""
    conflicted = proba_to_score(np.array([[0.5, 0.0, 0.5]]), CLASSES)[0]
    confident_medium = proba_to_score(np.array([[0.0, 1.0, 0.0]]), CLASSES)[0]
    assert conflicted == confident_medium
