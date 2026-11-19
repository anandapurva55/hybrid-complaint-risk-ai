import numpy as np

from hybrid_risk.decision import decide_levels
from training.tune_decision import MAX_HIGH_PREDICTED_LOW, tune_thresholds


def make_data(seed=0, n=600):
    rng = np.random.RandomState(seed)
    y = rng.choice(["low", "medium", "high"], size=n, p=[0.7, 0.15, 0.15])
    logits = {"low": [2, 0, -2], "medium": [0, 1, 0], "high": [-2, 0, 2]}
    z = np.array([logits[v] for v in y]) + rng.normal(0, 1.5, size=(n, 3))
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return p, y


def test_selected_thresholds_respect_both_constraints_when_feasible():
    p, y = make_data()
    th, tm, f1, recall = tune_thresholds(p, y)
    pred = decide_levels(p, th, tm)
    high = y == "high"
    assert recall >= 0.70
    assert ((pred == "low") & high).sum() / high.sum() <= MAX_HIGH_PREDICTED_LOW + 1e-9
    assert 0 < f1 <= 1


def test_tightening_the_cap_never_increases_missed_high_cases():
    p, y = make_data(seed=3)
    loose = decide_levels(p, *tune_thresholds(p, y, max_high_to_low=1.0)[:2])
    tight = decide_levels(p, *tune_thresholds(p, y, max_high_to_low=0.02)[:2])
    high = y == "high"
    assert ((tight == "low") & high).sum() <= ((loose == "low") & high).sum()
