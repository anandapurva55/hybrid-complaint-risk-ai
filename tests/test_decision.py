import numpy as np
import pytest

from hybrid_risk.decision import (DEFAULT_CONFIG, action_for, apply_temperature, assess, decide_levels,
                                  human_review_required, review_flags)

CFG = dict(DEFAULT_CONFIG, t_high=0.4, t_medium=0.5)


def P(*rows):
    return np.array(rows, dtype=float)


def test_decide_levels_boundaries():
    p = P([0.95, 0.04, 0.01], [0.40, 0.45, 0.15], [0.2, 0.3, 0.5], [0.6, 0.0, 0.4])
    assert decide_levels(p, 0.4, 0.5).tolist() == ["low", "medium", "high", "high"]


def test_high_is_reachable():
    """Regression for the old design where 'high' needed near-certainty from the text model."""
    assert decide_levels(P([0.2, 0.2, 0.6]), 0.4, 0.5)[0] == "high"


def test_temperature_preserves_normalisation_and_order():
    p = P([0.7, 0.2, 0.1])
    for t in (0.5, 1.0, 2.0):
        q = apply_temperature(p, t)
        assert q.sum() == pytest.approx(1.0)
        assert q.argmax() == 0
    assert apply_temperature(p, 2.0).max() < p.max()  # T>1 softens


def test_conflict_flag_catches_contradictory_evidence():
    flags = review_flags(P([0.5, 0.0, 0.5], [0.9, 0.05, 0.05]), cfg=CFG)
    assert flags["conflicted"].tolist() == [True, False]


def test_low_confidence_flag():
    flags = review_flags(P([0.34, 0.33, 0.33]), cfg=CFG)
    assert flags["low_confidence"][0]


def test_review_rules():
    levels = np.array(["low", "low", "medium", "high"])
    flags = {"x": np.array([False, True, False, False])}
    assert human_review_required(levels, flags).tolist() == [False, True, True, True]


def test_action_never_auto_replies_a_flagged_case():
    assert action_for("low", False) == "auto_reply"
    assert action_for("low", True) == "manual_review"
    assert action_for("medium", False) == "manual_review"
    assert action_for("high", True) == "escalate_immediately"


def test_assess_confident_positive_is_auto_reply():
    res = assess(P([0.97, 0.02, 0.01]), CFG)
    assert res["level"][0] == "low"
    assert res["action"][0] == "auto_reply"
    assert not res["review"][0]


def test_assess_confident_negative_escalates_and_requires_review():
    res = assess(P([0.02, 0.08, 0.90]), CFG)
    assert res["level"][0] == "high"
    assert res["action"][0] == "escalate_immediately"
    assert res["review"][0]


def test_assess_extra_flag_blocks_auto_reply():
    res = assess(P([0.97, 0.02, 0.01]), CFG, extra_flags={"short_text": np.array([True])})
    assert res["action"][0] == "manual_review"
    assert res["review"][0]


def test_assess_uses_the_temperature():
    sharp = assess(P([0.6, 0.3, 0.1]), dict(CFG, temp_text=0.5))["p_final"][0]
    soft = assess(P([0.6, 0.3, 0.1]), dict(CFG, temp_text=2.0))["p_final"][0]
    assert sharp.max() > soft.max()


def test_config_has_no_blend_or_disagreement_parameters():
    assert not {"w_classical", "temp_classical", "disagreement_tv"} & set(DEFAULT_CONFIG)
