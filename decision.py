"""
Pure (numpy-only) decision logic for the hybrid scorer.

Kept free of TensorFlow / sklearn imports so it is cheap to unit test and to
import from the evaluation and tuning scripts.

Pipeline, all on class-probability vectors in CLASSES order (low, medium, high):

  1. calibrate each text model (classical, fine-tuned DistilBERT) with a single temperature (fit on validation)
  2. blend:   p = w * p_classical + (1 - w) * p_text
  3. decide:  high   if p[high] >= t_high
              medium if p[high] + p[medium] >= t_medium
              low    otherwise
     (t_high / t_medium are tuned on validation data for high-class recall;
      they are NOT a threshold on a collapsed scalar score)
  4. flag for human review when the evidence is conflicted, low-confidence, or
     the two models disagree (see review_reasons)

"""

import json
import os

import numpy as np

from risk_common import CLASSES

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_config.json")

# Used only if decision_config.json is missing (e.g. unit tests). Real values
# are produced by tune_decision.py and written to decision_config.json.
DEFAULT_CONFIG = {
    "w_classical": 0.2,
    "temp_classical": 1.0,
    "temp_text": 1.0,
    "t_high": 0.35,
    "t_medium": 0.50,
    # human-review triggers
    "conflict_min": 0.30,        # min(P_low, P_high) above this => contradictory evidence
    "max_proba_floor": 0.45,     # top class probability below this => low confidence
    "disagreement_tv": 0.45,     # total-variation distance between the two models
}


def load_config(path=CONFIG_PATH):
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f).get("decision", {}))
    return cfg


def apply_temperature(proba, temperature):
    """Temperature-scale probabilities: p_i ** (1/T), renormalised. T>1 softens."""
    p = np.clip(np.asarray(proba, dtype="float64"), 1e-9, 1.0) ** (1.0 / temperature)
    return p / p.sum(axis=-1, keepdims=True)


def blend(p_classical, p_text, w_classical):
    return w_classical * np.asarray(p_classical) + (1.0 - w_classical) * np.asarray(p_text)


def decide_levels(proba, t_high, t_medium):
    """Vectorised risk-level decision for an (N, 3) probability array."""
    proba = np.atleast_2d(proba)
    p_low, p_med, p_high = proba[:, 0], proba[:, 1], proba[:, 2]
    out = np.where(p_high >= t_high, "high", np.where(p_high + p_med >= t_medium, "medium", "low"))
    return out


def total_variation(p, q):
    return 0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum(axis=-1)


def review_flags(p_final, p_classical=None, p_text=None, cfg=None):
    """Boolean arrays (N,) for each human-review trigger, keyed by name."""
    cfg = cfg or load_config()
    p_final = np.atleast_2d(p_final)
    flags = {
        "conflicted": np.minimum(p_final[:, 0], p_final[:, 2]) >= cfg["conflict_min"],
        "low_confidence": p_final.max(axis=1) < cfg["max_proba_floor"],
    }
    if p_classical is not None and p_text is not None:
        flags["models_disagree"] = total_variation(p_classical, p_text) > cfg["disagreement_tv"]
    else:
        flags["models_disagree"] = np.zeros(len(p_final), dtype=bool)
    return flags


def review_reasons(flags, i=0):
    return [name for name, arr in flags.items() if bool(arr[i])]


def human_review_required(levels, flags):
    """High risk is always reviewed; medium goes to manual review by definition;
    low is reviewed only if a trigger fired."""
    any_flag = np.zeros(len(levels), dtype=bool)
    for arr in flags.values():
        any_flag |= arr
    levels = np.asarray(levels)
    return (levels == "high") | (levels == "medium") | any_flag


ACTION_BY_RISK = {"low": "auto_reply", "medium": "manual_review", "high": "escalate_immediately"}


def action_for(level, needs_review):
    """Deterministic action. A low case that tripped a review trigger is NOT auto-replied."""
    if level == "low" and needs_review:
        return "manual_review"
    return ACTION_BY_RISK[level]


def final_score(proba):
    """Display score in [0,1]: P(medium)*0.5 + P(high). Not used for the decision."""
    proba = np.atleast_2d(proba)
    return proba[:, 1] * 0.5 + proba[:, 2]


def assess(p_classical, p_text, cfg=None, extra_flags=None):
    """Full deterministic assessment for (N, 3) probability arrays (CLASSES order).

    Returns a dict of arrays: p_final, level, review (bool), flags, action.
    `extra_flags` lets callers add input-level triggers (e.g. very short text).
    """
    cfg = cfg or load_config()
    pc = apply_temperature(p_classical, cfg["temp_classical"])
    pr = apply_temperature(p_text, cfg["temp_text"])
    p_final = blend(pc, pr, cfg["w_classical"])
    levels = decide_levels(p_final, cfg["t_high"], cfg["t_medium"])
    flags = review_flags(p_final, pc, pr, cfg)
    if extra_flags:
        flags.update(extra_flags)
    review = human_review_required(levels, flags)
    actions = np.array([action_for(l, r) for l, r in zip(levels, review)])
    return {"p_classical": pc, "p_text": pr, "p_final": p_final, "level": levels,
            "review": review, "flags": flags, "action": actions}
