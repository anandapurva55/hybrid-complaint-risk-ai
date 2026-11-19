"""
Measure the system's behaviour on the CheckList-style cases in behaviour_cases.py and write
results/behaviour_results.json.

For each category we report two numbers where they differ: what the MODEL's tier alone would do, and
what the whole system does once the review rules (including the noisy-text and non-English triggers)
are applied. "caught" means the feedback is NOT auto-replied; "auto_reply" is the share answered
automatically.

    python -m evaluation.evaluate_behaviour
"""
import json

import numpy as np

from evaluation import behaviour_cases as bc
from hybrid_risk import hybrid_inference as hi
from hybrid_risk.paths import RESULTS

BASE = {"positive_feedback_count": 1, "division_name": "General", "department_name": "Tops", "class_name": "Knits"}


def run(text):
    return hi.predict_hybrid({**BASE, "review_text": text}, explain=False)["model_outputs"]


def summarise(texts, want):
    """want: 'caught' (should not be auto-replied) or 'auto_reply' (should be)."""
    outs = [run(t) for t in texts]
    tier_flagged = np.mean([o["risk_level"] != "low" for o in outs])
    caught = np.mean([o["recommended_action"] != "auto_reply" for o in outs])
    auto = 1.0 - caught
    escalated = np.mean([o["risk_level"] == "high" for o in outs])
    failures = [t for t, o in zip(texts, outs) if (o["recommended_action"] == "auto_reply") == (want == "caught")][:3]
    res = {"n": len(texts), "want": want}
    if want == "caught":
        res.update({"caught_by_model_tier_alone": round(float(tier_flagged), 3), "caught_with_review_rules": round(float(caught), 3)})
    else:
        res.update({"auto_reply_rate": round(float(auto), 3), "escalated_as_high": round(float(escalated), 3)})
    res["examples_that_missed"] = failures
    return res


def main():
    out = {}
    neg, pos = bc.NEGATIVE, bc.POSITIVE
    typos = lambda texts, seeds, fn: [fn(t, s) for t in texts for s in seeds]
    out["negative_clean"] = summarise(neg, "caught")
    out["negative_mild_typos"] = summarise(typos(neg, (1, 2, 3), bc.add_typos), "caught")
    out["negative_heavy_typos"] = summarise(typos(neg, (1, 2, 3), bc.heavy_typos), "caught")
    out["negative_ALL_CAPS"] = summarise([t.upper() for t in neg], "caught")
    out["negative_non_english"] = summarise(bc.NON_ENGLISH_NEGATIVE, "caught")
    out["negative_mixed_sentiment"] = summarise(bc.MIXED, "caught")
    out["negative_sarcastic_KNOWN_LIMITATION"] = summarise(bc.SARCASTIC_NEGATIVE, "caught")
    out["positive_clean"] = summarise(pos, "auto_reply")
    out["positive_mild_typos_cost_of_the_noise_trigger"] = summarise(typos(pos, (1, 2, 3), bc.add_typos), "auto_reply")
    out["positive_ALL_CAPS"] = summarise([t.upper() for t in pos], "auto_reply")

    # negation: a negated negative should look worse than a negated positive
    p_high = lambda texts: [run(t)["probabilities"]["high"] for t in texts]
    nn, npos = p_high(bc.NEGATED_NEGATIVE), p_high(bc.NEGATED_POSITIVE)
    pairs = [a > b for a in nn for b in npos]
    out["negation_directional"] = {"n_pairs": len(pairs), "pass_rate": round(float(np.mean(pairs)), 3),
                                   "mean_P_high_negated_negative": round(float(np.mean(nn)), 3),
                                   "mean_P_high_negated_positive": round(float(np.mean(npos)), 3)}

    # invariance: shouting should not change the tier
    same = [run(t)["risk_level"] == run(t.upper())["risk_level"] for t in neg + pos]
    out["ALL_CAPS_keeps_the_same_tier"] = {"n": len(same), "rate": round(float(np.mean(same)), 3)}

    with open(RESULTS / "behaviour_results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for k, v in out.items():
        print(f"{k:48s} " + ", ".join(f"{a}={b}" for a, b in v.items() if a not in ("examples_that_missed", "want", "n", "n_pairs")))


if __name__ == "__main__":
    main()
