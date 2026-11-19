"""
Score the served system on the LLM-written sarcasm / negation set (data/synthetic/).

    python -m evaluation.evaluate_synthetic [--split test|dev] [--out results/synthetic_<split>.json]

Per kind we report the share the system gets RIGHT: complaints (sarcastic, negated-negative) count as
right when they are not auto-replied; satisfied-but-negated reviews count as right when they are not
escalated as high (a false alarm). The held-out TEST split comes from a different model family than the
training split, and these are synthetic texts, so read the numbers as a stress test, not as real-world rates.
"""
import argparse
import json

import numpy as np

from hybrid_risk import hybrid_inference as hi
from hybrid_risk.paths import RESULTS, ROOT


def evaluate(split):
    rows = [json.loads(l) for l in open(ROOT / "data" / "synthetic" / f"{split}.jsonl")]
    by_kind = {}
    for r in rows:
        mo = hi.predict_hybrid({"review_text": r["text"]}, explain=False)["model_outputs"]
        d = by_kind.setdefault(r["kind"], {"n": 0, "right": 0, "auto_reply": 0, "high": 0, "p_high": []})
        d["n"] += 1
        d["auto_reply"] += mo["recommended_action"] == "auto_reply"
        d["high"] += mo["risk_level"] == "high"
        d["p_high"].append(mo["probabilities"]["high"])
        if r["tier"] == "low":
            d["right"] += mo["risk_level"] != "high"          # satisfied: do not escalate
        else:
            d["right"] += mo["recommended_action"] != "auto_reply"  # unhappy: do not auto-reply
    out = {}
    for k, d in by_kind.items():
        out[k] = {"n": d["n"], "right": round(d["right"] / d["n"], 3),
                  "auto_replied": round(d["auto_reply"] / d["n"], 3), "escalated_as_high": round(d["high"] / d["n"], 3),
                  "mean_P_high": round(float(np.mean(d["p_high"])), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = evaluate(a.split)
    for k, v in res.items():
        print(f"{k:22s} {v}")
    if a.out:
        (RESULTS / a.out).write_text(json.dumps({"split": a.split, "results": res}, indent=2))


if __name__ == "__main__":
    main()
