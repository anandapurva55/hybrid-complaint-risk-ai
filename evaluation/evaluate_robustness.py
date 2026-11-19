"""
Uncertainty and subgroup checks for the served system.

  1. Bootstrap 95% CIs on the test split for macro-F1 / high recall / high precision, and the paired
     difference between the system and the same model with the classical baseline blended back in
     (each with its own validation-tuned thresholds). That interval is the evidence for removing the blend.
  2. Subgroup metrics (age band, department): high recall, macro-F1, auto-reply share, and the share of
     truly-high cases that would be auto-replied. Age is audited here but is NOT a model input.

Writes results/robustness_results.json.   python -m evaluation.evaluate_robustness
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from evaluation.evaluate_hybrid import BLEND_WEIGHT
from hybrid_risk import hybrid_inference as hi
from hybrid_risk.decision import apply_temperature, assess, decide_levels, load_config
from hybrid_risk.paths import DATA_PROCESSED, RESULTS
from hybrid_risk.risk_common import CLASSES, RANDOM_STATE, TARGET, TEXT_FEATURE
from training import baselines
from training.tune_decision import fit_temperature, tune_thresholds

N_BOOT = 1000


def stats(y, pred):
    return {
        "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro"),
        "high_recall": recall_score(y, pred, labels=["high"], average=None, zero_division=0)[0],
        "high_precision": precision_score(y, pred, labels=["high"], average=None, zero_division=0)[0],
    }


def ci(vals):
    lo, hi_ = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi_), 4)]


def main():
    cfg = load_config()
    val = pd.read_csv(DATA_PROCESSED / "val.csv")
    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    y, yv = test[TARGET].to_numpy(), val[TARGET].to_numpy()
    p_tx, p_tx_v = hi.transformer_proba(test[TEXT_FEATURE].astype(str)), hi.transformer_proba(val[TEXT_FEATURE].astype(str))
    res = assess(p_tx, cfg, extra_flags=hi.text_flags(test[TEXT_FEATURE], cfg))
    system = decide_levels(res["p_final"], cfg["t_high"], cfg["t_medium"])

    # the same model with the classical baseline blended back in, each tuned on validation
    pc_v, pc_t = baselines.classical_proba(val), baselines.classical_proba(test)
    t_cl = fit_temperature(pc_v, np.array([CLASSES.index(v) for v in yv]))
    c_v, c_t = apply_temperature(pc_v, t_cl), apply_temperature(pc_t, t_cl)
    tx_v = apply_temperature(p_tx_v, cfg["temp_text"])
    blend_v = BLEND_WEIGHT * c_v + (1 - BLEND_WEIGHT) * tx_v
    blend_t = BLEND_WEIGHT * c_t + (1 - BLEND_WEIGHT) * res["p_final"]
    th, tm, _, _ = tune_thresholds(blend_v, yv)
    blended = decide_levels(blend_t, th, tm)

    rng = np.random.RandomState(RANDOM_STATE)
    sys_b, blend_b, diff = [], [], []
    keys = ("macro_f1", "high_recall", "high_precision")
    for _ in range(N_BOOT):
        i = rng.randint(0, len(y), len(y))
        s, b = stats(y[i], system[i]), stats(y[i], blended[i])
        sys_b.append(s); blend_b.append(b); diff.append(b["macro_f1"] - s["macro_f1"])
    out = {"bootstrap_95ci": {
        "served_system": {k: ci([r[k] for r in sys_b]) for k in keys},
        "with_classical_blended_back_in": {k: ci([r[k] for r in blend_b]) for k in keys},
        "blend_minus_system_macro_f1": ci(diff), "n_boot": N_BOOT}}

    test = test.assign(pred=system, action=res["action"])
    test["age_band"] = pd.cut(test["Age"], [0, 29, 45, 60, 200], labels=["<30", "30-45", "46-60", "60+"])
    groups = {}
    for col in ("age_band", "Department Name"):
        rows = {}
        for g, d in test.groupby(col, observed=True):
            if len(d) < 30:
                continue
            high = d[TARGET] == "high"
            rows[str(g)] = {
                "n": int(len(d)),
                "macro_f1": round(f1_score(d[TARGET], d.pred, labels=CLASSES, average="macro"), 3),
                "high_recall": round(float((d.pred[high] == "high").mean()), 3) if high.sum() >= 10 else None,
                "n_true_high": int(high.sum()),
                "auto_reply_share": round(float((d.action == "auto_reply").mean()), 3),
                "true_high_auto_replied": round(float((d.action[high] == "auto_reply").mean()), 3) if high.sum() >= 10 else None,
            }
        groups[col] = rows
    out["subgroups"] = groups

    with open(RESULTS / "robustness_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["bootstrap_95ci"], indent=1))


if __name__ == "__main__":
    main()
