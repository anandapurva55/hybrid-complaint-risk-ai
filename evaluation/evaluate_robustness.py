"""
Uncertainty, subgroup and ordinal-modelling checks for the hybrid.

  1. Bootstrap 95% CIs on the test split for hybrid macro-F1 / high recall /
     high precision, and for the paired difference hybrid - transformer-alone (same thresholds, so slightly favourable to the hybrid).
  2. 5-fold cross-validation of the classical model on train+val.
  3. Subgroup metrics (age band, department): high recall, macro-F1, auto-reply
     share, and the share of truly-high cases that would be auto-replied.
  4. Ordinal experiment: two cumulative binary logistic models (>=medium,
     =high) vs the multinomial classical model.

Writes results/robustness_results.json.   python -m evaluation.evaluate_robustness
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from hybrid_risk import hybrid_inference as hi
from hybrid_risk.decision import assess, decide_levels, load_config
from hybrid_risk.paths import DATA_PROCESSED, RESULTS
from hybrid_risk.risk_common import CLASSES, RANDOM_STATE, TARGET, TEXT_FEATURE, FEATURE_COLUMNS
from training.train_classical_model import build_pipeline

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
    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    y = test[TARGET].to_numpy()
    p_cl = hi.classical_proba(test)
    p_rn = hi.transformer_proba(test[TEXT_FEATURE].astype(str))
    short = test[TEXT_FEATURE].astype(str).str.split().str.len().to_numpy() < hi.MIN_WORDS
    res = assess(p_cl, p_rn, cfg, extra_flags={"short_text": short})
    hyb = res["level"]
    cls = decide_levels(res["p_text"], cfg["t_high"], cfg["t_medium"])  # transformer alone, same thresholds
    out = {}

    # 1. bootstrap
    rng = np.random.RandomState(RANDOM_STATE)
    boots = {"hybrid": [], "transformer_alone": [], "diff_macro_f1": []}
    keys = ("macro_f1", "high_recall", "high_precision")
    for _ in range(N_BOOT):
        i = rng.randint(0, len(y), len(y))
        h, c = stats(y[i], hyb[i]), stats(y[i], cls[i])
        boots["hybrid"].append(h); boots["transformer_alone"].append(c)
        boots["diff_macro_f1"].append(h["macro_f1"] - c["macro_f1"])
    out["bootstrap_95ci"] = {
        "hybrid": {k: ci([b[k] for b in boots["hybrid"]]) for k in keys},
        "transformer_alone": {k: ci([b[k] for b in boots["transformer_alone"]]) for k in keys},
        "hybrid_minus_transformer_macro_f1": ci(boots["diff_macro_f1"]),
        "n_boot": N_BOOT,
    }

    # 2. cross-validation of the classical model
    trval = pd.concat([pd.read_csv(DATA_PROCESSED / "train.csv"), pd.read_csv(DATA_PROCESSED / "val.csv")])
    cv = cross_validate(build_pipeline(), trval[FEATURE_COLUMNS], trval[TARGET],
                        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                        scoring=["f1_macro", "accuracy"], n_jobs=1)
    out["classical_5fold_cv_argmax"] = {k: [round(float(cv[f"test_{k}"].mean()), 4), round(float(cv[f"test_{k}"].std()), 4)]
                                        for k in ("f1_macro", "accuracy")}

    # 3. subgroups
    test = test.assign(pred=hyb, action=res["action"])
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

    # 4. ordinal experiment on the classical feature pipeline
    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    pipe = build_pipeline().fit(train[FEATURE_COLUMNS], train[TARGET])
    pre = pipe.named_steps["preprocessor"]
    Xtr, Xte = pre.transform(train[FEATURE_COLUMNS]), pre.transform(test[FEATURE_COLUMNS])
    ge_med = LogisticRegression(max_iter=2000).fit(Xtr, train[TARGET].isin(["medium", "high"]))
    is_high = LogisticRegression(max_iter=2000).fit(Xtr, train[TARGET] == "high")
    a, b = ge_med.predict_proba(Xte)[:, 1], is_high.predict_proba(Xte)[:, 1]
    p_ord = np.column_stack([1 - a, np.clip(a - b, 0, 1), b])
    p_ord /= p_ord.sum(axis=1, keepdims=True)
    multi = res["p_classical"]
    def best_f1(p):
        best = 0
        for th in np.arange(0.1, 0.7, 0.025):
            for tm in np.arange(0.2, 0.8, 0.025):
                best = max(best, f1_score(y, decide_levels(p, th, tm), labels=CLASSES, average="macro"))
        return round(float(best), 4)
    out["ordinal_experiment"] = {
        "note": "best test macro-F1 over a threshold grid (optimistic, for comparing model forms only)",
        "multinomial_classical": best_f1(multi),
        "cumulative_binary_ordinal": best_f1(p_ord),
        "auc_high": {"multinomial": round(roc_auc_score(y == "high", multi[:, 2]), 4),
                     "ordinal": round(roc_auc_score(y == "high", p_ord[:, 2]), 4)},
    }

    with open(RESULTS / "robustness_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
