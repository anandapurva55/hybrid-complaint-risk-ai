"""
Fit calibration temperatures and tune the blend weight and decision thresholds
on the VALIDATION split only, then write decision_config.json.

Selection rules (explained in the README, "The calibrated decision layer"):
  1. blend weight w_classical = the weight minimising validation log-loss of the
     blended probabilities (a smooth criterion; macro-F1 under a recall
     constraint is too noisy to pick a weight);
  2. (t_high, t_medium) = among thresholds whose validation recall on "high" is
     >= TARGET_HIGH_RECALL, the pair with the best macro-F1; if none reach the
     target, the pair with the highest high-recall.
The test split is never touched here.

Run (after training both models):
    python tune_decision.py
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

import hybrid_inference as hi
from decision import DEFAULT_CONFIG, apply_temperature, blend, decide_levels
from risk_common import CLASSES, TARGET, TEXT_FEATURE

TARGET_HIGH_RECALL = 0.70
CONFIG_PATH = "decision_config.json"


def load_probas(split):
    df = pd.read_csv(f"{split}.csv")
    return df, hi.classical_proba(df), hi.transformer_proba(df[TEXT_FEATURE].astype(str))


def fit_temperature(proba, y_idx):
    best_t, best_nll = 1.0, np.inf
    for t in np.arange(0.5, 4.01, 0.05):
        p = apply_temperature(proba, t)
        nll = -np.log(p[np.arange(len(y_idx)), y_idx] + 1e-12).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return round(best_t, 2)


def main():
    df, p_cl, p_rn = load_probas("val")
    y = df[TARGET].to_numpy()
    y_idx = np.array([CLASSES.index(v) for v in y])

    t_cl = fit_temperature(p_cl, y_idx)
    t_rn = fit_temperature(p_rn, y_idx)
    print(f"Temperatures: classical={t_cl} transformer={t_rn}")
    pc, pr = apply_temperature(p_cl, t_cl), apply_temperature(p_rn, t_rn)

    nll_by_w = {}
    for w in np.round(np.arange(0.0, 1.01, 0.1), 2):
        p = blend(pc, pr, w)
        nll_by_w[float(w)] = float(-np.log(p[np.arange(len(y_idx)), y_idx] + 1e-12).mean())
    w = min(nll_by_w, key=nll_by_w.get)
    print(f"Blend weight (min val log-loss): w_classical={w}  nll={nll_by_w[w]:.4f}")

    rows = []
    p = blend(pc, pr, w)
    for t_high in np.round(np.arange(0.10, 0.71, 0.025), 3):
        for t_med in np.round(np.arange(0.20, 0.81, 0.025), 3):
            pred = decide_levels(p, t_high, t_med)
            rows.append({
                "w": float(w), "t_high": float(t_high), "t_medium": float(t_med),
                "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro"),
                "recall_high": recall_score(y, pred, labels=["high"], average=None)[0],
            })
    grid = pd.DataFrame(rows)
    ok = grid[grid.recall_high >= TARGET_HIGH_RECALL]
    best = (ok.sort_values("macro_f1", ascending=False) if len(ok)
            else grid.sort_values("recall_high", ascending=False)).iloc[0]
    print("Selected (validation):", best.to_dict())

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        "w_classical": float(best.w), "temp_classical": t_cl, "temp_text": t_rn,
        "t_high": float(best.t_high), "t_medium": float(best.t_medium),
    })
    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "decision": cfg,
            "selection": {
                "split": "val", "target_high_recall": TARGET_HIGH_RECALL,
                "val_macro_f1": round(float(best.macro_f1), 4),
                "val_recall_high": round(float(best.recall_high), 4),
            },
        }, f, indent=2)
    print(f"Wrote {CONFIG_PATH}")


if __name__ == "__main__":
    sys.exit(main())
