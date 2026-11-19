"""
Fit the calibration temperature and tune the decision thresholds on the VALIDATION split only,
then write models/decision_config.json.

Selection rules (explained in the README, "The calibrated decision layer"):
  1. temperature T = the value minimising validation log-loss of the calibrated probabilities;
  2. (t_high, t_medium) = among thresholds that satisfy BOTH
       - validation recall on "high" >= TARGET_HIGH_RECALL, and
       - at most MAX_HIGH_PREDICTED_LOW of the truly-high reviews predicted "low" (those would be
         auto-replied, the costliest mistake),
     the pair with the best macro-F1. If no pair satisfies both, the second constraint is dropped;
     if none reach the recall target either, the pair with the highest high-recall.
The test split is never touched here.

Run (after training the transformer):
    python -m training.tune_decision
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from hybrid_risk import hybrid_inference as hi
from hybrid_risk.decision import DEFAULT_CONFIG, apply_temperature, decide_levels
from hybrid_risk.paths import DATA_PROCESSED, MODELS
from hybrid_risk.risk_common import CLASSES, TARGET, TEXT_FEATURE

TARGET_HIGH_RECALL = 0.70
MAX_HIGH_PREDICTED_LOW = 0.02
CONFIG_PATH = MODELS / "decision_config.json"


def load_probas(split):
    df = pd.read_csv(DATA_PROCESSED / f"{split}.csv")
    return df, hi.transformer_proba(df[TEXT_FEATURE].astype(str))


def fit_temperature(proba, y_idx):
    best_t, best_nll = 1.0, np.inf
    for t in np.arange(0.5, 4.01, 0.05):
        p = apply_temperature(proba, t)
        nll = -np.log(p[np.arange(len(y_idx)), y_idx] + 1e-12).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return round(best_t, 2)


def tune_thresholds(p, y, target_recall=TARGET_HIGH_RECALL, max_high_to_low=MAX_HIGH_PREDICTED_LOW):
    """(t_high, t_medium, macro_f1, recall_high) chosen as described in the module docstring."""
    y = np.asarray(y)
    is_high = y == "high"
    rows = []
    for t_high in np.round(np.arange(0.10, 0.71, 0.025), 3):
        for t_med in np.round(np.arange(0.20, 0.81, 0.025), 3):
            pred = decide_levels(p, t_high, t_med)
            rows.append((float(t_high), float(t_med), f1_score(y, pred, labels=CLASSES, average="macro"),
                         recall_score(y, pred, labels=["high"], average=None, zero_division=0)[0],
                         float(((pred == "low") & is_high).sum() / max(is_high.sum(), 1))))
    grid = pd.DataFrame(rows, columns=["t_high", "t_medium", "macro_f1", "recall_high", "high_to_low"])
    ok = grid[(grid.recall_high >= target_recall) & (grid.high_to_low <= max_high_to_low)]
    if not len(ok):
        ok = grid[grid.recall_high >= target_recall]
    best = (ok.sort_values("macro_f1", ascending=False) if len(ok) else grid.sort_values("recall_high", ascending=False)).iloc[0]
    return float(best.t_high), float(best.t_medium), float(best.macro_f1), float(best.recall_high)


def main():
    df, p_tx = load_probas("val")
    y = df[TARGET].to_numpy()
    y_idx = np.array([CLASSES.index(v) for v in y])

    temperature = fit_temperature(p_tx, y_idx)
    print(f"Temperature: {temperature}")
    p = apply_temperature(p_tx, temperature)

    t_high, t_med, best_f1, best_recall = tune_thresholds(p, y)
    print(f"Selected (validation): t_high={t_high} t_medium={t_med} macro_f1={best_f1:.4f} recall_high={best_recall:.4f}")

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({"temp_text": temperature, "t_high": t_high, "t_medium": t_med})
    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "decision": cfg,
            "selection": {"split": "val", "target_high_recall": TARGET_HIGH_RECALL, "max_high_predicted_low": MAX_HIGH_PREDICTED_LOW,
                          "val_macro_f1": round(best_f1, 4), "val_recall_high": round(best_recall, 4)},
        }, f, indent=2)
    print(f"Wrote {CONFIG_PATH}")


if __name__ == "__main__":
    sys.exit(main())
