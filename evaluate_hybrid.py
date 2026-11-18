"""
Evaluate every component and the hybrid on the held-out TEST split, using the
parameters tuned on validation (decision_config.json). Writes
evaluation_results.json and prints a summary.

Reports: per-class precision/recall/F1 and confusion matrices (hybrid and each
component), calibration error (ECE) before/after temperature scaling, triage
metrics (auto-reply share, human-review share, and how many truly-high cases
would have been auto-replied), and a leakage ablation (classical model with
"Recommended IND" added back).

Run:  python evaluate_hybrid.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import hybrid_inference as hi
from decision import apply_temperature, assess, blend, decide_levels, load_config
from risk_common import (CATEGORICAL_FEATURES, CLASSES, FEATURE_COLUMNS, NUMERIC_FEATURES,
                         RANDOM_STATE, TARGET, TEXT_FEATURE, reorder_proba)

MIN_WORDS = hi.MIN_WORDS


def ece(proba, y_idx, bins=10):
    conf, pred = proba.max(axis=1), proba.argmax(axis=1)
    correct = (pred == y_idx).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi_ in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi_)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def metrics(y, pred):
    rep = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "macro_f1": round(f1_score(y, pred, labels=CLASSES, average="macro"), 4),
        "per_class": {c: {k: round(rep[c][k], 4) for k in ("precision", "recall", "f1-score")} for c in CLASSES},
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(y, pred, labels=CLASSES).tolist(),
    }


def ablation_with_recommended_ind(train_df, test_df):
    """Classical model with the leaky Recommended IND feature added back."""
    num = NUMERIC_FEATURES + ["Recommended IND"]
    pre = ColumnTransformer([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2), TEXT_FEATURE),
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    cols = FEATURE_COLUMNS + ["Recommended IND"]
    pipe.fit(train_df[cols], train_df[TARGET])
    return metrics(test_df[TARGET], pipe.predict(test_df[cols]))


def main():
    cfg = load_config()
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    y = test_df[TARGET].to_numpy()
    y_idx = np.array([CLASSES.index(v) for v in y])

    p_cl = hi.classical_proba(test_df)
    p_rn = hi.transformer_proba(test_df[TEXT_FEATURE].astype(str))
    short = test_df[TEXT_FEATURE].astype(str).str.split().str.len().to_numpy() < MIN_WORDS
    res = assess(p_cl, p_rn, cfg, extra_flags={"short_text": short})
    levels = res["level"]

    argmax = lambda p: np.array(CLASSES)[p.argmax(axis=1)]
    out = {
        "n_test": len(test_df),
        "class_share": {c: round(float((y == c).mean()), 4) for c in CLASSES},
        "decision_config": cfg,
        "components_argmax": {
            "majority_class_low": metrics(y, np.full(len(y), "low")),
            "classical": metrics(y, argmax(p_cl)),
            "transformer": metrics(y, argmax(p_rn)),
        },
        "components_with_tuned_thresholds": {
            "classical_alone": metrics(y, decide_levels(res["p_classical"], cfg["t_high"], cfg["t_medium"])),
            "transformer_alone": metrics(y, decide_levels(res["p_text"], cfg["t_high"], cfg["t_medium"])),
        },
        "hybrid": metrics(y, levels),
        "auc_high_vs_rest": {
            "classical": round(roc_auc_score(y == "high", res["p_classical"][:, 2]), 4),
            "transformer": round(roc_auc_score(y == "high", res["p_text"][:, 2]), 4),
            "hybrid": round(roc_auc_score(y == "high", res["p_final"][:, 2]), 4),
        },
        "calibration_ece": {
            "classical_raw": round(ece(p_cl, y_idx), 4), "classical_calibrated": round(ece(res["p_classical"], y_idx), 4),
            "transformer_raw": round(ece(p_rn, y_idx), 4), "transformer_calibrated": round(ece(res["p_text"], y_idx), 4),
            "hybrid": round(ece(res["p_final"], y_idx), 4),
        },
        "ablation_with_recommended_ind_classical_argmax": ablation_with_recommended_ind(train_df, test_df),
    }

    import baselines
    if baselines.rnn_available():
        p_lstm = baselines.rnn_proba(test_df[TEXT_FEATURE].astype(str))
        out["components_argmax"]["rnn_lstm_baseline_not_in_decision"] = metrics(y, argmax(p_lstm))
        out["auc_high_vs_rest"]["rnn_lstm_baseline"] = round(roc_auc_score(y == "high", p_lstm[:, 2]), 4)

    action = res["action"]
    true_high = y == "high"
    auto = action == "auto_reply"
    out["triage"] = {
        "action_share": {a: round(float((action == a).mean()), 4) for a in ("auto_reply", "manual_review", "escalate_immediately")},
        "human_review_share": round(float(res["review"].mean()), 4),
        "review_trigger_share": {k: round(float(v.mean()), 4) for k, v in res["flags"].items()},
        "truly_high_cases": int(true_high.sum()),
        "truly_high_auto_replied": int((auto & true_high).sum()),
        "truly_high_reaching_human": int((res["review"] & true_high).sum()),
        "auto_reply_precision_low": round(float((y[auto] == "low").mean()), 4) if auto.any() else None,
        "recall_of_truly_low_auto_replied": round(float((auto & (y == "low")).sum() / (y == "low").sum()), 4),
    }
    out["historical_reference_old_hybrid"] = {
        "note": "Measured on this same test split BEFORE the redesign (old models, constant CNN=0.4, thresholds 0.45/0.75).",
        "high_recall": 0.132, "high_precision": 0.681, "macro_f1": 0.543,
        "confusion_matrix_rows_true_cols_pred": [[2550, 66, 2], [176, 228, 20], [29, 279, 47]],
        "share_forced_to_human_review": 0.807, "share_scored_high": 0.020,
    }

    with open("monitoring_baseline.json", "w") as f:  # reference distribution for monitoring.py
        json.dump({"source": "test split", "action_share": out["triage"]["action_share"],
                   "mean_top_probability": round(float(res["p_final"].max(axis=1).mean()), 3)}, f, indent=2)

    with open("evaluation_results.json", "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({k: out[k] for k in ("hybrid", "triage", "auc_high_vs_rest", "calibration_ece")}, indent=2))
    for name, m in out["components_argmax"].items():
        print(name, m["accuracy"], m["macro_f1"], m["per_class"]["high"])
    for name, m in out["components_with_tuned_thresholds"].items():
        print(name, "tuned", m["accuracy"], m["macro_f1"], m["per_class"]["high"])
    a = out["ablation_with_recommended_ind_classical_argmax"]
    print("ablation +Recommended IND", a["accuracy"], a["macro_f1"], a["per_class"]["high"])


if __name__ == "__main__":
    main()
