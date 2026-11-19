"""
Evaluate the served system (DistilBERT + calibration + thresholds + review rules) on the held-out TEST
split, using the parameters tuned on validation (models/decision_config.json). Writes
results/evaluation_results.json and results/monitoring_baseline.json.

Reports: per-class precision/recall/F1 and the confusion matrix, calibration error (ECE) before and
after temperature scaling, triage numbers (auto-reply / review shares, where the truly-high cases go,
and what the review triggers withhold), the comparison with the classical and LSTM baselines, and three
ablations: blending the classical model back in (why it was removed), adding Recommended IND back to
the classical model (label leakage) and adding Age back (why it is not a feature).

Run:  python -m evaluation.evaluate_hybrid
"""

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hybrid_risk import hybrid_inference as hi
from hybrid_risk.decision import apply_temperature, assess, decide_levels, load_config
from hybrid_risk.paths import DATA_PROCESSED, RESULTS
from hybrid_risk.risk_common import (CATEGORICAL_FEATURES, CLASSES, NUMERIC_FEATURES, RANDOM_STATE, TARGET,
                                     TEXT_FEATURE, reorder_proba)
from training import baselines
from training.tune_decision import fit_temperature, tune_thresholds

BLEND_WEIGHT = 0.1  # the classical share the blend used to have (chosen then by validation log-loss)


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


def tuned_metrics(p_val, y_val, p_test, y_test):
    th, tm, _, _ = tune_thresholds(p_val, y_val)
    m = metrics(y_test, decide_levels(p_test, th, tm))
    return {"thresholds_t_high_t_medium": (round(th, 3), round(tm, 3)), "macro_f1": m["macro_f1"],
            "high_recall": m["per_class"]["high"]["recall"], "high_precision": m["per_class"]["high"]["precision"]}


def fit_classical(train_df, numeric):
    pre = ColumnTransformer([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2), TEXT_FEATURE),
        ("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    cols = [TEXT_FEATURE] + numeric + CATEGORICAL_FEATURES
    pipe.fit(train_df[cols], train_df[TARGET])
    return pipe, cols


def classical_argmax_macro_f1(pipe, cols, df):
    return round(f1_score(df[TARGET], pipe.predict(df[cols]), average="macro"), 4)


def main():
    cfg = load_config()
    train_df, val_df, test_df = [pd.read_csv(DATA_PROCESSED / f"{s}.csv") for s in ("train", "val", "test")]
    y, yv = test_df[TARGET].to_numpy(), val_df[TARGET].to_numpy()
    y_idx = np.array([CLASSES.index(v) for v in y])

    p_tx = hi.transformer_proba(test_df[TEXT_FEATURE].astype(str))
    p_tx_v = hi.transformer_proba(val_df[TEXT_FEATURE].astype(str))
    res = assess(p_tx, cfg, extra_flags=hi.text_flags(test_df[TEXT_FEATURE], cfg))
    levels = res["level"]
    p_cl, p_cl_v = baselines.classical_proba(test_df), baselines.classical_proba(val_df)
    argmax = lambda p: np.array(CLASSES)[p.argmax(axis=1)]

    out = {
        "n_test": len(test_df),
        "class_share": {c: round(float((y == c).mean()), 4) for c in CLASSES},
        "decision_config": cfg,
        "system": metrics(y, levels),
        "components_argmax": {
            "majority_class_low": metrics(y, np.full(len(y), "low")),
            "classical_baseline": metrics(y, argmax(p_cl)),
            "transformer": metrics(y, argmax(p_tx)),
        },
        "auc_high_vs_rest": {"classical_baseline": round(roc_auc_score(y == "high", p_cl[:, 2]), 4),
                             "transformer": round(roc_auc_score(y == "high", p_tx[:, 2]), 4)},
        "calibration_ece": {"transformer_raw": round(ece(p_tx, y_idx), 4), "transformer_calibrated": round(ece(res["p_final"], y_idx), 4),
                            "classical_baseline_raw": round(ece(p_cl, y_idx), 4)},
    }

    # ---- baseline comparison: each gets thresholds tuned on validation with the same rule
    t_tx_v, t_tx_t = apply_temperature(p_tx_v, cfg["temp_text"]), res["p_final"]
    t_cl = fit_temperature(p_cl_v, np.array([CLASSES.index(v) for v in yv]))
    c_v, c_t = apply_temperature(p_cl_v, t_cl), apply_temperature(p_cl, t_cl)
    out["each_tuned_on_validation"] = {"served_system": tuned_metrics(t_tx_v, yv, t_tx_t, y),
                                       "classical_baseline": tuned_metrics(c_v, yv, c_t, y)}
    blend_v, blend_t = BLEND_WEIGHT * c_v + (1 - BLEND_WEIGHT) * t_tx_v, BLEND_WEIGHT * c_t + (1 - BLEND_WEIGHT) * t_tx_t
    out["ablation_blend_classical_back_in"] = {"weight": BLEND_WEIGHT, **tuned_metrics(blend_v, yv, blend_t, y),
                                               "auc_high_vs_rest": round(roc_auc_score(y == "high", blend_t[:, 2]), 4)}

    # ---- ablations on the classical baseline
    base, cols = fit_classical(train_df, NUMERIC_FEATURES)
    leak, lcols = fit_classical(train_df, NUMERIC_FEATURES + ["Recommended IND"])
    aged, acols = fit_classical(train_df, ["Age"] + NUMERIC_FEATURES)
    out["ablation_with_recommended_ind_classical_argmax"] = {
        "without": metrics(y, base.predict(test_df[cols])), "with": metrics(y, leak.predict(test_df[lcols]))}
    out["age_ablation_classical_argmax_macro_f1"] = {"without_age": classical_argmax_macro_f1(base, cols, test_df),
                                                    "with_age": classical_argmax_macro_f1(aged, acols, test_df)}

    # ---- LSTM baseline (needs tensorflow and the locally regenerated weights)
    if baselines.rnn_available():
        p_lstm = baselines.rnn_proba(test_df[TEXT_FEATURE].astype(str))
        out["components_argmax"]["rnn_lstm_baseline"] = metrics(y, argmax(p_lstm))
        out["auc_high_vs_rest"]["rnn_lstm_baseline"] = round(roc_auc_score(y == "high", p_lstm[:, 2]), 4)

    # ---- triage
    action, true_high, auto = res["action"], y == "high", res["action"] == "auto_reply"
    would_auto = levels == "low"
    any_flag = np.any(list(res["flags"].values()), axis=0)
    withheld, rest = would_auto & any_flag, would_auto & ~any_flag
    out["triage"] = {
        "action_share": {a: round(float((action == a).mean()), 4) for a in ("auto_reply", "manual_review", "escalate_immediately")},
        "human_review_share": round(float(res["review"].mean()), 4),
        "review_trigger_share": {k: round(float(v.mean()), 4) for k, v in res["flags"].items()},
        "truly_high_cases": int(true_high.sum()),
        "truly_high_predicted_high": int(((levels == "high") & true_high).sum()),
        "truly_high_predicted_medium": int(((levels == "medium") & true_high).sum()),
        "truly_high_auto_replied": int((auto & true_high).sum()),
        "truly_high_reaching_human": int((res["review"] & true_high).sum()),
        "auto_reply_precision_low": round(float((y[auto] == "low").mean()), 4) if auto.any() else None,
        "recall_of_truly_low_auto_replied": round(float((auto & (y == "low")).sum() / (y == "low").sum()), 4),
        "review_triggers_value": {
            "cases_the_tier_alone_would_auto_reply": int(would_auto.sum()),
            "withheld_by_a_trigger": int(withheld.sum()),
            "share_not_low_among_withheld": round(float((y[withheld] != "low").mean()), 3) if withheld.any() else None,
            "share_not_low_among_still_auto_replied": round(float((y[rest] != "low").mean()), 3),
            "truly_high_among_withheld": int((y[withheld] == "high").sum()),
            "truly_high_among_still_auto_replied": int((y[rest] == "high").sum()),
        },
    }
    out["historical_reference_old_hybrid"] = {
        "note": "Measured on this same test split BEFORE the redesign (old models, constant CNN=0.4, thresholds 0.45/0.75).",
        "high_recall": 0.132, "high_precision": 0.681, "macro_f1": 0.543,
        "confusion_matrix_rows_true_cols_pred": [[2550, 66, 2], [176, 228, 20], [29, 279, 47]],
        "share_forced_to_human_review": 0.807, "share_scored_high": 0.020,
    }

    with open(RESULTS / "monitoring_baseline.json", "w") as f:  # reference distribution for monitoring.py
        json.dump({"source": "test split", "action_share": out["triage"]["action_share"],
                   "mean_top_probability": round(float(res["p_final"].max(axis=1).mean()), 3)}, f, indent=2)
    with open(RESULTS / "evaluation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("system", "each_tuned_on_validation", "ablation_blend_classical_back_in", "triage")}, indent=1)[:3500])


if __name__ == "__main__":
    main()
