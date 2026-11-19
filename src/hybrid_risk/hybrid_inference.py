"""
End-to-end inference: DistilBERT probabilities -> deterministic decision -> optional LLM explanation.

A fine-tuned DistilBERT (int8-compressed weights) reads the review text. Its calibrated class
probabilities are turned into a risk tier, an action and a human-review flag by decision.py, using
parameters tuned on the validation split (models/decision_config.json). The LLM (llm_explainer.py)
only EXPLAINS the result; it cannot change it.

Input: only `review_text` is used. The other fields of the original dataset (age, feedback count,
division/department/class, `recommended_ind`, images) are accepted for compatibility and ignored.

    {"review_text": "The zipper is broken and nobody answered my emails."}

Smoke test:  python -m hybrid_risk.hybrid_inference
"""

import json
import math
import os
import threading

import numpy as np

from hybrid_risk import text_checks
from hybrid_risk.decision import action_for, assess, final_score, load_config, review_reasons
from hybrid_risk.llm_explainer import explain_decision
from hybrid_risk.paths import SAMPLES
from hybrid_risk.risk_common import CLASSES

MIN_WORDS = text_checks.MIN_WORDS  # shorter texts give the text models too little to go on -> force review

_transformer = None
_infer_lock = threading.Lock()  # one transformer forward pass at a time: queueing beats thread thrashing


def cpu_limit():
    """CPUs this process may really use: TORCH_NUM_THREADS, else the container's cgroup quota, else all cores.
    (Inside a container os.cpu_count() reports the HOST's cores, which makes PyTorch oversubscribe a 2 vCPU task.)"""
    env = os.environ.get("TORCH_NUM_THREADS")
    if env:
        return max(1, int(env))
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:           # cgroup v2
            quota, period = f.read().split()
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:  # cgroup v1
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def _load_transformer():
    global _transformer
    if _transformer is None:
        import torch
        torch.set_num_threads(cpu_limit())
        from hybrid_risk import compact_transformer  # imported lazily: needs torch + transformers
        _transformer = compact_transformer.load()
    return _transformer


def transformer_proba(texts, batch_size=32, max_len=128):
    """(N, 3) probabilities in CLASSES order from the fine-tuned DistilBERT."""
    import torch
    model, tok = _load_transformer()
    texts = [str(t) for t in texts]
    out = []
    with _infer_lock, torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            out.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
    return np.concatenate(out)


def text_flags(texts, cfg=None):
    """Input-quality review triggers for a sequence of texts: short_text, noisy_text, non_english."""
    cfg = cfg or load_config()
    _, tok = _load_transformer()
    texts = [str(t) for t in texts]
    return {
        "short_text": np.array([len(t.split()) < MIN_WORDS for t in texts]),
        "noisy_text": np.array([text_checks.is_noisy(t, tok.tokenize, cfg["noisy_pieces_per_word"]) for t in texts]),
        "non_english": np.array([text_checks.is_non_english(t, cfg["min_english_share"]) for t in texts]),
    }


def predict_hybrid(input_data, explain=True):
    """Run DistilBERT, decide deterministically, then (optionally) explain."""
    cfg = load_config()
    text = str(input_data["review_text"])

    result = assess(transformer_proba([text]), cfg, extra_flags=text_flags([text], cfg))
    level = str(result["level"][0])
    review = bool(result["review"][0])
    reasons = review_reasons(result["flags"])
    p_final = result["p_final"]

    model_outputs = {
        "final_score": round(float(final_score(p_final)[0]), 3),
        "probabilities": {c: round(float(p), 3) for c, p in zip(CLASSES, p_final[0])},
        "risk_level": level,
        "recommended_action": action_for(level, review),
        "human_review_required": review,
        "review_reasons": reasons,
    }

    if explain:
        explanation = explain_decision({"complaint_text": text, "model_outputs": model_outputs})
    else:
        explanation = None

    source = (explanation or {}).pop("_source", None) if explanation else None
    return {
        "model_outputs": model_outputs,
        "llm_explanation": explanation,
        "_meta": {"explanation_source": source},
    }


if __name__ == "__main__":
    with open(SAMPLES / "sample_inputs.json") as f:
        samples = json.load(f)

    for sample in samples:
        result = predict_hybrid(sample)
        print(f"\n=== {sample.get('customer_id', '?')} ===")
        print(json.dumps(result, indent=2))
