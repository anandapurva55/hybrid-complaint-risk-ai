"""
Fine-tune DistilBERT on review text for the 3-class dissatisfaction tier.

Experimental component: compare against the classical/RNN models with
evaluate_hybrid.py. Saves the fp32 model to models/transformer_fp32/ (git-ignored; compact_transformer.py makes the
shipped int8 copy).

    python -m training.train_transformer_model            # the shipped model (seed 42)
    python -m training.train_transformer_model --seed 1   # variance study: only writes results/seed_runs/seed_1.json
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hybrid_risk.paths import DATA_PROCESSED, MODELS, RESULTS
from hybrid_risk.risk_common import CLASSES, RANDOM_STATE, TARGET, TEXT_FEATURE

BASE = "distilbert-base-uncased"
OUT_DIR = str(MODELS / "transformer_fp32")  # overridden by --out-dir
MAX_LEN, BATCH, EPOCHS, LR = 128, 32, 2, 3e-5

_ap = argparse.ArgumentParser()
_ap.add_argument("--seed", type=int, default=RANDOM_STATE)
_ap.add_argument("--synthetic", action="store_true", help="also train on data/synthetic/train.jsonl (LLM-written sarcasm and negation)")
_ap.add_argument("--out-dir", default=None, help="where to save the fp32 model (default models/transformer_fp32)")
_args = _ap.parse_args()
SEED = _args.seed
torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3).to(device)


def load(split):
    df = pd.read_csv(DATA_PROCESSED / f"{split}.csv")
    return df[TEXT_FEATURE].astype(str).tolist(), np.array([CLASSES.index(v) for v in df[TARGET]])


def batches(texts, labels=None, shuffle=False):
    idx = np.random.RandomState(RANDOM_STATE).permutation(len(texts)) if shuffle else np.arange(len(texts))
    for i in range(0, len(idx), BATCH):
        b = idx[i:i + BATCH]
        enc = tok([texts[j] for j in b], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        yield enc, (torch.tensor(labels[b]).to(device) if labels is not None else None)


@torch.no_grad()
def predict(texts):
    model.eval()
    out = []
    for enc, _ in batches(texts):
        out.append(torch.softmax(model(**enc).logits, -1).cpu().numpy())
    return np.concatenate(out)


def main():
    tr_t, tr_y = load("train"); va_t, va_y = load("val"); te_t, te_y = load("test")
    if _args.synthetic:
        import json as _json
        from hybrid_risk.paths import ROOT as _ROOT
        syn = [_json.loads(l) for l in open(_ROOT / "data" / "synthetic" / "train.jsonl")]
        tr_t = tr_t + [r["text"] for r in syn]
        tr_y = np.concatenate([tr_y, np.array([CLASSES.index(r["tier"]) for r in syn])])
        print(f"training on {len(tr_t) - len(syn)} real + {len(syn)} synthetic reviews", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = EPOCHS * ((len(tr_t) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total_iters=steps)
    for ep in range(EPOCHS):
        model.train()
        rng_seed = SEED + ep
        idx = np.random.RandomState(rng_seed).permutation(len(tr_t))
        for n, i in enumerate(range(0, len(idx), BATCH)):
            b = idx[i:i + BATCH]
            enc = tok([tr_t[j] for j in b], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            loss = model(**enc, labels=torch.tensor(tr_y[b]).to(device)).loss
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            if n % 100 == 0:
                print(f"epoch {ep} step {n} loss {loss.item():.3f}", flush=True)
        pv = predict(va_t)
        print(f"epoch {ep} val macro-F1 (argmax) {f1_score(va_y, pv.argmax(1), average='macro'):.3f}", flush=True)
    pt = predict(te_t)
    if SEED != RANDOM_STATE:  # variance-study run: record metrics only, never overwrite the shipped model
        from sklearn.metrics import roc_auc_score
        (RESULTS / "seed_runs").mkdir(parents=True, exist_ok=True)
        rec = {"seed": SEED,
               "val_macro_f1_argmax": round(f1_score(va_y, pv.argmax(1), average="macro"), 4),
               "test_macro_f1_argmax": round(f1_score(te_y, pt.argmax(1), average="macro"), 4),
               "test_auc_high": round(roc_auc_score(te_y == 2, pt[:, 2]), 4)}
        json.dump(rec, open(RESULTS / "seed_runs" / f"seed_{SEED}.json", "w"), indent=2)
        print(rec)
        return
    print("test macro-F1 (argmax)", round(f1_score(te_y, pt.argmax(1), average="macro"), 3))
    out_dir = _args.out_dir or OUT_DIR
    model.save_pretrained(out_dir); tok.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
