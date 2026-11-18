"""
Fine-tune DistilBERT on review text for the 3-class dissatisfaction tier.

Experimental component: compare against the classical/RNN models with
evaluate_hybrid.py. Saves the model to transformer_model/ and validation/test
probabilities (CLASSES order) to transformer_{val,test}_proba.npy.

    python train_transformer_model.py            # the shipped model (seed 42)
    python train_transformer_model.py --seed 1   # variance study: only writes seed_runs/seed_1.json
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from risk_common import CLASSES, RANDOM_STATE, TARGET, TEXT_FEATURE

BASE = "distilbert-base-uncased"
OUT_DIR = "transformer_model"
MAX_LEN, BATCH, EPOCHS, LR = 128, 32, 2, 3e-5

_ap = argparse.ArgumentParser()
_ap.add_argument("--seed", type=int, default=RANDOM_STATE)
SEED = _ap.parse_args().seed
torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3).to(device)


def load(split):
    df = pd.read_csv(f"{split}.csv")
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
        os.makedirs("seed_runs", exist_ok=True)
        rec = {"seed": SEED,
               "val_macro_f1_argmax": round(f1_score(va_y, pv.argmax(1), average="macro"), 4),
               "test_macro_f1_argmax": round(f1_score(te_y, pt.argmax(1), average="macro"), 4),
               "test_auc_high": round(roc_auc_score(te_y == 2, pt[:, 2]), 4)}
        json.dump(rec, open(f"seed_runs/seed_{SEED}.json", "w"), indent=2)
        print(rec)
        return
    np.save("transformer_val_proba.npy", pv); np.save("transformer_test_proba.npy", pt)
    print("test macro-F1 (argmax)", round(f1_score(te_y, pt.argmax(1), average="macro"), 3))
    model.save_pretrained(OUT_DIR); tok.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    main()
