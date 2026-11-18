"""
Portable int8 weight storage for the fine-tuned DistilBERT.

The fp32 weights are ~268 MB (over GitHub's 100 MB file limit). Every 2-D+ weight
tensor is stored as int8 with a per-row fp32 scale (symmetric quantisation);
1-D tensors (biases, LayerNorm) stay fp32. At load time the weights are
de-quantised back to fp32, so there is no dependency on a particular torch
quantisation engine / CPU architecture. evaluate_hybrid.py reports the test
metrics of the shipped (int8) model, so any accuracy cost is visible.

    python compact_transformer.py      # fp32 transformer_model/ -> transformer_assets/
"""
import os
import shutil

import numpy as np

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transformer_assets")
WEIGHTS = os.path.join(ASSET_DIR, "weights_int8.npz")


def export(fp32_dir="transformer_model", out_dir=ASSET_DIR):
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(fp32_dir)
    arrays = {}
    for name, t in model.state_dict().items():
        a = t.detach().cpu().numpy().astype("float32")
        if a.ndim >= 2:
            flat = a.reshape(a.shape[0], -1)
            scale = np.maximum(np.abs(flat).max(axis=1), 1e-12) / 127.0
            q = np.clip(np.round(flat / scale[:, None]), -127, 127).astype("int8")
            arrays[f"q::{name}"] = q.reshape(a.shape)
            arrays[f"s::{name}"] = scale.astype("float32")
        else:
            arrays[f"f::{name}"] = a
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "weights_int8.npz"), **arrays)
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(fp32_dir, f), os.path.join(out_dir, f))
    print("wrote", out_dir, round(os.path.getsize(os.path.join(out_dir, "weights_int8.npz")) / 1e6, 1), "MB")


def load(asset_dir=ASSET_DIR):
    """Return (model, tokenizer) with de-quantised fp32 weights, in eval mode."""
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    cfg = AutoConfig.from_pretrained(asset_dir)
    model = AutoModelForSequenceClassification.from_config(cfg)
    data = np.load(os.path.join(asset_dir, "weights_int8.npz"))
    state = {}
    for key in data.files:
        kind, name = key.split("::", 1)
        if kind == "f":
            state[name] = torch.from_numpy(data[key])
        elif kind == "q":
            q, s = data[key].astype("float32"), data[f"s::{name}"]
            state[name] = torch.from_numpy((q.reshape(q.shape[0], -1) * s[:, None]).reshape(q.shape))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, AutoTokenizer.from_pretrained(asset_dir)


if __name__ == "__main__":
    export()
