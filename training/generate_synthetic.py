"""
Generate LLM-written reviews for the two input styles the real data barely covers: SARCASTIC complaints
and NEGATION / understatement ("not bad at all", "wouldn't recommend").

    python -m training.generate_synthetic               # train/dev (Claude Haiku) and the held-out test (Nova Pro)
    python -m training.generate_synthetic --extra-train  # a second round of TRAIN-only data from Claude Sonnet

Writes data/synthetic/{train,dev,test}.jsonl with fields text, kind, tier, generator. These are SYNTHETIC,
not real customer text. To keep the evaluation honest:
  * train and dev come from Claude Haiku; the held-out TEST comes from a different model family
    (Amazon Nova Pro), so a gain on it cannot just be the model learning one generator's style;
  * test texts are never used for training, tuning or threshold selection;
  * near-duplicates across splits are removed.
Real validation and test reviews are untouched; calibration and thresholds are still tuned on them only.
"""
import json
import random
import re

import boto3
from botocore.config import Config

from hybrid_risk.paths import ROOT

OUT = ROOT / "data" / "synthetic"
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
NOVA_PRO = "us.amazon.nova-pro-v1:0"
SONNET = "us.anthropic.claude-sonnet-4-6"

PRODUCTS = ["dress", "sweater", "jeans", "blouse", "skirt", "jacket", "coat", "leggings", "tank top", "cardigan", "shorts",
            "swimsuit", "bra", "scarf", "boots", "sandals", "handbag", "pajamas", "hoodie", "blazer", "jumpsuit", "t-shirt",
            "socks", "belt", "gloves", "romper", "trousers", "tunic", "vest", "sneakers"]
PROBLEMS = ["it fell apart", "the sizing is wrong", "the fabric is scratchy", "it arrived late and damaged", "the color faded",
            "the zipper broke", "it shrank in the wash", "customer service ignored me", "it looks nothing like the photo",
            "the stitching came undone", "it pilled immediately", "the smell is awful", "the fit is terrible", "it was overpriced for the quality"]

KINDS = {
    "sarcastic_complaint": ("high",
        "customer reviews of fashion products where the customer is SARCASTIC: the wording sounds positive, grateful or ironic, "
        "but the customer is clearly unhappy and had a 1 or 2 star experience. Each mentions a concrete problem (for example: {problems})."),
    "negated_positive": ("low",
        "customer reviews that are genuinely SATISFIED (4 or 5 stars) but expressed through negation or understatement, such as "
        "\"not bad at all\", \"can't complain\", \"no regrets\", \"not disappointed\", \"nothing to fault\", \"wouldn't change a thing\"."),
    "negated_negative": ("high",
        "customer reviews that are genuinely UNHAPPY (1 or 2 stars) and expressed with negation, such as \"not good\", "
        "\"wouldn't recommend\", \"not worth it\", \"not what I expected\", \"not comfortable\", \"not happy at all\"."),
}
PLAN = {  # kind -> (train calls, dev calls) with Haiku, and test calls with Nova Pro; 15 texts per call
    "sarcastic_complaint": (24, 4, 6), "negated_positive": (10, 2, 3), "negated_negative": (8, 2, 3)}


def converse(client, model, prompt, temperature=1.0):
    r = client.converse(modelId=model, messages=[{"role": "user", "content": [{"text": prompt}]}],
                        inferenceConfig={"temperature": temperature, "maxTokens": 3000})
    return r["output"]["message"]["content"][0]["text"]


def parse_list(text):
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        items = []
    return [i.strip() for i in items if isinstance(i, str) and 20 <= len(i.strip()) <= 600]


def prompt_for(kind, n, rng):
    tier, desc = KINDS[kind]
    prods = ", ".join(rng.sample(PRODUCTS, 6))
    probs = ", ".join(rng.sample(PROBLEMS, 5))
    return (f"Write {n} different realistic {desc.format(problems=probs)} Products to draw from: {prods}. "
            "Vary the length (one to four sentences), tone, and wording; write like real online reviewers (casual, sometimes imperfect), "
            "with no emojis and no hashtags, and never use the words 'sarcasm' or 'sarcastic'. "
            "Return ONLY a JSON array of strings.")


def norm(t):
    return re.sub(r"[^a-z]+", " ", t.lower()).strip()


def extra_train():
    """Second round: extra TRAIN-only data from a second Claude model for diversity (test stays Nova-only and untouched)."""
    client = boto3.client("bedrock-runtime", region_name="us-east-1", config=Config(read_timeout=180, retries={"max_attempts": 3}))
    rng = random.Random(7)
    existing = [json.loads(l) for split in ("train", "dev", "test") for l in open(OUT / f"{split}.jsonl")]
    seen = {norm(r["text"]) for r in existing}
    added = []
    for kind, calls in (("sarcastic_complaint", 30), ("negated_positive", 5), ("negated_negative", 4)):
        got = 0
        for _ in range(calls):
            for t in parse_list(converse(client, SONNET, prompt_for(kind, 15, rng))):
                if norm(t) in seen:
                    continue
                seen.add(norm(t)); got += 1
                added.append({"text": t, "kind": kind, "tier": KINDS[kind][0], "generator": "claude-sonnet-4.6"})
        print(f"{kind:22s} extra train {got:4d} texts from claude-sonnet-4.6")
    train = [json.loads(l) for l in open(OUT / "train.jsonl")] + added
    rng.shuffle(train)
    with open(OUT / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("train now", len(train))


def main():
    client = boto3.client("bedrock-runtime", region_name="us-east-1", config=Config(read_timeout=120, retries={"max_attempts": 3}))
    rng = random.Random(42)
    seen, splits = set(), {"train": [], "dev": [], "test": []}
    for kind, (ntrain, ndev, ntest) in PLAN.items():
        tier = KINDS[kind][0]
        for split, calls, model, name in (("train", ntrain, HAIKU, "claude-haiku-4.5"), ("dev", ndev, HAIKU, "claude-haiku-4.5"),
                                          ("test", ntest, NOVA_PRO, "nova-pro")):
            got = 0
            for _ in range(calls):
                for t in parse_list(converse(client, model, prompt_for(kind, 15, rng))):
                    k = norm(t)
                    if k in seen:
                        continue
                    seen.add(k); got += 1
                    splits[split].append({"text": t, "kind": kind, "tier": tier, "generator": name})
            print(f"{kind:22s} {split:5s} {got:4d} texts from {name}")
    OUT.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        rng.shuffle(rows)
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    import sys
    extra_train() if "--extra-train" in sys.argv else main()
