"""
LLM-as-judge PROXY for the explanation comparison (LLM explanation vs templated baseline).

This is NOT human evidence. A model grading models can share their blind spots and favour fluent text,
so it is reported next to, never instead of, the human blind sheet in results/explanation_review_sheet.md.
Safeguards: the judge is a different model family (Claude) from the explainer (Amazon Nova); every pair is
scored twice with the A/B order swapped, and we report how often the two orders agree (position bias).

    python -m evaluation.judge_explanations         # reads results/llm_eval_samples.json
"""
import json
import re

import boto3
import numpy as np
from botocore.config import Config

from hybrid_risk.paths import RESULTS

JUDGE_MODEL = "us.anthropic.claude-sonnet-4-6"

PROMPT = """You are grading two explanations of an automated customer-feedback triage decision, written for a support reviewer.

COMPLAINT:
\"\"\"{complaint}\"\"\"

DECISION (already final): tier={tier}, action={action}, review reasons={reasons}, probabilities={probs}

EXPLANATION A:
{a}

EXPLANATION B:
{b}

Rate EACH explanation from 1 (poor) to 5 (excellent) on:
- faithful: says nothing the complaint or the decision does not support. Penalise invented facts or claims heavily.
- useful: would help a reviewer understand and act on this case.
- clear: concise and easy to read.
Then say which you would rather show a reviewer: "A", "B" or "tie".

Return ONLY JSON: {{"A": {{"faithful": n, "useful": n, "clear": n}}, "B": {{"faithful": n, "useful": n, "clear": n}}, "preferred": "A"|"B"|"tie"}}"""


def fmt(e):
    return f"{e['summary']}\n{e['reason']}\nNotes: {'; '.join(map(str, e['risk_notes'])) or 'none'}"


def judge(client, rec, llm_first):
    mo = rec["model_outputs"]
    a, b = (rec["llm"], rec["template"]) if llm_first else (rec["template"], rec["llm"])
    prompt = PROMPT.format(complaint=rec["text"][:1500], tier=mo["risk_level"], action=mo["recommended_action"],
                           reasons=", ".join(mo["review_reasons"]) or "none", probs=mo["probabilities"], a=fmt(a), b=fmt(b))
    r = client.converse(modelId=JUDGE_MODEL, messages=[{"role": "user", "content": [{"text": prompt}]}],
                        inferenceConfig={"temperature": 0, "maxTokens": 400})
    text = r["output"]["message"]["content"][0]["text"]
    d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    llm_key, tpl_key = ("A", "B") if llm_first else ("B", "A")
    pref = {"tie": "tie", llm_key: "llm", tpl_key: "template"}[d["preferred"]]
    return {"llm": d[llm_key], "template": d[tpl_key], "preferred": pref}


def judge_with_retry(client, rec, llm_first):
    try:
        return judge(client, rec, llm_first)
    except (AttributeError, KeyError, ValueError):  # no JSON / missing fields in the reply: one more attempt
        return judge(client, rec, llm_first)


def main():
    records = [r for r in json.load(open(RESULTS / "llm_eval_samples.json")) if r["source"] == "llm"]
    client = boto3.client("bedrock-runtime", region_name="us-east-1", config=Config(read_timeout=120, retries={"max_attempts": 3}))
    scores = {k: {d: [] for d in ("faithful", "useful", "clear")} for k in ("llm", "template")}
    prefs, agree, skipped = [], [], 0
    for rec in records:
        try:
            first, second = judge_with_retry(client, rec, True), judge_with_retry(client, rec, False)
        except Exception:  # the judge returned no usable JSON twice: skip the case and report it
            skipped += 1
            continue
        agree.append(first["preferred"] == second["preferred"])
        for res in (first, second):
            prefs.append(res["preferred"])
            for k in scores:
                for d in scores[k]:
                    scores[k][d].append(res[k][d])
    n = len(prefs)
    out = {"judge_model": JUDGE_MODEL, "n_cases": len(records) - skipped, "skipped_cases": skipped, "n_judgements": n,
           "note": "LLM judge, a PROXY and not human evidence; each case judged twice with A/B order swapped",
           "mean_scores_1_to_5": {k: {d: round(float(np.mean(v)), 2) for d, v in dims.items()} for k, dims in scores.items()},
           "preference": {p: round(prefs.count(p) / n, 3) for p in ("llm", "template", "tie")},
           "position_bias_check_same_preference_in_both_orders": round(float(np.mean(agree)), 3)}
    print(json.dumps(out, indent=1))
    with open(RESULTS / "explanation_judge_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
