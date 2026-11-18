# Hybrid Dissatisfaction-Risk Triage (Classical ML + fine-tuned DistilBERT + LLM explanation)

A triage pipeline that reads customer feedback, estimates how dissatisfied the customer is (low / medium / high), routes it to **auto-reply, manual review or escalation**, and explains the decision in plain language.

## What this is, and is not

- **It is** a worked example of a hybrid system: a classical model plus a fine-tuned transformer, a calibrated probability-based decision layer, deterministic review rules, and a local LLM (Ollama) that explains but cannot decide.
- **It is not** an escalation predictor. The dataset ([Women's E-Commerce Clothing Reviews](Womens%20Clothing%20E-Commerce%20Reviews.csv)) holds product *reviews*, not complaints, and has no escalation outcomes. The label is a **proxy** derived from star rating: 1-2 stars = `high`, 3 = `medium`, 4-5 = `low`. All metrics measure recovery of that rating-derived tier. Treat "risk" as "dissatisfaction", and validate against real escalation data before any real-world use. (Variable names such as `risk_level` are kept for continuity.)

## Architecture

```
 review_text + age + feedback count + division/department/class
                │
     ┌──────────┴──────────┐
     ▼                     ▼
 Classical model       Fine-tuned DistilBERT     each -> [P(low), P(medium), P(high)]
 (TF-IDF + tabular     (text only, int8 weights)
  LogReg)
     │  temperature calibration (fit on validation)
     └──────────┬──────────┘
                ▼
   p = 0.1*classical + 0.9*transformer     (weight = min validation log-loss)
                ▼
   high   if P(high) >= t_high             (thresholds tuned on validation
   medium elif P(high)+P(medium) >= t_med    for high-class recall >= 0.70)
   low    otherwise
                ▼
   review triggers: model disagreement, low confidence, conflicted evidence, very short text
                ▼
   action: escalate_immediately | manual_review | auto_reply   (deterministic)
                ▼
   LLM explanation (Ollama llama3.2 locally, or Bedrock)  -- cannot change level, action or review flag;
   output is rejected -> fallback if empty, contradictory or quoting invented numbers
```

### Why not the CNN or the LSTM?

**CNN.** The original design gave a CNN 30% of the score. It was trained on 28x28 Fashion-MNIST categories, so it cannot judge a real complaint photo, and with no image it returned a constant (0.4, or 0.6 if an image was flagged). That constant carried zero information, made "high" almost unreachable (high recall **13%**) and made the disagreement rule fire on ~81% of cases. It is excluded. No public dataset pairs complaint photos with outcome labels (only factory fabric-defect sets), so there is no honest way to build a real image signal here; it would need outcome-labelled photos from a real support system.

**LSTM.** Once a fine-tuned DistilBERT existed, the validation-optimal blend weight for the LSTM was 0, so it was dropped from the decision (it stays as a reported baseline in `baselines.py`). That also removed TensorFlow from serving and shrank the container from ~1.7 GB to ~550 MB.

## Results (test split, n = 3,397)

Full tables, confidence intervals and subgroups: [evaluation_report.md](evaluation_report.md).

| | v1 (original) | v3 (current) |
|---|---|---|
| Recall on high | 0.13 | **0.71** |
| Precision on high | 0.68 | 0.58 |
| Macro-F1 | 0.54 | **0.67** (95% CI 0.65-0.69) |
| Truly-high cases auto-replied | 29 | **5** of 355 |
| Cases forced to human review | 81% | **29%** |

- The gain comes from the fine-tuned transformer. The blend with the classical model adds only +0.002 to +0.012 macro-F1.
- About 4 in 10 "high" flags are false alarms, by design (recall-first).
- Removing the leaky `Recommended IND` feature explains why the classical model's numbers are lower than the first version's: that gap was ~0.05 macro-F1 of label leakage.
- Subgroup performance is fairly even across age bands; "Bottoms" has the lowest high recall (0.60). Details in the report.
- LLM layer evaluated live on 45 rows plus injection strings: 100% policy-compliant after validators, 0 injection successes; 6.7% of explanations still end up as the local fallback after one retry.
- Transformer seed variance: macro-F1 0.666 / 0.675 / 0.682 over three seeds.

## Decision rules

See [decision_policy.md](decision_policy.md). In short: the level and action are computed deterministically from calibrated probabilities; a low case with conflicting evidence, low confidence, disagreeing models or very short text is withheld from auto-reply; the LLM can only add caution.

## Setup

Python 3.11.

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt     # requirements.txt alone is enough to serve
ollama pull llama3.2                    # explanation LLM (needs `ollama serve`); or set LLM_PROVIDER=bedrock
cp .env.example .env                    # optional settings (see file); export them or use your shell
```

If the LLM is unreachable everything still works; explanations come from a local fallback and are labelled `_meta.explanation_source = "fallback"`.

## Usage

```bash
python make_splits.py            # train/val/test CSVs from the raw dataset (needed for training/eval)
python train_classical_model.py
python train_transformer_model.py   # ~20 min on a laptop; then: python compact_transformer.py
python tune_decision.py          # calibration + blend weight + thresholds, from validation only
python evaluate_hybrid.py        # test evaluation -> evaluation_results.json, monitoring_baseline.json
python evaluate_robustness.py    # bootstrap CIs, CV, subgroups, ordinal experiment
python hybrid_inference.py       # run the bundled samples
python generate_sample_outputs.py
python evaluate_llm.py --n 45    # needs the LLM provider reachable
python -m pytest                 # unit tests + end-to-end smoke tests
uvicorn app:app --reload         # HTTP API  (GET /stats = drift monitor)
```

The fp32 fine-tuned weights (`transformer_model/`, ~268 MB) are git-ignored; the int8 copy in `transformer_assets/` (60 MB) is what is committed and served.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \     # only needed if API_KEY is set
  -d '{"review_text":"The zipper broke after one wear and support never replied.","age":35,
       "positive_feedback_count":2,"division_name":"General","department_name":"Dresses",
       "class_name":"Dresses"}'
```

Container: `docker build -t hybrid-risk-api . && docker run --rm -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 hybrid-risk-api`. Built and smoke-tested locally (health, API-key auth, prediction, LLM explanation, `/stats`). AWS deployment (CloudFront HTTPS + WAF rate limit -> ALB -> ECS Fargate with autoscaling, Bedrock explanations, CloudWatch alarms; one CloudFormation stack) is in [deploy/README.md](deploy/README.md); it was deployed and verified live, see [aws_deployment_notes.md](aws_deployment_notes.md).

## Repository map

| File | Purpose |
|---|---|
| `risk_common.py` | constants, label proxy, probability ordering |
| `decision.py`, `decision_config.json` | pure decision logic and its tuned parameters |
| `hybrid_inference.py` | model loading, end-to-end `predict_hybrid` |
| `compact_transformer.py`, `transformer_assets/` | int8 DistilBERT weights + loader; `train_transformer_model.py` fine-tunes it |
| `llm_explainer.py` | Ollama / Bedrock explanation with sanitising, timeout, policy re-imposition, output validators, fallback |
| `monitoring.py` | rolling-window drift monitor behind `GET /stats` |
| `baselines.py`, `train_rnn_model.py`, `train_cnn_model.py` | LSTM baseline and CNN demo; not in the decision |
| `app.py` | FastAPI wrapper (validation, optional API key, no internal error leakage) |
| `make_splits.py`, `train_*.py`, `tune_decision.py`, `evaluate_*.py` | reproducible pipeline |
| `tests/`, `.github/workflows/ci.yml` | tests and CI |
| `evaluation_report.md`, `decision_policy.md`, `mini_model_card.md` | documentation |
| `model_manifest.json`, `requirements*.txt` | pinned versions and rebuild steps |
| `deploy/` | CloudFormation: `ecs-fargate.yaml` (the service) and `github-oidc.yaml` (keyless GitHub Actions deploy role); `.github/workflows/deploy.yml` |
| `aws_deployment_notes.md` | record of the first App Runner deployment, plus status of the current one |

## Known gaps

- Proxy label, unvalidated against real escalations (the main limitation).
- Medium class is the weakest (F1 0.44); high precision 0.58.
- One random data split (transformer seed variance is measured: macro-F1 0.666-0.682); no test on real support tickets.
- llama3.2 (3B) explanations: 6.7% rejected by validators after one retry; faithfulness beyond numbers/risk level not human-graded.
- No automatic retraining or label feedback loop. The CloudFront default certificate gives HTTPS on a `*.cloudfront.net` name; a custom domain needs your own ACM certificate.
- The AWS stack is currently left running (delete with `aws cloudformation delete-stack --stack-name hybrid-risk-api`).
