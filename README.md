# Hybrid Dissatisfaction-Risk Triage (Classical ML + RNN + LLM explanation)

A triage pipeline that reads customer feedback, estimates how dissatisfied the customer is (low / medium / high), routes it to **auto-reply, manual review or escalation**, and explains the decision in plain language.

## What this is, and is not

- **It is** a worked example of a hybrid system: two trained text models, a calibrated probability-based decision layer, deterministic review rules, and an LLM that explains but cannot decide.
- **It is not** an escalation predictor. The dataset ([Women's E-Commerce Clothing Reviews](Womens%20Clothing%20E-Commerce%20Reviews.csv)) holds product *reviews*, not complaints, and has no escalation outcomes. The label is a **proxy** derived from star rating: 1-2 stars = `high`, 3 = `medium`, 4-5 = `low`. All metrics measure recovery of that rating-derived tier. Treat "risk" as "dissatisfaction", and validate against real escalation data before any real-world use. (Variable names such as `risk_level` are kept for continuity.)

## Architecture

```
 review_text + age + feedback count + division/department/class
                │
     ┌──────────┴──────────┐
     ▼                     ▼
 Classical model        RNN / LSTM          each -> [P(low), P(medium), P(high)]
 (TF-IDF + tabular      (text only)
  LogReg)
     │  temperature calibration (fit on validation)
     └──────────┬──────────┘
                ▼
   p = 0.7*classical + 0.3*rnn            (weight = min validation log-loss)
                ▼
   high   if P(high) >= t_high            (thresholds tuned on validation
   medium elif P(high)+P(medium) >= t_med   for high-class recall >= 0.70)
   low    otherwise
                ▼
   review triggers: conflicted evidence, low confidence, model disagreement, very short text
                ▼
   action: escalate_immediately | manual_review | auto_reply   (deterministic)
                ▼
   Gemini explanation (summary, reason, notes)  -- cannot change level, action or review flag
```

### Why the CNN is not in the decision

The original design gave a CNN 30% of the score. It was trained on 28x28 Fashion-MNIST clothing categories, so it cannot judge a real complaint photo, and with no image it returned a constant (0.4, or 0.6 if an image was flagged). That constant carried zero information, merely attaching a photo moved the score by 0.06, it made "high" almost unreachable (high recall was **13%**), and it made the disagreement rule fire on ~81% of cases. The CNN is now excluded; [`train_cnn_model.py`](train_cnn_model.py) remains only as a documented demo of the architecture. A real image signal needs outcome-labelled complaint photos, which this dataset does not have.

## Results (test split, n = 3,397)

Full tables and caveats: [evaluation_report.md](evaluation_report.md).

| | Old hybrid | New hybrid |
|---|---|---|
| Recall on high | 0.13 | **0.73** |
| Macro-F1 | 0.54 | **0.62** |
| Cases forced to human review | 81% | **30%** |
| Truly-high cases that reach a human | n/a | 339 / 355 |

- Hybrid vs classical alone: only +0.003 macro-F1 / +0.003 AUC (within noise); the RNN adds a small signal.
- High precision is 0.50: about half of "high" flags are false alarms, by design (recall-first).
- Removing the leaky `Recommended IND` feature explains why headline metrics are lower than the first version's (0.845 / 0.666): that gap was ~0.05 macro-F1 of label leakage.
- The LLM layer is tested with fakes and an injection suite; a live-API evaluation script exists but has **not** been run (no key was available). See the report.

## Decision rules

See [decision_policy.md](decision_policy.md). In short: the level and action are computed deterministically from calibrated probabilities; a low case with conflicting evidence, low confidence, disagreeing models or very short text is withheld from auto-reply; the LLM can only add caution.

## Setup

Python 3.11.

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt     # requirements.txt alone is enough to serve
cp .env.example .env                    # optional: add GEMINI_API_KEY (and API_KEY to protect /predict)
```

Without `GEMINI_API_KEY` everything still works; explanations come from a local fallback and are labelled `_meta.explanation_source = "fallback"`.

## Usage

```bash
python make_splits.py            # train/val/test CSVs from the raw dataset (needed for training/eval)
python train_classical_model.py
python train_rnn_model.py
python tune_decision.py          # calibration + blend weight + thresholds, from validation only
python evaluate_hybrid.py        # test-split evaluation -> evaluation_results.json
python hybrid_inference.py       # run the bundled samples
python generate_sample_outputs.py
python evaluate_llm.py --n 60    # needs GEMINI_API_KEY
python -m pytest                 # unit tests + end-to-end smoke tests
uvicorn app:app --reload         # HTTP API
```

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \     # only needed if API_KEY is set
  -d '{"review_text":"The zipper broke after one wear and support never replied.","age":35,
       "positive_feedback_count":2,"division_name":"General","department_name":"Dresses",
       "class_name":"Dresses"}'
```

Container: `docker build -t hybrid-risk-api . && docker run --rm -p 8000:8000 --env-file .env hybrid-risk-api`. The rebuilt image was built and smoke-tested locally (health, API-key auth, a prediction); it has not been redeployed to AWS.

## Repository map

| File | Purpose |
|---|---|
| `risk_common.py` | constants, label proxy, probability ordering |
| `decision.py`, `decision_config.json` | pure decision logic and its tuned parameters |
| `hybrid_inference.py` | model loading, end-to-end `predict_hybrid` |
| `llm_explainer.py` | Gemini explanation with sanitising, timeout, policy re-imposition, fallback |
| `app.py` | FastAPI wrapper (validation, optional API key, no internal error leakage) |
| `make_splits.py`, `train_*.py`, `tune_decision.py`, `evaluate_*.py` | reproducible pipeline |
| `tests/`, `.github/workflows/ci.yml` | tests and CI |
| `evaluation_report.md`, `decision_policy.md`, `mini_model_card.md` | documentation |
| `model_manifest.json`, `requirements*.txt` | pinned versions and rebuild steps |
| `aws_deployment_notes.md` | record of an earlier AWS deployment (see its banner) |

## Known gaps

- Proxy label, unvalidated against real escalations (the main limitation).
- Medium class is weak (F1 ~0.35); one random split, no confidence intervals.
- Live-LLM evaluation not run (no API key available); AWS deployment not redone after the redesign (the image itself builds and serves locally).
- No drift monitoring, no built-in rate limiting.
