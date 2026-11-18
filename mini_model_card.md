# Model Card: Hybrid Dissatisfaction-Risk Triage

## Summary
Classifies customer feedback into low / medium / high **dissatisfaction risk** and explains the result. Intended as a first-pass triage aid with human oversight, not an autonomous decision maker.

## IMPORTANT: what the label is
The training label is derived from the star rating (1-2 stars = high, 3 = medium, 4-5 = low) in the Women's E-Commerce Clothing Reviews dataset. The data contains **no complaints and no escalation outcomes**. The model therefore predicts *how unhappy a review is*, used as a proxy for escalation risk. The proxy is unvalidated: an unhappy customer need not escalate, and an escalating customer may write a polite review. Do not describe outputs as "probability of escalation".

## Components
| Component | Role | Notes |
|---|---|---|
| Classical ML | TF-IDF (1-2 grams) + age, positive-feedback count, division/department/class -> Logistic Regression | `Recommended IND` removed (leaks the label) |
| RNN/LSTM | Text-only classifier | Full train split, early stopping |
| Decision layer | Blend calibrated probabilities, apply tuned thresholds, review triggers | `decision.py`, `decision_config.json` |
| LLM (Gemini) | Explanation only | Cannot change level, action or review flag |
| CNN | **Not used** | Fashion-MNIST demo only; cannot score complaint photos |

## Intended use
Prioritising and routing English-language product feedback for human reviewers; auto-acknowledging clearly positive messages.

## Out of scope
Fully automated customer-facing decisions on negative feedback; non-English text; other industries without re-validation; any use as a statistical estimate of churn or legal/regulatory escalation.

## Performance (test split, n = 3,397; see `evaluation_report.md`)
| | Value |
|---|---|
| Macro-F1 (hybrid, tuned) | 0.622 |
| Recall on high / precision on high | 0.730 / 0.498 |
| Recall on medium | 0.377 |
| Truly-high cases reaching a human | 339 / 355 |
| Share of cases sent to human review | 29.7% |
| Share auto-replied | 70.3% (96.3% truly low) |

Hybrid beats classical-alone only marginally (+0.003 macro-F1, within noise).

## Input schema
`review_text` (1-5000 chars), `age` (10-120), `positive_feedback_count` (>= 0), `division_name`, `department_name`, `class_name`; optional `customer_id`. Unknown fields (`image`, `recommended_ind`) are ignored.

## Output
`model_outputs` (probabilities, risk_level, recommended_action, human_review_required, review_reasons, display scores), `llm_explanation` (summary, recommended_action, reason, human_review_required, risk_notes), `_meta` (`explanation_source`: `llm` or `fallback`).

## Known limitations
- Proxy label (above); the biggest limitation.
- Medium class is weak (F1 0.35).
- About half of "high" flags are false alarms by design (recall-first).
- Reviews differ from support tickets in style and length.
- One random split, no confidence intervals; no cross-validation.
- LLM output quality is not yet evaluated against the live API (`evaluate_llm.py` needs a key).
- No drift monitoring; dependency versions are pinned and the pickled/Keras artifacts must be retrained if they change.
- Fairness: only a few demographic-adjacent features (age); no subgroup analysis has been done.

## Monitoring suggestions
Track class mix, `auto_reply`/review shares, mean top-class confidence, share of `fallback` explanations, and, once real escalation outcomes exist, recalibrate and re-validate against them.
