# Model Card: Hybrid Dissatisfaction-Risk Triage

## Summary
Classifies customer feedback into low / medium / high **dissatisfaction risk** and explains the result. Intended as a first-pass triage aid with human oversight, not an autonomous decision maker.

## IMPORTANT: what the label is
The training label is derived from the star rating (1-2 stars = high, 3 = medium, 4-5 = low) in the Women's E-Commerce Clothing Reviews dataset. The data contains **no complaints and no escalation outcomes**. The model therefore predicts *how unhappy a review is*, used as a proxy for escalation risk. The proxy is unvalidated: an unhappy customer need not escalate, and an escalating customer may write a polite review. Do not describe outputs as "probability of escalation".

## Components
| Component | Role | Notes |
|---|---|---|
| Classical ML | TF-IDF (1-2 grams) + age, positive-feedback count, division/department/class -> Logistic Regression | `Recommended IND` removed (leaks the label) |
| Fine-tuned DistilBERT | Text classifier, carries ~90% of the blend | int8-compressed weights (60 MB); LSTM baseline dropped (blend weight 0) |
| Decision layer | Blend calibrated probabilities, apply tuned thresholds, review triggers | `decision.py`, `decision_config.json` |
| LLM (Amazon Bedrock, Nova Lite) | Explanation only | Cannot change level, action or review flag; bad output is rejected and replaced by a fallback |
| LSTM, CNN | **Not used** | LSTM is a comparison baseline; the CNN is a Fashion-MNIST demo that cannot score complaint photos |

## Intended use
Prioritising and routing English-language product feedback for human reviewers; auto-acknowledging clearly positive messages.

## Out of scope
Fully automated customer-facing decisions on negative feedback; non-English text; other industries without re-validation; any use as a statistical estimate of churn or legal/regulatory escalation.

## Performance (test split, n = 3,397; see `evaluation_report.md`)
| | Value |
|---|---|
| Macro-F1 (hybrid, tuned) | 0.673 (95% CI 0.652-0.694) |
| Recall on high / precision on high | 0.710 / 0.583 |
| Recall / precision on medium | 0.507 / 0.395 |
| Truly-high cases reaching a human | 350 / 355 |
| Share of cases sent to human review | 29.3% |
| Share auto-replied | 70.7% (97.5% truly low) |

The transformer provides the gain; the blend with the classical model adds only +0.002 to +0.012 macro-F1.

## Input schema
`review_text` (1-5000 chars), `age` (10-120), `positive_feedback_count` (>= 0), `division_name`, `department_name`, `class_name`; optional `customer_id`. Unknown fields (`image`, `recommended_ind`) are ignored.

## Output
`model_outputs` (probabilities, risk_level, recommended_action, human_review_required, review_reasons, display scores), `llm_explanation` (summary, recommended_action, reason, human_review_required, risk_notes), `_meta` (`explanation_source`: `llm` or `fallback`).

## Known limitations
- Proxy label (above); the biggest limitation.
- Medium class is the weakest (F1 0.44).
- About 4 in 10 "high" flags are false alarms by design (recall-first).
- Reviews differ from support tickets in style and length.
- Bootstrap CIs and a 3-seed study of the transformer (macro-F1 0.666-0.682) are reported, but there is one random data split.
- Nova Lite explanations: 0 of 45 needed the local fallback in the live check; faithfulness beyond numbers/risk level is not human-graded and n is small.
- Drift monitoring is in-process only (`GET /stats`); dependency versions are pinned and the pickled/Keras artifacts must be retrained if they change.
- Fairness: subgroup metrics by age band and department are in the evaluation report (fairly even; Bottoms has lowest high recall, 0.60); no formal fairness test.

## Monitoring suggestions
`GET /stats` reports these over a rolling window and alerts on shifts versus the test-split baseline. Track class mix, `auto_reply`/review shares, mean top-class confidence, share of `fallback` explanations, and, once real escalation outcomes exist, recalibrate and re-validate against them.
