# Mini Model Card: Hybrid Complaint Escalation Risk System

## System name

**Hybrid AI System: Classical ML + CNN + RNN + LLM**, a customer complaint escalation-risk predictor for e-commerce support teams.

## Intended use

Given an e-commerce customer complaint (review text, customer profile fields, and optionally a product photo), the system estimates the risk that this complaint will escalate (`low`, `medium`, or `high`), recommends a business action, and produces a human-readable explanation, so that a support team can triage incoming complaints faster and route the highest-risk ones to a human before they escalate further.

**Out of scope, and not intended for:** fully autonomous customer responses without human oversight on anything above "low" risk; use on languages other than English (the dataset and vectorizers are English-only); or use as the sole basis for refund or compensation decisions (see [`decision_policy.md`](decision_policy.md), Section 5: the LLM and the models inform a recommendation, they do not make binding decisions).

## Components

| Component | Role | Architecture | Trained on |
|---|---|---|---|
| **Classical ML** ([`train_classical_model.py`](train_classical_model.py)) | Primary text and tabular risk classifier | A `ColumnTransformer` (TF-IDF on review text, `StandardScaler` on numeric fields, `OneHotEncoder` on categorical fields) feeding a `LogisticRegression(class_weight="balanced")` | The Women's E-Commerce Clothing Reviews dataset, stratified train split (15,848 rows) |
| **CNN** ([`train_cnn_model.py`](train_cnn_model.py)) | Visual-evidence signal from an attached complaint photo | `Conv2D(16) -> MaxPool -> Conv2D(32) -> MaxPool -> Flatten -> Dense(64) -> Dropout -> Dense(10, softmax)` | A Fashion-MNIST subset (8,000 training images, 2,000 test images); see "Known limitations" below for why this is a proxy task |
| **RNN / LSTM** ([`train_rnn_model.py`](train_rnn_model.py)) | Secondary, text-only risk classifier | `TextVectorization -> Embedding(64, mask_zero) -> LSTM(64) -> Dense(32) -> Dropout -> Dense(3, softmax)` | The same dataset, a stratified 6,000-row subset of the train split, chosen for compute-budget reasons |
| **LLM** ([`llm_explainer.py`](llm_explainer.py), Gemini `gemini-2.5-flash`) | Explanation and recommended-action layer, downstream of all scoring | A hosted generative model accessed via `google-genai`, used in JSON mode and prompted with the three component scores plus the complaint text | Not applicable; it is used through an API rather than trained as part of this project |

## Inputs and outputs

**Input** (an `input_data` object; see the docstring in [`hybrid_inference.py`](hybrid_inference.py) for the complete schema and the reasoning behind its field names): `review_text`, `age`, `positive_feedback_count`, `recommended_ind`, `division_name`, `department_name`, `class_name`, `image_available`, and an optional `image` (a 28 by 28 grayscale array).

**Output** (the return value of `predict_hybrid`):
```json
{
  "model_outputs": {
    "classical_score": 0.0,
    "cnn_score": 0.0,
    "rnn_score": 0.0,
    "final_score": 0.0,
    "risk_level": "low | medium | high"
  },
  "llm_explanation": {
    "summary": "...",
    "recommended_action": "...",
    "reason": "...",
    "human_review_required": true,
    "risk_notes": "..."
  },
  "_meta": { "cnn_mode": "real | simulated" }
}
```
See [`sample_inputs.json`](sample_inputs.json) and [`sample_outputs.json`](sample_outputs.json) for five fully worked examples.

## Metrics (held-out test split, identical for the classical model and the RNN; full analysis in [`evaluation_report.md`](evaluation_report.md))

| Component | Accuracy | Macro-F1 |
|---|---|---|
| Classical ML | 0.845 | 0.666 |
| RNN / LSTM | 0.741 | 0.542 |
| CNN (Fashion-MNIST proxy task; not directly comparable) | 0.832 | 0.818 |

The class distribution is imbalanced (roughly 77 percent low risk, 12.5 percent medium risk, and 10.5 percent high risk). Both classifiers were trained with `class_weight="balanced"`, and are reported here with macro-F1, which weights all three classes equally, alongside accuracy, for that reason.

## Hybrid scoring and thresholds

```
final_score = 0.4 * classical_score + 0.3 * cnn_score + 0.3 * rnn_score
```

This is mapped to a `risk_level` as follows: a score of 0.75 or above becomes "high," a score from 0.45 up to 0.75 becomes "medium," and anything below that becomes "low." The full rationale, the human-review trigger rules, and exactly what the LLM is and is not allowed to do are documented in [`decision_policy.md`](decision_policy.md).

## Known limitations and failure modes

1. **The CNN's contribution is currently simulated**, not learned from risk-labeled data. No public dataset pairs complaint photos with escalation outcomes, and the chosen text dataset has no images at all, so `cnn_score` for the common "no image attached" case is a constant derived purely from the `image_available` flag (either 0.6 or 0.4) rather than a genuine visual assessment. As shown in [`evaluation_report.md`](evaluation_report.md) (Q3), this was the single largest source of artificial disagreement between components in the sample runs: an unambiguously positive review, scored near zero by both text models, still pulled in a simulated `cnn_score` of 0.4, nudging the blended score upward for no text-grounded reason. A real production system must replace this with a CNN trained on outcome-labeled complaint photos. The architecture for this is already documented and ready in [`train_cnn_model.py`](train_cnn_model.py); only the training data is missing.
2. **The RNN underperforms the classical baseline** (macro-F1 of 0.542 versus 0.666), most likely due to the compute-budget training subset (6,000 of 15,848 rows) and its lack of access to the tabular features that the classical pipeline uses. See [`evaluation_report.md`](evaluation_report.md), Q1 and Q6, for more detail and proposed fixes.
3. **English-only, single-domain text.** The TF-IDF vocabulary, the LSTM's vectorizer, and the LLM prompt are all tuned and tested on English clothing reviews. Performance on other languages, other domains such as electronics or groceries, or text that is much longer or shorter than the training distribution, is untested and likely to be considerably worse.
4. **Class imbalance.** "High" and "medium" complaints are a minority (roughly 10.5 percent and 12.5 percent respectively). Despite `class_weight="balanced"`, both classifiers' macro-F1 trails their accuracy, meaning predictions in the minority classes (medium and high risk) are noticeably less reliable than majority-class (low risk) predictions, exactly the cases where being wrong matters most (see [`decision_policy.md`](decision_policy.md), Section 4, on the cost of false negatives).
5. **LLM availability and non-determinism.** The explanation layer depends on an external API. If it is unreachable, returns malformed JSON, or omits a required field, the system falls back to a templated local explanation (`_fallback_explanation`), so the pipeline never crashes, though that fallback is necessarily less specific than a real LLM response. The LLM's prose also varies from run to run, since generative models are inherently non-deterministic, even though the underlying scores never change.

## Human-review rules, in summary (full detail in [`decision_policy.md`](decision_policy.md))

A case is always routed to human review when `risk_level == "high"`, or when the three component scores disagree by more than 0.35 (`max - min > DISAGREEMENT_THRESHOLD`). This is enforced deterministically inside `llm_explainer._enforce_policy`, regardless of what the LLM itself outputs. The LLM may additionally flag a case for review on its own judgment, and did so, beyond what the policy required, in two of the five sample runs (see [`evaluation_report.md`](evaluation_report.md), Q5), but it can never turn the flag off when policy requires it to be on.

## Deployment assumptions

- The system runs as a batch or online inference pipeline behind an API (see [`app.py`](app.py), the FastAPI wrapper), or as a notebook or script for offline scoring.
- All three trained models ([`classical_model.pkl`](classical_model.pkl), [`cnn_model.keras`](cnn_model.keras), [`rnn_model.keras`](rnn_model.keras)) are loaded once and reused as lazy singletons inside [`hybrid_inference.py`](hybrid_inference.py); there is no per-request retraining.
- `GEMINI_API_KEY` is supplied through an environment variable, optionally via a git-ignored `.env` file loaded with `python-dotenv`, and is never hard-coded; see [`decision_policy.md`](decision_policy.md), Section 6, and the Setup section of [`README.md`](README.md).
- The system assumes English-language clothing-retail complaints structurally similar to the Women's E-Commerce Clothing Reviews dataset; see "Known limitations" above for what happens outside that distribution.

## Monitoring recommendations for production

- Track the **distribution of `final_score` and `risk_level`** over time. A drift toward more "medium" or "high" classifications, or a collapse toward "low," can signal either a real shift in complaint severity or model and data drift worth investigating.
- Track **how often `human_review_required` fires**, broken down by which rule fired (the high-risk rule, the score-disagreement rule, or the LLM's own judgment). A rising disagreement-triggered rate may mean the components are drifting apart and need recalibration or retraining.
- Track the **LLM fallback rate** (how often `_fallback_explanation` is invoked) as an availability and reliability signal for the explanation layer.
- Periodically **spot-check LLM explanations against ground truth**, the same kind of "did it invent facts" check performed in Q4 of [`evaluation_report.md`](evaluation_report.md), especially after any prompt or model-version change (`GEMINI_MODEL`).
- Once real outcome labels become available (did this complaint actually escalate or not), compute **true accuracy and F1 for the hybrid `risk_level`** itself, something that could not be done here for lack of ground truth (see [`evaluation_report.md`](evaluation_report.md), Q2), and use the result to retune `WEIGHTS` and the risk thresholds.
