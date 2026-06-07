# Decision Policy

This document defines how the hybrid system's numeric output (`final_score`, `risk_level`) is turned into a business action, when a human must review a case, and what the LLM explanation layer is, and is not, allowed to do. It documents the rules already encoded in [`hybrid_inference.py`](hybrid_inference.py) (`combine_scores`, `map_risk`) and [`llm_explainer.py`](llm_explainer.py) (`DISAGREEMENT_THRESHOLD`, `_enforce_policy`), so that the code and this document stay in sync. If they ever diverge, the code's deterministic checks are the source of truth; this document is the human-readable explanation of those checks, not a separate authority.

## 1. How the final score is produced

```
final_score = 0.4 * classical_score + 0.3 * cnn_score + 0.3 * rnn_score
```

Each component score is a value between 0 and 1 representing "expected escalation risk," derived from that model's class probabilities through the ordinal convention `0.0 * P(low) + 0.5 * P(medium) + 1.0 * P(high)` (see `risk_common.proba_to_score`, and for the CNN, `train_cnn_model.cnn_score_from_proba`). The weights are fixed by design: the classical model gets the largest share because it is the best-performing and most fully-grounded of the three components (see [`evaluation_report.md`](evaluation_report.md), Q1, for the evidence behind that ranking).

## 2. From risk thresholds to a default action

| `final_score` range | `risk_level` | Default business action |
|---|---|---|
| `score >= 0.75` | **high** | `escalate_immediately`: route to a senior support agent or escalation queue without delay |
| `0.45 <= score < 0.75` | **medium** | `manual_review`: route to a human reviewer before any reply is sent |
| `score < 0.45` | **low** | `auto_reply`: the system may send a templated acknowledgement or standard response automatically |

This is implemented in `hybrid_inference.map_risk`; the `ACTION_BY_RISK` mapping in [`llm_explainer.py`](llm_explainer.py) mirrors this table for the LLM's fallback path.

## 3. When a human must review a case

A case is forced into human review, meaning `human_review_required` is set to `true`, whenever any of the following is true. These checks are deterministic and are re-applied in code (`llm_explainer._enforce_policy`) regardless of what the LLM itself outputs:

1. **`risk_level == "high"`** (in other words, `final_score >= 0.75`). High-risk cases are never auto-actioned: the cost of a missed escalation, churn and reputational damage (see [`README.md`](README.md), Problem framing, "false negative impact"), is too high to leave to full automation.
2. **The component scores disagree strongly**, meaning the spread `max(classical_score, cnn_score, rnn_score) - min(...)` exceeds `DISAGREEMENT_THRESHOLD = 0.35`. Strong disagreement means the models are telling fundamentally different stories about the same complaint (one reads it as low risk, another as high risk); an automated system should not pick a winner silently. A person should look at the underlying text and decide.

In addition, the following should route to manual review even though they are not hard-coded as boolean checks. They are documented here so that reviewers and future engineers understand the intent (see also [`README.md`](README.md)'s human-review-trigger list):
- `risk_level == "medium"`. Anything that is not a confidently "low" case gets a look before an automated reply goes out, consistent with the action table above.
- The LLM itself flags `human_review_required = true` based on the complaint text, even when the deterministic checks above did not fire (see Section 5 below; this has been observed in practice and is a feature, not a bug).
- The complaint text is too short or ambiguous for the text models to have produced a confident probability distribution (for example, all class probabilities close to uniform), or required input fields are missing or malformed.
- An attached image is poor quality or unreadable, so `cnn_score` reflects the simulated fallback rather than a real visual assessment (see `_meta.cnn_mode` in `predict_hybrid`'s output, which is either `"simulated"` or `"real"`).

## 4. Why the cost of false positives and false negatives is treated differently

- **False positive** (the system flags a low-risk complaint as medium or high risk): wastes reviewer time and may cause the business to over-compensate a customer who was not actually at risk of escalating. Annoying and costly at scale, but recoverable.
- **False negative** (the system misses a genuinely high-risk complaint): the customer churns, and possibly escalates publicly through social media, chargebacks, or regulator complaints, causing reputational and revenue damage that is much harder to undo. This is the costlier failure mode by a wide margin.

This asymmetry is exactly why the thresholds are deliberately conservative. The "medium" band starts at 0.45, just below the midpoint, rather than at 0.5, and the "high" band only requires 0.75 rather than, say, 0.9, so that borderline cases are pulled toward review and escalation rather than toward silent auto-reply. It is also why Section 3 forces review on disagreement even when the combined score looks safely "low": see [`evaluation_report.md`](evaluation_report.md), Q3, for a concrete example (C001) where the text models confidently said "low" but the simulated CNN term disagreed enough to trigger review anyway.

## 5. What the LLM is allowed, and not allowed, to do

The LLM explanation layer (`llm_explainer.explain_with_gemini`) sits downstream of all scoring decisions, and both the prompt design and the surrounding code enforce a clear boundary around what it can and cannot influence.

**It is allowed to:**
- Summarize the complaint and the model outputs in plain business language (`summary`).
- Recommend a course of action in natural language (`recommended_action`).
- Explain why, in terms of the actual scores and complaint text it was given (`reason`, `risk_notes`).
- Independently set `human_review_required = true` even when the deterministic checks in Section 3 would not have forced it; in other words, the LLM is free to be more cautious than the floor requires. This was observed in practice for two of the five sample cases; see [`evaluation_report.md`](evaluation_report.md), Q5.

**It is not allowed to, and is structurally prevented from:**
- **Changing `final_score` or `risk_level`.** Those are computed in [`hybrid_inference.py`](hybrid_inference.py) before the LLM is ever called, and are passed to it as read-only context (`model_outputs`).
- **Inventing facts.** The prompt (`_build_prompt`) explicitly instructs it not to, and [`evaluation_report.md`](evaluation_report.md), Q4, confirms that no fabrication was observed across the sample runs: every detail it cited traces back to either the input text or the scores it was given.
- **Turning `human_review_required` to `false`** when policy requires it to be `true`. `_enforce_policy` runs on every output path, the real API response, a malformed or incomplete API response, and the local fallback used when the API call fails, and force-sets the flag to `true` whenever Section 3's conditions are met, overwriting whatever the LLM itself produced. A safety-critical flag must never depend solely on an LLM following instructions correctly.
- **Taking down the system if it is unavailable.** If the LLM call fails outright, returns invalid JSON, or omits a required field (`REQUIRED_FIELDS`), the system does not crash or silently drop the explanation. It falls back to `_fallback_explanation`, which is itself policy-correct (it derives `recommended_action` from `ACTION_BY_RISK` and still runs `_enforce_policy`). The rest of the hybrid pipeline, meaning the three trained models' scores, keeps working regardless of whether the LLM is available.

## 6. How the LLM's API key is handled

`GEMINI_API_KEY` is read only from the environment, optionally by way of a local, git-ignored `.env` file loaded with `python-dotenv`. It is never hard-coded in source and never logged. See [`.env.example`](.env.example) and the Setup section of [`README.md`](README.md) for how to configure it.
