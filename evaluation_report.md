# Evaluation Report

## 1. Component metrics

All three trained classifiers that operate on the `risk_level` target (classical ML and RNN; the CNN is discussed separately below) were evaluated on the same stratified 70/15/15 train, validation, and test split produced in [`data_exploration.ipynb`](data_exploration.ipynb) (`RANDOM_STATE = 42`), so their numbers are directly comparable. The CNN is evaluated on its own task; see the note below the table for why.

| Component | Task / target | Split | Accuracy | Macro-F1 | Notes |
|---|---|---|---|---|---|
| Classical ML (TF-IDF and tabular features through logistic regression) | `risk_level` (low, medium, high) | Validation | 0.842 | 0.668 | [`train_classical_model.py`](train_classical_model.py) |
| | | **Test** | **0.845** | **0.666** | |
| RNN / LSTM (review text only) | `risk_level` (low, medium, high) | Validation | 0.744 | 0.563 | trained on a stratified 6,000-row subset of the 15,848-row train split, for compute-budget reasons; see `TRAIN_SUBSET` in [`train_rnn_model.py`](train_rnn_model.py) |
| | | **Test** | **0.741** | **0.542** | |
| CNN (two Conv2D blocks plus dense layers, on Fashion-MNIST) | 10-class clothing image classification (a proxy task, not `risk_level`) | Test | 0.832 | 0.818 | [`train_cnn_model.py`](train_cnn_model.py); see note below |

**Why the CNN row is not directly comparable to the other two:** no publicly available dataset pairs customer-complaint photos with escalation-risk labels, and the chosen primary dataset, the Women's E-Commerce Clothing Reviews dataset, has no images at all. So the CNN was instead trained on a genuinely real task, Fashion-MNIST 10-class image classification, to demonstrate the architecture end to end with real training and real held-out accuracy and F1. Its 10-class output is then mapped onto a `[0, 1]` risk contribution through an illustrative `CATEGORY_RISK_WEIGHT` table, documented in [`train_cnn_model.py`](train_cnn_model.py). Its 0.832 accuracy and 0.818 macro-F1 say that the CNN learned its proxy task well; they say nothing about risk-prediction quality. This is intentional, and is documented at the exact point where the simulation happens; see Q6 below for how this would be replaced in a production system.

The **LLM** has no comparable accuracy or F1 metric: it is an explanation layer, not a classifier, and it is explicitly forbidden from changing scores or the `risk_level`. Its quality is assessed qualitatively in Q4 and Q5 below, by checking structural validity (does it return all required JSON fields), factual grounding (does it invent details), and policy alignment (does its `human_review_required` choice respect the deterministic rules).

## 2. Written analysis

### Q1: Which model performed best?

On the actual `risk_level` prediction task, the one both text-capable models were evaluated on, using identical splits, the **classical TF-IDF and logistic regression baseline performed best**: test accuracy of 0.845 versus 0.741 for the RNN, and macro-F1 of 0.666 versus 0.542, a 12-point macro-F1 gap on an imbalanced three-class problem.

Likely reasons:
- TF-IDF n-grams directly surface strong lexical cues for risk, words like "broken," "torn," "no response," and "disappointed," that a small LSTM trained on only 6,000 examples has less opportunity to learn reliably.
- The classical pipeline also folds in the tabular features (age, positive feedback count, recommended indicator, and division, department, and class name) through a `ColumnTransformer`, giving it signal that the text-only RNN simply does not have access to.
- LSTMs typically need much larger corpora to beat a strong linear baseline on short-text classification, and the compute-budget subset (6,000 of 15,848 train rows) limited how well it could generalize; see Q6 for how this gap could likely be closed.

The CNN's 0.832 and 0.818 figures look superficially like the strongest result, but as explained in the metrics note above, they measure a different task entirely (clothing image classification) and are not a fair comparison; they only confirm that the architecture trains and generalizes correctly on real image data.

### Q2: Did the hybrid combination improve on the individual components?

A clean accuracy lift for the combined `final_score` cannot be computed the way it can for the individual classifiers, because no ground-truth risk label exists for images in this dataset, exactly the gap that motivated simulating the CNN score in the first place. A fair hybrid-versus-component comparison would require a labeled set of complaints with both real text outcomes and real outcome-labeled photos, which does not exist here; see Q6.

What can be observed, from the five illustrative runs in [`sample_outputs.json`](sample_outputs.json):
- For clear-cut cases, the hybrid still lands in the right band: the unambiguously positive review (C001) scored `final_score = 0.132`, mapping to "low," and the multi-issue escalation (C005, "wrong size... broken zipper... third order in a row") scored `final_score = 0.774`, mapping to "high." Both are intuitively correct outcomes.
- For mixed or ambiguous cases (C002, C003, C004), the hybrid produced "medium," borderline scores that pushed the system toward `human_review_required = true`, arguably the right, conservative behaviour for a business-critical escalation system, even if it is not literally a single-metric accuracy improvement.

Bottom line: with a simulated CNN term, the hybrid cannot yet demonstrate a clean accuracy improvement over the classical baseline alone, and is not designed to be judged that way at this stage. What it does add is, first, a structured slot for a real visual-evidence signal once labeled complaint photos exist, and second, the LLM layer that turns three raw numbers into an actionable, explained, policy-checked recommendation: value that a single-component accuracy score does not capture.

### Q3: When did the models disagree, and why?

Across the five samples, the spread between the highest and lowest component score (`classical_score`, `cnn_score`, `rnn_score`) ranged from **0.309** (C002) to **0.607** (C003). Two of the five (C001 at 0.395, and C003 at 0.607) crossed the `DISAGREEMENT_THRESHOLD` of 0.35 and triggered the deterministic `human_review_required` override inside `_enforce_policy`.

The clearest pattern: **the largest disagreements were driven by the simulated CNN term diverging from the text-derived scores**, not by genuine model uncertainty. C001 is the starkest example: an unambiguously positive review ("Love this sweater!... fits true to size"), where the classical model and the RNN both correctly scored it near zero (0.026 and 0.005), but the simulated CNN contributed 0.4 purely because `image_available = false` maps to a fixed constant (see the simulated fallback inside `get_cnn_score`). That 0.395 spread is an artifact of the simulation rather than an independent visual signal catching something the text models missed; an important, honestly-documented limitation discussed further in Q6.

Disagreement between the two text models themselves (classical versus RNN, evaluated on identical input) was smaller but still meaningful, and more textually grounded: in C003 (a torn seam, a chemical smell, and two weeks of unanswered support emails), the classical model scored 0.796, likely picking up strong negative n-grams plus a `recommended_ind` of 0 and a low positive feedback count, while the RNN scored only 0.189, consistent with its weaker overall test macro-F1 (0.542 versus 0.666): on this dataset, it is simply the less reliable of the two text models.

### Q4: Did the LLM invent facts?

No fabrication was observed in any of the five runs. Spot-checking each `summary` and `reason` against the corresponding `review_text`:
- C005's explanation cites "incorrect size," "broken zipper," and "third consecutive order," all present, nearly verbatim, in the input text.
- C003's explanation cites "torn seam," "chemical smell," and "two emails... over two weeks," again fully grounded in the input.
- None of the five explanations introduced details absent from the input: no invented order numbers, dates, refund amounts, support-policy claims, or customer history beyond what the review text actually states.

The LLM also consistently quoted the actual numeric scores it was given (for example, "classical_score: 0.933 vs. cnn_score: 0.600") rather than rounding or paraphrasing them into something misleading, consistent with the prompt's explicit instruction not to change the model scores or invent facts (see `_build_prompt` in [`llm_explainer.py`](llm_explainer.py)).

### Q5: Were the LLM's recommendations aligned with the decision policy?

Yes, and in two of the five cases the LLM was actually **more conservative** than the deterministic floor required:
- The policy enforcement (`_enforce_policy`) force-sets `human_review_required = true` only when `risk_level == "high"` (C005) or the score spread exceeds 0.35 (C001 at 0.395, C003 at 0.607); three of the five samples.
- In the remaining two samples (C002 with a spread of 0.309, and C004 with a spread of 0.338, both below the 0.35 threshold), the **LLM independently chose** `human_review_required = true` anyway, reasoning from the same score-spread evidence plus the substance of the complaint text.

So across all five cases the final flag came out `true`; in three of five it was policy-enforced regardless of what the LLM said, and in two of five the LLM voluntarily erred on the side of caution. No run produced an attempt to set the flag to `false` against policy, exactly the scenario `_enforce_policy` exists to catch deterministically: a safety-critical flag should never depend solely on an LLM following instructions correctly.

`recommended_action` text was also directionally consistent with the project's risk-to-action mapping (`ACTION_BY_RISK` in the fallback path: low maps to an automated reply, medium to manual review, high to immediate escalation): C005 (high) drew escalation-style language ("Prioritize immediate contact... investigate the recurring... problems"), C001 (low, clearly positive) drew "No immediate action required... process as positive feedback," and the medium-risk cases (C002, C003, C004) drew review and escalate-for-investigation language. The system asks the LLM for natural-language `recommended_action` text rather than a literal enum, so exact-string matching is not expected, but the direction of every recommendation matched the policy mapping.

### Q6: What would you improve in a production version of this system?

1. **Replace the simulated CNN signal with a real one.** As Q3 shows, the simulated mapping from `image_available` to a constant of either 0.4 or 0.6 was the single largest source of artificial disagreement. It does not track the complaint's actual content, and it can pull a confidently-low-risk text prediction toward "medium." A production system should either train the CNN on real complaint photos labeled by escalation outcome (the path for this is already documented in [`train_cnn_model.py`](train_cnn_model.py)'s docstring), or, until such data exists, drop the CNN term and renormalize the classical and RNN weights for complaints with no attached image, rather than injecting a constant guess that dilutes a more-informed text signal.
2. **Train the RNN on more data.** Its test macro-F1 of 0.542 trails the classical baseline's 0.666 by a wide margin despite identical splits. Training on the full 15,848-row split rather than the 6,000-row compute-budget subset, trying pretrained embeddings, or feeding it the same tabular features the classical pipeline already uses (through a small auxiliary input branch) would likely close much of that gap.
3. **Re-derive the disagreement threshold empirically.** A value of 0.35 is a reasonable starting heuristic, but with a larger labeled sample one could tune it, or replace the simple max-minus-min spread with something like pairwise variance, so that "strong disagreement" reliably separates genuinely uncertain cases from noise, especially once the CNN term stops being a near-constant.
4. **Build a true multimodal evaluation set.** To honestly answer whether the hybrid improves on any single component (Q2), a set of complaints labeled with both real text outcomes and real outcome-labeled images would be needed. That does not exist for this dataset, and would be the highest-value data-collection investment for a production version of this system.
