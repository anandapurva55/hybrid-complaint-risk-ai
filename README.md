# Hybrid AI System: Classical ML + CNN + RNN + LLM

A complete, end-to-end AI system that predicts how likely a customer complaint is to escalate (low, medium, or high risk) and explains that prediction in plain business language. It works by combining four different kinds of models, a classical machine learning baseline, a convolutional neural network, a recurrent neural network, and a large language model, into a single decision pipeline, then wrapping that pipeline in a production-style API, a container, and a real cloud deployment.

This repository contains everything needed to understand, run, retrain, evaluate, and deploy the system: the trained models themselves, the scripts that produced them, the inference and explanation code, worked examples with real input and output, and detailed documentation covering the evaluation results, the decision-making policy, the system's strengths and limitations, and a real cloud deployment that was built, tested live, and torn down again as part of this project.

## Table of contents

1. [Problem framing](#problem-framing)
2. [System architecture](#system-architecture)
3. [Results at a glance](#results-at-a-glance)
4. [How the system makes decisions](#how-the-system-makes-decisions)
5. [About the model](#about-the-model)
6. [Project structure](#project-structure)
7. [Setup](#setup)
8. [How to run](#how-to-run)
9. [Deployment](#deployment)

## Problem framing

**Business scenario**

An e-commerce company receives customer complaints that combine customer profile data, free-text complaint or review content, and, in the full scenario, product images or screenshots. Support teams need an automated first pass that estimates how likely each complaint is to escalate, so that routine complaints can be handled automatically and serious ones reach a human quickly.

**Prediction target**

Complaint escalation risk level: low, medium, or high.

In this project, the target is derived from the Women's E-Commerce Clothing Reviews dataset, used here as a stand-in for the complaint scenario (see [data_exploration.ipynb](data_exploration.ipynb) for the full rationale). A review's star rating is mapped to risk as follows: 1 to 2 stars becomes high risk, 3 stars becomes medium risk, and 4 to 5 stars becomes low risk, treating a strongly negative customer experience as a proxy for "likely to escalate."

**Who uses the output**

The customer support and complaints triage team, through whatever queueing or ticketing tool routes incoming complaints to an automated reply, a manual review queue, or an escalation path.

**Business action**

- **Low risk**: send an automated reply (for example, an acknowledgement plus standard help links)
- **Medium risk**: route to manual support review
- **High risk**: escalate immediately to a senior agent or specialist queue

**False positive impact** (a low-risk complaint flagged as higher risk)

Reviewer time is wasted on a routine complaint, the support queue gets noisier, and the business may over-compensate (for example, with unnecessary refunds or expedited handling) for a customer who was never actually at risk of escalating.

**False negative impact** (a high-risk complaint flagged as low or medium risk)

A genuinely upset customer receives an automated reply, or sits in a routine queue, instead of being escalated. This is the most costly failure mode: it can directly cause customer churn, public complaints, or reputational damage, and is far harder to recover from after the fact.

**Human review is required when**

- the predicted risk level is medium or high
- the classical ML, CNN, and RNN component scores disagree strongly with one another
- the LLM explanation layer itself sets `human_review_required` to true
- the complaint text is short, ambiguous, or difficult to classify with confidence
- in the full multimodal scenario, the attached product image is missing or of poor quality

## System architecture

Here is how a single complaint moves through the system, from raw input to final, explained decision:

1. An incoming complaint (`input_data`) carries the review text, customer profile fields such as age and a recommendation flag, product category fields, and an optional product image.
2. Three independently trained models each look at this complaint from a different angle: the **classical model** combines the review text (through TF-IDF) with the structured profile fields in a logistic regression classifier; the **CNN** assesses an attached product image, falling back to a simulated estimate when no image is attached (see [About the model](#about-the-model) for why); and the **RNN/LSTM** reads the review text alone, as a sequence, through an embedding and recurrent layer.
3. Each model's three-class probability output is converted into a single risk score between 0 and 1. The three scores are then combined deterministically into one `final_score` using fixed weights, and that score is mapped to a `risk_level` of low, medium, or high using fixed thresholds (see [How the system makes decisions](#how-the-system-makes-decisions)).
4. The combined result, together with the original complaint, is handed to the **LLM explanation layer** (Google's Gemini model), which produces a plain-language summary, a recommended action, the reasoning behind it, a `human_review_required` flag, and supporting notes, all without ever being able to change the score or the risk level itself.

```
                         input_data: customer_id, review_text, age,
                         positive_feedback_count, recommended_ind,
                         division/department/class_name, image_available, image
                                            │
                ┌───────────────────────────┼───────────────────────────┐
                ▼                           ▼                           ▼
      get_classical_score          get_cnn_score              get_rnn_score
      (TF-IDF on the text,         (a real CNN when an        (an LSTM reads
       plus the structured         image is attached,         the review text
       profile fields, through     otherwise a simulated      as a sequence,
       logistic regression,        proxy score; see           producing a
       producing a 3-class         _meta.cnn_mode in the      3-class score)
       score)                      output)
                │                           │                           │
                └─────────────┬─────────────┴─────────────┬─────────────┘
                              ▼
           final_score = (0.4 × classical_score) + (0.3 × cnn_score) + (0.3 × rnn_score)
                              │
                              ▼
              risk_level: high (0.75 or above), medium (0.45 up to 0.75), or low (below 0.45)
                              │
                              ▼
               explain_with_gemini(input_data, model_outputs)
                 returns: summary, recommended_action, reason,
                          human_review_required, risk_notes
                              │
                              ▼
        predict_hybrid() returns: model_outputs, llm_explanation, and _meta
```

The three trained components each look at the complaint from a different angle: the classical model combines the structured profile fields with the review text, the CNN looks at the attached product image, and the RNN reads the free-text narrative on its own. Their scores are combined deterministically by `predict_hybrid`, using the fixed weights and thresholds described in [How the system makes decisions](#how-the-system-makes-decisions). The LLM never overrides that decision; its role is to explain the already-computed `risk_level` in plain business language and to flag cases that need a human look. [Results at a glance](#results-at-a-glance) walks through five real sample runs showing how often the components agree, how large their disagreements get, and where the LLM's own judgment was more cautious than the deterministic policy floor: the kind of evidence worth checking before trusting a system like this with real complaint traffic.

**How this is hosted in production**

The diagram above shows the AI decision pipeline itself. That entire pipeline, all four components plus the scoring and policy logic around them, runs inside a single container, served over HTTP by the FastAPI wrapper in [app.py](app.py). [Deployment](#deployment) describes how that container was built, deployed to AWS App Runner, and tested live in the cloud; the box labelled "container" in that section's architecture diagram is exactly the pipeline diagrammed here, simply running in the cloud rather than on a laptop.

## Results at a glance

Three of the four components are trained, evaluated classifiers. The fourth, the LLM, is an explanation layer with no comparable accuracy metric of its own; its quality is judged instead by whether it stays grounded in the facts it is given and respects the rules it is bound by (see below). Here is how the three classifiers performed on held-out test data. The full analysis, including exactly why each number looks the way it does, is in [evaluation_report.md](evaluation_report.md).

| Component | Task | Test accuracy | Test macro-F1 |
|---|---|---|---|
| Classical ML (TF-IDF and tabular features, logistic regression) | Risk-level prediction | 0.845 | 0.666 |
| RNN / LSTM (review text only) | Risk-level prediction | 0.741 | 0.542 |
| CNN (Fashion-MNIST image classification) | A proxy task, not risk-level prediction; see note below | 0.832 | 0.818 |

A few things worth knowing before reading these numbers at face value:

- **The classical baseline is the stronger of the two text-based classifiers** on the actual prediction task, beating the RNN by a meaningful margin (twelve points of macro-F1). It benefits from directly surfacing strong lexical cues such as "broken," "torn," or "no response," and from having access to structured profile fields that the RNN simply does not see.
- **The CNN's numbers measure a different task entirely.** No public dataset pairs customer complaint photos with escalation-risk labels, and the chosen text dataset has no images at all, so the CNN was trained on a genuinely real task instead, Fashion-MNIST clothing image classification, to demonstrate the architecture end to end with real training and real held-out accuracy. Its strong 0.832 accuracy and 0.818 macro-F1 say that the CNN learned its proxy task well; they say nothing about how well it would predict escalation risk from an actual complaint photo. This is a deliberate, clearly documented choice; see [About the model](#about-the-model) for what a production version would need instead.
- **Five worked examples** ([sample_inputs.json](sample_inputs.json) and [sample_outputs.json](sample_outputs.json)) show the full pipeline in action end to end. Across them, the spread between the highest and lowest component score ranged from 0.309 to 0.607; two of the five crossed the disagreement threshold and triggered mandatory human review automatically; and in two of the five cases, the LLM voluntarily flagged the case for human review even though the deterministic rules alone would not have required it, a sign that its judgment errs toward caution rather than away from it.
- **No fabricated facts were observed** in any of the LLM's explanations across the five runs: every detail it cited traced back to either the actual review text or the actual scores it was given.

The full report walks through all of this in depth, including a breakdown of exactly when and why the components disagreed, and a concrete list of what a production version of this system should improve first.

## How the system makes decisions

Once the three trained models have each produced a score between 0 and 1, the system combines them with fixed weights into a single number:

```
final_score = (0.4 × classical_score) + (0.3 × cnn_score) + (0.3 × rnn_score)
```

The classical model receives the largest share of the weight because it is the best-performing and most fully-grounded of the three; it is the only one with access to both the review text and the structured profile fields. That combined score is then mapped to a risk level and a default business action using fixed thresholds:

| `final_score` | `risk_level` | Default action |
|---|---|---|
| 0.75 or higher | High | Escalate immediately to a senior agent or specialist queue |
| 0.45 up to 0.75 | Medium | Route to a human reviewer before any reply is sent |
| Below 0.45 | Low | The system may send an automated, templated reply |

A case is **always** sent to a human reviewer, regardless of what the LLM itself says, whenever either of the following is true:

- the predicted risk level is high, because the cost of missing a genuine escalation is too high to leave to full automation, or
- the three component scores disagree by more than 0.35, because that level of disagreement means the models are telling fundamentally different stories about the same complaint, and an automated system should not silently pick a winner.

These two checks are deterministic, enforced directly in code, and cannot be overridden by the LLM. Beyond them, several softer signals, such as a "medium" risk level, the LLM's own independent judgment, ambiguous complaint text, or a poor-quality attached image, should also route a case to a human reviewer, even though they are not hard-coded boolean checks.

This asymmetry is intentional. A false positive (flagging a low-risk complaint as higher risk) wastes reviewer time and is merely annoying at scale. A false negative (missing a genuinely high-risk complaint) can cause real customer churn and reputational damage, and is far harder to undo. So the thresholds are deliberately conservative, pulling borderline cases toward review and escalation rather than toward silent automation.

The full policy document, [decision_policy.md](decision_policy.md), lays out the complete set of human-review trigger rules, the cost reasoning behind the threshold choices, and exactly what the LLM explanation layer is, and is not, permitted to do. In short: it can explain, summarize, and recommend, and it is free to be more cautious than the floor requires, but it can never change a score, invent a fact, or turn a required human-review flag off.

## About the model

**What this system is for**

Given a customer complaint (review text, customer profile fields, and optionally a product photo), the system estimates how likely it is to escalate, recommends a business action, and produces a plain-language explanation, so that a support team can triage incoming complaints faster and route the highest-risk ones to a human before they get worse.

**What it is not for**

Fully autonomous customer responses without human oversight on anything above "low" risk; use on languages other than English (the dataset, the vectorizers, and the prompts are all English-only); or use as the sole basis for refund or compensation decisions. The models and the LLM inform a recommendation; they do not make binding decisions on their own.

**The four components, in brief**

| Component | Role | What it is |
|---|---|---|
| Classical ML | Primary text-and-profile classifier | TF-IDF on the review text plus structured profile features, through a logistic regression classifier |
| CNN | Visual-evidence signal from an attached photo | A small convolutional network (two convolution-and-pooling blocks plus dense layers), trained on a real image-classification task as a proxy; see "Known limitations" below |
| RNN / LSTM | Secondary, text-only classifier | An embedding layer followed by an LSTM, reading the review text as a sequence |
| LLM | Explanation and recommendation layer | Google's Gemini model, prompted with the three component scores and the original complaint, returning a structured explanation in JSON |

**Known limitations, in brief**

The full detail behind each of these, including how it shows up concretely in the sample runs and how it could be addressed, is in [evaluation_report.md](evaluation_report.md) and [mini_model_card.md](mini_model_card.md).

1. **The CNN's contribution is currently simulated**, not learned from risk-labeled photos, simply because no such dataset exists publicly and the chosen text dataset has no images at all. When no image is attached, which is the common case, the system falls back to a constant estimate derived only from whether an image is present. In testing, this turned out to be the single largest source of artificial disagreement between the components. A production version would need to either train the CNN on real, outcome-labeled complaint photos (the architecture is ready and documented for exactly this), or remove its contribution entirely for complaints with no attached image.
2. **The RNN trails the classical baseline** by a meaningful margin, most likely because of the smaller training subset it was given for compute-budget reasons, and because, unlike the classical model, it has no access to the structured profile fields.
3. **English-only, single-domain text.** Everything text-related, the vocabulary, the vectorizer, and the LLM prompt, is tuned and tested on English clothing reviews; performance on other languages or product domains is untested and likely to be considerably worse.
4. **Class imbalance.** High-risk and medium-risk complaints are a minority of the dataset (roughly ten to twelve percent each), so predictions in those minority classes, which are exactly the cases where being wrong matters most, are noticeably less reliable than predictions of "low" risk.
5. **The LLM layer depends on an external API.** If it is unreachable, or returns something unusable, the system falls back to a templated local explanation rather than crashing, so the rest of the pipeline (the actual risk scores) keeps working regardless of whether the LLM is available. The LLM's wording also varies slightly from run to run, since generative models are inherently non-deterministic, even though the underlying scores never change.

For the complete picture, including the full input and output schema, monitoring recommendations for a production deployment, and a summary of the human-review rules, see [mini_model_card.md](mini_model_card.md).

## Project structure

```
hybrid-complaint-risk-ai/
├── README.md                                  you are here: overview, architecture, results, setup, and how to run
├── data_exploration.ipynb                     explores the data, derives the risk_level target, produces the train/val/test splits
├── Womens Clothing E-Commerce Reviews.csv     the raw source dataset
│
├── risk_common.py                             shared constants and the score-conversion helper every component relies on
├── train_classical_model.py                   trains the TF-IDF + logistic regression baseline
├── train_cnn_model.py                         trains the CNN on its proxy image-classification task
├── train_rnn_model.py                         trains the RNN/LSTM text classifier
├── classical_model.pkl                        the trained classical model (already included, ready to use)
├── cnn_model.keras                            the trained CNN (already included, ready to use)
├── rnn_model.keras                            the trained RNN/LSTM (already included, ready to use)
│
├── hybrid_inference.py                        combines the three trained models into one weighted score and risk level
├── llm_explainer.py                           turns that decision into a plain-language explanation, through Gemini
│
├── sample_inputs.json                         five realistic example complaints
├── sample_outputs.json                        the system's actual output for each of those five examples
│
├── evaluation_report.md                       full evaluation results and a detailed written analysis
├── decision_policy.md                         the rules that turn scores into business decisions
├── mini_model_card.md                         a complete model card: intended use, limitations, schema, monitoring
│
├── app.py                                     FastAPI wrapper exposing the pipeline as an HTTP API
├── Dockerfile                                 container definition for the API
├── .dockerignore                              files excluded from the container build context
├── aws_deployment_notes.md                    a real cloud deployment: built, tested live, and torn down again
│
├── requirements.txt                           Python dependencies
├── .env.example                               template for the one secret the system needs (an LLM API key)
└── .gitignore                                 version-control housekeeping
```

A bit more detail on each group, for anyone who wants the fuller picture:

**Getting started.** [`data_exploration.ipynb`](data_exploration.ipynb) is the natural entry point into the data itself. It explores the [Women's E-Commerce Clothing Reviews dataset](Womens%20Clothing%20E-Commerce%20Reviews.csv), explains exactly how the `risk_level` target is derived from it (see [Problem framing](#problem-framing)), and produces the train, validation, and test splits that every training script in this repository consumes.

**Training the three components.** [`risk_common.py`](risk_common.py) holds the small set of constants and the score-conversion helper (`proba_to_score`) that every trained component relies on, so that training and inference code can never drift out of sync with each other. [`train_classical_model.py`](train_classical_model.py), [`train_cnn_model.py`](train_cnn_model.py), and [`train_rnn_model.py`](train_rnn_model.py) train the three components and write [`classical_model.pkl`](classical_model.pkl), [`cnn_model.keras`](cnn_model.keras), and [`rnn_model.keras`](rnn_model.keras) respectively. All three trained models are already included in this repository, so nothing needs to be retrained just to use the system; see [How to run](#how-to-run).

**The hybrid pipeline.** [`hybrid_inference.py`](hybrid_inference.py) is the heart of the system. It loads the three trained models, combines their scores into a single decision, and calls [`llm_explainer.py`](llm_explainer.py), which turns that decision into a plain-language, policy-checked explanation through Gemini.

**Worked examples.** [`sample_inputs.json`](sample_inputs.json) holds five realistic example complaints, and [`sample_outputs.json`](sample_outputs.json) holds the system's actual output for each one. These five examples are used as concrete evidence throughout [Results at a glance](#results-at-a-glance) and the deeper documentation.

**Documentation.** [`evaluation_report.md`](evaluation_report.md), [`decision_policy.md`](decision_policy.md), and [`mini_model_card.md`](mini_model_card.md) are the three deep-dive documents summarized in [Results at a glance](#results-at-a-glance), [How the system makes decisions](#how-the-system-makes-decisions), and [About the model](#about-the-model) respectively.

**Serving and deployment.** [`app.py`](app.py) wraps the pipeline in an HTTP API. [`Dockerfile`](Dockerfile) and [`.dockerignore`](.dockerignore) define how it is containerised. [`aws_deployment_notes.md`](aws_deployment_notes.md) documents a real cloud deployment that was built, tested live, and fully torn down again, summarized in [Deployment](#deployment).

**Configuration.** [`requirements.txt`](requirements.txt) lists the Python dependencies, [`.env.example`](.env.example) is a template for the one secret the system needs (see [Setup](#setup)), and [`.gitignore`](.gitignore) keeps generated files and secrets out of version control.

## Setup

**Prerequisites:** Python 3.11 (the version range TensorFlow officially supports), and optionally Docker Desktop plus an AWS account if you want to build or deploy the container yourself.

```bash
# 1. Clone the repository, then create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate

# 2. Install the dependencies
pip install -r requirements.txt

# 3. Configure your Gemini API key, used by the LLM explanation layer
cp .env.example .env
```

Then open `.env` and replace the placeholder with your own key from Google AI Studio. Get it directly from Google, and never share it or paste it anywhere else, including chat windows, commits, or screenshots. `.env` is already listed in both [`.gitignore`](.gitignore) and [`.dockerignore`](.dockerignore) for exactly this reason: it should never leave your machine.

Both [`llm_explainer.py`](llm_explainer.py) and [`app.py`](app.py) load the key through `python-dotenv`'s `load_dotenv()`, which reads `.env` into the process environment without overriding any `GEMINI_API_KEY` you may already have exported in your shell.

## How to run

Each piece below can be run independently. They all share the trained model artifacts that are already included in this repository ([`classical_model.pkl`](classical_model.pkl), [`cnn_model.keras`](cnn_model.keras), [`rnn_model.keras`](rnn_model.keras)), so you do not need to train anything before trying the system.

**1. Explore the data and see how the risk target is derived**

```bash
jupyter notebook data_exploration.ipynb
```

**2. Retrain a component from scratch (optional)**

The trained models are already included, so this step is entirely optional, useful mainly if you want to reproduce the training process yourself or experiment with changes.

```bash
python train_classical_model.py   # writes classical_model.pkl
python train_cnn_model.py         # writes cnn_model.keras (trained on its proxy task)
python train_rnn_model.py         # writes rnn_model.keras
```

**3. Run the full hybrid pipeline on the bundled examples**

```bash
python hybrid_inference.py
```

This loads the five sample complaints from [`sample_inputs.json`](sample_inputs.json), runs each one through the complete pipeline (all three trained models, the score combination, the threshold mapping, and the LLM explanation layer), and prints the results, the same results already saved in [`sample_outputs.json`](sample_outputs.json).

**4. Try the LLM explanation layer directly**

```bash
python llm_explainer.py
```

**5. Serve predictions over HTTP**

```bash
uvicorn app:app --reload
```

Then, for example:

```bash
curl http://127.0.0.1:8000/

curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d @sample_inputs.json
```

(Note: send a single complaint object from the array as the request body, not the whole array.)

**6. Build and run the container**

```bash
docker build -t hybrid-risk-api .
docker run --rm -p 8000:8000 --env-file .env hybrid-risk-api
curl http://127.0.0.1:8000/
```

See the comments at the top of [Dockerfile](Dockerfile) for the reasoning behind how the API key is supplied: only at run time, and never baked into the image itself.

## Deployment

This system was deployed to the cloud, tested live with real traffic, and then fully torn down again, as a complete demonstration of how it would run in production. The deployment used **AWS App Runner**, **Amazon ECR**, **AWS Secrets Manager**, and two purpose-built **IAM roles**, all wired together so that the running container never needs the Gemini API key baked into it; the key is fetched securely from Secrets Manager at start-up instead.

A few highlights from that deployment. The full walkthrough, including every command that was run, the live request and response, the latency measurements, and the cleanup confirmation, is in [aws_deployment_notes.md](aws_deployment_notes.md):

- The live service answered health checks in roughly 1.3 seconds, and full predictions, including a real call to the Gemini API, in roughly five to nine seconds, with the Gemini round trip being the dominant cost. The deterministic parts of the pipeline, the three trained models and the score combination, ran in well under a second.
- A real request and response pair, captured directly from the live deployment, is included in the notes, alongside everything needed to reproduce the setup from scratch.
- After testing, every single resource that had been created (the App Runner service, the ECR repository, the Secrets Manager secret, and both IAM roles) was deleted, and that deletion was independently verified, so nothing from this deployment continues to run or to incur any cost.

As mentioned in [System architecture](#system-architecture), the container at the centre of this deployment runs exactly the pipeline diagrammed there. The cloud infrastructure is simply how that pipeline gets exposed to the world.
