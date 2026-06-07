"""
LLM explanation layer.

Wraps a Gemini call that turns the hybrid model's numeric outputs into a
structured business explanation (summary, recommended action, reasoning,
human-review flag, risk notes), without ever changing the underlying
model scores or risk_level (see decision_policy.md for the full rules
this layer must respect).

--------------------------------------------------------------------------
Security rule: never hard-code API keys
--------------------------------------------------------------------------
The Gemini key is read only from the GEMINI_API_KEY environment variable
(optionally via a local, untracked .env file loaded with python-dotenv):

    export GEMINI_API_KEY="your-key-here"
    # or create a .env file containing: GEMINI_API_KEY=your-key-here

If the key is missing or the API call fails for any reason, this module
falls back to a locally-generated, policy-correct explanation rather than
crashing the hybrid pipeline (see _fallback_explanation).
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"

# "Disagree strongly" threshold for the human-review rule below: the spread
# between the highest and lowest component score. 0.35 means e.g. 0.20 vs.
# 0.60 counts as strong disagreement, but 0.55 vs. 0.70 does not.
DISAGREEMENT_THRESHOLD = 0.35

REQUIRED_FIELDS = ["summary", "recommended_action", "reason", "human_review_required", "risk_notes"]

ACTION_BY_RISK = {"low": "auto_reply", "medium": "manual_review", "high": "escalate_immediately"}

_client = None


def _get_client():
    """Lazily creates the Gemini client. Raises if no API key is configured;
    callers (explain_with_gemini) catch this and fall back gracefully."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it as an environment variable, "
                "or add it to a local .env file. Never hard-code it in source."
            )
        from google import genai
        _client = genai.Client(api_key=api_key)
    return _client


def _build_prompt(input_payload):
    return f"""
You are a business AI decision assistant.

Given the model outputs below, generate a structured JSON response.

Rules:
- Return only valid JSON.
- Do not change the model scores.
- Do not invent facts.
- If risk_level is high, human_review_required must be true.
- If model scores disagree strongly, human_review_required must be true.

Input:
{json.dumps(input_payload, indent=2)}

Return JSON with these fields:
summary, recommended_action, reason, human_review_required, risk_notes
"""


def _scores_disagree_strongly(model_outputs):
    scores = [model_outputs[k] for k in ("classical_score", "cnn_score", "rnn_score") if k in model_outputs]
    if len(scores) < 2:
        return False
    return (max(scores) - min(scores)) > DISAGREEMENT_THRESHOLD


def _enforce_policy(explanation, model_outputs):
    """Safety net for the policy's hard rules: human_review_required must be
    True whenever risk_level is "high" or the components disagree strongly,
    regardless of what the LLM itself decided. A safety-critical flag like
    this should never depend solely on an LLM following instructions
    correctly, so we enforce it deterministically here as well."""
    must_review = model_outputs.get("risk_level") == "high" or _scores_disagree_strongly(model_outputs)
    if must_review:
        explanation["human_review_required"] = True
    return explanation


def _fallback_explanation(model_outputs, reason):
    """Used when the LLM call fails or returns something unusable. Keeps the
    hybrid pipeline functional and still emits a valid, policy-correct
    structure: the system is designed to keep working even when one component
    misbehaves, and an LLM outage should not crash a risk assessment that the
    other three models already produced."""
    risk_level = model_outputs.get("risk_level", "unknown")
    explanation = {
        "summary": f"Automated explanation unavailable ({reason}). "
                   f"The hybrid models assessed this complaint as {risk_level} risk "
                   f"(final_score={model_outputs.get('final_score', 'n/a')}).",
        "recommended_action": ACTION_BY_RISK.get(risk_level, "manual_review"),
        "reason": "Fallback explanation generated locally because the LLM call "
                  "could not be completed or returned an unusable response; "
                  "see model_outputs for the underlying scores.",
        "human_review_required": risk_level in ("medium", "high"),
        "risk_notes": ["LLM explanation unavailable; flagged for human review as a precaution."],
    }
    return _enforce_policy(explanation, model_outputs)


def explain_with_gemini(input_payload):
    """Generate a structured explanation for a hybrid prediction.

    `input_payload` should look like:
        {"complaint_text": "...", "model_outputs": {...from hybrid_inference...}}

    Always returns a dict with REQUIRED_FIELDS populated and
    human_review_required policy-enforced, falling back to a local
    explanation if the LLM call or its JSON response cannot be used.
    """
    model_outputs = input_payload.get("model_outputs", {})

    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=_build_prompt(input_payload),
            config={"response_mime_type": "application/json"},
        )
        explanation = json.loads(response.text)
    except Exception as exc:
        return _fallback_explanation(model_outputs, reason=str(exc))

    if not isinstance(explanation, dict) or any(field not in explanation for field in REQUIRED_FIELDS):
        return _fallback_explanation(model_outputs, reason="LLM response missing required fields")

    return _enforce_policy(explanation, model_outputs)


if __name__ == "__main__":
    sample_payload = {
        "complaint_text": "The product arrived broken and I have contacted support three times. Nobody has replied.",
        "model_outputs": {
            "classical_score": 0.71,
            "cnn_score": 0.4,
            "rnn_score": 0.82,
            "final_score": 0.65,
            "risk_level": "medium",
        },
    }
    print(json.dumps(explain_with_gemini(sample_payload), indent=2))
