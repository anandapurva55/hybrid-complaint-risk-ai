"""
FastAPI wrapper around the hybrid pipeline.

POST /predict accepts the `input_data` shape documented in hybrid_inference.py
and returns model_outputs (deterministic decision), llm_explanation and _meta.

Optional auth: if the API_KEY environment variable is set, every request to
/predict must send the same value in an `X-API-Key` header. (Rate limiting is
left to the platform in front of the service, e.g. API Gateway / WAF.)

Run:
    uvicorn hybrid_risk.app:app --reload
"""

import hmac
import logging
import os
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from hybrid_risk import hybrid_inference
from hybrid_risk.monitoring import Monitor

logger = logging.getLogger("hybrid-risk-api")

monitor = Monitor()

app = FastAPI(
    title="Hybrid Dissatisfaction-Risk Triage API",
    description="Fine-tuned DistilBERT decision with an LLM explanation. Risk is a proxy "
                "derived from star ratings, not an observed escalation outcome.",
    version="2.0.0",
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.environ.get("API_KEY")
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


class ComplaintInput(BaseModel):
    """Only `review_text` is used. The other fields mirror the original dataset's columns and are
    accepted (and validated if present) for compatibility, but ignored. Unknown fields are ignored too."""

    customer_id: Optional[str] = Field(None, max_length=64)
    review_text: str = Field(..., min_length=1, max_length=5000, description="Complaint / review text.")
    age: Optional[int] = Field(None, ge=10, le=120, description="Accepted for compatibility; not used.")
    positive_feedback_count: Optional[int] = Field(None, ge=0, description="Accepted for compatibility; not used.")
    division_name: Optional[str] = Field(None, max_length=100, description="Accepted for compatibility; not used.")
    department_name: Optional[str] = Field(None, max_length=100, description="Accepted for compatibility; not used.")
    class_name: Optional[str] = Field(None, max_length=100, description="Accepted for compatibility; not used.")


class ModelOutputs(BaseModel):
    final_score: float
    probabilities: Dict[str, float]
    risk_level: str
    recommended_action: str
    human_review_required: bool
    review_reasons: List[str]


class LlmExplanation(BaseModel):
    summary: str
    recommended_action: str
    reason: str
    human_review_required: bool
    risk_notes: List[str]


class Meta(BaseModel):
    explanation_source: Optional[str] = None


class PredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model_outputs: ModelOutputs
    llm_explanation: Optional[LlmExplanation] = None
    meta: Meta = Field(..., alias="_meta")


@app.get("/")
def health():
    return {"status": "ok", "service": "hybrid-complaint-risk-api"}


@app.get("/stats", dependencies=[Depends(require_api_key)])
def stats():
    """Rolling-window drift monitor (no complaint text or identifiers)."""
    return monitor.snapshot()


@app.post("/predict", response_model=PredictionResponse, response_model_by_alias=True,
          dependencies=[Depends(require_api_key)])
def predict(payload: ComplaintInput):
    """Runs the full hybrid pipeline on one complaint."""
    try:
        result = hybrid_inference.predict_hybrid(payload.model_dump())
        monitor.record(result)
        return result
    except FileNotFoundError:
        logger.exception("Model artifact missing")
        raise HTTPException(status_code=503, detail="Model artifacts unavailable.") from None
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from None
