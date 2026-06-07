"""
FastAPI wrapper around the hybrid pipeline.

Exposes a single POST /predict endpoint that accepts the same `input_data`
shape documented in hybrid_inference.py (the trained-model schema, matching
the Women's E-Commerce Clothing Reviews dataset's actual columns; see that
module's docstring for why) and returns predict_hybrid()'s full result:
model_outputs, llm_explanation, and _meta.

Run:
    uvicorn app:app --reload

Then, e.g.:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d @sample_inputs.json   # (send one object, not the whole list)
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hybrid_inference import predict_hybrid

app = FastAPI(
    title="Hybrid Complaint Escalation Risk API",
    description="Classical ML + CNN + RNN + LLM hybrid risk scoring for e-commerce complaints.",
    version="1.0.0",
)


class ComplaintInput(BaseModel):
    """Mirrors `input_data` in hybrid_inference.py: field names match the
    trained models' actual columns from the Women's E-Commerce Clothing
    Reviews dataset."""

    customer_id: Optional[str] = None
    review_text: str = Field(..., description="The complaint / review text.")
    age: int
    positive_feedback_count: int
    recommended_ind: int = Field(..., ge=0, le=1)
    division_name: str
    department_name: str
    class_name: str
    image_available: bool = False
    image: Optional[List[List[float]]] = Field(
        None, description="Optional 28x28 grayscale array (real CNN scoring path)."
    )


class ModelOutputs(BaseModel):
    classical_score: float
    cnn_score: float
    rnn_score: float
    final_score: float
    risk_level: str


class LlmExplanation(BaseModel):
    summary: str
    recommended_action: str
    reason: str
    human_review_required: bool
    risk_notes: object


class Meta(BaseModel):
    cnn_mode: str


class PredictionResponse(BaseModel):
    model_outputs: ModelOutputs
    llm_explanation: LlmExplanation
    meta: Meta = Field(..., alias="_meta")

    class Config:
        populate_by_name = True


@app.get("/")
def health():
    return {"status": "ok", "service": "hybrid-complaint-risk-api"}


@app.post("/predict", response_model=PredictionResponse, response_model_by_alias=True)
def predict(payload: ComplaintInput):
    """Runs the full hybrid pipeline (classical ML -> CNN -> RNN -> combine -> LLM)
    on a single complaint and returns the same structure predict_hybrid() does."""
    try:
        result = predict_hybrid(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model artifact missing: {exc}") from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from None
    return result
