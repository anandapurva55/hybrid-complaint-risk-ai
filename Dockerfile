# Containerised FastAPI wrapper around the hybrid pipeline.
#
# Build:
#   docker build -t hybrid-risk-api .
# Run (the key is supplied at runtime, never baked into the image; see the
# SECURITY note below):
#   docker run --rm -p 8000:8000 --env-file .env hybrid-risk-api
#
# SECURITY: GEMINI_API_KEY must be supplied at run time, through -e or
# --env-file, and never COPYed or hard-coded into the image. Anyone who can
# pull or inspect the image could otherwise extract it from a layer.
# .dockerignore excludes .env from the build context for the same reason.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what inference needs: the trained artifacts, plus the modules
# hybrid_inference.py imports from (risk_common, train_cnn_model for
# cnn_score_from_proba, and llm_explainer). Raw CSVs, notebooks, and the
# virtual environment are intentionally left out (see .dockerignore).
COPY risk_common.py train_cnn_model.py llm_explainer.py hybrid_inference.py app.py ./
COPY classical_model.pkl cnn_model.keras rnn_model.keras ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
