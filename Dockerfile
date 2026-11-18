# Containerised FastAPI wrapper around the hybrid pipeline.
#
# Build:
#   docker build -t hybrid-risk-api .
# Run (secrets such as API_KEY are supplied at runtime, never baked into the image).
# LLM provider (see llm_explainer.py): Amazon Bedrock via ambient AWS credentials / the task role
# (LLM_PROVIDER=bedrock is the default). Use -e LLM_PROVIDER=none to skip the LLM. If the LLM is
# unreachable the API still works and returns a local fallback explanation.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what inference needs: the hybrid_risk package, the trained artifacts, the tuned decision
# config and the monitoring baseline. Training and evaluation code, data, notebooks and the LSTM baseline are left out.
COPY src/ ./src/
COPY models/classical_model.pkl models/decision_config.json ./models/
COPY models/transformer_assets/ ./models/transformer_assets/
COPY results/monitoring_baseline.json ./results/
ENV PYTHONPATH=/app/src

# Optional: set API_KEY at run time to require an X-API-Key header on /predict.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "hybrid_risk.app:app", "--host", "0.0.0.0", "--port", "8000"]
