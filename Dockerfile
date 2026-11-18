# Containerised FastAPI wrapper around the hybrid pipeline.
#
# Build:
#   docker build -t hybrid-risk-api .
# Run (secrets such as API_KEY are supplied at runtime, never baked into the image).
# LLM provider (see llm_explainer.py): locally use Ollama on the host:
#   docker run --rm -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 \
#       -e API_KEY=... hybrid-risk-api
# In AWS use -e LLM_PROVIDER=bedrock with an IAM role. If the LLM is unreachable the
# API still works and returns a local fallback explanation.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what inference needs: the modules hybrid_inference.py imports
# (risk_common, decision, llm_explainer, compact_transformer, monitoring) and the trained artifacts + tuned
# decision config. The LSTM/CNN baselines, raw CSVs and notebooks are left out.
COPY risk_common.py decision.py llm_explainer.py compact_transformer.py monitoring.py hybrid_inference.py app.py ./
COPY classical_model.pkl decision_config.json monitoring_baseline.json ./
COPY transformer_assets/ ./transformer_assets/

# Optional: set API_KEY at run time to require an X-API-Key header on /predict.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
