# Containerised FastAPI wrapper around the hybrid pipeline.
#
# Build:
#   docker build -t hybrid-risk-api .
# Run (secrets such as API_KEY are supplied at runtime, never baked into the image).
# The LLM runs in a separate Ollama server; from the container reach the host's with:
#   docker run --rm -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 \
#       -e API_KEY=... hybrid-risk-api
# If Ollama is unreachable the API still works and returns a local fallback explanation.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what inference needs: the modules hybrid_inference.py imports
# (risk_common, decision, llm_explainer) and the trained artifacts + tuned
# decision config. The CNN demo model, raw CSVs and notebooks are left out.
COPY risk_common.py decision.py llm_explainer.py hybrid_inference.py app.py ./
COPY classical_model.pkl rnn_model.keras decision_config.json ./

# Optional: set API_KEY at run time to require an X-API-Key header on /predict.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
