"""
Regenerate sample_outputs.json by running every sample in sample_inputs.json
through the pipeline. Each record notes whether its explanation came from the
LLM or the local fallback (`_meta.explanation_source`); without a
working Bedrock access every explanation is a fallback and is labelled as such.

    python -m evaluation.generate_sample_outputs
"""
import json

from hybrid_risk.hybrid_inference import predict_hybrid
from hybrid_risk.paths import SAMPLES

with open(SAMPLES / "sample_inputs.json") as f:
    samples = json.load(f)

outputs = []
for s in samples:
    r = predict_hybrid(s)
    outputs.append({"customer_id": s.get("customer_id"), "review_text": s["review_text"], **r})

with open(SAMPLES / "sample_outputs.json", "w") as f:
    json.dump(outputs, f, indent=2)

for o in outputs:
    m = o["model_outputs"]
    print(o["customer_id"], m["risk_level"], m["recommended_action"], "review=", m["human_review_required"],
          m["review_reasons"], m["probabilities"], o["_meta"]["explanation_source"])
