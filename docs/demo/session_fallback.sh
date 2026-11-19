# The same angry request while the LLM is unavailable (the service runs with LLM_PROVIDER=none).
source "$(dirname "$0")/lib.sh"
clear; comment "# 4. same request with the LLM unavailable: same decision, local fallback explanation"
type_cmd "python -m hybrid_risk --url $URL \"Fell apart in the wash. Worst purchase ever, I want a refund!\"" 5
