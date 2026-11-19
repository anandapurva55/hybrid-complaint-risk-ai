# Three outcomes against a deployed service. Needs URL and HYBRID_RISK_API_KEY in the environment.
source "$(dirname "$0")/lib.sh"
clear; comment "# 1. clearly positive: answered automatically"
type_cmd "python -m hybrid_risk --url $URL \"Love this sweater, perfect fit and gorgeous color.\""
clear; comment "# 2. mixed: a person should look"
type_cmd "python -m hybrid_risk --url $URL \"Pretty dress but the material is not as soft as I hoped.\""
clear; comment "# 3. angry: escalate"
type_cmd "python -m hybrid_risk --url $URL \"Fell apart in the wash. Worst purchase ever, I want a refund!\"" 4
