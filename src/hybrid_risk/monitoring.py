"""
Lightweight in-process drift monitoring.

Tracks, over a rolling window of recent predictions: risk-level mix, action
mix, human-review share, mean top-class probability, and the share of LLM
explanations that fell back. snapshot() compares them with the baseline
measured on the test split (evaluation_results.json) and raises alerts when a
metric moves more than ALERT_DELTA (absolute) with at least MIN_N observations.

State is per process (each container instance has its own window); for a
fleet, export these counters to your metrics system instead. Exposed at
GET /stats in app.py. Contains no complaint text or customer identifiers.
"""
import json
import os
import sys
import threading
import time
from collections import deque

WINDOW = 1000
MIN_N = 100
ALERT_DELTA = 0.10
from hybrid_risk.paths import RESULTS


def load_baseline(path=str(RESULTS / "monitoring_baseline.json")):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return {}


def emit_metrics(row, stream=None):
    """One CloudWatch Embedded Metric Format line per prediction (stdout -> awslogs).

    Gives fleet-wide metrics and alarms without per-instance state: the Average of
    ReviewRequired / FallbackExplanation (0 or 1 per request) is the share over any
    period. No complaint text or identifiers are included."""
    line = {
        "_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [{
            "Namespace": "HybridRiskApi", "Dimensions": [["Service"]],
            "Metrics": [{"Name": "Predictions", "Unit": "Count"},
                        {"Name": "ReviewRequired", "Unit": "None"},
                        {"Name": "FallbackExplanation", "Unit": "None"},
                        {"Name": "AutoReply", "Unit": "None"},
                        {"Name": "EscalatedHigh", "Unit": "None"},
                        {"Name": "TopProbability", "Unit": "None"}]}]},
        "Service": "hybrid-risk-api",
        "Predictions": 1, "ReviewRequired": int(row["review"]), "FallbackExplanation": int(row["fallback"]),
        "AutoReply": int(row["action"] == "auto_reply"), "EscalatedHigh": int(row["level"] == "high"),
        "TopProbability": round(row["top_p"], 4),
    }
    print(json.dumps(line), file=stream or sys.stdout, flush=True)


class Monitor:
    def __init__(self, window=WINDOW, baseline=None):
        self._rows = deque(maxlen=window)
        self._lock = threading.Lock()
        self.baseline = load_baseline() if baseline is None else baseline

    def record(self, result):
        mo = result["model_outputs"]
        row = {
            "level": mo["risk_level"],
            "action": mo["recommended_action"],
            "review": bool(mo["human_review_required"]),
            "top_p": max(mo["probabilities"].values()),
            "fallback": (result.get("_meta") or {}).get("explanation_source") == "fallback",
        }
        with self._lock:
            self._rows.append(row)
        emit_metrics(row)

    def snapshot(self):
        with self._lock:
            rows = list(self._rows)
        n = len(rows)
        if n == 0:
            return {"n": 0, "alerts": []}
        share = lambda key, val: sum(r[key] == val for r in rows) / n
        cur = {
            "level_share": {l: round(share("level", l), 3) for l in ("low", "medium", "high")},
            "action_share": {a: round(share("action", a), 3) for a in ("auto_reply", "manual_review", "escalate_immediately")},
            "review_share": round(sum(r["review"] for r in rows) / n, 3),
            "mean_top_probability": round(sum(r["top_p"] for r in rows) / n, 3),
            "fallback_share": round(sum(r["fallback"] for r in rows) / n, 3),
        }
        alerts = []
        base = self.baseline
        if n >= MIN_N and base:
            for a, v in cur["action_share"].items():
                b = base.get("action_share", {}).get(a)
                if b is not None and abs(v - b) > ALERT_DELTA:
                    alerts.append(f"action_share[{a}] {v} vs baseline {b}")
            b = base.get("mean_top_probability")
            if b is not None and abs(cur["mean_top_probability"] - b) > 0.05:
                alerts.append(f"mean_top_probability {cur['mean_top_probability']} vs baseline {b}")
        if n >= MIN_N and cur["fallback_share"] > 0.40:
            alerts.append(f"fallback_share {cur['fallback_share']} > 0.40 (LLM unhealthy or rejecting output)")
        return {"n": n, "window": self._rows.maxlen, "current": cur, "baseline": base, "alerts": alerts}
