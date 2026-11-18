import json

from monitoring import Monitor


def result(level="low", action="auto_reply", review=False, top=0.9, source="llm"):
    return {"model_outputs": {"risk_level": level, "recommended_action": action, "human_review_required": review,
                              "probabilities": {"low": top, "medium": 0.05, "high": 0.05}},
            "_meta": {"explanation_source": source}}


BASE = {"action_share": {"auto_reply": 0.70, "manual_review": 0.15, "escalate_immediately": 0.15}, "mean_top_probability": 0.85}


def test_no_alerts_below_min_n():
    m = Monitor(baseline=BASE)
    for _ in range(10):
        m.record(result(action="escalate_immediately", level="high"))
    assert m.snapshot()["alerts"] == []


def test_drift_alert_when_action_mix_shifts():
    m = Monitor(baseline=BASE)
    for _ in range(150):
        m.record(result(action="escalate_immediately", level="high", review=True))
    alerts = m.snapshot()["alerts"]
    assert any("escalate_immediately" in a for a in alerts)


def test_fallback_alert():
    m = Monitor(baseline=BASE)
    for _ in range(120):
        m.record(result(source="fallback"))
    assert any("fallback_share" in a for a in m.snapshot()["alerts"])


def test_window_is_bounded():
    m = Monitor(window=5, baseline={})
    for _ in range(20):
        m.record(result())
    assert m.snapshot()["n"] == 5


def test_emf_line_is_valid_and_free_of_text():
    import io
    from monitoring import emit_metrics
    buf = io.StringIO()
    emit_metrics({"level": "high", "action": "escalate_immediately", "review": True, "top_p": 0.8, "fallback": False}, buf)
    line = json.loads(buf.getvalue())
    declared = {m["Name"] for m in line["_aws"]["CloudWatchMetrics"][0]["Metrics"]}
    assert declared <= set(line)            # every declared metric has a value
    assert line["EscalatedHigh"] == 1 and line["ReviewRequired"] == 1 and line["FallbackExplanation"] == 0
