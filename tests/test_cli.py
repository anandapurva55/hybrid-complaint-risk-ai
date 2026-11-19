from hybrid_risk.__main__ import format_result

RESULT = {
    "model_outputs": {"risk_level": "high", "recommended_action": "escalate_immediately", "human_review_required": True,
                      "review_reasons": ["low_confidence"], "probabilities": {"low": 0.1, "medium": 0.2, "high": 0.7}},
    "llm_explanation": {"summary": "The customer reports a broken zipper.", "reason": "Strong dissatisfaction.", "risk_notes": []},
    "_meta": {"explanation_source": "llm"},
}


def test_format_shows_tier_action_triggers_and_explanation():
    out = format_result("zipper broke", RESULT)
    for needle in ("Tier         high", "escalate_immediately", "human review: yes", "low_confidence", "broken zipper", "Source       llm"):
        assert needle in out


def test_format_without_explanation_omits_it():
    out = format_result("x", {**RESULT, "llm_explanation": None})
    assert "Explanation" not in out and "Triggers     low_confidence" in out


def test_remote_mode_posts_to_predict_and_sends_the_key_header():
    import http.server
    import json
    import threading

    from hybrid_risk.__main__ import predict_remote

    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"], seen["key"] = self.path, self.headers.get("X-API-Key")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            payload = json.dumps(RESULT).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    out = predict_remote(f"http://127.0.0.1:{server.server_port}/", "hello there friend", api_key="k")
    server.shutdown()
    assert out["model_outputs"]["risk_level"] == "high"
    assert seen == {"path": "/predict", "key": "k", "body": {"review_text": "hello there friend"}}
