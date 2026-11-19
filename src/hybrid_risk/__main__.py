"""
Command-line demo:  python -m hybrid_risk "The zipper broke after one wear and nobody answered my emails."

Prints the tier, the action, the review reasons and the explanation for one or more feedback texts
(or for texts piped in, one per line). Pass --no-explain to skip the LLM call.

To query a DEPLOYED service instead of the local models:
    HYBRID_RISK_API_KEY=... python -m hybrid_risk --url https://<distribution>.cloudfront.net "text"
The API key is read from the environment (never from the command line) and is never printed.
"""
import argparse
import json
import os
import sys
import textwrap
import urllib.error
import urllib.request


def format_result(text, result, width=88):
    mo, ex, meta = result["model_outputs"], result.get("llm_explanation"), result["_meta"]
    p = mo["probabilities"]
    reasons = ", ".join(mo["review_reasons"]) or "none"
    lines = [
        f"Feedback     {text}",
        f"Tier         {mo['risk_level']}   (low {p['low']:.2f} | medium {p['medium']:.2f} | high {p['high']:.2f})",
        f"Action       {mo['recommended_action']}   human review: {'yes' if mo['human_review_required'] else 'no'}",
        f"Triggers     {reasons}",
    ]
    if ex:
        body = f"{ex['summary']} {ex['reason']}"
        wrapped = textwrap.wrap(body, width=width - 13)
        lines.append(f"Explanation  {wrapped[0]}")
        lines += [" " * 13 + w for w in wrapped[1:]]
        lines.append(f"Source       {meta.get('explanation_source')}")
    return "\n".join(lines)


def predict_remote(url, text, api_key=None, timeout=60):
    """POST one feedback text to a deployed service and return its JSON response."""
    req = urllib.request.Request(url.rstrip("/") + "/predict", data=json.dumps({"review_text": text}).encode(),
                                 headers={"Content-Type": "application/json", **({"X-API-Key": api_key} if api_key else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"service returned HTTP {exc.code}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m hybrid_risk", description="Triage customer feedback and explain the decision.")
    ap.add_argument("text", nargs="*", help="feedback text(s); reads stdin (one per line) if omitted")
    ap.add_argument("--no-explain", action="store_true", help="skip the LLM explanation (local mode only)")
    ap.add_argument("--url", help="query a deployed service at this base URL instead of the local models")
    ap.add_argument("--api-key-env", default="HYBRID_RISK_API_KEY", help="environment variable holding the service's API key")
    args = ap.parse_args(argv)
    texts = args.text or [l.strip() for l in sys.stdin if l.strip()]
    if not texts:
        ap.error("give at least one feedback text")

    if args.url:
        key = os.environ.get(args.api_key_env)
        run = lambda t: predict_remote(args.url, t, key)
    else:
        from hybrid_risk.hybrid_inference import predict_hybrid  # imported late so --help is instant
        run = lambda t: predict_hybrid({"review_text": t}, explain=not args.no_explain)
    for i, t in enumerate(texts):
        if i:
            print()
        print(format_result(t, run(t)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
