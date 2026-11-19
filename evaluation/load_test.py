"""
Latency and throughput of POST /predict against a running service (model path only; start the
service with LLM_PROVIDER=template so the LLM call is excluded and measured separately).

Used with the container limited to 2 CPUs / 4 GB, the same sizing as the Fargate task:

    docker run --rm --cpus=2 --memory=4g -p 8013:8000 -e LLM_PROVIDER=template hybrid-risk-api
    python -m evaluation.load_test --url http://localhost:8013 --requests 300

Writes results/load_test_results.json. Real test-set reviews are the payloads.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd

from hybrid_risk.paths import DATA_PROCESSED, RESULTS


def payload(r):
    return {"review_text": r["Review Text"][:5000], "age": int(r["Age"]), "positive_feedback_count": int(r["Positive Feedback Count"]),
            "division_name": r["Division Name"], "department_name": r["Department Name"], "class_name": r["Class Name"]}


def run(url, rows, concurrency, headers):
    lat = []
    def one(r):
        with httpx.Client(timeout=60) as c:
            t0 = time.perf_counter()
            resp = c.post(f"{url}/predict", json=payload(r), headers=headers)
            resp.raise_for_status()
            return time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as ex:
        lat = list(ex.map(one, rows))
    wall = time.perf_counter() - t0
    return {"concurrency": concurrency, "requests": len(rows), "throughput_rps": round(len(rows) / wall, 2),
            "latency_ms_p50": round(float(np.percentile(lat, 50)) * 1000), "latency_ms_p95": round(float(np.percentile(lat, 95)) * 1000),
            "latency_ms_max": round(max(lat) * 1000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--api-key", default=None)
    a = ap.parse_args()
    headers = {"X-API-Key": a.api_key} if a.api_key else {}
    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    rows = [r for _, r in test.sample(a.requests, random_state=42).iterrows()]
    run(a.url, rows[:10], 1, headers)  # warm-up (lazy model load)
    results = [run(a.url, rows, c, headers) for c in (1, 4, 8)]
    for r in results:
        print(r)
    with open(RESULTS / "load_test_results.json", "w") as f:
        json.dump({"setup": "container limited to 2 CPUs / 4 GB on a laptop (Apple Silicon, arm64), LLM_PROVIDER=template, real test reviews",
                   "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
