"""
Live RAG Accuracy & Latency Benchmark
Hits the running backend at http://localhost:8000/chat
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import httpx
import time
import json

BASE_URL = "http://127.0.0.1:8000"

QUERIES = [
    {
        "id": "Q1",
        "label": "Identity",
        "query": "Who are you and what is your name?",
        "expected_keywords": ["prospectus", "ned", "ai", "assistant"],
        "must_not_contain": ["i don't know", "not found", "unable"],
    },
    {
        "id": "Q2",
        "label": "Seat Matrix",
        "query": "Give me the undergraduate seat distribution matrix",
        "expected_keywords": ["seat", "distribution", "pdf", "supabase"],
        "must_not_contain": ["not found", "unable"],
    },
    {
        "id": "Q3",
        "label": "UG Eligibility (SE)",
        "query": "What is the eligibility criteria for undergraduate Software Engineering admission?",
        "expected_keywords": ["intermediate", "fsc", "pre-engineering", "merit", "60%", "matric"],
        "must_not_contain": ["ms", "postgraduate", "phd", "research paper", "thesis"],
    },
    {
        "id": "Q4",
        "label": "Dean of Faculty",
        "query": "Who is the dean of the faculty of Electrical and Computer Engineering at NED?",
        "expected_keywords": ["dean", "electrical", "computer engineering"],
        "must_not_contain": ["not found", "unable to find", "couldn't find"],
    },
    {
        "id": "Q5",
        "label": "Course Code (CS)",
        "query": "What is the course code and credit hours for Programming Fundamentals in BS Computer Science?",
        "expected_keywords": ["credit", "hour", "programming fundamentals"],
        "must_not_contain": ["not found", "unable to find", "couldn't find"],
    },
    {
        "id": "Q6",
        "label": "Fee Structure (UG)",
        "query": "What are the tuition fees for undergraduate students at NED?",
        "expected_keywords": ["fee", "tuition", "per semester", "rupee", "rs", "pkr"],
        "must_not_contain": ["not found", "unable"],
    },
    {
        "id": "Q7",
        "label": "Admission Test",
        "query": "Is there an entry test for NED undergraduate admission?",
        "expected_keywords": ["test", "entry", "admission", "ecat", "merit"],
        "must_not_contain": ["not found", "unable"],
    },
]


def score_response(response_text: str, query_def: dict) -> tuple[int, list[str]]:
    """Returns (score 0-5, list of reasons)."""
    text_lower = response_text.lower()
    reasons = []
    score = 0

    # 1. Non-empty response
    if len(response_text.strip()) > 20:
        score += 1
    else:
        reasons.append("Response too short or empty")
        return 0, reasons

    # 2. No hallucination / refusal markers
    has_bad = any(bad in text_lower for bad in query_def["must_not_contain"])
    if not has_bad:
        score += 1
    else:
        reasons.append(f"Contains unwanted phrase(s): {[b for b in query_def['must_not_contain'] if b in text_lower]}")

    # 3-5. Expected keyword hits (up to 3 points)
    keywords_hit = [kw for kw in query_def["expected_keywords"] if kw.lower() in text_lower]
    kw_score = min(3, len(keywords_hit))
    score += kw_score
    if kw_score < 3:
        missed = [kw for kw in query_def["expected_keywords"] if kw.lower() not in text_lower]
        reasons.append(f"Missing keywords: {missed}")
    else:
        reasons.append(f"Keywords hit: {keywords_hit}")

    return score, reasons


def run_benchmark():
    print("\n" + "=" * 70)
    print("  NED Prospectus AI — Live Accuracy & Latency Benchmark")
    print("=" * 70)

    # Health check
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        print(f"\n✅ Server is UP — {BASE_URL}  (status {r.status_code})")
    except Exception as e:
        print(f"\n❌ Server not reachable at {BASE_URL}: {e}")
        return

    # Obtain session token
    print("\n🔑 Authenticating benchmark user...")
    username = "benchmark_tester"
    password = "secret_password_123"
    token = None
    try:
        # Try signing up first (ignore conflict errors)
        httpx.post(f"{BASE_URL}/auth/signup", json={"username": username, "password": password}, timeout=30)
        
        # Log in
        login_res = httpx.post(f"{BASE_URL}/auth/login", json={"username": username, "password": password}, timeout=30)
        if login_res.status_code == 200:
            token = login_res.json().get("token")
            print("✅ Authentication successful!")
        else:
            print(f"❌ Login failed: {login_res.status_code} - {login_res.text}")
            return
    except Exception as e:
        print(f"❌ Authentication exception: {e}")
        return

    headers = {"Authorization": f"Bearer {token}"}
    results = []

    for q in QUERIES:
        print(f"\n{'─'*70}")
        print(f"[{q['id']}] {q['label']}")
        print(f"  Query: {q['query']}")

        payload = {"query": q["query"], "history": []}

        # Collect streamed response
        full_response = ""
        start = time.perf_counter()

        try:
            with httpx.stream("POST", f"{BASE_URL}/user/query", json=payload, headers=headers, timeout=30) as stream:
                for chunk in stream.iter_text():
                    if chunk.strip():
                        # Strip SSE "data: " prefix if present
                        for line in chunk.split("\n"):
                            line = line.strip()
                            if line.startswith("data:"):
                                data = line[5:].strip()
                                if data and data != "[DONE]":
                                    try:
                                        obj = json.loads(data)
                                        if isinstance(obj, dict):
                                            token = obj.get("token") or obj.get("content") or obj.get("text") or ""
                                        else:
                                            token = str(obj)
                                        full_response += token
                                    except json.JSONDecodeError:
                                        full_response += data
                            elif line and not line.startswith("event:"):
                                try:
                                    obj = json.loads(line)
                                    if isinstance(obj, dict):
                                        token = obj.get("token") or obj.get("content") or obj.get("text") or obj.get("response") or ""
                                    else:
                                        token = str(obj)
                                    full_response += token
                                except json.JSONDecodeError:
                                    full_response += line

        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"  ❌ Request failed: {e}")
            results.append({"id": q["id"], "label": q["label"], "latency": elapsed, "score": 0, "response": ""})
            continue

        elapsed = time.perf_counter() - start

        if not full_response:
            # Fallback: try plain POST
            try:
                start2 = time.perf_counter()
                r2 = httpx.post(f"{BASE_URL}/user/query", json=payload, headers=headers, timeout=30)
                elapsed = time.perf_counter() - start2
                body = r2.json()
                full_response = (
                    body.get("response")
                    or body.get("answer")
                    or body.get("message")
                    or body.get("text")
                    or str(body)
                )
            except Exception as e2:
                print(f"  ❌ Fallback also failed: {e2}")
                results.append({"id": q["id"], "label": q["label"], "latency": elapsed, "score": 0, "response": ""})
                continue

        score, reasons = score_response(full_response, q)

        print(f"  ⏱  Latency  : {elapsed:.3f}s")
        print(f"  📊 Score    : {score}/5")
        for r in reasons:
            print(f"     • {r}")
        print(f"  💬 Response : {full_response[:300]}{'...' if len(full_response)>300 else ''}")

        results.append({
            "id": q["id"],
            "label": q["label"],
            "latency": elapsed,
            "score": score,
            "response": full_response,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n{'ID':<5} {'Label':<25} {'Score':>7} {'Latency':>10}")
    print(f"{'─'*5} {'─'*25} {'─'*7} {'─'*10}")

    total_score = 0
    total_possible = len(results) * 5
    latencies = []

    for r in results:
        stars = "★" * r["score"] + "☆" * (5 - r["score"])
        print(f"{r['id']:<5} {r['label']:<25} {stars}  {r['latency']:>7.3f}s")
        total_score += r["score"]
        latencies.append(r["latency"])

    pct = (total_score / total_possible * 100) if total_possible else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    print(f"\n  Overall Accuracy : {total_score}/{total_possible}  ({pct:.0f}%)")
    print(f"  Avg Latency      : {avg_lat:.3f}s")
    print(f"  P95 Latency      : {p95_lat:.3f}s")
    print(f"  Min / Max        : {min(latencies):.3f}s / {max(latencies):.3f}s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_benchmark()
