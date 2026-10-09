"""Quick API check. Start the server first, then:  python checking/test_api.py
Add --rate to also test the rate limit (sends 12 quick requests)."""
import codecs
import json
import sys
import time
import urllib.error
import urllib.request

import os
BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")  # Docker: http://127.0.0.1:8080/api


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        return urllib.request.urlopen(req, timeout=180)
    except urllib.error.HTTPError as err:   # 4xx / 5xx still has a body
        return err


def ask(question, session_id=None, show=True):
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id

    start = time.time()
    first = None
    resp = post("/chat", payload)
    status = getattr(resp, "status", None) or resp.code
    sid = resp.headers.get("X-Session-Id")

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    parts = []
    while True:
        chunk = resp.read(32)
        if not chunk:
            break
        if first is None:
            first = time.time() - start
        parts.append(decoder.decode(chunk))
    answer = "".join(parts).strip()

    if show:
        print(f"\nQ: {question}")
        print(f"HTTP {status} | session {sid and sid[:8]} | "
              f"first byte {first or 0:.1f}s | total {time.time() - start:.1f}s")
        print(f"A: {answer}")
    return status, sid, answer


def main():
    print("GET /health")
    with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
        print(r.read().decode())

    # 1. basic question
    _, sid, _ = ask("what is your name?")

    # 2. memory: follow-up in the SAME session
    _, sid, _ = ask("What did you do at InstaVC?")
    ask("What technologies did you use there?", sid)
    ask("How long were you there?", sid)

    # 3. a new session must not remember InstaVC
    ask("What technologies did you use there?")

    # 4. validation
    print("\n--- validation (expect HTTP 422) ---")
    status, _, _ = ask("   ", show=False)
    print("empty question ->", status)
    status, _, _ = ask("x" * 600, show=False)
    print("600 characters ->", status)

    # 5. rate limit (optional)
    if "--rate" in sys.argv:
        print("\n--- rate limit (expect 429 after ~10) ---")
        for i in range(12):
            status, _, _ = ask("what is your name?", show=False)
            print(i + 1, status)


if __name__ == "__main__":
    main()