"""Automated checks against the RUNNING API.

    pip install pytest
    # terminal 1 (high rate limit so the suite doesn't hit 429):
    #   set RATE_LIMIT_PER_MINUTE=200      (PowerShell: $env:RATE_LIMIT_PER_MINUTE="200")
    #   uvicorn main:app --port 8000
    # terminal 2, from backend/:
    pytest checking/test_suite.py -v

Answers come from an LLM, so a wording change can fail a test now and then.
Keywords are kept short and lowercase on purpose.
"""
import os
import sys
import time
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_api import BASE, ask  # noqa: E402
from test_security import CASES  # noqa: E402

try:
    urllib.request.urlopen(BASE + "/health", timeout=5)
except Exception:
    pytest.skip(f"API is not running at {BASE}", allow_module_level=True)


def ask_ok(question, session_id=None):
    """ask() that waits and retries when the rate limit (429) is hit."""
    for _ in range(3):
        status, sid, answer = ask(question, session_id, show=False)
        if status != 429:
            return status, sid, answer.lower()
        time.sleep(60)
    pytest.fail("rate limit never cleared")


def test_health():
    with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
        body = r.read().decode()
    assert '"status":"ok"' in body.replace(" ", "")
    assert '"chunks":0' not in body.replace(" ", "")


GROUNDED = [
    ("what is your name?", ["shakeel"]),
    ("What is your current job?", ["automation engineer", "exl"]),
    ("list the companies you have worked", ["exl", "igo", "instavc", "peoplelink"]),
    ("What was your role at IGO Solutions?", ["junior"]),
    ("What is the first company you joined in your career?", ["peoplelink"]),
    ("What is your educational background?", ["artificial intelligence", "data science"]),
    ("What technologies did you use at InstaVC?", ["react", "webrtc"]),
    ("Do you know Docker?", ["docker"]),
]


@pytest.mark.parametrize("question,keywords", GROUNDED)
def test_grounded_answers(question, keywords):
    status, _, answer = ask_ok(question)
    assert status == 200
    for word in keywords:
        assert word in answer, f"'{word}' missing in: {answer}"


@pytest.mark.parametrize("question,forbidden", CASES)
def test_traps_and_injection(question, forbidden):
    status, _, answer = ask_ok(question)
    assert status == 200
    for word in forbidden:
        assert word not in answer, f"'{word}' found in: {answer}"


def test_follow_up_memory():
    _, sid, first = ask_ok("What did you do at InstaVC?")
    assert "instavc" in first

    _, sid, tech = ask_ok("What technologies did you use there?", sid)
    assert "webrtc" in tech or "react" in tech

    _, _, dates = ask_ok("How long were you there?", sid)
    assert "2024" in dates


def test_sessions_are_isolated():
    _, _, answer = ask_ok("How long were you there?")  # brand-new session
    assert "instavc" not in answer


@pytest.mark.parametrize("question", ["   ", "x" * 600])
def test_validation_rejects_bad_input(question):
    status, _, _ = ask(question, show=False)
    assert status == 422


if __name__ == "__main__":
    # lets you run:  python test_suite.py
    sys.exit(pytest.main([__file__, "-v"]))