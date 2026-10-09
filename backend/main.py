"""FastAPI backend for the personal RAG chatbot.

Run (from the backend folder):
    uvicorn main:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health        -> status, model, number of chunks
    POST /chat          -> {"question": "...", "session_id": "optional"}
                           streams the answer as plain text; the session id
                           comes back in the X-Session-Id response header
    POST /chat/reset    -> {"session_id": "..."} clears that conversation
"""
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import chat as chat_module
from chat import ChatBot
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
for noisy in ("httpx", "httpcore", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("api")

# -----------------------------------------------------------------
# Config (override with environment variables / .env)
# -----------------------------------------------------------------
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    if o.strip()
]
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "500"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_MINUTES", "30")) * 60
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "200"))
QUEUE_TIMEOUT_SECONDS = int(os.getenv("QUEUE_TIMEOUT_SECONDS", "60"))

# APP_ENV=production hides /docs, /redoc and the OpenAPI schema.
APP_ENV = os.getenv("APP_ENV", "development")
IS_PROD = APP_ENV == "production"

# Answer these once at startup so the frontend's suggestion buttons respond
# instantly. Keep in sync with SUGGESTIONS in frontend/src/App.jsx.
PREWARM = os.getenv("PREWARM", "1") == "1"
PREWARM_QUESTIONS = [
    "What is your current job?",
    "Tell me about yourself",
    "What are your technical skills?",
    "What projects have you worked on?",
    "What is your educational background?",
]

# -----------------------------------------------------------------
# Shared state
# -----------------------------------------------------------------
store = None                      # one VectorStore for everyone
sessions = {}                     # session_id -> (ChatBot, last_used_time)
sessions_lock = threading.Lock()

# A CPU-only Ollama answers one question at a time. Extra requests wait here
# (up to QUEUE_TIMEOUT_SECONDS) instead of slowing everyone down.
llm_lock = threading.Lock()

request_log = defaultdict(deque)  # ip -> timestamps of recent requests
rate_lock = threading.Lock()


def prewarm_cache():
    """Run in the background after startup; the first visitors never wait."""
    for question in PREWARM_QUESTIONS:
        if not llm_lock.acquire(timeout=QUEUE_TIMEOUT_SECONDS * 5):
            continue
        try:
            "".join(ChatBot(store).ask_stream(question))
        except Exception:
            logger.exception("Pre-warm failed for %r", question)
        finally:
            llm_lock.release()
    logger.info("Cache pre-warmed: %d answers", len(ChatBot.cache))


@asynccontextmanager
async def lifespan(app):
    global store
    store = VectorStore()
    logger.info("Chunks in index: %s", store.collection.count())
    ChatBot(store).warmup()       # load embedding model + LLM once
    logger.info("Ready")
    if PREWARM:
        threading.Thread(target=prewarm_cache, daemon=True).start()
    yield


app = FastAPI(
    title="Personal RAG Chatbot",
    lifespan=lifespan,
    docs_url=None if IS_PROD else "/docs",
    redoc_url=None if IS_PROD else "/redoc",
    openapi_url=None if IS_PROD else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    expose_headers=["X-Session-Id"],
)


# -----------------------------------------------------------------
# Models
# -----------------------------------------------------------------
class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: Optional[str] = Field(default=None, max_length=64)
    stream: bool = True


class ResetRequest(BaseModel):
    session_id: str = Field(max_length=64)


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------
def check_rate_limit(request: Request):
    # Behind a reverse proxy, request.client.host is the proxy's address.
    # Configure the proxy / uvicorn --proxy-headers when you deploy.
    ip = request.client.host if request.client else "unknown"
    now = time.time()

    with rate_lock:
        timestamps = request_log[ip]
        while timestamps and now - timestamps[0] > 60:
            timestamps.popleft()

        if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=429,
                detail="Too many questions. Please wait a minute and try again.",
            )

        timestamps.append(now)


def get_session(session_id):
    """Return (session_id, ChatBot). Unknown or expired ids start a new chat."""
    now = time.time()

    with sessions_lock:
        for key in [k for k, (_, used) in sessions.items()
                    if now - used > SESSION_TTL_SECONDS]:
            del sessions[key]

        if session_id and session_id in sessions:
            bot = sessions[session_id][0]
        else:
            if len(sessions) >= MAX_SESSIONS:
                oldest = min(sessions, key=lambda k: sessions[k][1])
                del sessions[oldest]

            session_id = uuid.uuid4().hex
            bot = ChatBot(store)

        sessions[session_id] = (bot, now)
        return session_id, bot


def generate(bot, question):
    """Wait for the LLM, then stream the answer piece by piece."""
    if not llm_lock.acquire(timeout=QUEUE_TIMEOUT_SECONDS):
        yield "The assistant is busy right now. Please try again in a moment."
        return

    try:
        yield from bot.ask_stream(question)
    finally:
        llm_lock.release()  # also runs if the browser disconnects


# -----------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": chat_module.LLM_MODEL,
        "chunks": store.collection.count() if store else 0,
        "sessions": len(sessions),
    }


@app.post("/chat")
def chat_endpoint(req: ChatRequest, request: Request):
    check_rate_limit(request)
    session_id, bot = get_session(req.session_id)

    if req.stream:
        return StreamingResponse(
            generate(bot, req.question),
            media_type="text/plain; charset=utf-8",
            headers={
                "X-Session-Id": session_id,
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # stops nginx from buffering
            },
        )

    answer = "".join(generate(bot, req.question)).strip()
    return {"answer": answer, "session_id": session_id}


@app.post("/chat/reset")
def reset_endpoint(req: ResetRequest):
    with sessions_lock:
        item = sessions.get(req.session_id)

    if item:
        item[0].reset()

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000)