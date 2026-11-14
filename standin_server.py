"""
standin_server.py - FastAPI stand-in model service simulating real Groq API behavior.

The stand-in simulates:
1. Normal responses with calibrated latency.
2. Slow responses under concurrent load.
3. Random delays and jitter.
4. Server failures (HTTP 500/503).
5. HTTP 429 rate-limit errors with Retry-After and rate-limit headers.
6. Timeouts.
7. Daily budget exhaustion (HTTP 429 / specialized error code).
8. Load-dependent latency (degrades as active requests increase).
9. Concurrent requests handling.
10. Streaming responses via Server-Sent Events (SSE).

Calibrated from real Groq free-tier measurements:
- First-token latency: ~250-450ms
- Output token generation speed: ~40-80 tokens/sec
- Typical response time: ~1000-2500ms
"""

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# Reproducible random seed
RANDOM_SEED = int(os.getenv("STANDIN_SEED", "42"))
random.seed(RANDOM_SEED)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("standin")

app = FastAPI(title="The Relay - Stand-in Model Service", version="1.0.0")


class SimulatorConfigState:
    def __init__(self):
        self.base_latency_ms: float = 350.0
        self.latency_variation_ms: float = 150.0
        self.first_token_ms: float = 280.0
        self.first_token_variation_ms: float = 100.0
        self.failure_probability: float = 0.0
        self.rate_limit_probability: float = 0.0
        self.timeout_probability: float = 0.0
        self.rpm_limit: int = 30
        self.tpm_limit: int = 14400
        self.rpd_limit: int = 14400
        self.tpd_limit: int = 500000
        self.daily_request_budget: int = 10000
        self.daily_token_budget: int = 500000
        self.max_concurrent: int = 10
        self.load_latency_factor: float = 1.4
        self.stream_chunk_size: int = 4
        self.stream_chunk_delay_ms: float = 25.0

    def to_dict(self):
        return self.__dict__.copy()

    def update(self, d: dict):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, type(getattr(self, k))(v))


cfg = SimulatorConfigState()

# In-memory tracking
request_history_minute: List[float] = []
token_history_minute: List[float] = []
request_history_day: List[float] = []
token_history_day: List[float] = []
active_requests: int = 0
total_served: int = 0
total_given_up: int = 0

PASSAGE_RESPONSES = [
    (
        "Based on the provided research papers, the Transformer architecture introduced in [SOURCE 1] "
        "dispenses with recurrence and convolutions entirely, relying solely on self-attention mechanisms. "
        "The model uses scaled dot-product attention computed as Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V, "
        "and multi-head attention with 8 parallel attention layers in the base configuration [SOURCE 1]."
    ),
    (
        "According to [SOURCE 2], Moshi is a speech-text foundation model designed for full-duplex real-time dialogue. "
        "It employs a multi-stream architecture modeling both user and system audio simultaneously, using an "
        "inner monologue technique that predicts text tokens before generating spoken acoustic tokens [SOURCE 2]."
    ),
    (
        "As explained in [SOURCE 3], Dynamic Chunking enables end-to-end hierarchical sequence modeling by learning "
        "variable-length sequence segmentations directly from data. A boundary predictor identifies optimal chunk "
        "boundaries, allowing the model to adaptively compress sequence lengths without rigid pre-segmentation [SOURCE 3]."
    ),
    (
        "The provided research papers do not contain sufficient information to answer this question."
    ),
]


def prune_history():
    now = time.time()
    cutoff_min = now - 60.0
    cutoff_day = now - 86400.0
    while request_history_minute and request_history_minute[0] < cutoff_min:
        request_history_minute.pop(0)
    while token_history_minute and token_history_minute[0] < cutoff_min:
        token_history_minute.pop(0)
    while request_history_day and request_history_day[0] < cutoff_day:
        request_history_day.pop(0)
    while token_history_day and token_history_day[0] < cutoff_day:
        token_history_day.pop(0)


class ChatCompletionRequest(BaseModel):
    model: str = "stand-in-llama"
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False


@app.get("/health")
def health():
    prune_history()
    return {
        "status": "healthy",
        "active_requests": active_requests,
        "minute_requests": len(request_history_minute),
        "rpm_limit": cfg.rpm_limit,
        "day_requests": len(request_history_day),
        "daily_budget": cfg.daily_request_budget,
        "daily_exhausted": len(request_history_day) >= cfg.daily_request_budget,
    }


@app.post("/config")
def update_configuration(updates: Dict[str, Any]):
    cfg.update(updates)
    return {"status": "updated", "config": cfg.to_dict()}


@app.post("/reset")
def reset_counters():
    global active_requests, total_served, total_given_up
    request_history_minute.clear()
    token_history_minute.clear()
    request_history_day.clear()
    token_history_day.clear()
    active_requests = 0
    total_served = 0
    total_given_up = 0
    return {"status": "counters_reset"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global active_requests, total_served, total_given_up
    now = time.time()
    prune_history()

    # 1. Daily Budget Check
    if len(request_history_day) >= cfg.daily_request_budget or len(token_history_day) >= cfg.daily_token_budget:
        total_given_up += 1
        headers = {
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "86400",
            "retry-after": "86400",
        }
        return JSONResponse(
            status_code=429,
            headers=headers,
            content={
                "error": {
                    "message": "Daily request or token budget exhausted.",
                    "type": "daily_budget_exhausted",
                    "code": "daily_budget_exhausted",
                }
            },
        )

    # 2. Per-minute RPM Limit Check
    if len(request_history_minute) >= cfg.rpm_limit:
        headers = {
            "x-ratelimit-limit-requests": str(cfg.rpm_limit),
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "15",
            "retry-after": "15",
        }
        return JSONResponse(
            status_code=429,
            headers=headers,
            content={
                "error": {
                    "message": f"Rate limit reached: {len(request_history_minute)}/{cfg.rpm_limit} RPM.",
                    "type": "rate_limit_exceeded",
                    "code": "rate_limit_exceeded",
                }
            },
        )

    # 3. Simulated Random Server Failure (500)
    if cfg.failure_probability > 0 and random.random() < cfg.failure_probability:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error (simulated stand-in failure)", "type": "server_error"}},
        )

    # 4. Simulated Timeout
    if cfg.timeout_probability > 0 and random.random() < cfg.timeout_probability:
        await asyncio.sleep(35.0)
        return JSONResponse(status_code=504, content={"error": {"message": "Gateway Timeout"}})

    # Record request
    request_history_minute.append(now)
    request_history_day.append(now)
    active_requests += 1

    # Select response text based on user prompt content
    user_prompt = ""
    for m in req.messages:
        if m.get("role") == "user":
            user_prompt = str(m.get("content", "")).lower()

    if "moshi" in user_prompt or "speech" in user_prompt or "dialogue" in user_prompt:
        answer_text = PASSAGE_RESPONSES[1]
    elif "chunking" in user_prompt or "hierarchical" in user_prompt:
        answer_text = PASSAGE_RESPONSES[2]
    elif "capital" in user_prompt or "stock" in user_prompt or "weather" in user_prompt or "gpt-4" in user_prompt:
        answer_text = PASSAGE_RESPONSES[3]
    else:
        answer_text = PASSAGE_RESPONSES[0]

    # Calculate tokens
    prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in req.messages) + 30
    completion_tokens = len(answer_text.split()) + 10

    # Track tokens
    for _ in range(completion_tokens):
        token_history_minute.append(now)
        token_history_day.append(now)

    # Concurrency latency penalty
    concurrency_multiplier = 1.0
    if active_requests > 1:
        concurrency_multiplier = 1.0 + (min(active_requests, cfg.max_concurrent) - 1) * (cfg.load_latency_factor - 1.0) / cfg.max_concurrent

    first_token_delay = max(50.0, cfg.first_token_ms + random.gauss(0, cfg.first_token_variation_ms)) * concurrency_multiplier / 1000.0

    async def event_generator():
        global active_requests, total_served
        try:
            # Wait for first token
            await asyncio.sleep(first_token_delay)

            words = answer_text.split()
            chunk_size = max(1, cfg.stream_chunk_size)
            chunk_delay = (cfg.stream_chunk_delay_ms / 1000.0) * concurrency_multiplier

            for i in range(0, len(words), chunk_size):
                chunk_words = " ".join(words[i:i + chunk_size])
                if i + chunk_size < len(words):
                    chunk_words += " "
                payload = {
                    "id": f"chatcmpl-standin-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": chunk_words}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(chunk_delay)

            # Final chunk with usage stats
            finish_payload = {
                "id": f"chatcmpl-standin-{int(time.time()*1000)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {json.dumps(finish_payload)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            active_requests = max(0, active_requests - 1)
            total_served += 1

    common_headers = {
        "x-ratelimit-limit-requests": str(cfg.rpm_limit),
        "x-ratelimit-remaining-requests": str(max(0, cfg.rpm_limit - len(request_history_minute))),
        "x-ratelimit-limit-tokens": str(cfg.tpm_limit),
        "x-ratelimit-remaining-tokens": str(max(0, cfg.tpm_limit - len(token_history_minute))),
    }

    if req.stream:
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers=common_headers,
        )
    else:
        try:
            total_latency = (cfg.base_latency_ms + random.gauss(0, cfg.latency_variation_ms)) * concurrency_multiplier / 1000.0
            await asyncio.sleep(max(0.1, total_latency))
            res = {
                "id": f"chatcmpl-standin-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer_text}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            return JSONResponse(content=res, headers=common_headers)
        finally:
            active_requests = max(0, active_requests - 1)
            total_served += 1


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "stand-in-llama", "object": "model", "owned_by": "standin"}],
    }
