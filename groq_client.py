"""
groq_client.py - Async Groq API client with streaming, metrics collection, and error handling.
"""

import os
import time
import json
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator, Dict, Any

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
FALLBACK_MODEL = "openai/gpt-oss-20b"


@dataclass
class GroqResponse:
    content: str
    model: str
    tokens_in: int
    tokens_out: int
    first_token_ms: float
    total_ms: float
    http_status: int
    request_id: Optional[str] = None
    rate_limit_headers: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    success: bool = True
    is_streaming: bool = False


class GroqClient:
    """Async Groq API client with full observability."""

    SYSTEM_PROMPT = """You are a precise research assistant. Answer questions ONLY using the provided source passages.

Rules:
1. Base every factual claim on the provided [SOURCE N] passages.
2. After each factual claim, cite the source like: [SOURCE 1] or [SOURCE 2].
3. If the sources do not contain enough information, respond EXACTLY:
   "The provided research papers do not contain sufficient information to answer this question."
4. Never invent citations, paper names, or facts not in the sources.
5. Be concise and accurate."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        api_base: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.api_base = api_base or os.getenv("GROQ_API_BASE", GROQ_API_BASE)
        if not self.api_key and "127.0.0.1" not in self.api_base and "localhost" not in self.api_base:
            logger.warning("GROQ_API_KEY not set. Real API calls will fail.")

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        question: str,
        context: str,
        stream: bool = True,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> GroqResponse:
        """
        Generate a response using the Groq API.
        
        Args:
            question: User question.
            context: Retrieved source passages.
            stream: Whether to use streaming.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (low for factual answers).
        
        Returns:
            GroqResponse with full metrics.
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Source passages:\n\n{context}\n\nQuestion: {question}",
            },
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        t_start = time.time()
        first_token_ms = 0.0

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if stream:
                    return await self._stream_response(
                        client, payload, t_start
                    )
                else:
                    return await self._sync_response(client, payload, t_start)

        except httpx.TimeoutException as e:
            total_ms = (time.time() - t_start) * 1000
            logger.error("Groq request timed out after %.0f ms", total_ms)
            return GroqResponse(
                content="",
                model=self.model,
                tokens_in=0,
                tokens_out=0,
                first_token_ms=0,
                total_ms=total_ms,
                http_status=408,
                error=f"Timeout: {e}",
                success=False,
            )
        except httpx.HTTPStatusError as e:
            total_ms = (time.time() - t_start) * 1000
            status = e.response.status_code
            err_text = ""
            try:
                err_text = e.response.text
            except Exception:
                err_text = str(e)
            
            # Auto-fallback if the specified model is not supported on this specific Groq tier
            if status == 404 and ("model_not_found" in err_text or "does not exist" in err_text) and self.model != FALLBACK_MODEL:
                logger.warning(
                    "Configured model '%s' not found on Groq account. Auto-falling back to available model '%s'",
                    self.model,
                    FALLBACK_MODEL,
                )
                self.model = FALLBACK_MODEL
                payload["model"] = FALLBACK_MODEL
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as fallback_client:
                        if stream:
                            return await self._stream_response(fallback_client, payload, t_start)
                        else:
                            return await self._sync_response(fallback_client, payload, t_start)
                except Exception as fallback_e:
                    logger.error("Fallback call also encountered error: %s", fallback_e)

            logger.error("Groq HTTP error %d: %s", status, err_text)
            rate_limit_headers = {
                k: v for k, v in e.response.headers.items()
                if "rate" in k.lower() or "retry" in k.lower() or "limit" in k.lower()
            }
            return GroqResponse(
                content="",
                model=self.model,
                tokens_in=0,
                tokens_out=0,
                first_token_ms=0,
                total_ms=total_ms,
                http_status=status,
                rate_limit_headers=rate_limit_headers,
                error=f"HTTP {status}: {err_text}",
                success=False,
            )
        except Exception as e:
            total_ms = (time.time() - t_start) * 1000
            logger.error("Groq unexpected error: %s", e)
            return GroqResponse(
                content="",
                model=self.model,
                tokens_in=0,
                tokens_out=0,
                first_token_ms=0,
                total_ms=total_ms,
                http_status=0,
                error=str(e),
                success=False,
            )

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        payload: dict,
        t_start: float,
    ) -> GroqResponse:
        """Handle streaming SSE response."""
        url = f"{self.api_base}/chat/completions"
        content_parts = []
        first_token_ms = 0.0
        http_status = 200
        rate_limit_headers = {}
        tokens_in = 0
        tokens_out = 0
        request_id = None

        async with client.stream("POST", url, headers=self._headers(), json=payload) as resp:
            http_status = resp.status_code
            rate_limit_headers = {
                k: v for k, v in resp.headers.items()
                if "rate" in k.lower() or "retry" in k.lower() or "limit" in k.lower() or "x-groq" in k.lower()
            }
            request_id = resp.headers.get("x-request-id")
            if resp.status_code >= 400:
                await resp.aread()
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"]
                        token = delta.get("content", "")
                        if token:
                            if not content_parts:
                                first_token_ms = (time.time() - t_start) * 1000
                            content_parts.append(token)
                        # Extract usage if present (some models send it in final chunk)
                        if "usage" in data:
                            usage = data["usage"]
                            tokens_in = usage.get("prompt_tokens", 0)
                            tokens_out = usage.get("completion_tokens", 0)
                    except (json.JSONDecodeError, KeyError):
                        pass

        total_ms = (time.time() - t_start) * 1000
        content = "".join(content_parts)
        return GroqResponse(
            content=content,
            model=self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            http_status=http_status,
            request_id=request_id,
            rate_limit_headers=rate_limit_headers,
            is_streaming=True,
            success=True,
        )

    async def _sync_response(
        self,
        client: httpx.AsyncClient,
        payload: dict,
        t_start: float,
    ) -> GroqResponse:
        """Handle non-streaming response."""
        url = f"{self.api_base}/chat/completions"
        resp = await client.post(url, headers=self._headers(), json=payload)
        http_status = resp.status_code
        rate_limit_headers = {
            k: v for k, v in resp.headers.items()
            if "rate" in k.lower() or "retry" in k.lower() or "limit" in k.lower() or "x-groq" in k.lower()
        }
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        total_ms = (time.time() - t_start) * 1000
        return GroqResponse(
            content=content,
            model=self.model,
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            first_token_ms=total_ms,  # no streaming, so first token = total
            total_ms=total_ms,
            http_status=http_status,
            request_id=resp.headers.get("x-request-id"),
            rate_limit_headers=rate_limit_headers,
            is_streaming=False,
            success=True,
        )

    async def list_models(self) -> list:
        """List available models from Groq."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{GROQ_API_BASE}/models",
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
