"""
test_rate_limiter.py - Tests for the rolling window rate limiter.
"""

import asyncio
import pytest
import time
from src.rate_limiter import RateLimiter, RateLimitConfig


@pytest.mark.asyncio
async def test_rate_limiter_allows_requests_within_limit():
    cfg = RateLimitConfig(rpm=10, tpm=5000, rpd=100, tpd=50000, max_wait_s=2.0)
    limiter = RateLimiter(cfg)

    # Acquire 3 requests
    for _ in range(3):
        res = await limiter.acquire(estimated_tokens=100)
        assert res is True

    state = limiter.get_state()
    assert state["rpm_used"] == 3
    assert state["tpm_used"] == 300
    assert state["daily_exhausted"] is False


@pytest.mark.asyncio
async def test_rate_limiter_detects_daily_budget_exhaustion():
    cfg = RateLimitConfig(rpm=100, tpm=50000, rpd=2, tpd=50000, max_wait_s=0.5)
    limiter = RateLimiter(cfg)

    # First two requests succeed
    assert await limiter.acquire(estimated_tokens=50) is True
    assert await limiter.acquire(estimated_tokens=50) is True

    # Third request must be rejected because daily limit is reached
    assert limiter.is_daily_exhausted() is True
    assert await limiter.acquire(estimated_tokens=50) is False
