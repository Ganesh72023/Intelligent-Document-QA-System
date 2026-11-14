"""
rate_limiter.py - Token bucket rate limiter tracking RPM, TPM, RPD, TPD.
"""

import time
import logging
import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    rpm: int = 30            # requests per minute
    tpm: int = 14400         # tokens per minute
    rpd: int = 14400         # requests per day
    tpd: int = 500000        # tokens per day
    max_wait_s: float = 60.0  # max seconds to wait for a slot


@dataclass
class RateLimitState:
    rpm_remaining: int = 0
    tpm_remaining: int = 0
    rpd_remaining: int = 0
    tpd_remaining: int = 0
    daily_budget_exhausted: bool = False
    last_reset_minute: float = 0.0
    last_reset_day: float = 0.0


class RateLimiter:
    """
    Rolling-window rate limiter for RPM, TPM, RPD, TPD.
    
    Uses deques to track timestamps of recent requests/tokens
    within the last 60 seconds (minute) or 86400 seconds (day).
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._lock = asyncio.Lock()

        # Rolling windows: deque of (timestamp, value) tuples
        self._minute_requests: deque = deque()
        self._minute_tokens: deque = deque()
        self._day_requests: deque = deque()
        self._day_tokens: deque = deque()

    def _prune(self, dq: deque, window_s: float) -> None:
        """Remove entries older than window_s seconds."""
        cutoff = time.time() - window_s
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _count(self, dq: deque) -> float:
        """Sum values in the deque."""
        return sum(v for _, v in dq)

    def is_daily_exhausted(self) -> bool:
        self._prune(self._day_requests, 86400)
        self._prune(self._day_tokens, 86400)
        req_count = self._count(self._day_requests)
        tok_count = self._count(self._day_tokens)
        return req_count >= self.config.rpd or tok_count >= self.config.tpd

    async def acquire(self, estimated_tokens: int = 500) -> bool:
        """
        Wait until a request slot is available.
        
        Returns True if acquired, False if daily budget is exhausted.
        Raises asyncio.TimeoutError if max_wait exceeded.
        """
        async with self._lock:
            # Check daily limits first
            if self.is_daily_exhausted():
                logger.warning("Daily budget exhausted. Request rejected.")
                return False

            deadline = time.time() + self.config.max_wait_s
            while True:
                now = time.time()
                if now > deadline:
                    raise asyncio.TimeoutError("Rate limit wait exceeded max_wait_s")

                self._prune(self._minute_requests, 60)
                self._prune(self._minute_tokens, 60)

                rpm_used = self._count(self._minute_requests)
                tpm_used = self._count(self._minute_tokens)

                rpm_ok = rpm_used < self.config.rpm
                tpm_ok = tpm_used + estimated_tokens <= self.config.tpm

                if rpm_ok and tpm_ok:
                    # Record the request
                    t = time.time()
                    self._minute_requests.append((t, 1))
                    self._minute_tokens.append((t, estimated_tokens))
                    self._day_requests.append((t, 1))
                    self._day_tokens.append((t, estimated_tokens))
                    logger.debug(
                        "Rate limiter: RPM=%d/%d, TPM=%d/%d",
                        rpm_used + 1, self.config.rpm,
                        tpm_used + estimated_tokens, self.config.tpm,
                    )
                    return True

                # Calculate wait time
                wait = 0.1
                if not rpm_ok and self._minute_requests:
                    oldest = self._minute_requests[0][0]
                    wait = max(wait, oldest + 60.0 - time.time())
                if not tpm_ok and self._minute_tokens:
                    oldest = self._minute_tokens[0][0]
                    wait = max(wait, oldest + 60.0 - time.time())

                wait = min(wait, deadline - time.time())
                if wait <= 0:
                    raise asyncio.TimeoutError("Rate limit wait exceeded")

                logger.info(
                    "Rate limited (RPM=%d/%d, TPM=%d/%d). Waiting %.1fs",
                    rpm_used, self.config.rpm, tpm_used, self.config.tpm, wait,
                )
                await asyncio.sleep(wait)

    def get_state(self) -> dict:
        """Return current usage state."""
        self._prune(self._minute_requests, 60)
        self._prune(self._minute_tokens, 60)
        self._prune(self._day_requests, 86400)
        self._prune(self._day_tokens, 86400)
        return {
            "rpm_used": self._count(self._minute_requests),
            "rpm_limit": self.config.rpm,
            "tpm_used": self._count(self._minute_tokens),
            "tpm_limit": self.config.tpm,
            "rpd_used": self._count(self._day_requests),
            "rpd_limit": self.config.rpd,
            "tpd_used": self._count(self._day_tokens),
            "tpd_limit": self.config.tpd,
            "daily_exhausted": self.is_daily_exhausted(),
        }
