"""
retry_manager.py - Exponential backoff retry manager with jitter and Retry-After support.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Any, Optional, Type, Tuple

logger = logging.getLogger(__name__)

# Error types that should NOT be retried
NON_RETRYABLE_HTTP = {400, 401, 403, 404, 422}
# Error types that SHOULD be retried
RETRYABLE_HTTP = {429, 500, 502, 503, 504}


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    backoff_factor: float = 2.0
    jitter: float = 0.5  # random fraction of delay
    respect_retry_after: bool = True


@dataclass
class RetryOutcome:
    success: bool
    result: Any
    attempts: int
    total_wait_ms: float
    gave_up: bool = False
    final_error: Optional[str] = None


class RetryManager:
    """
    Exponential backoff retry manager.
    
    Example of when retrying makes performance WORSE:
    
    If a server is overloaded and all clients retry simultaneously after
    their backoff period, they all hit the server at the same time — creating
    a "retry storm" that makes the overload worse. This is known as the
    "thundering herd" problem. Jitter is specifically added to prevent this.
    
    However, if the server has a strict rate limit window (e.g., 1 req/sec),
    and a request is retried immediately after failure, it will ALWAYS fail
    again until the window resets. In this case, retrying increases total
    request time and wastes quota. The correct behavior is to wait for the
    window to reset (Retry-After header) or give up and report the failure.
    """

    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()

    def _compute_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """
        Compute delay for the next retry.
        
        Uses exponential backoff with jitter:
        delay = min(base * factor^attempt, max_delay) * (1 + random jitter)
        """
        if retry_after and self.config.respect_retry_after:
            return min(retry_after, self.config.max_delay_s)

        base = self.config.base_delay_s * (self.config.backoff_factor ** attempt)
        base = min(base, self.config.max_delay_s)
        jitter = random.uniform(0, self.config.jitter * base)
        return base + jitter

    def is_retryable(self, http_status: int, error: Optional[str] = None) -> bool:
        """Determine if an error is retryable."""
        if http_status in NON_RETRYABLE_HTTP:
            return False
        if http_status in RETRYABLE_HTTP:
            return True
        if http_status == 408:  # timeout
            return True
        if http_status == 0:  # network error
            return True
        return False

    async def execute_with_retry(
        self,
        fn: Callable,
        *args,
        request_id: str = "unknown",
        **kwargs,
    ) -> RetryOutcome:
        """
        Execute a coroutine with retry logic.
        
        The function `fn` must return an object with:
        - .success: bool
        - .http_status: int
        - .error: Optional[str]
        - .rate_limit_headers: dict (may contain Retry-After)
        """
        total_wait_ms = 0.0
        last_error = None
        result = None

        for attempt in range(self.config.max_retries + 1):
            if attempt > 0:
                logger.info(
                    "Retry %d/%d for request %s",
                    attempt, self.config.max_retries, request_id,
                )

            result = await fn(*args, **kwargs)

            if result.success:
                return RetryOutcome(
                    success=True,
                    result=result,
                    attempts=attempt + 1,
                    total_wait_ms=total_wait_ms,
                )

            last_error = result.error
            http_status = result.http_status

            if not self.is_retryable(http_status, last_error):
                logger.warning(
                    "Non-retryable error (HTTP %d) for %s: %s",
                    http_status, request_id, last_error,
                )
                return RetryOutcome(
                    success=False,
                    result=result,
                    attempts=attempt + 1,
                    total_wait_ms=total_wait_ms,
                    gave_up=True,
                    final_error=last_error,
                )

            if attempt >= self.config.max_retries:
                break

            # Extract Retry-After header if present
            retry_after = None
            headers = getattr(result, "rate_limit_headers", {})
            if "retry-after" in headers:
                try:
                    retry_after = float(headers["retry-after"])
                except (ValueError, TypeError):
                    pass
            if "x-ratelimit-reset-requests" in headers:
                try:
                    retry_after = float(headers["x-ratelimit-reset-requests"])
                except (ValueError, TypeError):
                    pass

            delay = self._compute_delay(attempt, retry_after)
            logger.info(
                "Will retry %s in %.2fs (HTTP %d, error: %s)",
                request_id, delay, http_status, last_error,
            )
            t_wait_start = time.time()
            await asyncio.sleep(delay)
            total_wait_ms += (time.time() - t_wait_start) * 1000

        logger.error(
            "Gave up on %s after %d attempts. Last error: %s",
            request_id, self.config.max_retries + 1, last_error,
        )
        return RetryOutcome(
            success=False,
            result=result,
            attempts=self.config.max_retries + 1,
            total_wait_ms=total_wait_ms,
            gave_up=True,
            final_error=last_error,
        )
