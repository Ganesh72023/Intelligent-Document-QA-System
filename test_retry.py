"""
test_retry.py - Tests for the exponential backoff retry manager with jitter and Retry-After.
"""

import asyncio
import pytest
from src.retry_manager import RetryManager, RetryConfig


class MockCall:
    def __init__(self, fail_count=2, status_code=429):
        self.attempts = 0
        self.fail_count = fail_count
        self.status_code = status_code

    async def __call__(self):
        self.attempts += 1
        if self.attempts <= self.fail_count:
            class MockResp:
                success = False
                http_status = self.status_code
                error = "Rate limit hit"
                rate_limit_headers = {"retry-after": "0.1"}
            return MockResp()
        else:
            class MockSuccess:
                success = True
                http_status = 200
                error = None
                rate_limit_headers = {}
            return MockSuccess()


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failure():
    mgr = RetryManager(RetryConfig(max_retries=3, base_delay_s=0.05, max_delay_s=0.2))
    mock = MockCall(fail_count=2, status_code=429)

    outcome = await mgr.execute_with_retry(mock, request_id="req-test-1")
    assert outcome.success is True
    assert outcome.attempts == 3
    assert outcome.gave_up is False


@pytest.mark.asyncio
async def test_retry_stops_on_non_retryable_error():
    mgr = RetryManager(RetryConfig(max_retries=3, base_delay_s=0.05))
    mock = MockCall(fail_count=2, status_code=401)  # Unauthorized: non-retryable

    outcome = await mgr.execute_with_retry(mock, request_id="req-test-2")
    assert outcome.success is False
    assert outcome.attempts == 1  # Should not retry 401
    assert outcome.gave_up is True
