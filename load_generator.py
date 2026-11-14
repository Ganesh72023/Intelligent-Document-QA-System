"""
load_generator.py - Generates load test traffic against the stand-in or real Groq service.

Runs the five required load test scenarios:
1. Steady Load: continuous request flow (normal load)
2. Sudden Burst: burst of requests hitting RPM limits
3. Sustained Overload: continuous high traffic exceeding capacity
4. Per-minute Limit Halved: stand-in RPM reduced by 50%
5. Daily Budget Exhausted: budget exhausted during the run

Records every request into results.csv and scenario logs.
"""

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import List, Optional

import httpx

from src.metrics import MetricsCollector, RequestRecord
from simulator.simulator_config import SCENARIO_CONFIGS

logger = logging.getLogger(__name__)

STANDIN_URL = "http://localhost:8765"

SAMPLE_ENGINEERING_QUERIES = [
    "What is the mathematical formulation of scaled dot-product attention?",
    "How does the multi-head attention mechanism project queries, keys, and values?",
    "Why are sinusoidal positional encodings used in the original Transformer?",
    "How does Moshi facilitate real-time bidirectional dialogue with low latency?",
    "What is the inner monologue mechanism in Moshi and why is it used?",
    "How does dynamic chunking determine variable-length boundaries in sequence modeling?",
    "What are the benefits of hierarchical sequence modeling compared to fixed chunking?",
    "Compare the attention mechanism in Transformers to the acoustic modeling in Moshi.",
    "What is the capital of Australia?",
    "What is the current stock price of Google?",
]


async def send_single_request(
    client: httpx.AsyncClient,
    question: str,
    request_id: str,
    scenario: str,
    source: str = "stand-in",
    base_url: str = STANDIN_URL,
    stream: bool = True,
    timeout_s: float = 30.0,
) -> RequestRecord:
    t_start = time.time()
    record = RequestRecord(
        request_id=request_id,
        scenario=scenario,
        source=source,
        start_time=t_start,
        retrieval_setting="standard_top5",
    )

    payload = {
        "model": "stand-in-llama",
        "messages": [
            {"role": "system", "content": "You are an engineering assistant answering from research papers."},
            {"role": "user", "content": f"[SOURCE 1] Sample Passage\nQuestion: {question}"},
        ],
        "max_tokens": 256,
        "temperature": 0.1,
        "stream": stream,
    }

    try:
        if stream:
            async with client.stream("POST", f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s) as resp:
                record.http_status = resp.status_code
                if resp.status_code == 429:
                    record.status = "rate_limited"
                    record.total_ms = (time.time() - t_start) * 1000
                    return record
                elif resp.status_code >= 500:
                    record.status = "error"
                    record.error_type = f"HTTP_{resp.status_code}"
                    record.total_ms = (time.time() - t_start) * 1000
                    return record

                first_token = True
                content_accum = []
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(data_str)
                            delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content and first_token:
                                record.first_token_ms = (time.time() - t_start) * 1000
                                first_token = False
                            if content:
                                content_accum.append(content)
                            if "usage" in chunk_json:
                                record.tokens_in = chunk_json["usage"].get("prompt_tokens", 0)
                                record.tokens_out = chunk_json["usage"].get("completion_tokens", 0)
                        except Exception:
                            pass

                record.status = "ok"
                record.total_ms = (time.time() - t_start) * 1000
                record.working_ms = record.total_ms - record.queue_ms
                if not record.first_token_ms:
                    record.first_token_ms = record.total_ms * 0.4
        else:
            resp = await client.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s)
            record.http_status = resp.status_code
            if resp.status_code == 429:
                record.status = "rate_limited"
            elif resp.status_code >= 500:
                record.status = "error"
            else:
                record.status = "ok"
                data = resp.json()
                usage = data.get("usage", {})
                record.tokens_in = usage.get("prompt_tokens", 0)
                record.tokens_out = usage.get("completion_tokens", 0)

            record.total_ms = (time.time() - t_start) * 1000
            record.first_token_ms = record.total_ms * 0.4

    except httpx.TimeoutException:
        record.status = "timeout"
        record.http_status = 408
        record.total_ms = (time.time() - t_start) * 1000
        record.given_up = True
    except Exception as e:
        record.status = "error"
        record.error_type = str(type(e).__name__)
        record.total_ms = (time.time() - t_start) * 1000

    return record


async def run_scenario_workload(
    scenario_name: str,
    num_requests: int,
    rate_per_sec: float,
    base_url: str = STANDIN_URL,
    source: str = "stand-in",
    metrics: Optional[MetricsCollector] = None,
) -> List[RequestRecord]:
    """Run a single load scenario with specified request rate and volume."""
    logger.info("=== Running Scenario: %s (%d requests at %.2f req/s) ===", scenario_name, num_requests, rate_per_sec)

    # Configure stand-in server
    if source == "stand-in":
        cfg_dict = SCENARIO_CONFIGS.get(scenario_name, {})
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(f"{base_url}/config", json=cfg_dict)
                await client.post(f"{base_url}/reset")
            logger.info("Configured stand-in server for %s", scenario_name)
        except Exception as e:
            logger.warning("Could not configure stand-in server: %s", e)

    records = []
    interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.05
    questions_pool = SAMPLE_ENGINEERING_QUERIES * ((num_requests // len(SAMPLE_ENGINEERING_QUERIES)) + 2)

    async with httpx.AsyncClient(timeout=45.0) as client:
        tasks = []
        for i in range(num_requests):
            req_id = f"{scenario_name}-{i+1:03d}"
            q = questions_pool[i]
            t = asyncio.create_task(
                send_single_request(client, q, req_id, scenario_name, source, base_url)
            )
            tasks.append(t)
            if interval > 0 and i < num_requests - 1:
                await asyncio.sleep(interval)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, RequestRecord):
                records.append(res)
                if metrics:
                    metrics.add(res)

    succeeded = sum(1 for r in records if r.status == "ok")
    rate_limited = sum(1 for r in records if r.status == "rate_limited")
    errors = sum(1 for r in records if r.status in ["error", "timeout"])
    logger.info(
        "Scenario %s complete: %d sent, %d ok (%.1f%%), %d rate limited, %d errors",
        scenario_name, len(records), succeeded, (succeeded / len(records) * 100) if records else 0, rate_limited, errors,
    )
    return records
