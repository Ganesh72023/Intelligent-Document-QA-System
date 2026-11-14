"""
simulator_config.py - Configuration profiles for the stand-in model simulator.

Each profile corresponds to one of the five required load test scenarios:
1. Steady load (normal traffic)
2. Sudden burst (high request volume over a short time)
3. Sustained overload (traffic continuously exceeds capacity)
4. Per-minute limit halved (50% reduction in RPM limit)
5. Daily budget exhausted (budget exhausted mid-run)
"""

SCENARIO_CONFIGS = {
    "steady_load": {
        "description": "Normal continuous flow of requests.",
        "rpm_limit": 30,
        "tpm_limit": 14400,
        "daily_request_budget": 10000,
        "daily_token_budget": 500000,
        "failure_probability": 0.0,
        "rate_limit_probability": 0.0,
        "base_latency_ms": 350.0,
        "latency_variation_ms": 150.0,
        "max_concurrent": 10,
    },
    "sudden_burst": {
        "description": "Many requests over a short period exceeding RPM.",
        "rpm_limit": 30,
        "tpm_limit": 14400,
        "daily_request_budget": 10000,
        "daily_token_budget": 500000,
        "failure_probability": 0.02,
        "rate_limit_probability": 0.0,
        "base_latency_ms": 450.0,
        "latency_variation_ms": 250.0,
        "max_concurrent": 10,
    },
    "sustained_overload": {
        "description": "Traffic above capacity for extended period.",
        "rpm_limit": 20,
        "tpm_limit": 8000,
        "daily_request_budget": 10000,
        "daily_token_budget": 500000,
        "failure_probability": 0.05,
        "rate_limit_probability": 0.0,
        "base_latency_ms": 600.0,
        "latency_variation_ms": 300.0,
        "max_concurrent": 8,
    },
    "rpm_halved": {
        "description": "Per-minute limit reduced by 50% (15 RPM).",
        "rpm_limit": 15,
        "tpm_limit": 14400,
        "daily_request_budget": 10000,
        "daily_token_budget": 500000,
        "failure_probability": 0.0,
        "rate_limit_probability": 0.0,
        "base_latency_ms": 350.0,
        "latency_variation_ms": 150.0,
        "max_concurrent": 10,
    },
    "daily_budget_exhausted": {
        "description": "Daily budget is consumed during execution.",
        "rpm_limit": 30,
        "tpm_limit": 14400,
        "daily_request_budget": 12,  # Set low so it exhausts after 12 requests
        "daily_token_budget": 6000,
        "failure_probability": 0.0,
        "rate_limit_probability": 0.0,
        "base_latency_ms": 350.0,
        "latency_variation_ms": 150.0,
        "max_concurrent": 10,
    },
}
