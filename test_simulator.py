"""
test_simulator.py - Unit tests for the stand-in simulator endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from simulator.standin_server import app, cfg


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "rpm_limit" in data


def test_config_update(client):
    res = client.post("/config", json={"rpm_limit": 45, "base_latency_ms": 200.0})
    assert res.status_code == 200
    assert res.json()["config"]["rpm_limit"] == 45


def test_chat_completions_non_streaming(client):
    client.post("/reset")
    payload = {
        "model": "stand-in-llama",
        "messages": [{"role": "user", "content": "What is attention?"}],
        "stream": False,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "attention" in data["choices"][0]["message"]["content"].lower()


def test_chat_completions_rate_limit_enforced(client):
    client.post("/reset")
    client.post("/config", json={"rpm_limit": 2})

    payload = {
        "model": "stand-in-llama",
        "messages": [{"role": "user", "content": "Test prompt"}],
        "stream": False,
    }

    # 1st: ok
    r1 = client.post("/v1/chat/completions", json=payload)
    assert r1.status_code == 200

    # 2nd: ok
    r2 = client.post("/v1/chat/completions", json=payload)
    assert r2.status_code == 200

    # 3rd: rate limit 429
    r3 = client.post("/v1/chat/completions", json=payload)
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers

    # Reset config back to default 30
    client.post("/config", json={"rpm_limit": 30})
    client.post("/reset")
