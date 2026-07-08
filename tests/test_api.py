from fastapi.testclient import TestClient

import time

from research_brief.api.main import app
from research_brief.models import JobStatus


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_research_endpoint():
    client = TestClient(app)
    response = client.post(
        "/research",
        json={
            "topic": "How do LLM agents use tools?",
            "max_sources": 2,
            "search_provider": "mock",
            "synthesis_mode": "template",
            "fetch_mode": "snippet",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["topic"].startswith("How do LLM agents")
    assert len(payload["trace"]) == 4
    assert payload["brief_template"]
    assert "[agent-1]" in payload["brief"]


def test_research_job_endpoints(job_store):
    client = TestClient(app)
    create = client.post(
        "/research/jobs",
        json={
            "topic": "How do LLM agents use tools?",
            "max_sources": 2,
            "search_provider": "mock",
            "synthesis_mode": "template",
            "fetch_mode": "snippet",
        },
    )
    assert create.status_code == 200
    job_id = create.json()["job_id"]

    final = None
    deadline = time.time() + 10
    while time.time() < deadline:
        status = client.get(f"/research/jobs/{job_id}")
        assert status.status_code == 200
        final = status.json()
        if final["status"] in {JobStatus.COMPLETED.value, JobStatus.FAILED.value}:
            break
        time.sleep(0.2)

    assert final is not None
    assert final["status"] == JobStatus.COMPLETED.value
    assert "[agent-1]" in final["result"]["brief"]
