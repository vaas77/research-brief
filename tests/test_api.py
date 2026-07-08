from fastapi.testclient import TestClient

from research_brief.api.main import app


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
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["topic"].startswith("How do LLM agents")
    assert len(payload["trace"]) == 4
    assert "[agent-1]" in payload["brief"]
