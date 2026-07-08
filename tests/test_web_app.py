from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

pytest.importorskip("streamlit")
pytest.importorskip("pandas")

from app import _run_cached  # noqa: E402
from research_brief.models import ResearchRequest  # noqa: E402


def test_streamlit_cached_run_mock_mode():
    request = ResearchRequest(
        topic="What are the tradeoffs of RAG vs fine-tuning?",
        max_sources=2,
        search_provider="mock",
        synthesis_mode="template",
        fetch_mode="snippet",
    )
    payload = _run_cached(request.model_dump_json(), _version="test")
    assert "[rag-1]" in payload


@pytest.mark.parametrize("topic,needle", [
    ("How do LLM agents use tools?", "[agent-1]"),
    ("What is MLOps CI/CD?", "[mlops-1]"),
])
def test_streamlit_cached_topics(topic: str, needle: str):
    request = ResearchRequest(
        topic=topic,
        max_sources=2,
        search_provider="mock",
        synthesis_mode="template",
        fetch_mode="snippet",
    )
    payload = _run_cached(request.model_dump_json(), _version="test")
    assert needle in payload
