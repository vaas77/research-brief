from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
import yaml

from research_brief.agent.pipeline import run_research
from research_brief.models import Citation, ResearchRequest
from research_brief.search.fetch import fetch_excerpts, html_to_text
from research_brief.search.tavily import TavilySearchProvider
from research_brief.synthesis.brief import build_template_brief, synthesize_brief
from research_brief.validation.grounding import validate_llm_grounding


def _load_cases() -> dict:
    path = Path(__file__).resolve().parents[1] / "eval" / "test_cases.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_demo_pipeline_produces_trace_and_citations():
    result = run_research(
        ResearchRequest(
            topic="What are the tradeoffs of RAG vs fine-tuning?",
            max_sources=3,
            search_provider="mock",
            synthesis_mode="template",
            fetch_mode="snippet",
        )
    )
    assert len(result.sources) >= 2
    assert result.citation_coverage_pct >= 80.0
    assert result.brief_template
    assert [s.name.value for s in result.trace] == [
        "search",
        "fetch",
        "synthesize",
        "validate",
    ]
    assert "[rag-1]" in result.brief


@pytest.mark.parametrize("case_id", list(_load_cases().keys()))
def test_eval_cases_from_yaml(case_id: str):
    case = _load_cases()[case_id]
    result = run_research(
        ResearchRequest(
            topic=case["topic"],
            max_sources=case.get("max_sources", 3),
            search_provider="mock",
            synthesis_mode="template",
            fetch_mode="snippet",
        )
    )
    exp = case["expectations"]
    assert len(result.sources) >= exp["min_sources"]
    assert result.citation_coverage_pct >= exp["min_citation_coverage_pct"]
    assert [s.name.value for s in result.trace] == exp["trace_steps"]
    for phrase in exp["brief_contains"]:
        assert phrase in result.brief


def test_html_to_text_strips_tags():
    html = "<html><body><h1>Title</h1><p>Hello <b>world</b>.</p></body></html>"
    text = html_to_text(html)
    assert "Title" in text
    assert "Hello world" in text
    assert "<" not in text


def test_tavily_search_parses_results(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "title": "Example",
                        "url": "https://example.com/a",
                        "content": "Example content about RAG.",
                    }
                ]
            }

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    with patch("research_brief.search.tavily.httpx.post", return_value=FakeResponse()):
        sources, step = TavilySearchProvider().search("RAG systems", 1)
    assert len(sources) == 1
    assert sources[0].url == "https://example.com/a"
    assert step.metadata["provider"] == "tavily"


def test_tavily_missing_key_falls_back_to_mock():
    with patch("research_brief.search.tavily.get_settings") as mock_settings:
        mock_settings.return_value.tavily_api_key = None
        sources, step = TavilySearchProvider().search("RAG", 2)
    assert len(sources) >= 1
    assert "mock" in step.detail.lower()


def test_fetch_live_mode_uses_http(monkeypatch):
    from research_brief.models import SourceDocument

    source = SourceDocument(
        id="src-test",
        title="Example",
        url="https://example.com",
        snippet="snippet only",
    )

    def fake_fetch(url: str, max_chars: int) -> str:
        return "Fetched excerpt about RAG and retrieval quality."

    monkeypatch.setattr("research_brief.search.fetch.fetch_url_text", fake_fetch)
    enriched, step = fetch_excerpts([source], fetch_mode="live")
    assert "Fetched excerpt" in enriched[0].excerpt
    assert step.metadata["fetched"] == 1


def test_grounding_validation_requires_source_ids():
    template = "Point A [rag-1]. Point B [rag-2]."
    citations = [
        Citation(source_id="rag-1", label="A", url="https://a"),
        Citation(source_id="rag-2", label="B", url="https://b"),
    ]
    assert validate_llm_grounding(template, template, citations)
    assert not validate_llm_grounding(template, "Missing ids", citations)


def test_openai_synthesis_falls_back_without_api_key():
    from research_brief.search.mock import get_mock_sources

    sources = get_mock_sources("RAG", 2)
    with patch("research_brief.synthesis.brief.get_settings") as mock_settings:
        mock_settings.return_value.openai_api_key = None
        mock_settings.return_value.synthesis_mode = "template"
        brief, template, _, _, step, mode, llm_used = synthesize_brief(
            "RAG",
            sources,
            mode="openai",
        )
    assert mode == "template"
    assert llm_used is False
    assert brief == template
