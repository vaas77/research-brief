from __future__ import annotations

import time
from typing import Protocol

from research_brief.config import get_settings
from research_brief.models import AgentStep, AgentStepName, SourceDocument
from research_brief.search.mock import get_mock_sources
from research_brief.search.tavily import TavilySearchProvider


class SearchProvider(Protocol):
    def search(self, topic: str, max_sources: int) -> tuple[list[SourceDocument], AgentStep]:
        ...


class MockSearchProvider:
    def search(self, topic: str, max_sources: int) -> tuple[list[SourceDocument], AgentStep]:
        start = time.perf_counter()
        sources = get_mock_sources(topic, max_sources)
        duration_ms = int((time.perf_counter() - start) * 1000)
        step = AgentStep(
            name=AgentStepName.SEARCH,
            status="ok",
            detail=f"Found {len(sources)} mock source(s) for topic.",
            duration_ms=duration_ms,
            metadata={"provider": "mock", "topic": topic},
        )
        return sources, step


def get_search_provider(name: str | None = None) -> SearchProvider:
    settings = get_settings()
    provider = (name or settings.search_provider).lower()
    if provider == "auto":
        provider = "tavily" if settings.tavily_api_key else "mock"
    if provider == "mock":
        return MockSearchProvider()
    if provider == "tavily":
        return TavilySearchProvider()
    raise ValueError(f"Unknown search provider '{provider}'. Use auto, mock, or tavily.")
