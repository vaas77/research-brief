from __future__ import annotations

import hashlib
import time

import httpx

from research_brief.config import get_settings
from research_brief.models import AgentStep, AgentStepName, SourceDocument


def _source_id(url: str) -> str:
    return "src-" + hashlib.sha256(url.encode()).hexdigest()[:10]


class TavilySearchProvider:
    def search(self, topic: str, max_sources: int) -> tuple[list[SourceDocument], AgentStep]:
        settings = get_settings()
        api_key = settings.tavily_api_key
        if not api_key:
            from research_brief.search.provider import MockSearchProvider

            sources, step = MockSearchProvider().search(topic, max_sources)
            step.detail = "Tavily API key missing; used mock search."
            step.metadata = {**step.metadata, "fallback": "mock"}
            return sources, step

        start = time.perf_counter()
        try:
            response = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": topic,
                    "max_results": max_sources,
                    "search_depth": "basic",
                    "include_answer": False,
                },
                timeout=settings.http_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            sources: list[SourceDocument] = []
            for item in payload.get("results", [])[:max_sources]:
                url = item.get("url", "")
                if not url:
                    continue
                sources.append(
                    SourceDocument(
                        id=_source_id(url),
                        title=item.get("title") or "Untitled source",
                        url=url,
                        snippet=(item.get("content") or "")[:600],
                    )
                )
            status = "ok" if sources else "error"
            detail = f"Tavily returned {len(sources)} source(s)."
            metadata = {"provider": "tavily", "topic": topic, "result_count": len(sources)}
        except Exception as exc:
            from research_brief.search.provider import MockSearchProvider

            sources, step = MockSearchProvider().search(topic, max_sources)
            duration_ms = int((time.perf_counter() - start) * 1000)
            return sources, AgentStep(
                name=AgentStepName.SEARCH,
                status="ok",
                detail=f"Tavily failed ({exc}); used mock search.",
                duration_ms=duration_ms,
                metadata={"provider": "tavily", "fallback": "mock", "error": str(exc)},
            )

        duration_ms = int((time.perf_counter() - start) * 1000)
        step = AgentStep(
            name=AgentStepName.SEARCH,
            status=status,
            detail=detail,
            duration_ms=duration_ms,
            metadata=metadata,
        )
        return sources, step
