from __future__ import annotations

import time

from research_brief.models import AgentStep, AgentStepName, SourceDocument


def fetch_excerpts(sources: list[SourceDocument]) -> tuple[list[SourceDocument], AgentStep]:
    """Normalize source excerpts for synthesis (mock fetch uses preloaded text)."""
    start = time.perf_counter()
    enriched: list[SourceDocument] = []
    for source in sources:
        excerpt = source.excerpt or source.snippet
        enriched.append(source.model_copy(update={"excerpt": excerpt}))
    duration_ms = int((time.perf_counter() - start) * 1000)
    step = AgentStep(
        name=AgentStepName.FETCH,
        status="ok",
        detail=f"Prepared excerpts for {len(enriched)} source(s).",
        duration_ms=duration_ms,
        metadata={"source_ids": [s.id for s in enriched]},
    )
    return enriched, step
