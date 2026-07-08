from __future__ import annotations

from research_brief.models import ResearchRequest, ResearchResult
from research_brief.search.fetch import fetch_excerpts
from research_brief.search.provider import get_search_provider
from research_brief.synthesis.brief import synthesize_brief
from research_brief.validation.citations import validate_citations


def run_research(request: ResearchRequest) -> ResearchResult:
    trace = []

    search = get_search_provider(request.search_provider)
    sources, search_step = search.search(request.topic, request.max_sources)
    trace.append(search_step)

    sources, fetch_step = fetch_excerpts(sources)
    trace.append(fetch_step)

    brief, key_points, citations, synth_step, mode = synthesize_brief(
        request.topic,
        sources,
        mode=request.synthesis_mode,
    )
    trace.append(synth_step)

    coverage, validate_step = validate_citations(brief, citations)
    trace.append(validate_step)

    return ResearchResult(
        topic=request.topic,
        brief=brief,
        key_points=key_points,
        sources=sources,
        citations=citations,
        trace=trace,
        synthesis_mode=mode,
        citation_coverage_pct=coverage,
    )
