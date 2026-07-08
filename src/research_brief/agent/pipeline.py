from __future__ import annotations

from research_brief.config import get_settings
from research_brief.models import ResearchRequest, ResearchResult
from research_brief.search.fetch import fetch_excerpts
from research_brief.search.provider import get_search_provider
from research_brief.synthesis.brief import synthesize_brief
from research_brief.tracing import current_trace_id, maybe_wrap_langsmith, start_span
from research_brief.validation.citations import validate_citations


def _run_research_impl(request: ResearchRequest) -> ResearchResult:
    settings = get_settings()
    fetch_mode = request.fetch_mode or settings.fetch_mode

    trace = []

    with start_span("search", attributes={"topic": request.topic}):
        search = get_search_provider(request.search_provider)
        sources, search_step = search.search(request.topic, request.max_sources)
    trace.append(search_step)
    search_provider = search_step.metadata.get("provider", "mock")
    if search_step.metadata.get("fallback"):
        search_provider = f"{search_provider}->{search_step.metadata['fallback']}"

    with start_span("fetch", attributes={"source_count": len(sources)}):
        sources, fetch_step = fetch_excerpts(sources, fetch_mode=fetch_mode)
    trace.append(fetch_step)

    with start_span("synthesize", attributes={"topic": request.topic}):
        brief, template, key_points, citations, synth_step, mode, llm_used = synthesize_brief(
            request.topic,
            sources,
            mode=request.synthesis_mode,
        )
    trace.append(synth_step)

    with start_span("validate", attributes={"citation_count": len(citations)}):
        coverage, validate_step = validate_citations(brief, citations)
    trace.append(validate_step)

    return ResearchResult(
        topic=request.topic,
        brief=brief,
        brief_template=template,
        key_points=key_points,
        sources=sources,
        citations=citations,
        trace=trace,
        search_provider=search_provider,
        synthesis_mode=mode,
        synthesis_llm_used=llm_used,
        citation_coverage_pct=coverage,
        trace_id=current_trace_id(),
    )


run_research = maybe_wrap_langsmith(_run_research_impl)
