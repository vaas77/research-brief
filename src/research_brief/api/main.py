from __future__ import annotations

from fastapi import FastAPI

from research_brief.agent.pipeline import run_research
from research_brief.models import ResearchRequest, ResearchResult

app = FastAPI(
    title="Research Brief API",
    description="Multi-step research agent with cited briefs and tool traces.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResult)
def research(request: ResearchRequest) -> ResearchResult:
    return run_research(request)
