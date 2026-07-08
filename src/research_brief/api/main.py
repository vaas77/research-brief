from __future__ import annotations

from fastapi import FastAPI, HTTPException

from research_brief.agent.pipeline import run_research
from research_brief.jobs.worker import enqueue_research, get_job
from research_brief.models import (
    JobStatus,
    ResearchJob,
    ResearchJobCreated,
    ResearchRequest,
    ResearchResult,
)
from research_brief.tracing import configure_tracing

configure_tracing()

app = FastAPI(
    title="Research Brief API",
    description="Multi-step research agent with cited briefs and tool traces.",
    version="0.3.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResult)
def research(request: ResearchRequest) -> ResearchResult:
    return run_research(request)


@app.post("/research/jobs", response_model=ResearchJobCreated)
def research_job(request: ResearchRequest) -> ResearchJobCreated:
    job = enqueue_research(request)
    return ResearchJobCreated(job_id=job.id, status=job.status)


@app.get("/research/jobs/{job_id}", response_model=ResearchJob)
def research_job_status(job_id: str) -> ResearchJob:
    try:
        return get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
