from __future__ import annotations

import time

import pytest

from research_brief.jobs.worker import enqueue_research, get_job
from research_brief.models import JobStatus, ResearchRequest


def test_enqueue_and_complete_job(job_store):
    request = ResearchRequest(
        topic="What are the tradeoffs of RAG vs fine-tuning?",
        max_sources=2,
        search_provider="mock",
        synthesis_mode="template",
        fetch_mode="snippet",
    )
    job = enqueue_research(request)
    assert job.status == JobStatus.PENDING

    deadline = time.time() + 10
    final = job
    while time.time() < deadline:
        final = get_job(job.id)
        if final.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
            break
        time.sleep(0.2)

    assert final.status == JobStatus.COMPLETED
    assert final.result is not None
    assert "[rag-1]" in final.result.brief


def test_get_missing_job_raises():
    with pytest.raises(KeyError):
        get_job("missing-job-id")
