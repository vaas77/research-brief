from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

from research_brief.agent.pipeline import run_research
from research_brief.jobs.store import JobStore
from research_brief.models import JobStatus, ResearchJob, ResearchRequest

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="research-job")
_store = JobStore()


def get_job_store() -> JobStore:
    return _store


def _execute_job(job_id: str) -> None:
    job = _store.get(job_id)
    _store.update_status(job_id, JobStatus.RUNNING)
    try:
        result = run_research(job.request)
        _store.complete(job_id, result)
    except Exception as exc:
        _store.update_status(job_id, JobStatus.FAILED, error=str(exc))


def enqueue_research(request: ResearchRequest) -> ResearchJob:
    job_id = str(uuid.uuid4())
    job = _store.create(job_id, request)
    _executor.submit(_execute_job, job_id)
    return job


def get_job(job_id: str) -> ResearchJob:
    return _store.get(job_id)
