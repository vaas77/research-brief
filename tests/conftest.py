from __future__ import annotations

import pytest

from research_brief.jobs.store import JobStore


@pytest.fixture()
def job_store(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    import research_brief.jobs.worker as worker

    original = worker._store
    worker._store = store
    yield store
    worker._store = original
