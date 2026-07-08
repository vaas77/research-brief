from __future__ import annotations

import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from research_brief.models import JobStatus, ResearchJob, ResearchRequest, ResearchResult

DEFAULT_DB_PATH = Path(tempfile.gettempdir()) / "research_brief_jobs.db"


class JobStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create(self, job_id: str, request: ResearchRequest) -> ResearchJob:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_jobs (id, status, request_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, JobStatus.PENDING.value, request.model_dump_json(), now, now),
            )
        return self.get(job_id)

    def update_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE research_jobs
                SET status = ?, error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status.value, error, now, job_id),
            )

    def complete(self, job_id: str, result: ResearchResult) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE research_jobs
                SET status = ?, result_json = ?, error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (JobStatus.COMPLETED.value, result.model_dump_json(), now, job_id),
            )

    def get(self, job_id: str) -> ResearchJob:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Job '{job_id}' not found")

        request = ResearchRequest.model_validate_json(row["request_json"])
        result = (
            ResearchResult.model_validate_json(row["result_json"])
            if row["result_json"]
            else None
        )
        return ResearchJob(
            id=row["id"],
            status=JobStatus(row["status"]),
            request=request,
            result=result,
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
