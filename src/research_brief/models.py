from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AgentStepName(str, Enum):
    SEARCH = "search"
    FETCH = "fetch"
    SYNTHESIZE = "synthesize"
    VALIDATE = "validate"


class AgentStep(BaseModel):
    name: AgentStepName
    status: Literal["ok", "error", "skipped"] = "ok"
    detail: str = ""
    duration_ms: int = 0
    metadata: dict = Field(default_factory=dict)


class SourceDocument(BaseModel):
    id: str
    title: str
    url: str
    snippet: str
    excerpt: str = ""


class Citation(BaseModel):
    source_id: str
    label: str
    url: str


class ResearchRequest(BaseModel):
    topic: str
    max_sources: int = 3
    synthesis_mode: Literal["template", "openai"] | None = None
    search_provider: Literal["auto", "mock", "tavily"] | None = None
    fetch_mode: Literal["live", "snippet"] | None = None


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchJob(BaseModel):
    id: str
    status: JobStatus
    request: ResearchRequest
    result: ResearchResult | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ResearchJobCreated(BaseModel):
    job_id: str
    status: JobStatus


class ResearchResult(BaseModel):
    topic: str
    brief: str
    brief_template: str = ""
    key_points: list[str] = Field(default_factory=list)
    sources: list[SourceDocument] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    trace: list[AgentStep] = Field(default_factory=list)
    search_provider: str = "mock"
    synthesis_mode: str = "template"
    synthesis_llm_used: bool = False
    citation_coverage_pct: float = 0.0
    trace_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BriefDraft(BaseModel):
    topic: str
    sources: list[SourceDocument]
    sections: list[str] = Field(default_factory=list)


ResearchJob.model_rebuild()
