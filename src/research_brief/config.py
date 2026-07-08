from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    search_provider: str = "auto"
    synthesis_mode: str = "template"
    tavily_api_key: str | None = None
    max_sources: int = 3
    fetch_mode: str = "live"
    max_excerpt_chars: int = 1500
    http_timeout_seconds: float = 20.0
    http_user_agent: str = "ResearchBrief/1.0 (+https://github.com/vaas77/research-brief)"
    job_store_path: str | None = None
    otel_enabled: bool = False
    otel_service_name: str = "research-brief"
    langsmith_api_key: str | None = None
    langsmith_project: str = "research-brief"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()


def langsmith_api_key(settings: Settings | None = None) -> str | None:
    settings = settings or get_settings()
    return (
        settings.langsmith_api_key
        or os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
    )


def langsmith_tracing_enabled(settings: Settings | None = None) -> bool:
    if os.getenv("LANGSMITH_TRACING", "").lower() in {"0", "false", "no"}:
        return False
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() in {"0", "false", "no"}:
        return False
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() in {"1", "true", "yes"}:
        return bool(langsmith_api_key(settings))
    if os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"}:
        return bool(langsmith_api_key(settings))
    return bool(langsmith_api_key(settings))
