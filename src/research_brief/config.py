from __future__ import annotations

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
