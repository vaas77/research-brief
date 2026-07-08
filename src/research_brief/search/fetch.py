from __future__ import annotations

import re
import time

import httpx

from research_brief.config import get_settings
from research_brief.models import AgentStep, AgentStepName, SourceDocument

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|noscript)[^>]*>.*?</\1>",
    flags=re.IGNORECASE | re.DOTALL,
)


def html_to_text(html: str) -> str:
    cleaned = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _HTML_TAG_RE.sub(" ", cleaned)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_url_text(url: str, max_chars: int) -> str:
    settings = get_settings()
    response = httpx.get(
        url,
        follow_redirects=True,
        timeout=settings.http_timeout_seconds,
        headers={"User-Agent": settings.http_user_agent},
    )
    response.raise_for_status()
    text = html_to_text(response.text)
    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def _needs_live_fetch(source: SourceDocument) -> bool:
    if source.excerpt and len(source.excerpt) >= 120:
        return False
    if not source.url.startswith(("http://", "https://")):
        return False
    return True


def fetch_excerpts(
    sources: list[SourceDocument],
    *,
    fetch_mode: str | None = None,
) -> tuple[list[SourceDocument], AgentStep]:
    """Fetch page excerpts for live URLs; reuse bundled snippets when present."""
    start = time.perf_counter()
    settings = get_settings()
    mode = (fetch_mode or settings.fetch_mode).lower()
    enriched: list[SourceDocument] = []
    fetched = 0
    failed = 0

    for source in sources:
        excerpt = source.excerpt
        if mode == "live" and _needs_live_fetch(source):
            try:
                excerpt = fetch_url_text(source.url, settings.max_excerpt_chars)
                fetched += 1
            except Exception:
                failed += 1
                excerpt = source.excerpt or source.snippet

        if not excerpt:
            excerpt = source.snippet

        enriched.append(source.model_copy(update={"excerpt": excerpt}))

    duration_ms = int((time.perf_counter() - start) * 1000)
    detail = f"Prepared excerpts for {len(enriched)} source(s)"
    if mode == "live":
        detail += f" ({fetched} fetched, {failed} fallback to snippets)."
    else:
        detail += " (snippet mode)."

    step = AgentStep(
        name=AgentStepName.FETCH,
        status="ok",
        detail=detail,
        duration_ms=duration_ms,
        metadata={
            "source_ids": [s.id for s in enriched],
            "fetched": fetched,
            "failed": failed,
            "fetch_mode": mode,
        },
    )
    return enriched, step
