from __future__ import annotations

import os
import time

from research_brief.config import get_settings
from research_brief.models import AgentStep, AgentStepName, Citation, SourceDocument


def _build_citations(sources: list[SourceDocument]) -> list[Citation]:
    return [
        Citation(source_id=s.id, label=s.title, url=s.url)
        for s in sources
    ]


def build_template_brief(topic: str, sources: list[SourceDocument]) -> tuple[str, list[str]]:
    lines = [f"Research brief: {topic}", ""]
    key_points: list[str] = []

    for idx, source in enumerate(sources, start=1):
        point = source.excerpt.split(".")[0].strip() + "."
        key_points.append(point)
        lines.append(f"{idx}. {point} [{source.id}]")
        lines.append(f"   Source: {source.title} ({source.url})")
        lines.append("")

    lines.append("Summary")
    lines.append(
        "Across the selected sources, the strongest shared theme is that retrieval and "
        "orchestration quality materially affect answer faithfulness in GenAI systems."
    )
    return "\n".join(lines).strip(), key_points


def _polish_with_openai(topic: str, template: str) -> str:
    from openai import OpenAI

    settings = get_settings()
    if not settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
        return template

    client = OpenAI(api_key=settings.openai_api_key or os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the research brief for clarity. Do not add new facts or sources. "
                    "Preserve all source IDs in square brackets exactly as written."
                ),
            },
            {"role": "user", "content": template},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content
    return content.strip() if content else template


def synthesize_brief(
    topic: str,
    sources: list[SourceDocument],
    mode: str | None = None,
) -> tuple[str, list[str], list[Citation], AgentStep, str]:
    start = time.perf_counter()
    resolved_mode = (mode or get_settings().synthesis_mode).lower()
    citations = _build_citations(sources)
    template, key_points = build_template_brief(topic, sources)
    brief = template
    status = "ok"
    detail = "Template brief generated from fetched excerpts."

    if resolved_mode == "openai":
        try:
            brief = _polish_with_openai(topic, template)
            detail = "OpenAI polish applied with template fallback on errors."
        except Exception as exc:
            brief = template
            resolved_mode = "template"
            detail = f"OpenAI polish failed; used template. ({exc})"

    duration_ms = int((time.perf_counter() - start) * 1000)
    step = AgentStep(
        name=AgentStepName.SYNTHESIZE,
        status=status,
        detail=detail,
        duration_ms=duration_ms,
        metadata={"mode": resolved_mode, "source_count": len(sources)},
    )
    return brief, key_points, citations, step, resolved_mode
