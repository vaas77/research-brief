from __future__ import annotations

import os
import time

from research_brief.config import get_settings
from research_brief.models import AgentStep, AgentStepName, Citation, SourceDocument
from research_brief.validation.grounding import validate_llm_grounding


def _build_citations(sources: list[SourceDocument]) -> list[Citation]:
    return [Citation(source_id=s.id, label=s.title, url=s.url) for s in sources]


def _first_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return "No excerpt available."
    sentence = text.split(".")[0].strip()
    return sentence + "." if sentence else text[:180]


def build_template_brief(topic: str, sources: list[SourceDocument]) -> tuple[str, list[str]]:
    lines = [f"Research brief: {topic}", ""]
    key_points: list[str] = []

    for idx, source in enumerate(sources, start=1):
        point = _first_sentence(source.excerpt or source.snippet)
        key_points.append(point)
        lines.append(f"{idx}. {point} [{source.id}]")
        lines.append(f"   Source: {source.title} ({source.url})")
        lines.append("")

    lines.append("Summary")
    if key_points:
        joined = " ".join(key_points[:3])
        lines.append(
            f"Across {len(sources)} source(s), the research on '{topic}' highlights: {joined}"
        )
    else:
        lines.append(f"No substantive excerpts were available for '{topic}'.")

    return "\n".join(lines).strip(), key_points


def _polish_with_openai(topic: str, template: str) -> str:
    from openai import OpenAI

    settings = get_settings()
    api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research analyst. Rewrite the brief below for clarity and flow. "
                    "Rules:\n"
                    "- Use only facts present in the brief\n"
                    "- Do NOT add new sources, claims, or recommendations\n"
                    "- Keep every source ID in square brackets exactly as written (e.g. [src-abc])\n"
                    "- Keep source titles and URLs unchanged\n"
                    "- Use short paragraphs"
                ),
            },
            {"role": "user", "content": f"Topic: {topic}\n\n{template}"},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from OpenAI")
    return content.strip()


def synthesize_brief(
    topic: str,
    sources: list[SourceDocument],
    mode: str | None = None,
) -> tuple[str, str, list[str], list[Citation], AgentStep, str, bool]:
    """
    Returns:
        brief, template, key_points, citations, step, resolved_mode, llm_used
    """
    start = time.perf_counter()
    settings = get_settings()
    resolved_mode = (mode or settings.synthesis_mode).lower()
    citations = _build_citations(sources)
    template, key_points = build_template_brief(topic, sources)
    brief = template
    llm_used = False
    status = "ok"
    detail = "Template brief generated from fetched excerpts."

    if resolved_mode == "openai":
        if not (settings.openai_api_key or os.getenv("OPENAI_API_KEY")):
            resolved_mode = "template"
            detail = "OPENAI_API_KEY missing; used template brief."
        else:
            try:
                polished = _polish_with_openai(topic, template)
                if validate_llm_grounding(template, polished, citations):
                    brief = polished
                    llm_used = True
                    detail = "OpenAI brief accepted after grounding validation."
                else:
                    brief = template
                    resolved_mode = "template"
                    detail = "OpenAI brief failed grounding check; used template."
            except Exception as exc:
                brief = template
                resolved_mode = "template"
                detail = f"OpenAI synthesis failed; used template. ({exc})"

    duration_ms = int((time.perf_counter() - start) * 1000)
    step = AgentStep(
        name=AgentStepName.SYNTHESIZE,
        status=status,
        detail=detail,
        duration_ms=duration_ms,
        metadata={"mode": resolved_mode, "source_count": len(sources), "llm_used": llm_used},
    )
    return brief, template, key_points, citations, step, resolved_mode, llm_used
