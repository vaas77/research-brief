from __future__ import annotations

import re

from research_brief.models import Citation

_CITATION_RE = re.compile(r"\[([^\]]+)\]")


def extract_grounding_tokens(template: str, citations: list[Citation]) -> list[str]:
    """Facts that must survive LLM polish unchanged."""
    tokens: list[str] = []
    for citation in citations:
        tokens.append(f"[{citation.source_id}]")
    tokens.extend(m.group(0) for m in _CITATION_RE.finditer(template))

    seen: set[str] = set()
    unique: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def validate_llm_grounding(
    template: str,
    polished: str,
    citations: list[Citation],
    *,
    min_ratio: float = 0.85,
) -> bool:
    must_preserve = extract_grounding_tokens(template, citations)
    if not must_preserve:
        return True
    preserved = sum(1 for token in must_preserve if token in polished)
    threshold = max(1, int(len(must_preserve) * min_ratio))
    return preserved >= threshold
