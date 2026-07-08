from __future__ import annotations

import re
import time

from research_brief.models import AgentStep, AgentStepName, Citation


_CITATION_RE = re.compile(r"\[([^\]]+)\]")


def validate_citations(brief: str, citations: list[Citation]) -> tuple[float, AgentStep]:
    start = time.perf_counter()
    valid_ids = {c.source_id for c in citations}
    found = {m.group(1) for m in _CITATION_RE.finditer(brief)}
    matched = found & valid_ids
    coverage = (len(matched) / len(valid_ids) * 100.0) if valid_ids else 100.0
    duration_ms = int((time.perf_counter() - start) * 1000)

    status = "ok" if coverage >= 80.0 else "error"
    detail = (
        f"Citation coverage {coverage:.1f}% ({len(matched)}/{len(valid_ids)} source IDs cited)."
    )
    step = AgentStep(
        name=AgentStepName.VALIDATE,
        status=status,
        detail=detail,
        duration_ms=duration_ms,
        metadata={"matched_ids": sorted(matched), "found_ids": sorted(found)},
    )
    return coverage, step
