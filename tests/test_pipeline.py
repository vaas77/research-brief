from __future__ import annotations

from pathlib import Path

import yaml

from research_brief.agent.pipeline import run_research
from research_brief.models import ResearchRequest


def _load_cases() -> dict:
    path = Path(__file__).resolve().parents[1] / "eval" / "test_cases.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_demo_pipeline_produces_trace_and_citations():
    result = run_research(
        ResearchRequest(
            topic="What are the tradeoffs of RAG vs fine-tuning?",
            max_sources=3,
            search_provider="mock",
            synthesis_mode="template",
        )
    )
    assert len(result.sources) >= 2
    assert result.citation_coverage_pct >= 80.0
    assert [s.name.value for s in result.trace] == [
        "search",
        "fetch",
        "synthesize",
        "validate",
    ]
    assert "[rag-1]" in result.brief


def test_eval_cases_from_yaml():
    for case in _load_cases().values():
        result = run_research(
            ResearchRequest(
                topic=case["topic"],
                max_sources=case.get("max_sources", 3),
                search_provider="mock",
                synthesis_mode="template",
            )
        )
        exp = case["expectations"]
        assert len(result.sources) >= exp["min_sources"]
        assert result.citation_coverage_pct >= exp["min_citation_coverage_pct"]
        assert [s.name.value for s in result.trace] == exp["trace_steps"]
        for phrase in exp["brief_contains"]:
            assert phrase in result.brief
