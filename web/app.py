"""Research Brief - Streamlit web UI."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from research_brief.agent.pipeline import run_research
from research_brief.config import get_settings
from research_brief.models import ResearchRequest

load_dotenv()

st.set_page_config(
    page_title="Research Brief",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Running research agent...")
def _run_cached(request_json: str, _version: str = "v1") -> str:
    del _version
    request = ResearchRequest.model_validate_json(request_json)
    result = run_research(request)
    return result.model_dump_json()


def main() -> None:
    st.title("Research Brief")
    st.caption("Multi-step research agent with cited briefs and tool traces.")

    settings = get_settings()

    with st.sidebar:
        st.header("Configuration")
        topic = st.text_input(
            "Research topic",
            value="What are the tradeoffs of RAG vs fine-tuning?",
        )
        max_sources = st.slider("Max sources", min_value=1, max_value=8, value=3)
        search_provider = st.selectbox(
            "Search provider",
            options=["auto", "mock", "tavily"],
            index=["auto", "mock", "tavily"].index(
                settings.search_provider if settings.search_provider in {"auto", "mock", "tavily"} else "auto"
            ),
        )
        fetch_mode = st.selectbox("Fetch mode", options=["live", "snippet"], index=0)
        synthesis_mode = st.selectbox("Synthesis mode", options=["template", "openai"], index=0)
        if synthesis_mode == "openai" and not (settings.openai_api_key or os.getenv("OPENAI_API_KEY")):
            st.warning("Set OPENAI_API_KEY in .env to enable OpenAI synthesis.")
        if search_provider in {"auto", "tavily"} and not settings.tavily_api_key:
            st.info("No TAVILY_API_KEY set. Auto/mock fallback will be used.")

        run = st.button("Generate brief", type="primary", use_container_width=True)

    if not run:
        st.info("Enter a topic in the sidebar, then click **Generate brief**.")
        return

    if not topic.strip():
        st.error("Topic cannot be empty.")
        return

    request = ResearchRequest(
        topic=topic.strip(),
        max_sources=max_sources,
        search_provider=search_provider,  # type: ignore[arg-type]
        fetch_mode=fetch_mode,  # type: ignore[arg-type]
        synthesis_mode=synthesis_mode,  # type: ignore[arg-type]
    )

    try:
        payload = _run_cached(request.model_dump_json())
        from research_brief.models import ResearchResult

        result = ResearchResult.model_validate_json(payload)
    except Exception as exc:
        st.error(f"Research failed: {exc}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sources", len(result.sources))
    c2.metric("Citation coverage", f"{result.citation_coverage_pct:.0f}%")
    c3.metric("Search", result.search_provider)
    c4.metric("Synthesis", result.synthesis_mode)

    st.subheader("Brief")
    if result.synthesis_llm_used:
        st.caption("LLM-polished brief (grounding validated)")
    else:
        st.caption("Template brief (deterministic)")
    st.markdown(result.brief)

    if result.synthesis_llm_used:
        with st.expander("View original template brief"):
            st.markdown(result.brief_template)

    st.subheader("Sources")
    rows = [
        {
            "ID": s.id,
            "Title": s.title,
            "URL": s.url,
            "Excerpt": (s.excerpt or s.snippet)[:220] + "...",
        }
        for s in result.sources
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Agent trace")
    trace_rows = [
        {
            "Step": step.name.value,
            "Status": step.status,
            "Duration (ms)": step.duration_ms,
            "Detail": step.detail,
        }
        for step in result.trace
    ]
    st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)

    with st.expander("Full JSON"):
        st.json(result.model_dump(mode="json"))


if __name__ == "__main__":
    main()
