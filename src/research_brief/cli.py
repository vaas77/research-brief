from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from research_brief.agent.pipeline import run_research
from research_brief.models import ResearchRequest

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _render_trace(result) -> None:
    table = Table(title="Agent trace")
    table.add_column("Step")
    table.add_column("Status")
    table.add_column("Duration (ms)")
    table.add_column("Detail")
    for step in result.trace:
        table.add_row(step.name.value, step.status, str(step.duration_ms), step.detail)
    console.print(table)


@app.command("demo")
def demo() -> None:
    """Run the default demo topic with mock sources."""
    request = ResearchRequest(
        topic="What are the tradeoffs of RAG vs fine-tuning?",
        max_sources=3,
        synthesis_mode="template",
        search_provider="mock",
    )
    result = run_research(request)
    console.print(
        Panel(
            f"[bold]Research Brief Demo[/bold]\n\n"
            f"Topic: {result.topic}\n"
            f"Citation coverage: {result.citation_coverage_pct:.1f}%",
            border_style="blue",
        )
    )
    console.print(Panel(result.brief, title="Brief", border_style="green"))
    _render_trace(result)


@app.command("run")
def run(
    topic: str = typer.Argument(..., help="Research topic or question."),
    max_sources: int = typer.Option(3, help="Maximum sources to retrieve."),
    output_json: Path | None = typer.Option(None, help="Optional JSON output path."),
) -> None:
    """Run research for a custom topic."""
    request = ResearchRequest(topic=topic, max_sources=max_sources)
    result = run_research(request)
    console.print(Panel(result.brief, title=result.topic, border_style="green"))
    _render_trace(result)
    if output_json:
        output_json.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[dim]JSON written to {output_json}[/dim]")


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
) -> None:
    """Start the FastAPI server."""
    console.print(f"[green]Starting API on http://{host}:{port}[/green]")
    uvicorn.run("research_brief.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
