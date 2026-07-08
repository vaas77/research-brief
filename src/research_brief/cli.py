from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from research_brief.agent.pipeline import run_research
from research_brief.config import get_settings, langsmith_tracing_enabled
from research_brief.jobs.worker import enqueue_research, get_job
from research_brief.models import JobStatus, ResearchRequest

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


def _render_sources(result) -> None:
    for source in result.sources:
        console.print(f"- [bold]{source.title}[/bold] ({source.url}) [{source.id}]")


@app.command("demo")
def demo() -> None:
    """Run the default demo topic with mock sources."""
    request = ResearchRequest(
        topic="What are the tradeoffs of RAG vs fine-tuning?",
        max_sources=3,
        synthesis_mode="template",
        search_provider="mock",
        fetch_mode="snippet",
    )
    result = run_research(request)
    console.print(
        Panel(
            f"[bold]Research Brief Demo[/bold]\n\n"
            f"Topic: {result.topic}\n"
            f"Search: {result.search_provider}\n"
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
    search_provider: str = typer.Option("auto", help="auto, mock, or tavily"),
    synthesis_mode: str = typer.Option("template", help="template or openai"),
    fetch_mode: str = typer.Option("live", help="live or snippet"),
    output_json: Path | None = typer.Option(None, help="Optional JSON output path."),
) -> None:
    """Run research for a custom topic."""
    request = ResearchRequest(
        topic=topic,
        max_sources=max_sources,
        search_provider=search_provider,  # type: ignore[arg-type]
        synthesis_mode=synthesis_mode,  # type: ignore[arg-type]
        fetch_mode=fetch_mode,  # type: ignore[arg-type]
    )
    result = run_research(request)
    console.print(Panel(result.brief, title=result.topic, border_style="green"))
    _render_sources(result)
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


@app.command("job")
def job(
    topic: str = typer.Argument(..., help="Research topic or question."),
    max_sources: int = typer.Option(3, help="Maximum sources to retrieve."),
    search_provider: str = typer.Option("auto", help="auto, mock, or tavily"),
    synthesis_mode: str = typer.Option("template", help="template or openai"),
    fetch_mode: str = typer.Option("live", help="live or snippet"),
    timeout_seconds: int = typer.Option(180, help="Max seconds to wait for completion."),
) -> None:
    """Enqueue research as a background job and wait for completion."""
    import time

    request = ResearchRequest(
        topic=topic,
        max_sources=max_sources,
        search_provider=search_provider,  # type: ignore[arg-type]
        synthesis_mode=synthesis_mode,  # type: ignore[arg-type]
        fetch_mode=fetch_mode,  # type: ignore[arg-type]
    )
    created = enqueue_research(request)
    console.print(f"[cyan]Job {created.id} queued ({created.status.value})[/cyan]")

    deadline = time.time() + timeout_seconds
    final = created
    while time.time() < deadline:
        final = get_job(created.id)
        if final.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
            break
        time.sleep(0.5)

    if final.status == JobStatus.FAILED:
        console.print(f"[red]{final.error or 'Job failed'}[/red]")
        raise typer.Exit(code=1)
    if final.status != JobStatus.COMPLETED or final.result is None:
        console.print("[red]Job timed out[/red]")
        raise typer.Exit(code=1)

    result = final.result
    console.print(Panel(result.brief, title=result.topic, border_style="green"))
    if result.trace_id:
        console.print(f"[dim]trace_id={result.trace_id}[/dim]")
    _render_sources(result)
    _render_trace(result)


@app.command("web")
def web(port: int = typer.Option(8501, help="Port for Streamlit server.")) -> None:
    """Launch the Streamlit web UI."""
    app_path = Path(__file__).resolve().parents[2] / "web" / "app.py"
    console.print(f"[green]Starting Research Brief UI on port {port}...[/green]")
    raise typer.Exit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                str(port),
                "--browser.gatherUsageStats",
                "false",
            ]
        )
    )


@app.command("config")
def show_config() -> None:
    """Show resolved runtime configuration."""
    settings = get_settings()
    console.print(
        Panel(
            f"search_provider={settings.search_provider}\n"
            f"synthesis_mode={settings.synthesis_mode}\n"
            f"fetch_mode={settings.fetch_mode}\n"
            f"tavily_api_key={'set' if settings.tavily_api_key else 'missing'}\n"
            f"openai_api_key={'set' if settings.openai_api_key else 'missing'}\n"
            f"otel_enabled={settings.otel_enabled}\n"
            f"langsmith_tracing={langsmith_tracing_enabled(settings)}",
            title="Settings",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    app()
