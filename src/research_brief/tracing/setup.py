from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from research_brief.config import get_settings, langsmith_tracing_enabled


@contextmanager
def start_span(name: str, *, attributes: dict | None = None) -> Iterator[object | None]:
    """Start an OpenTelemetry and/or LangSmith child span when enabled."""
    attributes = attributes or {}
    settings = get_settings()

    langsmith_run = None
    otel_cm = None
    span = None

    if langsmith_tracing_enabled(settings):
        try:
            from langsmith.run_trees import RunTree

            langsmith_run = RunTree(
                name=name,
                run_type="tool",
                inputs=attributes,
                project_name=settings.langsmith_project,
            )
            langsmith_run.post()
        except Exception:
            langsmith_run = None

    if settings.otel_enabled:
        try:
            from opentelemetry import trace

            tracer = trace.get_tracer(settings.otel_service_name)
            otel_cm = tracer.start_as_current_span(name)
            span = otel_cm.__enter__()
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        except Exception:
            otel_cm = None
            span = None

    try:
        yield span or langsmith_run
    finally:
        if otel_cm is not None:
            try:
                otel_cm.__exit__(None, None, None)
            except Exception:
                pass
        if langsmith_run is not None:
            try:
                langsmith_run.end()
                langsmith_run.patch()
            except Exception:
                pass


def configure_tracing() -> None:
    """Initialize OpenTelemetry provider/exporter when enabled."""
    settings = get_settings()
    if not settings.otel_enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        resource = Resource.create({"service.name": settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
    except Exception:
        return


def current_trace_id() -> str | None:
    if not get_settings().otel_enabled:
        return None
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            return format(ctx.trace_id, "032x")
    except Exception:
        return None
    return None


def maybe_wrap_langsmith(func):
    """Decorator: wrap top-level pipeline in a LangSmith run when configured."""
    settings = get_settings()
    if not langsmith_tracing_enabled(settings):
        return func

    try:
        from langsmith import traceable

        return traceable(
            name="run_research",
            run_type="chain",
            project_name=settings.langsmith_project,
        )(func)
    except Exception:
        return func
