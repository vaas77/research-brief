from __future__ import annotations

from research_brief.tracing import configure_tracing, maybe_wrap_langsmith, start_span


def test_start_span_noop_when_tracing_disabled():
    with start_span("search", attributes={"topic": "test"}) as span:
        assert span is None


def test_configure_tracing_noop_when_disabled():
    configure_tracing()


def test_maybe_wrap_langsmith_returns_callable():
    def fn(x):
        return x

    wrapped = maybe_wrap_langsmith(fn)
    assert wrapped(1) == 1
