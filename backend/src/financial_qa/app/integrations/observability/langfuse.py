"""Optional Langfuse tracing.

Enabled only when LANGFUSE_ENABLED is truthy *and* keys are present, so the app runs fully without
it. All Langfuse calls are defensive: if anything goes wrong, tracing is skipped, never the request.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from financial_qa.app.infrastructure.settings import get_settings


def enabled() -> bool:
    settings = get_settings()
    return bool(settings.langfuse_enabled and settings.langfuse_public_key and settings.langfuse_secret_key)


def _ensure_env() -> None:
    """Map our settings onto the env vars the Langfuse SDK reads (note: it expects LANGFUSE_HOST)."""
    settings = get_settings()
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_base_url)


def callbacks() -> list:
    if not enabled():
        return []
    try:
        _ensure_env()
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception:
        return []


@contextmanager
def trace_run(name: str = "financial-qa-chat"):
    """Yield the Langfuse trace id for this run (or None when tracing is off/unavailable)."""
    if not enabled():
        yield None
        return
    try:
        _ensure_env()
        from langfuse import get_client

        client = get_client()
        with client.start_as_current_span(name=name):
            yield client.get_current_trace_id()
    except Exception:
        yield None
