"""Progress-event helpers shared by tools, nodes, and the streaming service.

Nodes/tools call ``emit(kind, **data)`` to push a structured progress event through LangGraph's
custom stream channel. The service turns each custom chunk into one NDJSON line (``kind`` -> ``type``).
"""

from __future__ import annotations

import json
from typing import Any


def emit(kind: str, **payload: Any) -> None:
    """Write a custom progress event to the active LangGraph stream (no-op if not streaming)."""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    writer({"kind": kind, **payload})


def custom_to_event(chunk: dict[str, Any]) -> dict[str, Any]:
    """Map a custom stream chunk ``{"kind": ..., ...}`` to an NDJSON event ``{"type": ..., ...}``."""
    data = dict(chunk)
    kind = data.pop("kind", "event")
    return {"type": kind, **data}


def to_ndjson(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
