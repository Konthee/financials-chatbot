"""Pinecone Local query client.

Connects to the control plane (``PINECONE_HOST``), resolves the per-index data-plane host, and
exposes an async similarity search. The index is created/populated by ``scripts/load_vectors.py``.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

from pinecone import Pinecone

from financial_qa.app.infrastructure.settings import get_settings


@lru_cache
def _index():
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key, host=settings.pinecone_host)
    host = pc.describe_index(settings.pinecone_index_name).host
    if not host.startswith("http"):  # Pinecone Local serves over plain HTTP, no scheme in host
        host = f"http://{host}"
    return pc.Index(host=host)


def _query_sync(vector: list[float], top_k: int, metadata_filter: dict | None) -> list[dict[str, Any]]:
    settings = get_settings()
    result = _index().query(
        vector=vector,
        top_k=top_k,
        namespace=settings.pinecone_namespace,
        include_metadata=True,
        filter=metadata_filter or None,
    )
    matches = result.get("matches", []) if isinstance(result, dict) else result.matches
    return [
        {"id": m["id"], "score": m["score"], "metadata": m.get("metadata", {})}
        if isinstance(m, dict)
        else {"id": m.id, "score": m.score, "metadata": dict(m.metadata or {})}
        for m in matches
    ]


async def query(
    vector: list[float],
    *,
    top_k: int = 6,
    metadata_filter: dict | None = None,
) -> list[dict[str, Any]]:
    """Async cosine search; runs the synchronous Pinecone REST call off the event loop."""
    return await asyncio.to_thread(_query_sync, vector, top_k, metadata_filter)
