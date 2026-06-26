"""Query-time embeddings via an OpenAI-compatible endpoint.

Must produce 512-dim vectors to match the loaded fixture (text-embedding-3-small @ dimensions=512).
Only queries are embedded at runtime — the corpus uses the prebuilt fixture — so the cost is tiny.
"""

from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI

from financial_qa.app.infrastructure.settings import get_settings


@lru_cache
def _client() -> AsyncOpenAI:
    settings = get_settings()
    return AsyncOpenAI(base_url=settings.embedding_base_url, api_key=settings.embedding_api_key)


async def embed_query(text: str) -> list[float]:
    settings = get_settings()
    response = await _client().embeddings.create(
        model=settings.embedding_model_name,
        input=text,
        dimensions=settings.embedding_dim,
    )
    return response.data[0].embedding
