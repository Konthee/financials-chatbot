"""Application configuration.

Every runtime value comes from the environment (the repo-root ``.env`` locally, or the
process environment injected by docker-compose). Nothing is hardcoded here beyond the
sensible defaults that mirror ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_repo_root() -> Path:
    """Walk up from this file to the project root.

    The root is the first ancestor that carries a project marker. This resolves to the repo
    root for local runs (``backend/src/...``) and to ``/app`` inside the container (where the
    package is copied to ``/app/src`` and ``data/`` is mounted at ``/app/data``).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docker-compose.yml").is_file() or (parent / "data").is_dir():
            return parent
    # Fallback: backend/ -> repo root is two levels above ``src``.
    return here.parents[4]


REPO_ROOT = _find_repo_root()


class Settings(BaseSettings):
    """Strongly-typed view over the environment."""

    # --- Chat LLM (OpenAI-compatible endpoint) -------------------------------------------
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-4o-mini"

    # --- Embeddings (must produce 512-dim vectors to match the fixture) ------------------
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = ""
    embedding_model_name: str = "text-embedding-3-small"
    embedding_dim: int = 512

    # --- Langfuse (optional; app runs fully with this disabled) --------------------------
    langfuse_enabled: bool = False
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # --- PostgreSQL ----------------------------------------------------------------------
    database_url: str = "postgresql+asyncpg://postgres:postgres@postgres:5432/financial_qa"

    # --- Pinecone Local ------------------------------------------------------------------
    pinecone_api_key: str = "pclocal"
    pinecone_host: str = "http://pinecone-local:5080"
    pinecone_index_name: str = "tenk-filings"
    pinecone_namespace: str = "__default__"

    # --- Auth / JWT ----------------------------------------------------------------------
    jwt_secret: str = "change-me-to-a-long-random-string"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440
    demo_user_email: str = "demo@example.com"
    demo_user_password: str = "demo1234"

    # --- CORS ----------------------------------------------------------------------------
    cors_origins: str = "http://localhost:3000"

    # --- Chat history --------------------------------------------------------------------
    db_history_turns: int = 10

    # --- Data fixtures (loaders) ---------------------------------------------------------
    data_dir: str = str(REPO_ROOT / "data")

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a clean list (``.env`` stores them comma-separated)."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def psycopg_dsn(self) -> str:
        """A plain libpq DSN for synchronous psycopg (used by the SQL loader)."""
        return self.database_url.replace("+asyncpg", "").replace("+psycopg", "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
