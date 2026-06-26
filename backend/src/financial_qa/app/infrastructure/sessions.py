"""Persistence helpers for chat sessions (threads).

A session's ``id`` doubles as the conversation key the graph sees as ``state["user_id"]``, so the AI
layer needs no knowledge of sessions — ownership and lifecycle live entirely here and in the routes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_qa.app.infrastructure.models import ChatSession, ChatTurn

_TITLE_MAX = 54


def derive_title(text: str) -> str:
    """First user message, trimmed to a rail-friendly title (mirrors the old frontend rule)."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "New chat"
    return cleaned if len(cleaned) <= _TITLE_MAX else f"{cleaned[:_TITLE_MAX]}…"


async def create_session(db: AsyncSession, *, user_id: int, title: str) -> ChatSession:
    chat_session = ChatSession(id=uuid.uuid4().hex, user_id=user_id, title=title)
    db.add(chat_session)
    await db.commit()
    await db.refresh(chat_session)
    return chat_session


async def list_sessions(db: AsyncSession, *, user_id: int) -> list[ChatSession]:
    result = await db.execute(
        select(ChatSession).where(ChatSession.user_id == user_id).order_by(ChatSession.updated_at.desc())
    )
    return list(result.scalars().all())


async def get_messages(db: AsyncSession, *, session_id: str) -> list[dict[str, str]]:
    """Persisted user/assistant turns for a session, oldest first (mirrors history.get_history)."""
    result = await db.execute(
        select(ChatTurn.role, ChatTurn.content)
        .where(ChatTurn.session_id == session_id)
        .order_by(ChatTurn.created_at.asc(), ChatTurn.id.asc())
    )
    return [{"role": role, "content": content} for role, content in result.all()]


async def delete_session(db: AsyncSession, *, session_id: str) -> None:
    """Remove a session and its persisted turns in one transaction."""
    await db.execute(delete(ChatTurn).where(ChatTurn.session_id == session_id))
    await db.execute(delete(ChatSession).where(ChatSession.id == session_id))
    await db.commit()


async def finalize_turn(db: AsyncSession, *, session_id: str, last_render: dict[str, Any] | None) -> None:
    """After a run, attach the reload payload (stream only) and bump ``updated_at`` for rail sorting."""
    chat_session = (
        await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    ).scalar_one_or_none()
    if chat_session is None:
        return
    if last_render is not None:
        chat_session.last_render = last_render
    chat_session.updated_at = datetime.now(UTC)
    await db.commit()
