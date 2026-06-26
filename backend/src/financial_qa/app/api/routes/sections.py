"""Chat sections (conversation threads), scoped to the signed-in user.

Creation is lazy — a session is created by the chat endpoint on the first message — so this router
only lists, loads, and deletes. Every route is isolated by user via the ownership dependency.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from financial_qa.app.api.deps import get_current_user, get_owned_session
from financial_qa.app.infrastructure import sessions as sessions_repo
from financial_qa.app.infrastructure.db import get_session
from financial_qa.app.infrastructure.models import ChatSession, User

router = APIRouter(prefix="/api/v1/sections", tags=["sections"])


class SectionSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class SectionMessage(BaseModel):
    role: str
    content: str


class SectionDetail(SectionSummary):
    messages: list[SectionMessage]
    last_render: dict[str, Any] | None = None


@router.get("", response_model=list[SectionSummary])
async def list_sections(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> list[ChatSession]:
    return await sessions_repo.list_sessions(db, user_id=user.id)


@router.get("/{session_id}", response_model=SectionDetail)
async def get_section(
    chat_session: Annotated[ChatSession, Depends(get_owned_session)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> SectionDetail:
    messages = await sessions_repo.get_messages(db, session_id=chat_session.id)
    return SectionDetail(
        id=chat_session.id,
        title=chat_session.title,
        created_at=chat_session.created_at,
        updated_at=chat_session.updated_at,
        messages=[SectionMessage(**message) for message in messages],
        last_render=chat_session.last_render,
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_section(
    chat_session: Annotated[ChatSession, Depends(get_owned_session)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> None:
    await sessions_repo.delete_session(db, session_id=chat_session.id)
