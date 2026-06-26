"""Chat endpoints. Thin adapters over the workflow service.

Both routes resolve (or lazily create) the user-owned chat session, then run the *unchanged* graph
keyed by the session id. ``/runs/stream`` streams NDJSON progress + answer and persists a
full-fidelity reload payload; ``/runs`` returns only the final answer.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_qa.app.agent.events import to_ndjson
from financial_qa.app.agent.schemas import ChatMessage, ChatRunRequest
from financial_qa.app.agent.service import get_chat_workflow_service
from financial_qa.app.api.deps import get_current_user
from financial_qa.app.infrastructure import sessions as sessions_repo
from financial_qa.app.infrastructure.db import SessionLocal, get_session
from financial_qa.app.infrastructure.models import ChatSession, User

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Progress events worth replaying to rebuild the Thought-process timeline + evidence on reload.
_RENDER_EVENTS = {
    "reasoning.delta",
    "node.finished",
    "tool.selected",
    "sql.query",
    "vector.search",
    "coverage.notice",
    "evidence",
    "validation",
}


class ChatRunResponse(BaseModel):
    session_id: str
    answer: str
    grounded: bool | None = None
    usage: dict[str, Any]
    trace_id: str | None = None


def _first_user_text(messages: list[ChatMessage]) -> str:
    return next((m.content for m in messages if m.role == "user"), "")


async def _resolve_session(db: AsyncSession, user: User, payload: ChatRunRequest) -> ChatSession:
    """Return the requested session (owned by the user) or lazily create a new one."""
    if payload.session_id:
        chat_session = (
            await db.execute(
                select(ChatSession).where(
                    ChatSession.id == payload.session_id, ChatSession.user_id == user.id
                )
            )
        ).scalar_one_or_none()
        if chat_session is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
        return chat_session
    title = sessions_repo.derive_title(_first_user_text(payload.messages))
    return await sessions_repo.create_session(db, user_id=user.id, title=title)


@router.post("/runs/stream")
async def stream_run(
    payload: ChatRunRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> StreamingResponse:
    service = get_chat_workflow_service()
    chat_session = await _resolve_session(db, user, payload)
    session_id = chat_session.id
    messages = [message.model_dump() for message in payload.messages]

    async def generate():
        seq = 0
        captured: list[dict[str, Any]] = []
        trace_id: str | None = None
        # Graph history is keyed by the session id (passed as user_id) — no AI-layer change.
        async for event in service.astream_events(user_id=session_id, messages=messages):
            line = {"seq": seq, **event}
            if event.get("type") == "run.started":
                line["session_id"] = session_id
                trace_id = event.get("trace_id")
            if event.get("type") in _RENDER_EVENTS:
                captured.append(line)
            yield to_ndjson(line)
            seq += 1
        async with SessionLocal() as finalize_db:
            await sessions_repo.finalize_turn(
                finalize_db, session_id=session_id, last_render={"events": captured, "trace_id": trace_id}
            )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/runs", response_model=ChatRunResponse)
async def run(
    payload: ChatRunRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_session)],
) -> ChatRunResponse:
    """Non-streaming: run the same graph to completion and return only the final answer."""
    service = get_chat_workflow_service()
    chat_session = await _resolve_session(db, user, payload)
    messages = [message.model_dump() for message in payload.messages]

    result = await service.arun(user_id=chat_session.id, messages=messages)
    await sessions_repo.finalize_turn(db, session_id=chat_session.id, last_render=None)
    return ChatRunResponse(session_id=chat_session.id, **result)
