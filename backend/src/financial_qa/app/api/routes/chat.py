"""The single streaming chat endpoint. Thin adapter over the workflow service."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from financial_qa.app.agent.events import to_ndjson
from financial_qa.app.agent.schemas import ChatRunRequest
from financial_qa.app.agent.service import get_chat_workflow_service
from financial_qa.app.api.deps import get_current_user
from financial_qa.app.infrastructure.models import User

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("/runs/stream")
async def stream_run(
    payload: ChatRunRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> StreamingResponse:
    service = get_chat_workflow_service()
    messages = [message.model_dump() for message in payload.messages]

    async def generate():
        seq = 0
        async for event in service.astream_events(user_id=user.email, messages=messages):
            yield to_ndjson({"seq": seq, **event})
            seq += 1

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
