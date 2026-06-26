"""Turn-based chat history on async SQLAlchemy.

The frontend sends only the new user message each turn; prior turns are loaded here and prepended.
We persist user messages and the final (validated) assistant answer — never tool messages, tool
calls, or usage embedded in the message text (usage goes to dedicated columns).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.utils import convert_to_messages
from sqlalchemy import select

from financial_qa.app.agent.schemas import State
from financial_qa.app.infrastructure.db import SessionLocal
from financial_qa.app.infrastructure.models import ChatTurn
from financial_qa.app.infrastructure.settings import get_settings


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


async def get_history(state: State) -> dict:
    session_id = (state.get("user_id") or "").strip()
    request_messages: list[BaseMessage] = convert_to_messages(state.get("input") or [])
    if not session_id:
        return {"messages": request_messages, "persist_start_index": 0}

    limit = max(get_settings().db_history_turns, 0) * 2
    async with SessionLocal() as session:
        result = await session.execute(
            select(ChatTurn.role, ChatTurn.content)
            .where(ChatTurn.session_id == session_id)
            .order_by(ChatTurn.created_at.asc(), ChatTurn.id.asc())
        )
        rows = result.all()

    selected = rows[-limit:] if limit else []
    history_messages = convert_to_messages([{"role": role, "content": content} for role, content in selected])
    merged = [*history_messages, *request_messages]
    return {"messages": merged, "persist_start_index": len(history_messages)}


async def save_history(state: State) -> dict:
    session_id = (state.get("user_id") or "").strip()
    if not session_id:
        return {}

    messages: list[BaseMessage] = state.get("messages") or []
    start = int(state.get("persist_start_index", 0) or 0)
    usage = state.get("usage") or {}

    new_rows: list[ChatTurn] = []
    for message in messages[start:]:
        if isinstance(message, HumanMessage):
            new_rows.append(ChatTurn(session_id=session_id, role="user", content=_message_text(message.content)))

    answer = (state.get("output") or "").strip()
    if answer:
        new_rows.append(
            ChatTurn(
                session_id=session_id,
                role="assistant",
                content=answer,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
                cost_usd=usage.get("cost"),
            )
        )

    if new_rows:
        async with SessionLocal() as session:
            session.add_all(new_rows)
            await session.commit()
    return {}
