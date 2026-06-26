"""Chat workflow service: drives the compiled graph and yields NDJSON-ready events."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from financial_qa.app.agent.events import custom_to_event
from financial_qa.app.agent.graph import get_compiled_graph
from financial_qa.app.agent.schemas import empty_usage
from financial_qa.app.infrastructure.settings import get_settings
from financial_qa.app.integrations.observability import langfuse


class ChatWorkflowService:
    def __init__(self) -> None:
        self._graph = get_compiled_graph()
        self._model_name = get_settings().model_name

    async def astream_events(self, *, user_id: str, messages: list[dict]) -> AsyncIterator[dict[str, Any]]:
        run_id = "run_" + uuid.uuid4().hex[:12]
        initial = {"user_id": user_id, "input": messages, "usage": empty_usage()}
        last_usage = empty_usage()

        with langfuse.trace_run() as trace_id:
            config = {"callbacks": langfuse.callbacks()}
            yield {"type": "run.started", "run_id": run_id, "trace_id": trace_id, "model": self._model_name}
            try:
                async for stream_mode, chunk in self._graph.astream(
                    initial, config=config, stream_mode=["updates", "custom"]
                ):
                    if stream_mode == "custom":
                        yield custom_to_event(chunk)
                    elif stream_mode == "updates":
                        for node_name, patch in chunk.items():
                            if isinstance(patch, dict) and patch.get("usage"):
                                last_usage = patch["usage"]
                            yield {"type": "node.finished", "node": node_name}
            except Exception as error:
                yield {"type": "error", "message": str(error), "recoverable": False}
                return
            yield {
                "type": "run.finished",
                "run_id": run_id,
                "trace_id": trace_id,
                "usage": last_usage,
                "finish_reason": "stop",
            }


@lru_cache
def get_chat_workflow_service() -> ChatWorkflowService:
    return ChatWorkflowService()
