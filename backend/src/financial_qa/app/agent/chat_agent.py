"""Model + tool handling: binds tools for the agent and exposes the grounding judge."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import ToolMessage

from financial_qa.app.agent.schemas import GroundingVerdict, RouteDecision
from financial_qa.app.agent.tools import CHAT_TOOL_MAP, CHAT_TOOLS
from financial_qa.app.integrations.llm.models import create_chat_model


class ChatAgent:
    def __init__(self) -> None:
        base = create_chat_model(temperature=0.0, streaming=True)
        self._chat_model = base                  # plain model (no tools) — chat route
        self._tool_model = base.bind_tools(CHAT_TOOLS)  # tool-enabled — financial-qa route
        decision_model = create_chat_model(temperature=0.0, streaming=False)
        # function_calling avoids OpenAI strict json_schema's "all fields required" constraint,
        # which clashes with the defaulted fields on GroundingVerdict / RouteDecision.
        self._judge = decision_model.with_structured_output(GroundingVerdict, method="function_calling")
        self._router = decision_model.with_structured_output(RouteDecision, method="function_calling")

    def astream(self, messages: list, config: dict | None = None):
        """Stream the tool-enabled model (yields AIMessageChunk)."""
        return self._tool_model.astream(messages, config=config)

    def astream_chat(self, messages: list, config: dict | None = None):
        """Stream the plain (no-tools) model for the chat route (yields AIMessageChunk)."""
        return self._chat_model.astream(messages, config=config)

    async def route(self, messages: list, config: dict | None = None) -> RouteDecision:
        return await self._router.ainvoke(messages, config=config)

    async def arun_tool_calls(self, tool_calls: list[dict], config: dict | None = None) -> list[ToolMessage]:
        results: list[ToolMessage] = []
        for call in tool_calls:
            name = call.get("name") or ""
            args = call.get("args") or {}
            tool = CHAT_TOOL_MAP.get(name)
            if tool is None:
                content: Any = f"Unknown tool: {name}"
            else:
                try:
                    content = str(await tool.ainvoke(args, config=config))
                except Exception as error:  # surfaced to the model, never crashes the run
                    content = f"Tool {name} error: {error}"
            results.append(ToolMessage(content=content, tool_call_id=str(call.get("id")), name=name))
        return results

    async def judge(self, messages: list, config: dict | None = None) -> GroundingVerdict:
        return await self._judge.ainvoke(messages, config=config)
