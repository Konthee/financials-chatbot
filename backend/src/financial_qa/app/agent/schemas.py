"""Public API schemas and the internal LangGraph state."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# --- API request -------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""


class ChatRunRequest(BaseModel):
    """Body of POST /api/v1/chat/runs/stream and /api/v1/chat/runs."""

    messages: list[ChatMessage] = Field(default_factory=list)
    session_id: str | None = Field(
        default=None, description="Target chat session; omit to create a new one."
    )


# --- Routing decision (orchestrator node structured output) ------------------------------


class RouteDecision(BaseModel):
    """How the orchestrator routes a turn: the grounded financial-qa flow or plain chat."""

    route: Literal["financial_qa", "chat"] = Field(
        description=(
            "'financial_qa' when answering needs retrieved data (SQL financial figures or 10-K "
            "filing content); 'chat' when it is answerable from the conversation or the message alone."
        )
    )
    reason: str = Field(default="", description="One short sentence explaining the choice.")


# --- Grounding verdict (validate node structured output) ---------------------------------


class GroundingVerdict(BaseModel):
    grounded: bool = Field(description="True only if every factual/numeric/causal claim is backed by evidence.")
    unsupported_claims: list[str] = Field(default_factory=list)


# --- LangGraph state ---------------------------------------------------------------------


class State(TypedDict, total=False):
    user_id: str
    input: list[dict[str, Any]]          # raw request messages
    messages: Annotated[list, add_messages]  # working transcript (LangChain messages)
    route: str                           # orchestrator decision: "financial_qa" | "chat"
    output: str                          # final answer text
    grounded: bool                       # validate verdict
    usage: dict[str, Any]                # normalized token usage
    persist_start_index: int             # index into messages from which to persist
    preloaded_evidence: list[dict[str, Any]]  # deterministic tool payloads gathered before agent


def empty_usage() -> dict[str, Any]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0.0}
