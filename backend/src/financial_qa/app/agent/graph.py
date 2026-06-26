"""LangGraph topology, compilation, and workflow-asset rendering.

Shape:

    START -> get_history -> orchestrator
                              ├─ FINANCIAL_QA -> preflight -> agent -> (tools -> agent)* -> validate ┐
                              └─ CHAT ────────-> chat_agent ───────────────────────────────────────-┤
                                                                                       save_history -> END

The orchestrator routes each turn: ``financial_qa`` runs the grounded retrieval flow, ``chat``
answers from conversation history / user input with no tools.

Run ``python -m financial_qa.app.agent.graph`` to (re)render assets/workflow.{mmd,jpeg} from the
compiled graph — the diagram is always generated, never hand-written. The ``tools`` node is expanded
to list its real bound tools so the rendered diagram shows what the agent can call.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from financial_qa.app.agent.chat_agent import ChatAgent
from financial_qa.app.agent.nodes import ChatNodes
from financial_qa.app.agent.schemas import State
from financial_qa.app.agent.tools import CHAT_TOOLS

GRAPH_NAME = "financial-qa-chat"


def route_after_orchestrator(state: State) -> str:
    """Branch on the orchestrator's decision. Keys double as the diagram's edge labels."""
    return "CHAT" if state.get("route") == "chat" else "FINANCIAL_QA"


def route_after_agent(state: State) -> str:
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if last is not None and getattr(last, "tool_calls", None):
        return "tools"
    return "validate"


def build_graph(nodes: ChatNodes):
    builder = StateGraph(State)
    builder.add_node("get_history", nodes.get_history_node)
    builder.add_node("orchestrator", nodes.orchestrator_node)
    builder.add_node("preflight", nodes.preflight_node)
    builder.add_node("agent", nodes.agent_node)
    builder.add_node("tools", nodes.tools_node)
    builder.add_node("validate", nodes.validate_node)
    builder.add_node("chat_agent", nodes.chat_node)
    builder.add_node("save_history", nodes.save_history_node)

    builder.add_edge(START, "get_history")
    builder.add_edge("get_history", "orchestrator")
    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {"FINANCIAL_QA": "preflight", "CHAT": "chat_agent"},
    )
    builder.add_edge("preflight", "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "validate": "validate"})
    builder.add_edge("tools", "agent")
    builder.add_edge("validate", "save_history")
    builder.add_edge("chat_agent", "save_history")
    builder.add_edge("save_history", END)
    return builder.compile(name=GRAPH_NAME)


@lru_cache
def get_compiled_graph():
    return build_graph(ChatNodes(ChatAgent()))


def _assets_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "assets").is_dir():
            return parent / "assets"
    return here.parents[3] / "assets"


def _inject_tool_list(mermaid_source: str, tool_names: list[str]) -> str:
    """Expand the single ``tools`` node into a labelled box that lists the real bound tools.

    LangGraph renders ``tools`` as one opaque node; this rewrites that node's label so the diagram
    shows the actual callable tools (title + divider + italic names), driven by ``CHAT_TOOLS``.
    """
    if not tool_names:
        return mermaid_source
    items = "<br/>".join(f"<i>{name}</i>" for name in tool_names)
    rich_label = f"tools(<b>tools</b><hr/>{items})"
    return mermaid_source.replace("tools(tools)", rich_label, 1)


def _write_local_jpeg(path: Path, mermaid_source: str, reason: Exception) -> None:
    """Create a readable local JPEG fallback when remote Mermaid rendering is unavailable."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    lines = ["Workflow diagram source", f"JPEG fallback reason: {reason}", "", *mermaid_source.splitlines()]
    width = 1400
    line_height = 16
    height = max(600, 40 + line_height * len(lines))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = 20
    for line in lines:
        draw.text((20, y), line[:180], fill="black", font=font)
        y += line_height
    image.save(path, "JPEG", quality=90)


def render_assets() -> None:
    graph = get_compiled_graph().get_graph(xray=True)
    assets = _assets_dir()
    assets.mkdir(parents=True, exist_ok=True)

    mermaid_source = _inject_tool_list(graph.draw_mermaid(), [tool_obj.name for tool_obj in CHAT_TOOLS])
    (assets / "workflow.mmd").write_text(mermaid_source, encoding="utf-8")
    try:
        # Render from the (tool-expanded) mermaid string, not graph.draw_mermaid_png(), so the JPEG
        # matches workflow.mmd. PNG render needs network (mermaid.ink); .mmd is always written.
        from langchain_core.runnables.graph import MermaidDrawMethod
        from langchain_core.runnables.graph_mermaid import draw_mermaid_png
        from PIL import Image

        png_bytes = draw_mermaid_png(mermaid_source, draw_method=MermaidDrawMethod.API)
        Image.open(BytesIO(png_bytes)).convert("RGB").save(assets / "workflow.jpeg", "JPEG")
        print(f"Wrote {assets/'workflow.mmd'} + {assets/'workflow.jpeg'}")
    except Exception as error:
        _write_local_jpeg(assets / "workflow.jpeg", mermaid_source, error)
        print(f"Wrote {assets/'workflow.mmd'} + local fallback {assets/'workflow.jpeg'}")


if __name__ == "__main__":
    render_assets()
