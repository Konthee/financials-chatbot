"""LangGraph node handlers: get_history -> agent -> tools? -> agent -> validate -> save_history."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from financial_qa.app.agent.chat_agent import ChatAgent
from financial_qa.app.agent.events import emit
from financial_qa.app.agent.prompt import (
    CHAT_SYSTEM_PROMPT,
    ORCHESTRATOR_PROMPT,
    VALIDATE_PROMPT,
    build_system_prompt,
)
from financial_qa.app.agent.schemas import State
from financial_qa.app.infrastructure import history as history_store

_QUALITATIVE_TERMS = (
    "factor",
    "factors",
    "reason",
    "reasons",
    "why",
    "strategy",
    "strategic",
    "business model",
    "revenue structure",
    "strength",
    "weakness",
    "ปัจจัย",
    "เหตุผล",
    "ทำไม",
    "กลยุทธ์",
    "โครงสร้างรายได้",
    "จุดแข็ง",
    "จุดอ่อน",
)

_COMPANY_TERMS: dict[str, str] = {
    "microsoft": "Microsoft",
    "msft": "Microsoft",
    "apple": "Apple",
    "aapl": "Apple",
    "google": "Google",
    "alphabet": "Google",
    "googl": "Google",
    "facebook": "Facebook",
    "meta": "Facebook",
    "amazon": "Amazon",
    "amzn": "Amazon",
}

_FINANCIAL_TERMS = (
    "revenue",
    "net income",
    "income",
    "growth",
    "compare",
    "comparison",
    "rank",
    "รายได้",
    "กำไร",
    "เติบโต",
    "เปรียบเทียบ",
    "สูงสุด",
)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _chunk_to_message(acc: AIMessageChunk | None) -> AIMessage:
    if acc is None:
        return AIMessage(content="")
    return AIMessage(
        content=acc.content,
        tool_calls=list(acc.tool_calls or []),
        usage_metadata=getattr(acc, "usage_metadata", None),
        additional_kwargs=acc.additional_kwargs,
        response_metadata=acc.response_metadata,
    )


def _merge_usage(previous: dict | None, message: AIMessage) -> dict:
    usage = dict(previous or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0.0})
    meta = getattr(message, "usage_metadata", None) or {}
    usage["input_tokens"] += int(meta.get("input_tokens", 0) or 0)
    usage["output_tokens"] += int(meta.get("output_tokens", 0) or 0)
    usage["total_tokens"] += int(meta.get("total_tokens", 0) or 0)
    return usage


def _describe_tool_calls(tool_calls: list[dict]) -> str:
    parts = []
    for call in tool_calls:
        args = call.get("args") or {}
        rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
        parts.append(f"{call.get('name')}({rendered})")
    return "Gathering evidence via " + "; ".join(parts)


def _describe_preflight(request_text: str, companies: list[str]) -> str:
    sources: list[str] = []
    if _has_financial_intent(request_text):
        sources.append("ตัวเลขการเงินจากฐานข้อมูล SQL")
    if _has_qualitative_intent(request_text):
        sources.append("ข้อความ 10-K สำหรับกลยุทธ์ โครงสร้างรายได้ หรือปัจจัยเชิงธุรกิจ")
    if not sources:
        sources.append("คำตอบจากหลักฐานที่มีในระบบ")

    company_text = ", ".join(companies) if companies else "บริษัทที่เกี่ยวข้องจากคำถาม"
    year_text = ", ".join(str(year) for year in (_years(request_text) or [2025]))
    return f"ต้องตรวจ {', '.join(sources)} สำหรับ {company_text} ในปี {year_text} ก่อนสรุปคำตอบ"


def _tool_payloads(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            try:
                payloads.append(json.loads(_text(message.content)))
            except (ValueError, TypeError):
                continue
    return payloads


def _request_text(state: State) -> str:
    parts: list[str] = []
    for message in state.get("input") or []:
        if isinstance(message, dict):
            parts.append(str(message.get("content") or ""))
    return "\n".join(parts)


def _has_qualitative_intent(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _QUALITATIVE_TERMS)


def _has_financial_intent(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _FINANCIAL_TERMS)


def _mentioned_companies(text: str) -> list[str]:
    lowered = text.lower()
    companies = {company for term, company in _COMPANY_TERMS.items() if re.search(rf"\b{re.escape(term)}\b", lowered)}
    return sorted(companies)


def _years(text: str) -> list[int]:
    years = {int(year) for year in re.findall(r"\b(20\d{2})\b", text)}
    for start, end in re.findall(r"\b(20\d{2})\s*[-–]\s*(20\d{2})\b", text):
        years.update(range(int(start), int(end) + 1))
    return sorted(years)


def _metric(text: str) -> str:
    lowered = text.lower()
    if "net income" in lowered:
        return "net_income"
    if "operating income" in lowered:
        return "operating_income"
    if "gross profit" in lowered:
        return "gross_profit"
    return "revenue"


def _operation(text: str) -> str:
    lowered = text.lower()
    if "growth" in lowered or "เติบโต" in lowered:
        return "growth"
    if "rank" in lowered or "สูงสุด" in lowered:
        return "rank"
    return "values"


def _vector_matches(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for payload in payloads for item in payload.get("matches", []) if item.get("text")]


def _no_filing_for(payloads: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for payload in payloads:
        missing.extend(str(company) for company in payload.get("no_filing_for", []))
    return sorted(set(missing))


def _filing_companies(payloads: list[dict[str, Any]]) -> list[str]:
    companies = {
        item.get("company")
        for payload in payloads
        for item in payload.get("matches", [])
        if item.get("company")
    }
    return sorted(companies)


def _sql_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        rows.extend(payload.get("rows", []))
    return rows


def _growth_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        rows.extend(payload.get("growth", []))
    return rows


def _is_thai(text: str) -> bool:
    return any("\u0e00" <= char <= "\u0e7f" for char in text)


def _corrected_answer(request_text: str, payloads: list[dict[str, Any]], unsupported_claims: list[str]) -> str:
    growth = _growth_rows(payloads)
    sql_rows = _sql_rows(payloads)
    no_filing = _no_filing_for(payloads)
    thai = _is_thai(request_text)

    if thai:
        lines = ["คำตอบฉบับแก้ไข: ผมตัดส่วนที่ไม่มีหลักฐานรองรับออกแล้ว"]
        if growth:
            lines.append("\nข้อมูลตัวเลขจาก SQL:")
            for row in sorted(growth, key=lambda item: item.get("growth_pct", 0), reverse=True):
                lines.append(
                    f"- {row['company']} ({row['ticker']}): {row['growth_pct']}% "
                    f"จาก FY{row['from_year']} {row['from_value']} เป็น FY{row['to_year']} {row['to_value']}."
                )
        elif sql_rows:
            lines.append("\nข้อมูลตัวเลขจาก SQL:")
            for row in sql_rows:
                lines.append(f"- {row['company']} ({row['ticker']}) FY{row['year']}: {row['value']}.")
        if no_filing:
            lines.append("\nไม่มี 10-K ในชุดข้อมูลสำหรับ: " + ", ".join(no_filing) + ".")
        lines.append("ผมไม่สามารถให้ปัจจัยเชิงคุณภาพที่ไม่มีข้อความ 10-K รองรับได้จากข้อมูลที่มีอยู่.")
        lines.append("Claims ที่ถูกตัดออก: " + "; ".join(unsupported_claims))
        return "\n".join(lines)

    lines = ["Corrected answer: I removed claims that were not grounded in the provided evidence."]
    if growth:
        lines.append("\nSQL figures:")
        for row in sorted(growth, key=lambda item: item.get("growth_pct", 0), reverse=True):
            lines.append(
                f"- {row['company']} ({row['ticker']}): {row['growth_pct']}% "
                f"from FY{row['from_year']} {row['from_value']} to FY{row['to_year']} {row['to_value']}."
            )
    elif sql_rows:
        lines.append("\nSQL figures:")
        for row in sql_rows:
            lines.append(f"- {row['company']} ({row['ticker']}) FY{row['year']}: {row['value']}.")
    if no_filing:
        lines.append("\nNo 10-K filing is available in the provided data for: " + ", ".join(no_filing) + ".")
    lines.append("I cannot provide qualitative factors without supporting 10-K text evidence.")
    lines.append("Removed claims: " + "; ".join(unsupported_claims))
    return "\n".join(lines)


def _render_validation(draft: str, filing_companies: list[str], sql_rows: list[dict[str, Any]]) -> str:
    return (
        "COMPANIES WITH 10-K EVIDENCE: " + (", ".join(filing_companies) or "(none)") + "\n\n"
        "SQL FIGURES (the only valid numbers):\n"
        f"{json.dumps(sql_rows, ensure_ascii=False, default=str)}\n\n"
        "DRAFT ANSWER:\n"
        f"{draft}"
    )


def _answer_language_instruction(request_text: str) -> str:
    if _is_thai(request_text):
        return (
            "FINAL ANSWER LANGUAGE CHECK: The user's current question is Thai. Write the final "
            "answer in Thai only. Markdown headings, bullet labels, summaries, caveats, and "
            "explanations must be Thai. Keep only company names, tickers, filenames, metric names, "
            "citations, and exact numeric values unchanged. Do not use English headings such as "
            "'Revenue Growth Comparison', 'Revenue Figures', 'Highest Growth Rate', 'Key Factors', "
            "or 'Summary'."
        )
    return (
        "FINAL ANSWER LANGUAGE CHECK: Write the final answer in the same language as the user's "
        "current question. Evidence and tool outputs may use another language, but the final "
        "headings, labels, summaries, caveats, and explanations must follow the user's question."
    )


def _fallback_route(request_text: str) -> str:
    """Deterministic route used only when the LLM router call fails."""
    if _mentioned_companies(request_text) or _has_financial_intent(request_text) or _has_qualitative_intent(request_text):
        return "financial_qa"
    return "chat"


def _describe_route(route: str, reason: str) -> str:
    label = "สายงานตอบคำถามการเงิน (financial-qa)" if route == "financial_qa" else "สายงานสนทนาทั่วไป (chat)"
    text = f"เลือกเส้นทาง: {label}"
    return f"{text} — {reason}" if reason else text


class ChatNodes:
    def __init__(self, agent: ChatAgent) -> None:
        self._agent = agent
        self._system_prompt = build_system_prompt()
        self._orchestrator_prompt = ORCHESTRATOR_PROMPT
        self._chat_system_prompt = CHAT_SYSTEM_PROMPT

    async def get_history_node(self, state: State, config: RunnableConfig | None = None) -> dict:
        return await history_store.get_history(state)

    async def orchestrator_node(self, state: State, config: RunnableConfig | None = None) -> dict:
        """Route the turn to the grounded financial-qa flow or to plain chat."""
        request_text = _request_text(state)
        transcript = [SystemMessage(content=self._orchestrator_prompt), *state["messages"]]
        try:
            decision = await self._agent.route(transcript, config=config)
            route, reason = decision.route, decision.reason
        except Exception:  # LLM/router failure -> deterministic intent heuristic, never crash the run
            route, reason = _fallback_route(request_text), ""
        emit("reasoning.delta", phase="orchestrator", text=_describe_route(route, reason))
        return {"route": route}

    async def preflight_node(self, state: State, config: RunnableConfig | None = None) -> dict:
        request_text = _request_text(state)
        companies = _mentioned_companies(request_text)
        emit("reasoning.delta", phase="preflight", text=_describe_preflight(request_text, companies))
        if not companies:
            return {}

        payloads: list[dict[str, Any]] = []
        if _has_financial_intent(request_text):
            from financial_qa.app.agent.tools import query_financials

            financial_payload = await query_financials.ainvoke(
                {
                    "companies": companies,
                    "metric": _metric(request_text),
                    "years": _years(request_text) or [2025],
                    "op": _operation(request_text),
                },
                config=config,
            )
            payloads.append(json.loads(financial_payload))

        if _has_qualitative_intent(request_text):
            from financial_qa.app.agent.tools import search_filings

            filing_payload = await search_filings.ainvoke(
                {"query": request_text, "companies": companies, "top_k": 5},
                config=config,
            )
            payloads.append(json.loads(filing_payload))

        if not payloads:
            return {}

        evidence_block = (
            "PRELOADED TOOL EVIDENCE (already gathered for this request; treat it exactly like tool "
            "output and cite it in the final answer):\n"
            f"{json.dumps(payloads, ensure_ascii=False, default=str)}"
        )
        return {"messages": [SystemMessage(content=evidence_block)], "preloaded_evidence": payloads}

    async def agent_node(self, state: State, config: RunnableConfig | None = None) -> dict:
        emit("reasoning.delta", phase="agent", text="กำลังสังเคราะห์คำตอบจากหลักฐานที่ดึงมา และจัดรูปแบบเป็น Markdown")
        request_text = _request_text(state)
        transcript = [
            SystemMessage(content=self._system_prompt),
            *state["messages"],
            SystemMessage(content=_answer_language_instruction(request_text)),
        ]
        acc: AIMessageChunk | None = None
        async for chunk in self._agent.astream(transcript, config=config):
            acc = chunk if acc is None else acc + chunk

        message = _chunk_to_message(acc)
        usage = _merge_usage(state.get("usage"), message)
        if message.tool_calls:
            emit("reasoning.delta", phase="agent", text=_describe_tool_calls(message.tool_calls))
            return {"messages": [message], "usage": usage}
        return {"messages": [message], "output": _text(message.content), "usage": usage}

    async def chat_node(self, state: State, config: RunnableConfig | None = None) -> dict:
        """Chat route: answer from conversation history and the user's input, with no tools."""
        emit("reasoning.delta", phase="chat", text="กำลังตอบจากบริบทการสนทนาและคำถามของผู้ใช้ (ไม่เรียกเครื่องมือ)")
        request_text = _request_text(state)
        transcript = [
            SystemMessage(content=self._chat_system_prompt),
            *state["messages"],
            SystemMessage(content=_answer_language_instruction(request_text)),
        ]
        acc: AIMessageChunk | None = None
        async for chunk in self._agent.astream_chat(transcript, config=config):
            acc = chunk if acc is None else acc + chunk

        message = _chunk_to_message(acc)
        usage = _merge_usage(state.get("usage"), message)
        answer = _text(message.content)
        emit("answer.delta", text=answer)
        return {"messages": [message], "output": answer, "grounded": True, "usage": usage}

    async def tools_node(self, state: State, config: RunnableConfig | None = None) -> dict:
        last = state["messages"][-1]
        tool_messages = await self._agent.arun_tool_calls(last.tool_calls, config=config)
        return {"messages": tool_messages}

    async def validate_node(self, state: State, config: RunnableConfig | None = None) -> dict:
        draft = (state.get("output") or "").strip()
        if not draft:
            emit("validation", grounded=True, unsupported_claims=[])
            return {"grounded": True}

        payloads = [*state.get("preloaded_evidence", []), *_tool_payloads(state["messages"])]
        request_text = _request_text(state)
        unsupported_claims: list[str] = []
        if _has_qualitative_intent(request_text) and not _vector_matches(payloads):
            unsupported_claims.append("Qualitative factors or strategy were requested, but no 10-K vector evidence was retrieved.")

        if not unsupported_claims:
            emit("validation", grounded=True, unsupported_claims=[])
            emit("answer.delta", text=draft)
            return {"grounded": True, "output": draft}

        corrected = _corrected_answer(request_text, payloads, unsupported_claims)
        emit("validation", grounded=False, unsupported_claims=unsupported_claims, action="corrected")
        emit("answer.delta", text=corrected)
        return {"grounded": False, "output": corrected}

    async def save_history_node(self, state: State, config: RunnableConfig | None = None) -> dict:
        return await history_store.save_history(state)
