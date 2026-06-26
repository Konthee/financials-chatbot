"""Grounded tools. Each returns a JSON string built only from real data, emits progress + evidence
to the stream, and reports missing data explicitly so the model can never fabricate."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import sqlglot
from langchain_core.tools import tool
from sqlalchemy import text
from sqlglot import exp

from financial_qa.app.agent import coverage
from financial_qa.app.agent.events import emit
from financial_qa.app.infrastructure.db import SessionLocal
from financial_qa.app.integrations.llm.embeddings import embed_query
from financial_qa.app.integrations.vectorstore import pinecone_client

_ALLOWED_METRICS = ("revenue", "net_income", "operating_income", "gross_profit")
_ALLOWED_TABLE = "financial_data"
_SELECT_ROW_CAP = 200


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------------------------
# query_financials — structured, allowlisted, fully grounded
# --------------------------------------------------------------------------------------------
@tool
async def query_financials(
    companies: list[str],
    metric: Literal["revenue", "net_income", "operating_income", "gross_profit"],
    years: list[int],
    op: Literal["values", "growth", "rank"] = "values",
    sector: str | None = None,
) -> str:
    """Get structured income-statement figures from the financial_data table.

    Use this for any numeric question about U.S. public companies. `metric` is one of revenue,
    net_income, operating_income, gross_profit (USD). `op="growth"` additionally returns the percent
    change from the earliest to the latest requested year per company; `op="rank"` ranks returned
    companies by the metric within each requested year. `sector` optionally restricts rows to one
    sector. Company names or tickers both work (e.g. "Apple" or "AAPL", "Google" resolves to
    Alphabet/GOOGL). Missing companies/years and NULL metrics are reported explicitly in the result —
    never guess them.
    """
    if metric not in _ALLOWED_METRICS:
        return _json({"error": f"Unsupported metric '{metric}'."})

    emit(
        "tool.selected",
        tool="query_financials",
        args={"companies": companies, "metric": metric, "years": years, "op": op, "sector": sector},
    )

    names: set[str] = set()
    for company in companies:
        names.update(coverage.sql_match_terms(company))

    sector_clause = "AND lower(sector) = :sector " if sector else ""
    sql = (
        f"SELECT company, ticker, year, {metric} AS value "
        f"FROM {_ALLOWED_TABLE} "
        "WHERE (lower(company) = ANY(:names) OR lower(ticker) = ANY(:names)) "
        f"AND year = ANY(:years) {sector_clause}ORDER BY ticker, year"
    )
    params: dict[str, Any] = {"names": sorted(names), "years": years}
    if sector:
        params["sector"] = sector.strip().lower()
    async with SessionLocal() as session:
        result = await session.execute(text(sql), params)
        rows = [dict(row) for row in result.mappings().all()]

    emit("sql.query", sql=sql, params=params, row_count=len(rows))

    # Per-company growth (computed server-side so the model never does ungrounded arithmetic).
    growth: list[dict[str, Any]] = []
    if op == "growth":
        by_ticker: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_ticker.setdefault(row["ticker"], []).append(row)
        for ticker, group in by_ticker.items():
            valid = sorted((r for r in group if r["value"] is not None), key=lambda r: r["year"])
            if len(valid) >= 2 and valid[0]["value"]:
                first, last = valid[0], valid[-1]
                pct = (last["value"] - first["value"]) / first["value"] * 100
                growth.append(
                    {
                        "company": first["company"],
                        "ticker": ticker,
                        "from_year": first["year"],
                        "from_value": first["value"],
                        "to_year": last["year"],
                        "to_value": last["value"],
                        "growth_pct": round(pct, 2),
                    }
                )

    rankings: list[dict[str, Any]] = []
    if op == "rank":
        for year in sorted(set(years)):
            year_rows = [row for row in rows if row["year"] == year and row["value"] is not None]
            ranked = sorted(year_rows, key=lambda row: row["value"], reverse=True)
            for idx, row in enumerate(ranked, start=1):
                rankings.append(
                    {
                        "rank": idx,
                        "company": row["company"],
                        "ticker": row["ticker"],
                        "year": row["year"],
                        "value": row["value"],
                    }
                )

    # Explicit gaps.
    missing: list[str] = []
    for company in companies:
        cands = set(coverage.sql_match_terms(company))
        matched_rows = [r for r in rows if r["company"].lower() in cands or r["ticker"].lower() in cands]
        if not matched_rows:
            missing.append(f"No financial_data for '{company}'.")
            continue
        present_years = {r["year"] for r in matched_rows}
        for year in sorted(set(years) - present_years):
            missing.append(f"{company} FY{year}: no financial_data row.")
    for row in rows:
        if row["value"] is None:
            missing.append(f"{row['company']} FY{row['year']}: {metric} is not available (NULL).")

    emit("evidence", source="sql", items=rows)
    return _json(
        {
            "metric": metric,
            "op": op,
            "sector": sector,
            "rows": rows,
            "growth": growth,
            "rankings": rankings,
            "missing": missing,
        }
    )


# --------------------------------------------------------------------------------------------
# run_sql_select — guarded read-only SELECT escape hatch
# --------------------------------------------------------------------------------------------
def _validate_select(raw_sql: str) -> str:
    statements = sqlglot.parse(raw_sql, read="postgres")
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise ValueError("Provide exactly one SQL statement.")
    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise ValueError("Only SELECT statements are allowed.")
    for forbidden in (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command):
        if statement.find(forbidden):
            raise ValueError("Only read-only SELECT statements are allowed.")
    tables = {t.name.lower() for t in statement.find_all(exp.Table)}
    if tables - {_ALLOWED_TABLE}:
        raise ValueError(f"Only the {_ALLOWED_TABLE} table may be queried.")
    if statement.args.get("limit") is None:
        statement = statement.limit(_SELECT_ROW_CAP)
    return statement.sql(dialect="postgres")


@tool
async def run_sql_select(sql: str) -> str:
    """Run a single read-only SELECT against the `financial_data` table for ad-hoc or ranking
    questions the structured tool can't express (e.g. "top 5 Technology companies by revenue in
    2025"). Columns: company, ticker, sector, year, revenue, net_income, operating_income,
    gross_profit. Only SELECT is permitted; a row limit is enforced. Returns rows only.
    """
    emit("tool.selected", tool="run_sql_select", args={"sql": sql})
    try:
        safe_sql = _validate_select(sql)
    except Exception as error:
        return _json({"error": f"Rejected SQL: {error}"})

    async with SessionLocal() as session:
        result = await session.execute(text(safe_sql))
        rows = [dict(row) for row in result.mappings().all()]

    emit("sql.query", sql=safe_sql, params={}, row_count=len(rows))
    emit("evidence", source="sql", items=rows)
    return _json({"sql": safe_sql, "rows": rows, "row_count": len(rows)})


# --------------------------------------------------------------------------------------------
# search_filings — vector search over FY2025 10-K text
# --------------------------------------------------------------------------------------------
def _company_from_source(source: str) -> str:
    return os.path.basename(source).split("_", 1)[0]


def _page_label(metadata: dict[str, Any]) -> Any:
    label = metadata.get("page_label")
    if label:
        return label
    page = metadata.get("page")
    return page + 1 if isinstance(page, int) else page


@tool
async def search_filings(query: str, companies: list[str] | None = None, top_k: int = 6) -> str:
    """Semantic search over FY2025 10-K filing text for qualitative content (strategy, business
    model, revenue segments, risks, the "why"). Filings exist ONLY for Alphabet (Google), Amazon,
    Apple, Meta. If you request a company without a filing, the result says so — do not invent
    qualitative explanations for it. Returns chunks with their source filename and page for citation.
    """
    emit("tool.selected", tool="search_filings", args={"query": query, "companies": companies, "top_k": top_k})

    targets: list[str] = []
    no_filing_for: list[str] = []
    if companies:
        for company in companies:
            resolved = coverage.to_filing_company(company)
            (targets.append(resolved) if resolved else no_filing_for.append(company))
        if no_filing_for:
            emit(
                "coverage.notice",
                message=f"No 10-K filing available for: {', '.join(no_filing_for)}.",
                missing=no_filing_for,
            )
        if not targets:
            return _json({"matches": [], "no_filing_for": no_filing_for})

    target_set = set(targets)
    pool_k = min(max(top_k * (len(target_set) or 1) * 3, 20), 40)
    vector = await embed_query(query)
    emit("vector.search", query=query, top_k=top_k, pool_k=pool_k)
    raw = await pinecone_client.query(vector, top_k=pool_k)

    selected: list[dict[str, Any]] = []
    per_company: dict[str, int] = {}
    for match in raw:
        metadata = match["metadata"]
        company = _company_from_source(metadata.get("source", ""))
        if target_set and company not in target_set:
            continue
        if target_set and per_company.get(company, 0) >= top_k:
            continue
        per_company[company] = per_company.get(company, 0) + 1
        selected.append(
            {
                "text": metadata.get("text", ""),
                "doc": os.path.basename(metadata.get("source", "")),
                "company": company,
                "page": _page_label(metadata),
                "score": round(float(match["score"]), 4),
            }
        )
        if not target_set and len(selected) >= top_k:
            break

    emit("evidence", source="vector", items=selected)
    payload: dict[str, Any] = {"matches": selected}
    if no_filing_for:
        payload["no_filing_for"] = no_filing_for
    if not selected:
        payload["note"] = "No relevant filing passages found."
    return _json(payload)


CHAT_TOOLS = [query_financials, run_sql_select, search_filings]
CHAT_TOOL_MAP = {tool_obj.name: tool_obj for tool_obj in CHAT_TOOLS}
