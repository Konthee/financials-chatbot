"""Data-coverage and company-alias resolution — the single source of truth for the few
real-world name differences in the corpus.

Filing coverage is *derived from the data* (the PDFs present in ``data/10k_filings/``), not
hardcoded. The only static config is the small alias table for names that genuinely differ across
sources: the SQL table stores "Google"/"Meta" while the filings are "Alphabet"/"Meta", and users
say "Facebook" for Meta. No question answers are hardcoded anywhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from financial_qa.app.infrastructure.settings import get_settings

# Lowercased common term -> company name as stored in financial_data.company.
_ALIAS_TO_SQL: dict[str, str] = {
    "alphabet": "google",
    "facebook": "meta",
}

# Lowercased common term -> company name as used in the 10-K filenames (filing identity).
_ALIAS_TO_FILING: dict[str, str] = {
    "google": "Alphabet",
    "googl": "Alphabet",
    "facebook": "Meta",
}


@lru_cache
def filing_companies() -> tuple[str, ...]:
    """Companies that have a 10-K in the corpus, derived from ``data/10k_filings/``."""
    filings_dir = Path(get_settings().data_dir) / "10k_filings"
    if not filings_dir.is_dir():
        return ()
    names = {pdf.stem.split("_", 1)[0] for pdf in filings_dir.glob("*.pdf")}
    return tuple(sorted(names))


def sql_match_terms(company: str) -> list[str]:
    """Lowercased candidates to match against ``company`` and ``ticker`` columns."""
    term = company.strip().lower()
    candidates = {term}
    if term in _ALIAS_TO_SQL:
        candidates.add(_ALIAS_TO_SQL[term])
    return sorted(candidates)


def to_filing_company(company: str) -> str | None:
    """Resolve a user term to the filing-company name, or None if no filing exists for it."""
    term = company.strip().lower()
    resolved = _ALIAS_TO_FILING.get(term, company.strip().title())
    return resolved if resolved in filing_companies() else None


def has_filing(company: str) -> bool:
    return to_filing_company(company) is not None


def coverage_prompt_block() -> str:
    """A data-derived coverage statement injected into the agent's system prompt."""
    filings = filing_companies()
    filings_str = ", ".join(filings) if filings else "(none)"
    return (
        "DATA COVERAGE (read carefully):\n"
        "- Structured financials (the financial tools) cover ~48 large U.S. public companies for "
        "fiscal years 2022-2025: revenue, net_income, operating_income, gross_profit (USD).\n"
        f"- Qualitative 10-K filing text (the document search tool) exists ONLY for: {filings_str} "
        "(all FY2025). Note: 'Google' = Alphabet, and 'Facebook' = Meta.\n"
        "- A company can have financial numbers but NO filing text (e.g. Microsoft). For such a "
        "company you may report its SQL figures, but you MUST state plainly that no 10-K filing is "
        "available to ground any qualitative/strategy/'why' explanation. Never invent such factors."
    )
