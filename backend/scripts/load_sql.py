"""Load the provided PostgreSQL dump into the ``financial_data`` table.

``data/financial_data.sql`` is a pg_dump-style file: a DDL prelude (DROP/CREATE) followed by a
single ``COPY financial_data (...) FROM stdin`` block of tab-separated rows terminated by ``\\.``.
We run the DDL, then stream the COPY block through psycopg3's binary-safe COPY API. Re-running is
idempotent (the DDL drops and recreates the table).

    python scripts/load_sql.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

from financial_qa.app.infrastructure.settings import get_settings

EXPECTED_ROWS = 192


def _split_dump(sql_text: str) -> tuple[list[str], str, str]:
    """Return (ddl_statements, copy_statement, data_block) from a pg_dump-style file."""
    lines = sql_text.splitlines()
    copy_idx = next(
        i
        for i, line in enumerate(lines)
        if line.strip().upper().startswith("COPY ") and "FROM STDIN" in line.upper()
    )
    end_idx = next(i for i in range(copy_idx + 1, len(lines)) if lines[i].strip() == r"\.")

    ddl_text = "\n".join(lines[:copy_idx])
    ddl_statements = [stmt.strip() for stmt in ddl_text.split(";") if stmt.strip()]
    copy_statement = lines[copy_idx].strip().rstrip(";")
    data_block = "\n".join(lines[copy_idx + 1 : end_idx]) + "\n"
    return ddl_statements, copy_statement, data_block


def main() -> int:
    settings = get_settings()
    sql_path = Path(settings.data_dir) / "financial_data.sql"
    if not sql_path.is_file():
        print(f"ERROR: dump not found at {sql_path}", file=sys.stderr)
        return 1

    ddl_statements, copy_statement, data_block = _split_dump(sql_path.read_text(encoding="utf-8"))

    with psycopg.connect(settings.psycopg_dsn) as conn:
        with conn.cursor() as cur:
            for statement in ddl_statements:
                cur.execute(statement)
            with cur.copy(copy_statement) as copy:
                copy.write(data_block)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM financial_data")
            (count,) = cur.fetchone()

    print(f"Loaded financial_data: {count} rows")
    if count != EXPECTED_ROWS:
        print(f"WARNING: expected {EXPECTED_ROWS} rows, got {count}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
