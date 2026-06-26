"""Rebuild Pinecone vectors directly from the source 10-K PDFs.

This is the Option B ingestion path from the take-home PDF. It extracts text from
``DATA_DIR/10k_filings/*.pdf``, splits it with the same recursive-character style settings
documented for the provided fixture (chunk_size=1000, overlap=200), embeds each chunk with the
configured OpenAI-compatible embedding model, and upserts the vectors into Pinecone Local.

The output record shape matches the provided JSONL fixture: each line contains ``id``, ``namespace``,
``values``, and ``metadata``. By default this script writes one vector per unique chunk to avoid
duplicate records. Use ``--replicas-per-chunk 2`` only when you intentionally want duplicate vectors
for compatibility testing.

By default the script clears the configured namespace before upserting because regenerated IDs will
not match the random IDs in the provided fixture. Use ``--append`` to keep existing vectors.

Examples:
    python scripts/reembed_vectors_from_pdfs.py --dry-run
    python scripts/reembed_vectors_from_pdfs.py
    python scripts/reembed_vectors_from_pdfs.py --output-jsonl /tmp/pinecone_vectors.reembedded.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

from financial_qa.app.infrastructure.settings import get_settings

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 96
UPSERT_BATCH_SIZE = 50
SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-embed 10-K PDFs and upsert them into Pinecone Local.")
    parser.add_argument("--pdf-dir", type=Path, default=None, help="Directory of source PDFs. Defaults to DATA_DIR/10k_filings.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--embed-batch-size", type=int, default=EMBED_BATCH_SIZE)
    parser.add_argument("--upsert-batch-size", type=int, default=UPSERT_BATCH_SIZE)
    parser.add_argument("--replicas-per-chunk", type=int, default=1, help="Vector records to write per unique chunk.")
    parser.add_argument("--append", action="store_true", help="Append to the namespace instead of clearing it first.")
    parser.add_argument("--dry-run", action="store_true", help="Extract and chunk only; do not embed or upsert.")
    parser.add_argument("--output-jsonl", type=Path, default=None, help="Optional path to write embedded records as JSONL.")
    return parser.parse_args()


def _pdf_date(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    if not text.startswith("D:"):
        return text
    raw = text[2:]
    try:
        year = int(raw[0:4])
        month = int(raw[4:6] or "1")
        day = int(raw[6:8] or "1")
        hour = int(raw[8:10] or "0")
        minute = int(raw[10:12] or "0")
        second = int(raw[12:14] or "0")
        tzinfo = UTC
        if len(raw) >= 20 and raw[14] in "+-":
            sign = 1 if raw[14] == "+" else -1
            tz_hour = int(raw[15:17])
            tz_minute = int(raw[18:20])
            tzinfo = timezone(sign * timedelta(hours=tz_hour, minutes=tz_minute))
        return datetime(year, month, day, hour, minute, second, tzinfo=tzinfo).astimezone(UTC).isoformat()
    except Exception:
        return text


def _meta_value(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key) or metadata.get("/" + key.capitalize())
    return str(value) if value else None


def _split_recursive(text: str, *, chunk_size: int, overlap: int, separators: tuple[str, ...] = SEPARATORS) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=list(separators),
        length_function=len,
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def _page_records(pdf_path: Path, *, chunk_size: int, overlap: int) -> Iterator[dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    metadata = dict(reader.metadata or {})
    total_pages = len(reader.pages)
    base_metadata = {
        "creationdate": _pdf_date(metadata.get("/CreationDate")),
        "creator": _meta_value(metadata, "Creator"),
        "moddate": _pdf_date(metadata.get("/ModDate")),
        "producer": _meta_value(metadata, "Producer"),
        "source": str(pdf_path.resolve()),
        "title": _meta_value(metadata, "Title") or pdf_path.stem,
        "total_pages": total_pages,
    }

    for page_index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for chunk_index, chunk in enumerate(_split_recursive(text, chunk_size=chunk_size, overlap=overlap)):
            chunk_key = f"{pdf_path.name}:{page_index}:{chunk_index}:{chunk[:80]}"
            yield {
                "chunk_key": chunk_key,
                "metadata": {
                    **base_metadata,
                    "page": page_index,
                    "page_label": str(page_index + 1),
                    "text": chunk,
                },
            }


def _iter_records(pdf_dir: Path, *, chunk_size: int, overlap: int) -> Iterator[dict[str, Any]]:
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        yield from _page_records(pdf_path, chunk_size=chunk_size, overlap=overlap)


def _batched(items: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _index_host(pc: Pinecone, name: str) -> str:
    host = pc.describe_index(name).host
    if not host.startswith("http"):
        host = f"http://{host}"
    return host


def _embed_records(records: list[dict[str, Any]], *, batch_size: int, replicas_per_chunk: int) -> Iterator[dict[str, Any]]:
    settings = get_settings()
    client = OpenAI(base_url=settings.embedding_base_url, api_key=settings.embedding_api_key)
    for batch in _batched(records, batch_size):
        response = client.embeddings.create(
            model=settings.embedding_model_name,
            input=[record["metadata"]["text"] for record in batch],
            dimensions=settings.embedding_dim,
        )
        for record, item in zip(batch, response.data, strict=True):
            for replica in range(replicas_per_chunk):
                yield {
                    "id": str(uuid.uuid4()),
                    "namespace": settings.pinecone_namespace,
                    "values": item.embedding,
                    "metadata": record["metadata"],
                }


def _write_jsonl(path: Path, records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            yield record


def _upsert(records: Iterator[dict[str, Any]], *, clear: bool, batch_size: int) -> int:
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key, host=settings.pinecone_host)
    if not pc.has_index(settings.pinecone_index_name):
        print(f"Creating index '{settings.pinecone_index_name}' (dim={settings.embedding_dim}, cosine)")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(host=_index_host(pc, settings.pinecone_index_name))
    if clear:
        try:
            index.delete(delete_all=True, namespace=settings.pinecone_namespace)
            print(f"Cleared namespace '{settings.pinecone_namespace}'")
        except Exception as error:
            print(f"Namespace clear skipped: {error}")

    total = 0
    pending: list[dict[str, Any]] = []
    for record in records:
        pending.append({"id": record["id"], "values": record["values"], "metadata": record["metadata"]})
        if len(pending) >= batch_size:
            index.upsert(vectors=pending, namespace=settings.pinecone_namespace)
            total += len(pending)
            print(f"  upserted {total} vectors", end="\r", flush=True)
            pending = []
    if pending:
        index.upsert(vectors=pending, namespace=settings.pinecone_namespace)
        total += len(pending)
        print(f"  upserted {total} vectors", end="\r", flush=True)
    print()
    return total


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    pdf_dir = args.pdf_dir or Path(settings.data_dir) / "10k_filings"
    if not pdf_dir.is_dir():
        print(f"ERROR: PDF directory not found at {pdf_dir}", file=sys.stderr)
        return 1

    records = list(_iter_records(pdf_dir, chunk_size=args.chunk_size, overlap=args.chunk_overlap))
    print(f"Extracted {len(records)} chunks from {pdf_dir}")
    if args.dry_run:
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            count = sum(1 for record in records if record["metadata"]["source"] == str(pdf_path.resolve()))
            print(f"  {pdf_path.name}: {count} chunks, {count * args.replicas_per_chunk} vectors")
        print(f"Expected vector records with replicas_per_chunk={args.replicas_per_chunk}: {len(records) * args.replicas_per_chunk}")
        return 0

    if args.replicas_per_chunk < 1:
        print("ERROR: --replicas-per-chunk must be at least 1", file=sys.stderr)
        return 1

    embedded: Iterator[dict[str, Any]] = _embed_records(
        records,
        batch_size=args.embed_batch_size,
        replicas_per_chunk=args.replicas_per_chunk,
    )
    if args.output_jsonl:
        embedded = _write_jsonl(args.output_jsonl, embedded)

    total = _upsert(embedded, clear=not args.append, batch_size=args.upsert_batch_size)
    print(f"Upserted {total} re-embedded vectors into '{settings.pinecone_index_name}'/{settings.pinecone_namespace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
