"""Upsert the provided 10-K embedding fixture into a local Pinecone index.

``data/pinecone_vectors.jsonl.gz`` is one JSON object per line:
``{id, namespace, values[512], metadata{text, source, page, ...}}``. We create a cosine index of
dimension 512 (matching the fixture) and upsert every record into its namespace. Using the fixture
means no embeddings are recomputed — keeping us well under the OpenAI budget.

    python scripts/load_vectors.py
"""

from __future__ import annotations

import gzip
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec

from financial_qa.app.infrastructure.settings import get_settings

EXPECTED_VECTORS = 4072
BATCH_SIZE = 200


def _read_records(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _batched(records: Iterator[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for record in records:
        batch.append(record)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _index_host(pc: Pinecone, name: str) -> str:
    """Resolve the data-plane host. Pinecone Local serves each index on its own port and the host
    carries no scheme, so prepend http:// for the (non-TLS) local emulator."""
    host = pc.describe_index(name).host
    if not host.startswith("http"):
        host = f"http://{host}"
    return host


def main() -> int:
    settings = get_settings()
    fixture = Path(settings.data_dir) / "pinecone_vectors.jsonl.gz"
    if not fixture.is_file():
        print(f"ERROR: fixture not found at {fixture}", file=sys.stderr)
        return 1

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

    total = 0
    for batch in _batched(_read_records(fixture), BATCH_SIZE):
        by_namespace: dict[str, list[dict]] = {}
        for record in batch:
            namespace = record.get("namespace") or settings.pinecone_namespace
            by_namespace.setdefault(namespace, []).append(
                {
                    "id": record["id"],
                    "values": record["values"],
                    "metadata": record.get("metadata", {}),
                }
            )
        for namespace, vectors in by_namespace.items():
            index.upsert(vectors=vectors, namespace=namespace)
            total += len(vectors)
        print(f"  upserted {total} vectors", end="\r", flush=True)

    print(f"\nUpserted {total} vectors into '{settings.pinecone_index_name}'")
    stats = index.describe_index_stats()
    reported = stats.get("total_vector_count", 0) if isinstance(stats, dict) else stats.total_vector_count
    print(f"Index reports total_vector_count = {reported}")
    if total != EXPECTED_VECTORS:
        print(f"WARNING: expected {EXPECTED_VECTORS} records, processed {total}", file=sys.stderr)
        return 1
    if reported != EXPECTED_VECTORS:
        print(f"WARNING: expected Pinecone to report {EXPECTED_VECTORS} vectors, got {reported}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
