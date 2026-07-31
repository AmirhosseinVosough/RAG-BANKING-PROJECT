"""Load prepared JSONL chunks into the active vector store."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import settings
from modules.document_processor import ProcessedChunk
from modules.vector_store import VectorStore


logger = logging.getLogger(__name__)


def ingest_jsonl(
    vector_store: VectorStore,
    jsonl_path: str | Path | None = None,
    batch_size: int = 50,
) -> int:
    """Load text and metadata from JSONL, regenerating embeddings in batches."""
    path = Path(jsonl_path) if jsonl_path else Path(settings.chunks_export_path)
    if not path.exists():
        return 0
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    total = 0
    batch: list[ProcessedChunk] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("record must be a JSON object")
                source_name = str(item.get("source_name") or item.get("source_filename") or "unknown")
                metadata = item.get("metadata", {})
                batch.append(
                    ProcessedChunk(
                        source_id=str(item.get("source_id") or source_name),
                        source_name=source_name,
                        chunk_id=str(item["chunk_id"]),
                        regulation_title=str(item.get("regulation_title") or source_name),
                        jurisdiction=str(item.get("jurisdiction") or "unknown"),
                        effective_date=str(item.get("effective_date") or "unknown"),
                        section=str(item.get("section") or "unknown"),
                        text=str(item["text"]),
                        metadata=metadata if isinstance(metadata, dict) else {},
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Skipping invalid JSONL record at %s:%d: %s", path, line_number, exc)
                continue

            if len(batch) == batch_size:
                vector_store.add_chunks(batch)
                total += len(batch)
                batch = []

    if batch:
        vector_store.add_chunks(batch)
        total += len(batch)

    logger.info("Ingested %d prepared chunks from %s", total, path)
    return total
