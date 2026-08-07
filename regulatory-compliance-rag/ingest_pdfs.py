# ingest_pdfs.py
import json
import logging
from pathlib import Path

from config import REGULATIONS_DIR, REGULATIONS_MANIFEST, settings
from modules.document_processor import iter_supported_files, process_document, read_document_bytes
from modules.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        logger.warning(
            "Manifest file not found at %s — all PDFs will fall back to "
            "jurisdiction='unknown'. Create the manifest to fix this.",
            manifest_path,
        )
        return {}
    try:
        with manifest_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        # Strip the optional _comment key so callers never see it.
        data.pop("_comment", None)
        logger.info("Loaded regulations manifest with %d entries.", len(data))
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to parse manifest %s: %s", manifest_path, exc)
        return {}


def main():
    manifest = _load_manifest(REGULATIONS_MANIFEST)
    store = VectorStore()
    all_chunks = []

    for file_path in iter_supported_files(REGULATIONS_DIR):
        text = read_document_bytes(file_path)
        if not text:
            logger.warning("No text extracted from %s", file_path.name)
            continue

        # Look up pre-defined metadata from the manifest (keyed by filename).
        entry = manifest.get(file_path.name, {})
        if not entry:
            logger.warning(
                "No manifest entry for '%s' — jurisdiction and title will be "
                "'unknown'. Add an entry to regulations_manifest.json to fix this.",
                file_path.name,
            )

        source_id = file_path.stem
        chunks = process_document(
            source_id=source_id,
            source_name=file_path.name,
            text=text,
            jurisdiction=entry.get("jurisdiction"),
            regulation_title=entry.get("regulation_title"),
            effective_date=entry.get("effective_date"),
        )
        logger.info("Processed %s -> %d chunks", file_path.name, len(chunks))
        all_chunks.extend(chunks)
        store.add_chunks(chunks)

    # Save to JSONL so future startups don't need to re-parse PDFs.
    store.save_chunks_to_jsonl(all_chunks)
    logger.info("Done. Total chunks: %d", len(all_chunks))


if __name__ == "__main__":
    main()