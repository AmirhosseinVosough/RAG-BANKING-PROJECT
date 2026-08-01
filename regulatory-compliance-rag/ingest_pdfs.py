# ingest_pdfs.py
import logging
from config import REGULATIONS_DIR, settings
from modules.document_processor import iter_supported_files, process_document, read_document_bytes
from modules.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    store = VectorStore()
    all_chunks = []

    for file_path in iter_supported_files(REGULATIONS_DIR):
        text = read_document_bytes(file_path)
        if not text:
            logger.warning("No text extracted from %s", file_path.name)
            continue
        source_id = file_path.stem
        chunks = process_document(source_id=source_id, source_name=file_path.name, text=text)
        logger.info("Processed %s -> %d chunks", file_path.name, len(chunks))
        all_chunks.extend(chunks)
        store.add_chunks(chunks)

    # Save to JSONL so future startups don't need to re-parse PDFs
    store.save_chunks_to_jsonl(all_chunks)
    logger.info("Done. Total chunks: %d", len(all_chunks))

if __name__ == "__main__":
    main()