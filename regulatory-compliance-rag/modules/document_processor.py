from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import re
from typing import Optional
from importlib import import_module

from config import settings


METADATA_PATTERN = re.compile(r"^(?P<key>[A-Za-z_]+)\s*:\s*(?P<value>.+)$")


@dataclass(frozen=True)
class ProcessedChunk:
	source_id: str
	source_name: str
	chunk_id: str
	regulation_title: str
	jurisdiction: str
	effective_date: str
	section: str
	text: str
	metadata: dict[str, object] = field(default_factory=dict)


# Normalize whitespace so downstream chunking and retrieval stay consistent.
def _clean_text(text: str) -> str:
	return " ".join(text.split())


# Read a regulation file from disk and convert it into searchable text.
def read_document_bytes(file_path: Path) -> str:
	if file_path.suffix.lower() == ".pdf":
		reader = _get_pdf_reader()(str(file_path))
		pages = [page.extract_text() or "" for page in reader.pages]
		return _clean_text("\n".join(pages))
	return _clean_text(file_path.read_text(encoding="utf-8"))


# Read uploaded file bytes and convert them into searchable text.
def read_uploaded_bytes(data: bytes, filename: str) -> str:
	if filename.lower().endswith(".pdf"):
		import io

		reader = _get_pdf_reader()(io.BytesIO(data))
		pages = [page.extract_text() or "" for page in reader.pages]
		return _clean_text("\n".join(pages))
	return _clean_text(data.decode("utf-8"))


# Create a PDF reader lazily so the app only needs PyPDF2 when PDFs are used.
def _get_pdf_reader():
	try:
		return import_module("PyPDF2").PdfReader
	except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
		raise RuntimeError("PDF support requires PyPDF2 to be installed.") from exc


# Split long text into overlapping character windows when sentence packing is not enough.
def _chunk_by_window(text: str, chunk_size: int, overlap: int) -> list[str]:
	chunks: list[str] = []
	start = 0
	while start < len(text):
		end = min(len(text), start + chunk_size)
		chunk = text[start:end].strip()
		if chunk:
			chunks.append(chunk)
		if end >= len(text):
			break
		start = max(end - overlap, start + 1)
	return chunks


# Split regulations into sentence-aware chunks so retrieval keeps clause boundaries together.
def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
	chunk_size = chunk_size or settings.max_chunk_size
	overlap = overlap if overlap is not None else settings.chunk_overlap
	normalized = _clean_text(text)
	if len(normalized) <= chunk_size:
		return [normalized] if normalized else []

	sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence.strip()]
	if len(sentences) <= 1:
		return _chunk_by_window(normalized, chunk_size, overlap)

	chunks: list[str] = []
	current: list[str] = []
	current_length = 0
	for sentence in sentences:
		if len(sentence) > chunk_size:
			if current:
				chunks.append(" ".join(current))
				current = []
				current_length = 0
			chunks.extend(_chunk_by_window(sentence, chunk_size, overlap))
			continue 

		proposed_length = current_length + len(sentence) + (1 if current else 0)
		if current and proposed_length > chunk_size:
			chunks.append(" ".join(current))
			current = [sentence]
			current_length = len(sentence)
		else:
			current.append(sentence)
			current_length = proposed_length

	if current:
		chunks.append(" ".join(current))
	return [chunk for chunk in chunks if chunk]


# Parse leading metadata lines from a regulation body when they are present.
def parse_metadata_lines(text: str) -> tuple[dict[str, str], str]:
	metadata: dict[str, str] = {}
	content_lines: list[str] = []
	for line in text.splitlines():
		match = METADATA_PATTERN.match(line.strip())
		if match and not content_lines:
			metadata[match.group("key").lower()] = match.group("value").strip()
		else:
			content_lines.append(line)
	return metadata, "\n".join(content_lines).strip()


# Turn a raw regulation document into normalized chunk records with metadata.
def process_document(
	source_id: str,
	source_name: str,
	text: str,
	regulation_title: Optional[str] = None,
	jurisdiction: Optional[str] = None,
	effective_date: Optional[str] = None,
	section: Optional[str] = None,
) -> list[ProcessedChunk]:
	metadata, body = parse_metadata_lines(text)
	resolved_title = regulation_title or metadata.get("regulation_title") or metadata.get("title") or source_name
	resolved_jurisdiction = jurisdiction or metadata.get("jurisdiction") or "unknown"
	resolved_effective_date = effective_date or metadata.get("effective_date") or "unknown"
	resolved_section = section or metadata.get("section") or "general"
	chunks = chunk_text(body)
	return [
		ProcessedChunk(
			source_id=source_id,
			source_name=source_name,
			chunk_id=f"{source_id}-chunk-{index + 1}",
			regulation_title=resolved_title,
			jurisdiction=resolved_jurisdiction,
			effective_date=resolved_effective_date,
			section=resolved_section,
			text=chunk,
		)
		for index, chunk in enumerate(chunks)
	]


# Yield supported regulation file types from a folder in a stable order.
def iter_supported_files(folder: Path) -> Iterable[Path]:
	for path in sorted(folder.iterdir()):
		if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
			yield path
