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
		try:
			reader = _get_pdf_reader()(str(file_path))
			pages = [page.extract_text() or "" for page in reader.pages]
			return _clean_text("\n".join(pages))
		except Exception as exc:
			import logging
			logging.getLogger(__name__).warning("WARNING: %s is not a valid PDF file (%s)", file_path.name, exc)
			return ""
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







"""Input file: data/regulations/mifid-ii-suitability.txt

regulation_title: MiFID II Suitability Requirements
jurisdiction: EU
effective_date: 2018-01-03
section: Article 25
Investment firms must assess the suitability of financial instruments for retail clients. Firms must gather information on the client's knowledge, experience, financial situation, and investment objectives before providing advice. Failure to conduct adequate suitability assessments may result in regulatory sanctions and client compensation claims.

Step 1 — read_document_bytes() reads the raw file → passes text into process_document()

Step 2 — parse_metadata_lines(text) runs first, inside process_document():

Walking line by line:

"regulation_title: MiFID II Suitability Requirements" → matches pattern, content_lines still empty → stored as metadata: {"regulation_title": "MiFID II Suitability Requirements"}
"jurisdiction: EU" → matches, still empty → metadata["jurisdiction"] = "EU"
"effective_date: 2018-01-03" → matches → metadata["effective_date"] = "2018-01-03"
"section: Article 25" → matches → metadata["section"] = "Article 25"
"Investment firms must assess..." → does not match key: value pattern → goes into content_lines. From here on, not content_lines is False forever.

Result:

python
metadata = {
    "regulation_title": "MiFID II Suitability Requirements",
    "jurisdiction": "EU",
    "effective_date": "2018-01-03",
    "section": "Article 25",
}
body = "Investment firms must assess the suitability of financial instruments for retail clients. Firms must gather information on the client's knowledge, experience, financial situation, and investment objectives before providing advice. Failure to conduct adequate suitability assessments may result in regulatory sanctions and client compensation claims."

Step 3 — back in process_document(), resolve final values:

python
resolved_title = regulation_title or metadata.get("regulation_title") or ... 
# → "MiFID II Suitability Requirements" (since no explicit arg was passed, falls back to metadata)
resolved_jurisdiction = "EU"
resolved_effective_date = "2018-01-03"
resolved_section = "Article 25"

Step 4 — chunk_text(body) splits the body. This body is short (under max_chunk_size=1200), so chunk_text just returns it as one single chunk:

python
chunks = ["Investment firms must assess the suitability..."]

Step 5 — build ProcessedChunk objects:

python
ProcessedChunk(
    source_id="mifid-ii-suitability",
    source_name="mifid-ii-suitability.txt",
    chunk_id="mifid-ii-suitability-chunk-1",
    regulation_title="MiFID II Suitability Requirements",
    jurisdiction="EU",
    effective_date="2018-01-03",
    section="Article 25",
    text="Investment firms must assess the suitability...",
)

Step 6 — save_chunks_to_jsonl() writes this to chunks.jsonl as one line:

json
{"source_id": "mifid-ii-suitability", "source_name": "mifid-ii-suitability.txt", "chunk_id": "mifid-ii-suitability-chunk-1", "regulation_title": "MiFID II Suitability Requirements", "jurisdiction": "EU", "effective_date": "2018-01-03", "section": "Article 25", "text": "Investment firms must assess the suitability...", "metadata": {}}

Step 7 — later, ingest_jsonl() reads that line back, reconstructs a ProcessedChunk, and calls vector_store.add_chunks([chunk]).

Step 8 — inside add_chunks(), _build_stored_chunks() calls _embed_texts([chunk.text]):

The embedding model (fastembed, all-MiniLM-L6-v2) turns the chunk's text into a 384-dimensional vector: [0.0231, -0.114, 0.0056, ...] (384 numbers representing the semantic "meaning" of that sentence).

Step 9 — StoredChunk is built, same fields as before, plus the new embedding list, and gets saved either to Postgres (vector_literal()) or your in-memory store.

Step 10 — at search time, e.g. search("suitability requirements"):

Your query also gets embedded into a 384-dim vector.
_cosine_similarity_batch() compares your query vector against this chunk's stored embedding — since they're semantically close (both about "suitability"), you'd get a high confidence score like 0.7+.
BM25 lexical search would also match strongly here since the literal word "suitability" appears in both.
Both signals agree → this chunk ranks near the top after RRF fusion and reranking."""












"""Q1 — confirming your understanding, with one correction:

You're right that source_id, source_name, and text are different from the others — but notice they're not optional in the function signature:

python
def process_document(
	source_id: str,        # required, no default, no fallback
	source_name: str,      # required, no default, no fallback
	text: str,              # required, no default, no fallback
	regulation_title: Optional[str] = None,   # optional, has a fallback chain
	jurisdiction: Optional[str] = None,        # optional, has a fallback chain
	...

source_id/source_name/text always come directly from whoever calls the function — there's no or metadata.get(...) fallback for them because they don't need one; they're mandatory, always explicitly supplied (a filename, an upload's file, the raw text itself). It'd make no sense to guess those from a metadata header.

The or-chain fallback pattern only applies to the four optional fields (regulation_title, jurisdiction, effective_date, section) — because those are things a file might self-declare in a header, but might not, and the caller might also know them explicitly (e.g. via an upload form), but might not either. So yes — your instinct is right, just to be precise: it's specifically these four optional fields that have the "user's explicit choice wins, otherwise fall back to the file's own header" logic, not the required three.

Q2 — how does regex avoid mixing up which value belongs to which key?

The regex has zero semantic understanding — it doesn't "know" what jurisdiction or title mean at all. It's purely mechanical, and it works because of one simple fact: each line is parsed completely independently, one at a time, and each line only has one key: value pair on it.

Look at the input:

regulation_title: MiFID II Suitability Requirements
jurisdiction: EU
effective_date: 2018-01-03
section: Article 25

The loop processes line 1 first:

python
match = METADATA_PATTERN.match("regulation_title: MiFID II Suitability Requirements")

The regex ^(?P<key>[A-Za-z_]+)\s*:\s*(?P<value>.+)$ just says: "capture everything before the first : as key, capture everything after as value." On this specific line, that means key = "regulation_title", value = "MiFID II Suitability Requirements". Stored as metadata["regulation_title"] = "MiFID II Suitability Requirements".

Then the loop moves to line 2, completely fresh, no memory of line 1's content:

python
match = METADATA_PATTERN.match("jurisdiction: EU")

Same regex, applied to this line only → key = "jurisdiction", value = "EU" → metadata["jurisdiction"] = "EU".

There's no possibility of mixing them up, because the regex never sees multiple lines at once — it's re-run fresh on each individual line string, and each line only contains one key and one value, separated by exactly one colon. The "key" is literally whatever word(s) appear before the colon on that line — the regex doesn't validate that jurisdiction should look like "EU" or that title should look like a sentence; it would happily accept jurisdiction: banana if that's what the line said. It's pure syntax matching (position relative to the colon), not any kind of understanding of what a valid jurisdiction value should look like.

Where a real mistake could happen: if someone wrote a badly-formatted header like this:

jurisdiction, title: EU MiFID II

That's one line with no colon after "jurisdiction" — it wouldn't match the pattern at all (regex requires key to be only [A-Za-z_]+ immediately followed by :), so this entire line would fail to match and get dumped into body content instead. The regex is strict about format, not about meaning — it either matches the exact word: value shape per line, or it doesn't match at all."""