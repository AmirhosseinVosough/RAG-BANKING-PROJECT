from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
import json
from dataclasses import asdict, dataclass
import logging
import math
import re
from pathlib import Path
from typing import Iterable, cast

from config import settings
from modules.document_processor import ProcessedChunk

import numpy as np


logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class StoredChunk:
	source_id: str
	source_name: str
	chunk_id: str
	regulation_title: str
	jurisdiction: str
	effective_date: str
	section: str
	text: str
	embedding: list[float]
	metadata: dict[str, object]


# Split text into normalized tokens for embedding and scoring.
def tokenize(text: str) -> list[str]:
	return TOKEN_PATTERN.findall(text.lower())


# Format embedding values as a Postgres vector literal.
def vector_literal(values: list[float]) -> str:
	return "[" + ",".join(f"{value:.6f}" for value in values) + "]"



def _embedding_to_blob(vec: list[float]) -> bytes:
    """Serialize a float vector to a compact binary blob."""
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()


def _blob_to_embedding(blob: bytes) -> list[float]:
    """Deserialize a blob back into a float list."""
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr.tolist()


def _cosine_similarity_batch(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and an (N × D) matrix.

    Uses numpy matrix multiplication — ~100x faster than a Python loop.
    """
	#nor : Norm/length = a single number computed FROM those 384 values — specifically sqrt(x1² + x2² + ... + x384²). This measures the vector's magnitude in 384-dimensional space.
    query_norm = np.linalg.norm(query_vec) # Calculates the mathematical length (magnitude) of your vector. Cosine similarity only cares about direction (the angle between them), not raw magnitude. Dividing by the norms cancels out magnitude differences so you're comparing pure direction — that's the whole point of the norm calculation.
    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32) # If the query vector has zero length, return a zero array of the same length as the number of rows in the matrix.
    row_norms = np.linalg.norm(matrix, axis=1) # Calculates the mathematical length (magnitude) of each row in the matrix.
    row_norms = np.where(row_norms == 0, 1.0, row_norms) # Replace any zero norms with 1.0 to avoid division by zero.
    return (matrix @ query_vec) / (row_norms * query_norm) #Calculates the cosine similarity by taking the dot product of the matrix and the query vector, and then dividing by the product of their norms.



class VectorStore:
	# Set up the active vector store backend and load any cached chunks.
	def __init__(self, database_url: str | None = None, embedding_dim: int | None = None) -> None:
		self.database_url = database_url or settings.database_url
		self.embedding_dim = embedding_dim or settings.embedding_dim
		self.embedding_model_name = settings.embedding_model_name
		self._embedding_model = None
		self.cache_path = Path(settings.vector_store_cache_path)
		self._memory_chunks: dict[str, StoredChunk] = {}
		self._pending_postgres_writes: set[Future[None]] = set()
		self._postgres_write_executor: ThreadPoolExecutor | None = None
		self._use_postgres = False

		if self.database_url:
			try:
				self._initialize_postgres()
				self._use_postgres = True
			except Exception as exc:  # pragma: no cover - fallback path
				logger.warning("Postgres vector store unavailable, using in-memory fallback: %s", exc)

		if not self._use_postgres:
			self._load_memory_cache()

	# Load and cache the sentence embedding model on first use.
	def _get_embedding_model(self):
		if self._embedding_model is None:
			try:
				import fastembed as fastembed_module
			except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
				raise RuntimeError(
					"Semantic embeddings require fastembed. Install dependencies with `pip install -r requirements.txt`."
				) from exc

			self._embedding_model = fastembed_module.TextEmbedding(model_name=self.embedding_model_name)
		return self._embedding_model

	# Open a PostgreSQL connection with vector support when configured.
	def _connect(self):
		try:
			import psycopg2
			from pgvector.psycopg2 import register_vector
		except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
			raise RuntimeError("Postgres vector support requires psycopg2-binary and pgvector.") from exc

		conn = psycopg2.connect(self.database_url)
		register_vector(conn)
		return conn

	# Create the Postgres table and indexes used by the persistent vector store.
	def _initialize_postgres(self) -> None:
		conn = self._connect()
		cur = conn.cursor()
		try:
			cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
			cur.execute(
				f"""
				CREATE TABLE IF NOT EXISTS regulation_chunks (
					id SERIAL PRIMARY KEY,
					source_id TEXT NOT NULL,
					source_name TEXT NOT NULL,
					chunk_id TEXT NOT NULL UNIQUE,
					regulation_title TEXT NOT NULL,
					jurisdiction TEXT NOT NULL,
					effective_date TEXT NOT NULL,
					section TEXT NOT NULL,
					content TEXT NOT NULL,
					embedding vector({self.embedding_dim}) NOT NULL,
					metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
					created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
					updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
				)
				"""
			)
			cur.execute("CREATE INDEX IF NOT EXISTS regulation_chunks_source_id_idx ON regulation_chunks (source_id)")
			cur.execute("CREATE INDEX IF NOT EXISTS regulation_chunks_jurisdiction_idx ON regulation_chunks (jurisdiction)")
			cur.execute("CREATE INDEX IF NOT EXISTS regulation_chunks_embedding_idx ON regulation_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)") #Without it, finding the "nearest" vectors means comparing your query against every single embedding in the table (brute-force). With ivfflat, Postgres pre-clusters similar vectors together so it only checks a subset of likely candidates — much faster at scale, at a small cost to accuracy.
			conn.commit()
		finally:
			cur.close()
			conn.close()


	# Build embeddings for many chunks in one model call.
	def _embed_texts(self, texts: list[str]) -> list[list[float]]:
		if not texts:
			return []
		model = self._get_embedding_model()
		embeddings = [[float(value) for value in values] for values in model.embed(texts)]
		if self.embedding_dim:
			for embedding in embeddings:
			
				if len(embedding) != self.embedding_dim:
					raise RuntimeError(
						f"Embedding dimension mismatch: model {self.embedding_model_name} returned {len(embedding)} values, expected {self.embedding_dim}."
					)
		return embeddings

	# Materialize stored chunks after batch embedding has completed.
	def _build_stored_chunks(self, chunks: list[ProcessedChunk]) -> list[StoredChunk]:
		embeddings = self._embed_texts([chunk.text for chunk in chunks])
		return [
			StoredChunk(
				source_id=chunk.source_id,
				source_name=chunk.source_name,
				chunk_id=chunk.chunk_id,
				regulation_title=chunk.regulation_title,
				jurisdiction=chunk.jurisdiction,
				effective_date=chunk.effective_date,
				section=chunk.section,
				text=chunk.text,
				embedding=embedding,
				metadata=self._build_metadata(chunk),
			)
			for chunk, embedding in zip(chunks, embeddings)
		]

	# Replace any existing in-memory chunk with the same chunk id.
	def _memory_replace(self, chunk: StoredChunk) -> None:
		self._memory_chunks[chunk.chunk_id] = chunk

	# Add processed chunks to the active store and persist the in-memory fallback.      Write-Through Cache !!!!!!!!!!!!
	def add_chunks(self, chunks: Iterable[ProcessedChunk]) -> None:
		chunks = list(chunks)
		if not chunks:
			return

		stored_chunks = self._build_stored_chunks(chunks)
		if self._use_postgres:
			# Postgres is the persistent primary. Keep an in-process replica so a
			# later fallback has the chunks already ingested during this session.
			self._queue_postgres_write(stored_chunks)
			self._update_memory_cache(stored_chunks)
		else:
			# Without Postgres, the memory store is primary and must survive restarts.
			self._update_memory_cache(stored_chunks)
			self._save_memory_cache()

	# Note: NOT calling _save_memory_cache() - keeps it in RAM only
	def _update_memory_cache(self, chunks: list[StoredChunk]) -> None:
		for chunk in chunks:
			self._memory_replace(chunk)
		

	# Queue a Postgres write in the background so ingestion can return quickly.
	def _queue_postgres_write(self, chunks: list[StoredChunk]) -> None:
		future = self._get_postgres_write_executor().submit(self._write_chunks_postgres, chunks)
		self._pending_postgres_writes.add(future)
		future.add_done_callback(self._pending_postgres_writes.discard)

	# Lazily create the background executor used for Postgres ingestion.
	def _get_postgres_write_executor(self) -> ThreadPoolExecutor:
		if self._postgres_write_executor is None:
			self._postgres_write_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vector-store-writes")
		return self._postgres_write_executor

	# Wait for queued Postgres writes before serving reads.
	def _flush_pending_postgres_writes(self) -> None:
		if not self._pending_postgres_writes:
			return
		pending = list(self._pending_postgres_writes)
		wait(pending)
		for future in pending:
			future.result()

	# Insert or update chunks in the Postgres-backed vector store.
	def _write_chunks_postgres(self, chunks: list[StoredChunk]) -> None:
		conn = self._connect()
		cur = conn.cursor()
		try:
			from psycopg2.extras import Json, execute_values

			rows = [
				(
					chunk.source_id,
					chunk.source_name,
					chunk.chunk_id,
					chunk.regulation_title,
					chunk.jurisdiction,
					chunk.effective_date,
					chunk.section,
					chunk.text,
					vector_literal(chunk.embedding),
					Json(chunk.metadata),
				)
				for chunk in chunks
			]
			if rows:
				execute_values(
					cur,
					"""
					INSERT INTO regulation_chunks (
						source_id, source_name, chunk_id, regulation_title, jurisdiction, effective_date, section, content, embedding, metadata
					)
					VALUES %s
					ON CONFLICT (chunk_id) DO UPDATE SET
						source_id = EXCLUDED.source_id,
						source_name = EXCLUDED.source_name,
						regulation_title = EXCLUDED.regulation_title,
						jurisdiction = EXCLUDED.jurisdiction,
						effective_date = EXCLUDED.effective_date,
						section = EXCLUDED.section,
						content = EXCLUDED.content,
						embedding = EXCLUDED.embedding,
						metadata = EXCLUDED.metadata,
						updated_at = NOW()
					""",
					rows   ,
					template="(%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)",
				)
			conn.commit()
		finally:
			cur.close()
			conn.close()

	# Return every stored chunk regardless of whether the backend is memory or Postgres.
	def all_chunks(self) -> list[StoredChunk]:
		if self._use_postgres:
			self._flush_pending_postgres_writes()
			return self._all_chunks_postgres()
		return list(self._memory_chunks.values())

	# Load all chunks from Postgres for inspection or startup bookkeeping.
	def _all_chunks_postgres(self) -> list[StoredChunk]:
		conn = self._connect()
		cur = conn.cursor()
		try:
			cur.execute(
				"""
				SELECT source_id, source_name, chunk_id, regulation_title, jurisdiction, effective_date, section, content, embedding::text, metadata
				FROM regulation_chunks
				ORDER BY created_at ASC, id ASC
				"""
			)
			rows = cur.fetchall()
			return [
				StoredChunk(
					source_id=row[0],
					source_name=row[1],
					chunk_id=row[2],
					regulation_title=row[3],
					jurisdiction=row[4],
					effective_date=row[5],
					section=row[6],
					text=row[7],
					embedding=self._parse_embedding_string(row[8]),
					metadata=cast(dict[str, object], row[9] or {}),
				)
				for row in rows
			]
		finally:
			cur.close()
			conn.close()

	# Search the active store with the configured retrieval strategy.
	def search(
		self,
		query: str,
		top_k: int = 5,
		jurisdiction: str | None = None,
		source_id: str | None = None,
	) -> list[dict[str, object]]:
		if self._use_postgres:
			self._flush_pending_postgres_writes()
			return self._search_postgres(query, top_k=top_k, jurisdiction=jurisdiction, source_id=source_id)
		return self._search_memory(query, top_k=top_k, jurisdiction=jurisdiction, source_id=source_id)

	# Search Postgres by vector distance and apply optional scope filters. <=> is cosine distance (0 = identical direction, 2 = opposite), so 1 - distance gives you cosine similarity, which is what confidence should represent.
	def _search_postgres(
		self,
		query: str,
		top_k: int,
		jurisdiction: str | None,
		source_id: str | None,
	) -> list[dict[str, object]]:
		conn = self._connect()
		cur = conn.cursor()
		try:
			query_embedding = vector_literal(self._embed_texts([query])[0])
			cur.execute(
				"""
				SELECT
					source_id,
					source_name,
					regulation_title,
					jurisdiction,
					effective_date,
					section,
					chunk_id,
					content,
					1 - (embedding <=> %s::vector) AS confidence 
				FROM regulation_chunks
				WHERE (%s IS NULL OR jurisdiction = %s)
				  AND (%s IS NULL OR source_id = %s)
				ORDER BY embedding <=> %s::vector
				LIMIT %s
				""",
				(query_embedding, jurisdiction, jurisdiction, source_id, source_id, query_embedding, top_k),
			)
			rows = cur.fetchall()
			return [self._row_to_result(row) for row in rows]
		finally:
			cur.close()
			conn.close()



	def _search_memory(
		self,
		query: str,
		top_k: int,
		jurisdiction: str | None,
		source_id: str | None,
	) -> list[dict[str, object]]:
		query_embedding = np.asarray(self._embed_texts([query])[0], dtype=np.float32)
		query_tokens = set(tokenize(query))

		chunks = list(self._memory_chunks.values())
		if jurisdiction:
			chunks = [c for c in chunks if c.jurisdiction.lower() == jurisdiction.lower()]
		if source_id:
			chunks = [c for c in chunks if c.source_id.lower() == source_id.lower()]
		if not chunks:
			return []

		matrix = np.array([c.embedding for c in chunks], dtype=np.float32)
		cosine_scores = _cosine_similarity_batch(query_embedding, matrix)

		scored: list[dict[str, object]] = []
		for chunk, cos_score in zip(chunks, cosine_scores):
			lexical_score = self._overlap_score(query_tokens, chunk.text)
			confidence = min(0.99, round((0.7 * float(cos_score)) + (0.3 * lexical_score), 4))
			scored.append(self._chunk_to_result(chunk, confidence))

		scored.sort(key=self._confidence_sort_key, reverse=True)
		return scored[:top_k]



	# Group chunks by source so the API can list indexed regulations.  
	def grouped_regulations(self) -> list[dict[str, object]]:
		if self._use_postgres:
			return self._grouped_regulations_postgres()

		grouped: dict[str, dict[str, object]] = {}
		for chunk in self._memory_chunks.values():
			record = grouped.setdefault(
				chunk.source_id,
				{
					"source_id": chunk.source_id,
					"source_name": chunk.source_name,
					"regulation_title": chunk.regulation_title,
					"jurisdiction": chunk.jurisdiction,
					"effective_date": chunk.effective_date,
					"chunk_count": 0,
					"text_preview": chunk.text[:180],
				},
			)
			record["chunk_count"] = cast(int, record["chunk_count"]) + 1
		return list(grouped.values())






	# Group Postgres-backed chunks by regulation source for listing.  Show users what's available without flooding them with data.
	def _grouped_regulations_postgres(self) -> list[dict[str, object]]:
		conn = self._connect()
		cur = conn.cursor()
		try:
			cur.execute(
				"""
				SELECT
					source_id,
					MIN(source_name) AS source_name,
					MIN(regulation_title) AS regulation_title,
					MIN(jurisdiction) AS jurisdiction,
					MIN(effective_date) AS effective_date,
					COUNT(*) AS chunk_count,
					MIN(LEFT(content, 180)) AS text_preview
				FROM regulation_chunks
				GROUP BY source_id
				ORDER BY source_id ASC
				"""
			)
			rows = cur.fetchall()
			return [
				{
					"source_id": row[0],
					"source_name": row[1],
					"regulation_title": row[2],
					"jurisdiction": row[3],
					"effective_date": row[4],
					"chunk_count": cast(int, row[5]),
					"text_preview": row[6] or "",
				}
				for row in rows
			]
		finally:
			cur.close()
			conn.close()









	@staticmethod
	# Convert a database row into the API-friendly search result shape.
	def _row_to_result(row: tuple[object, ...]) -> dict[str, object]:
		return {
			"source_id": row[0],
			"source_name": row[1],
			"regulation_title": row[2],
			"jurisdiction": row[3],
			"effective_date": row[4],
			"section": row[5],
			"chunk_id": row[6],
			"text": row[7],
			"confidence": cast(float, row[8]),
		}

	@staticmethod
	# Convert a stored chunk into the API-friendly search result shape.
	def _chunk_to_result(chunk: StoredChunk, confidence: float) -> dict[str, object]:
		return {
			"source_id": chunk.source_id,
			"source_name": chunk.source_name,
			"regulation_title": chunk.regulation_title,
			"jurisdiction": chunk.jurisdiction,
			"effective_date": chunk.effective_date,
			"section": chunk.section,
			"chunk_id": chunk.chunk_id,
			"text": chunk.text,
			"confidence": round(confidence, 4),
		}



	@staticmethod
	# Build the minimal metadata payload we persist with each chunk.
	def _build_metadata(chunk: ProcessedChunk) -> dict[str, object]:
		metadata= chunk.metadata
		metadata.update({
			"source_name": chunk.source_name,
			"regulation_title": chunk.regulation_title,
			"jurisdiction": chunk.jurisdiction,
			"effective_date": chunk.effective_date,
			"section": chunk.section,
		})
		return metadata

	@staticmethod
	# Compute Euclidean distance between two embedding vectors.
	def _euclidean_distance(left: list[float], right: list[float]) -> float:
		return math.sqrt(sum((l_value - r_value) ** 2 for l_value, r_value in zip(left, right)))

	@staticmethod
	# Score token overlap so exact topical matches rank higher than noisy matches.
	def _overlap_score(query_tokens: set[str], text: str) -> float:
		if not query_tokens: 
			return 0.0
		text_tokens = set(tokenize(text))
		if not text_tokens:
			return 0.0
		return len(query_tokens & text_tokens) / len(query_tokens)

	@staticmethod
	# Sort results by confidence in descending order.
	def _confidence_sort_key(item: dict[str, object]) -> float:
		return cast(float, item["confidence"])

	# Restore the in-memory cache from disk when no database is configured.
	def _load_memory_cache(self) -> None:
		if not self.cache_path.exists():
			return
		try:
			payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
			if isinstance(payload, dict):
				self._memory_chunks = {
					chunk_id: StoredChunk(**item)
					for chunk_id, item in payload.items()
					if isinstance(chunk_id, str) and isinstance(item, dict)
				}
			elif isinstance(payload, list):
				self._memory_chunks = {
					str(item["chunk_id"]): StoredChunk(**item)
					for item in payload
					if isinstance(item, dict) and item.get("chunk_id")
				}
			else:
				return
		except Exception as exc:  # pragma: no cover - defensive guard
			logger.warning("Could not load vector store cache: %s", exc)

	# Persist the in-memory cache so regulations survive application restarts.
	def _save_memory_cache(self) -> None:
		try:
			self.cache_path.parent.mkdir(parents=True, exist_ok=True)
			payload = {chunk_id: asdict(chunk) for chunk_id, chunk in self._memory_chunks.items()}
			self.cache_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
		except Exception as exc:  # pragma: no cover - defensive guard
			logger.warning("Could not save vector store cache: %s", exc)

	# Parse embeddings stored as JSON text or a Postgres vector literal.
	@staticmethod
	def _parse_embedding_string(emb_str: str) -> list[float]:
		if not emb_str:
			return []

		try:
			parsed = json.loads(emb_str)
			if isinstance(parsed, list):
				return [float(value) for value in parsed]
		except (json.JSONDecodeError, TypeError, ValueError):
			pass

		cleaned_str = emb_str.strip("[]{} ")
		if not cleaned_str:
			return []
		return [float(value) for value in cleaned_str.split(",")]

	def save_chunks_to_jsonl(self, chunk_path: Path) -> None:
		"""
		Save all stored chunks to a JSONL file for inspection or backup.
		"""
		path = Path(chunk_path) if chunk_path else settings.DEFAULT_CHUNK_PATH
		if not path.exists():
			logger.error("Chunks JSONL not found: %s", path)
			return
		records = []
		with path.open(encoding="utf-8") as f:
			for line in f:
				if not line.strip():
					continue
				item = json.loads(line)
				records.append(
					ProcessedChunk(
						chunk_id=item["chunk_id"],
						source_id=item.get("source_filename", "unknown"),
						source_name=item.get("source_filename", "unknown"),
						text=item["text"],
						page_start=item.get("page_start"),
						page_end=item.get("page_end"),
						metadata=item.get("metadata", {}),
					)
				)
		            
		store = VectorStore()
		logger.info("Ingesting %d chunks to SQLite with background threading...", len(records))
    
    # Process in batches to not block RAM too hard
		batch_size = 50
		for i in range(0, len(records), batch_size):
			store.add_chunks(records[i:i+batch_size])
			logger.info("Queued batch %d/%d", i, len(records))
			
		store._flush_pending_sqlite_writes()
		logger.info("Ingestion complete.")

	if __name__ == "__main__":
		logging.basicConfig(level=logging.INFO)
		save_chunks_to_jsonl()

"""
Why JSONL-first ingestion is useful:
You chunk a document once, not every time the API starts.
You can inspect, version, and reuse the exact chunks.
You can rebuild Postgres or RAM from chunks.jsonl if the database is deleted.
Startup becomes “load prepared chunks,” rather """

"""Q: Pros of Postgres + pgvector (DB+index) method?

Scales to millions of vectors without RAM limits
ANN index (ivfflat/hnsw) makes queries sub-linear, not full-scan
Filtering (jurisdiction/source_id) happens in SQL before/with vector search
Concurrent access, transactions, durability built in
Standard production pattern — recognizable on a CV

Q: Cons of Postgres + pgvector?

Requires running/maintaining a Postgres instance
Index needs tuning (lists, probes) or recall suffers
Slightly more setup/ops complexity than a flat file
Extra network round-trip per query vs. in-process RAM

Q: Pros of SQLite + numpy brute-force method?

Zero infra — single file, no server to run
Simple to read/debug, no index tuning needed
Fast enough for small datasets (thousands of rows)
Good as a local dev/offline fallback

Q: Cons of SQLite + numpy brute-force?

O(N) every query — doesn't scale past tens of thousands of rows
Entire embedding matrix must fit in RAM
No real ANN index — SQLite BLOB storage isn't vector-aware
Single-threaded compute-bound search competes with app for CPU
Weaker signal for "vector DB experience" on a resume"""




#Euclidean measures straight-line distance between two points; cosine measures the angle between two vectors, ignoring their length.
"""Concrete example:

Say you have two document vectors (simplified to 2D):

A = [1, 1] — short vector
B = [4, 4] — same direction, just "longer" (e.g. a longer document repeating the same words)

These point in exactly the same direction — semantically identical topic.

Cosine similarity between A and B = 1.0 (perfect match — correctly says "same meaning")
Euclidean distance between A and B = √((4-1)² + (4-1)²) = √18 ≈ 4.24 (says "very far apart" — wrong, because it's reacting to magnitude, not meaning)

Now compare A = [1,1] to C = [1,-1] (different direction, same length):

Cosine similarity = 0 (correctly says "unrelated")
Euclidean distance = √((1-1)² + (-1-1)²) = 2 (smaller than A-to-B, even though A-to-B was the actual semantic match)"""

#rn Imma change the Euclidean distance to cosine similarity for the in-memory search, since cosine is more appropriate for semantic embeddings. but aint gonna use numpoy since it is unscaled and only good for small datasets but imma replace iut with db indexiung usin ivfflat
