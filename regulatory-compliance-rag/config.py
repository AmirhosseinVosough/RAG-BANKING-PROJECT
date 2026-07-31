from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REGULATIONS_DIR = DATA_DIR / "regulations"


@dataclass(frozen=True)
class Settings:
	app_name: str = "Regulatory Compliance RAG"
	app_version: str = "0.1.0"
	api_prefix: str = "/compliance"
	database_url: str | None = os.getenv("DATABASE_URL")
	embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
	vector_store_cache_path: Path = DATA_DIR / "vector_store_index.json"

	embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
	max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "1200"))
	chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
	top_k: int = int(os.getenv("TOP_K", "5"))
	reranking_enabled: bool = os.getenv("RERANKING_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
	reranker_model_name: str = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
	rerank_candidate_count: int = int(os.getenv("RERANK_CANDIDATE_COUNT", "20"))
	rrf_constant : int = int(os.getenv("RRF_CONSTANT", "60"))
	DEFAULT_CHUNK_PATH : Path = Path(os.getenv("DEFAULT_CHUNK_PATH", "data/chunks.jsonl"))
settings = Settings()
