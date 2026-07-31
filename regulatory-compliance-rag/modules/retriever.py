from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re

from config import settings
from modules.reranker import CrossEncoder, Reranker
from modules.vector_store import VectorStore

from modules.vector_store import StoredChunk, VectorStore, tokenize


import logging 
reranker = Reranker()

logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Normalize text for lexical BM25 matching."""
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class RetrievedRegulation:
    """A regulation chunk returned to the compliance decision layer."""

    source_id: str
    source_name: str
    regulation_title: str
    jurisdiction: str
    effective_date: str
    section: str
    chunk_id: str
    text: str
    confidence: float

"""
def _get_fastembed_client(self) -> CrossEncoder:
    if self._model is None:
        if self._model_factory is not None:
            self._model = self._model_factory()
        else:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Cross-encoder reranking requires fastembed. "
                    "Install dependencies with `pip install -r requirements.txt`."
                ) from exc
            self._model = TextCrossEncoder(model_name=self.model_name)
    return self._model
"""



"""
class API_Reranker:
    

    def __init__(self) -> None:
        self.client = _get_fastembed_client()
        self.model = "Xenova/ms-marco-MiniLM-L-6-v2"

    def rerank(self, query: str, chunks: list[dict[str, object]]) -> list[dict[str, object]]: #Cross-encoder reranking (fastembed's TextCrossEncoder.rerank, or Cohere's .rerank) doesn't work on pre-computed embeddings at all. It takes the raw query string and raw document strings, and internally runs them together through a transformer to produce a relevance score directly.
        if not chunks:
            return []

        docs = [str(chunk["text"]) for chunk in chunks] 

        scores = list(self.client.rerank(query, docs))

        # Map back scores based on original indices
        for i, score in enumerate(scores):
            chunks[i]["rerank_score"] = float(score)

        return sorted(chunks, key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
"""
	



class Retriever:
    """Combine BM25 lexical search with semantic (cosine) search, fused via RRF."""

    def __init__(self, vector_store: VectorStore, reranker=None) -> None:
        self.vector_store = vector_store
        self._chunks: list[StoredChunk] = []
        self._bm25 = None
        self.reranker = reranker
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the BM25 index after chunks are added to the vector store."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "BM25 dependency is missing. Run 'pip install rank_bm25'."
            ) from exc

        self._chunks = self.vector_store.all_chunks()
        if not self._chunks:
            self._bm25 = None
            logger.warning("BM25 index is empty; index chunks before retrieval.")
            return
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self._chunks])

    def lexical_search(
        self, query: str, top_k: int = 20, jurisdiction: str | None = None
    ) -> list[dict[str, object]]:
        """Retrieve exact-term matches using BM25."""
        if not query.strip():
            raise ValueError("query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        chunk_score_pairs = zip(self._chunks, scores)
        ranked_pairs = sorted(chunk_score_pairs, key=lambda pair: float(pair[1]), reverse=True)

        results: list[dict[str, object]] = []
        for chunk, score in ranked_pairs:
            if score <= 0:
                break
            if jurisdiction and chunk.jurisdiction.lower() != jurisdiction.lower():
                continue
            results.append(self._chunk_to_result(chunk, float(score)))
            if len(results) == top_k:
                break
        return results

    def retrieve(
        self,
        query: str,
        semantic_k: int = 20,
        lexical_k: int = 20,
        rrf_k: int = 20,
        final_k: int = 8,
        jurisdiction: str | None = None,
        source_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Run the three-stage hybrid pipeline: semantic + lexical + RRF fusion, then rerank."""
        if final_k <= 0:
            raise ValueError("final_k must be greater than zero.")

        semantic_results = self.vector_store.search(
            query, top_k=semantic_k, jurisdiction=jurisdiction, source_id=source_id
        )
        lexical_results = self.lexical_search(query, top_k=lexical_k, jurisdiction=jurisdiction)

        # Stage: fuse both rankings with RRF
        fused_scores: dict[str, float] = defaultdict(float)
        result_by_id: dict[str, dict[str, object]] = {}
        for rank, result in enumerate(semantic_results, start=1):
            chunk_id = str(result["chunk_id"])
            fused_scores[chunk_id] += 1 / (settings.rrf_constant + rank)
            result_by_id[chunk_id] = dict(result)
        for rank, result in enumerate(lexical_results, start=1):
            chunk_id = str(result["chunk_id"])
            fused_scores[chunk_id] += 1 / (settings.rrf_constant + rank)
            result_by_id.setdefault(chunk_id, dict(result))

        # Each ID comes from fused_scores, so direct indexing always returns a float.
        candidate_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)[:rrf_k]
        candidates = []
        for chunk_id in candidate_ids:
            result = result_by_id[chunk_id]
            result["retrieval_score"] = round(fused_scores[chunk_id], 6)
            candidates.append(result)

        if self.reranker:
            candidates = self.reranker.rerank(query, candidates, top_k=final_k)

        return candidates[:final_k]

    @staticmethod
    def _chunk_to_result(chunk: StoredChunk, bm25_score: float) -> dict[str, object]:
        return {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "source_name": chunk.source_name,
            "regulation_title": chunk.regulation_title,
            "jurisdiction": chunk.jurisdiction,
            "effective_date": chunk.effective_date,
            "section": chunk.section,
            "text": chunk.text,
            "bm25_score": round(bm25_score, 6),
        }






	


"""
	==================================================================================================
PHASE 1: STARTUP — BUILD THE LEXICAL INDEX
==================================================================================================

[Function: HybridRetriever.__init__ → refresh()] (Target: SYSTEM RAM)
 ├──► Calls 'self.vector_store.all_chunks()' — pulls every StoredChunk from Postgres (or memory).
 ├──► Extracts just '.text' from each chunk, tokenizes it, and builds a BM25Okapi index.
 └──► Stores the full StoredChunk list in 'self._chunks' (embeddings ride along unused here).

==================================================================================================
PHASE 2: USER QUERY ARRIVES
==================================================================================================

[QUERY EXAMPLE]: "What is the minimum capital requirement for banks?"

[Function: retrieve(query, jurisdiction="EU")] (Target: ORCHESTRATOR)
 └──► Kicks off two independent retrieval legs in sequence.

==================================================================================================
PHASE 3A: SEMANTIC LEG (VECTOR SEARCH)
==================================================================================================

[Function: vector_store.search(query, jurisdiction="EU")]
 ├──► Embeds the query text into a vector via '_embed_texts([query])'.
 ├──► ROUTES: Postgres path → SQL 'ORDER BY embedding <=> query_vector' using the ivfflat/cosine index.
 │             Memory path  → '_search_memory': builds an (N×D) matrix, runs '_cosine_similarity_batch'.
 └──► Returns top 'semantic_k' chunks ranked by cosine similarity, each with a 'confidence' score.

==================================================================================================
PHASE 3B: LEXICAL LEG (BM25)
==================================================================================================

[Function: lexical_search(query, jurisdiction="EU")]
 ├──► Tokenizes the query into words.
 ├──► 'self._bm25.get_scores(...)' scores every chunk in 'self._chunks' by term overlap/frequency.
 ├──► Zips scores back to their chunks (same positional order as index build), sorts descending.
 ├──► Filters by jurisdiction, stops early once score hits 0.
 └──► Returns top 'lexical_k' chunks ranked by BM25 score.

==================================================================================================
PHASE 4: FUSION (RECIPROCAL RANK FUSION)
==================================================================================================

[Inside retrieve()] (Target: RANK MERGER)
 ├──► Walks semantic_results: chunk at rank r gets '+= 1 / (rrf_constant + r)'.
 ├──► Walks lexical_results: same formula, added to the SAME 'fused_scores' dict by chunk_id.
 │     (A chunk appearing in both lists accumulates score from both — ranks higher.)
 ├──► Sorts all chunk_ids by fused RRF score, keeps top 'rrf_k' as candidates.
 └──► Attaches 'retrieval_score' and 'citation' (via '_citation()') to each candidate.

==================================================================================================
PHASE 5: RERANKING (CROSS-ENCODER)
==================================================================================================

[Function: self.reranker.rerank(query, candidates)] (Target: PRECISION PASS)
 ├──► Local fastembed TextCrossEncoder scores query against each candidate's full text.
 ├──► Re-sorts candidates by cross-encoder relevance score (more accurate than RRF alone).
 └──► If reranker fails/unavailable, falls back to the RRF order untouched.

==================================================================================================
PHASE 6: FINAL OUTPUT
==================================================================================================

[Function: retrieve() returns] → 'candidates[:final_k]'

[FINAL MATCH OUTPUT]:
 -> Chunk: "Banks must maintain minimum capital requirements..." 
    Regulation: Basel III | Jurisdiction: EU | Section 4.2
    Citation: [Source: Basel III: Finalising post-crisis reforms (EU), eff. 2023-01-01, §4.2]
    Retrieval Score: 0.0328 (RRF) → Reranked #1"""
