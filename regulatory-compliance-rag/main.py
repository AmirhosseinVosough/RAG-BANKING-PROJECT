from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4
from typing import cast

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import REGULATIONS_DIR, settings
from modules.chunk_ingestion import ingest_jsonl
from modules.agent import ComplianceAgent
from modules.document_processor import iter_supported_files, process_document, read_document_bytes, read_uploaded_bytes
from modules.models import Citation, ErrorResponse, RegulationInfo, RegulationSearchResponse, StrategyAuditRecord, StrategyCheckRequest, StrategyCheckResponse, UploadRegulationResponse
from modules.retriever import Retriever
from modules.vector_store import VectorStore

from modules.reranker import Reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("regulatory-compliance-rag")

vector_store = VectorStore()
reranker = Reranker()
retriever = Retriever(vector_store, reranker=reranker)




app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)


vector_store = VectorStore()
retriever = Retriever(vector_store)
agent = ComplianceAgent()
audit_trail: list[StrategyAuditRecord] = []


# Initialize the vector store from prepared JSONL, or create that JSONL from raw documents.
def load_regulations_from_disk() -> None:
	if vector_store.all_chunks():
		logger.info("Using %s chunks already loaded from the active store", len(vector_store.all_chunks()))
		return

	prepared_chunk_count = ingest_jsonl(vector_store)
	if prepared_chunk_count:
		retriever.refresh()
		return

	if not REGULATIONS_DIR.exists():
		return

	all_chunks = []
	for file_path in iter_supported_files(REGULATIONS_DIR):
		text = read_document_bytes(file_path)
		if not text:
			continue
		source_id = file_path.stem
		chunks = process_document(source_id=source_id, source_name=file_path.name, text=text)
		all_chunks.extend(chunks)
		vector_store.add_chunks(chunks)

	if all_chunks:
		vector_store.save_chunks_to_jsonl(all_chunks)

	retriever.refresh()



# Start the app by loading regulations and reporting the indexed chunk count.
@app.on_event("startup")
def startup_event() -> None:
	load_regulations_from_disk()
	logger.info("Loaded %s regulation chunks", len(vector_store.all_chunks()))


# Return a simple health check payload for liveness probes.
@app.get("/health")
def health_check() -> dict[str, str]:
	return {"status": "ok"}


@app.post(
	f"{settings.api_prefix}/check-strategy",
	response_model=StrategyCheckResponse,
	responses={400: {"model": ErrorResponse}},
)
# Check a proposed strategy against the indexed regulations.
def check_strategy(payload: StrategyCheckRequest) -> StrategyCheckResponse:
	if not payload.strategy.strip():
		raise HTTPException(status_code=400, detail="Strategy description cannot be empty.")
	query_text = " ".join(
		value
		for value in [
			payload.strategy_name or "",
			payload.strategy,
			payload.description or "",
			payload.asset_class or "",
			" ".join(payload.geographic_scope),
		]
		if value
	)
	retrieved = retriever.retrieve(query_text)
	risk_profile = payload.model_dump(exclude={"strategy", "strategy_name", "description", "asset_class", "geographic_scope"})
	risk_profile["use_insider_info"] = payload.use_insider_info
	risk_profile["leverage_ratio"] = payload.leverage_ratio
	risk_profile["short_selling"] = payload.short_selling
	response = agent.evaluate(query_text, retrieved, risk_profile=risk_profile)
	
	audit_trail.append(
		StrategyAuditRecord(
			strategy_id=response.strategy_id,
			timestamp=response.timestamp,
			decision=response.decision,
			confidence=response.confidence,
			strategy=query_text,
			retrieved_count=response.retrieved_count,
			reason=response.reason,
		)
	)
	return response


# Upload a regulation file and index it immediately for later searches.
@app.post(
	f"{settings.api_prefix}/upload-regulation",
	response_model=UploadRegulationResponse,
	responses={400: {"model": ErrorResponse}},
)
async def upload_regulation(
	file: UploadFile = File(...),
	regulation_title: str | None = Form(default=None),
	jurisdiction: str | None = Form(default=None),
	effective_date: str | None = Form(default=None),
	section: str | None = Form(default=None),
) -> UploadRegulationResponse:

	if not file.filename:
		raise HTTPException(status_code=400, detail="A regulation file name is required.")
	content = await file.read()
	if not content:
		raise HTTPException(status_code=400, detail="Uploaded regulation file is empty.")

	try:
		text = read_uploaded_bytes(content, file.filename)
	except Exception as exc:  # pragma: no cover - defensive guard
		raise HTTPException(status_code=400, detail=f"Could not parse regulation file: {exc}") from exc

	source_id = f"uploaded-{uuid4().hex[:12]}"
	chunks = process_document(
		source_id=source_id,
		source_name=file.filename,
		text=text,
		regulation_title=regulation_title,
		jurisdiction=jurisdiction,
		effective_date=effective_date,
		section=section,
	)
	vector_store.add_chunks(chunks)
	retriever.refresh()
	first_chunk = chunks[0] if chunks else None
	return UploadRegulationResponse(
		source_id=source_id,
		source_name=file.filename,
		regulation_title=first_chunk.regulation_title if first_chunk else file.filename,
		jurisdiction=first_chunk.jurisdiction if first_chunk else "unknown",
		effective_date=first_chunk.effective_date if first_chunk else "unknown",
		chunk_count=len(chunks),
	)


# List the indexed regulations with optional jurisdiction and source filters.
@app.get(f"{settings.api_prefix}/regulations", response_model=list[RegulationInfo])
def list_regulations(jurisdiction: str | None = Query(default=None), source_id: str | None = Query(default=None)) -> list[RegulationInfo]:
	items = vector_store.grouped_regulations()
	if jurisdiction:
		items = [item for item in items if str(item["jurisdiction"]).lower() == jurisdiction.lower()]
	if source_id:
		items = [item for item in items if str(item["source_id"]).lower() == source_id.lower()]
	return [
		RegulationInfo(
			source_id=str(item["source_id"]),
			source_name=str(item["source_name"]),
			regulation_title=str(item["regulation_title"]),
			jurisdiction=str(item["jurisdiction"]),
			effective_date=str(item["effective_date"]),
			chunk_count=cast(int, item["chunk_count"]),
			text_preview=str(item["text_preview"]),
		)
		for item in items
	]


# Search regulations directly without running a compliance decision.
@app.get(f"{settings.api_prefix}/search", response_model=RegulationSearchResponse)
def search_regulations(query: str = Query(..., min_length=2), jurisdiction: str | None = None) -> RegulationSearchResponse:
	started_at = perf_counter()
	results = retriever.retrieve(query, jurisdiction=jurisdiction)
	citations = [result_to_citation(result) for result in results]
	retrieval_time_ms = int((perf_counter() - started_at) * 1000)
	return RegulationSearchResponse(query=query, results=citations, retrieval_time_ms=retrieval_time_ms)


# Return the recorded compliance decisions for audit review.
@app.get(f"{settings.api_prefix}/audit-trail", response_model=list[StrategyAuditRecord])
def get_audit_trail() -> list[StrategyAuditRecord]:
	return list(audit_trail)


# Convert a retrieval result into the API citation shape.
def result_to_citation(result: dict) -> Citation:
	return Citation(
		source_id=result["source_id"],
		source_name=result["source_name"],
		regulation_title=result["regulation_title"],
		jurisdiction=result["jurisdiction"],
		effective_date=result["effective_date"],
		section=result["section"],
		chunk_id=result["chunk_id"],
		confidence=result["confidence"],
		excerpt=result["text"][:240],
	)