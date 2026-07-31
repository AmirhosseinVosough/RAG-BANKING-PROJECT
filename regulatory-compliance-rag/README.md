# Regulatory Compliance RAG

FastAPI MVP for evaluating proposed trading strategies against regulatory constraints.

## Features

- Ingests text and PDF regulations
- Stores regulation chunks in a local in-memory vector store
- Retrieves a bounded candidate set with pretrained semantic embeddings and hybrid keyword scoring, then cross-encoder reranks it for relevance
- Returns APPROVED or BLOCKED decisions with citations
- Exposes FastAPI endpoints for checking strategies, uploading regulations, and listing indexed rules

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`, which produces 384-dimensional semantic vectors.
The default reranker is `Xenova/ms-marco-MiniLM-L-6-v2`. Set `RERANKING_ENABLED=false` to use first-stage retrieval only, or tune `RERANK_CANDIDATE_COUNT` (default: `20`) to control the number of chunks considered by the reranker.

## Endpoints

- `POST /compliance/check-strategy`
- `POST /compliance/upload-regulation`
- `GET /compliance/regulations`

## Run

```bash
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

`modules/vector_store.py` uses NumPy directly. If your editor reports that it
cannot resolve `numpy`, select the same interpreter used to install the
requirements. In this workspace that is `.venv/bin/python` at the workspace
root; verify it with:

```bash
../.venv/bin/python -c "import numpy; print(numpy.__version__)"
```

## Example request

```json
{
	"strategy": "Use customer data to target EU clients with personalized trading signals."
}
```
