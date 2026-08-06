Regulatory Compliance RAG

Ever wondered whether a trading strategy is about to get you in trouble with GDPR, MiFID II, or Basel III before you actually run it? That's exactly what this project does — it reads your strategy description, digs through real regulatory text, and hands you back a verdict with receipts.

Why I Built This

Compliance review is one of those things everyone agrees is important and nobody enjoys doing manually. A human has to sit down, cross-reference a strategy against hundreds of pages of dense regulatory text, and hope they didn't miss the one clause that matters. It's slow, it's easy to get wrong, and it doesn't scale.

I wanted to see if a properly built retrieval system — not just "throw everything at an LLM and hope" — could actually surface the right regulatory clause fast enough to be useful as a first-pass filter. Turns out, with the right retrieval architecture, it can get pretty close. This isn't magic and it isn't a lawyer, but it's a genuinely useful triage layer that tells you where to look and why.

What It Actually Does

Feed it a strategy description. It searches through indexed regulations using a mix of keyword matching and semantic understanding, pulls out the most relevant clauses, checks the strategy against a set of compliance rules, and returns one of three verdicts — APPROVED, BLOCKED, or NEEDS REVIEW — along with the exact regulatory text that backs up the decision.

No black box. Every call is traceable back to a real chunk of a real regulation.
Features

Finds the right regulation, not just a matching keyword. Combines BM25 lexical search with semantic embeddings, fused together with Reciprocal Rank Fusion (RRF), then re-ranked by a cross-encoder for a real precision pass. Keyword search alone misses meaning; embeddings alone miss exact terminology. This uses both.
Reads your actual documents. Ingests .txt, .md, and .pdf regulations, with built-in handling for corrupted or invalid files so a bad PDF doesn't take down your whole ingestion run.
Explains itself. Every decision comes with citations back to the specific regulatory chunks that drove it — jurisdiction, section, effective date, the works.
Flags real risk categories. Insider trading, market abuse, GDPR-conflicting data use, high leverage, short-selling exposure — a rule-based compliance layer checks for all of it and tells you exactly which rule fired.
Fast restarts. Chunks get cached to JSONL after the first ingestion, so you're not re-parsing PDFs and re-embedding everything every time you restart the server.
Add regulations on the fly. Upload a new regulation through the API and it's searchable immediately — no redeploy needed.
Docs you can actually click through. Full interactive Swagger UI, so you can test every endpoint without writing a single curl command.
Who This Is For

Anyone building compliance tooling, exploring hybrid RAG architectures, or just curious what it looks like when you combine lexical search, semantic search, and reranking instead of reaching straight for "vector search and call it a day." If you're evaluating RAG patterns for a regulated industry — finance, healthcare, legal — this is a solid reference architecture to poke at.

Tech Stack

Framework: FastAPI (Python 3.10+)
Embeddings & Reranking: Sentence-Transformers, HuggingFace Transformers
Vector Math: NumPy
PDF Parsing: PyPDF2
Validation: Pydantic v2
How It Works, Roughly

Your strategy text
      │
      ├──► Semantic search     (does this mean the same thing?)
      ├──► BM25 lexical search (does this use the same words?)
      │
      ▼
Reciprocal Rank Fusion (merge both rankings)
      │
      ▼
Cross-encoder reranking (double-check what actually matters)
      │
      ▼
Rule-based compliance check ──► decision + citations
Quick Start

1. Installation

# Grab the code and get into the project folder
cd regulatory-compliance-rag

# Set up a clean environment
python3 -m venv ../.venv
source ../.venv/bin/activate

# Install dependencies
pip install -r requirements.txt
2. Run the Server

uvicorn main:app --reload
Once it's running, head to http://127.0.0.1:8000/docs — that's the interactive Swagger UI, and honestly it's the easiest way to poke around and see what this thing can do without writing any client code.

Sanity check: Verify NumPy resolved correctly in your environment:

python -c "import numpy; print('NumPy Version:', numpy.__version__)"
API Endpoints

Method	Endpoint	What it does
POST	/compliance/check-strategy	Submit a strategy, get back a decision with citations
POST	/compliance/upload-regulation	Index a new regulation document
GET	/compliance/search	Search indexed regulations directly, no compliance decision attached
GET	/compliance/regulations	List everything currently indexed
GET	/compliance/audit-trail	Pull up the history of past compliance checks
GET	/health	Liveness check, for when you just need to know it's alive
Example

Send this:

POST /compliance/check-strategy
{
  "strategy_name": "Customer Data Targeting",
  "strategy": "Use customer personal data to target EU clients with automated trading signals.",
  "asset_class": "equities",
  "leverage_ratio": 1.5,
  "geographic_scope": ["EU"]
}
Get back something like this:

{
  "strategy_id": "strategy-a8f1b94d21e0",
  "decision": "NEEDS_REVIEW",
  "reason": "NEEDS_REVIEW: The strategy appears to process personal data in a way that may conflict with GDPR controls.",
  "citations": [
    {
      "regulation_title": "GDPR General Data Protection Regulation",
      "jurisdiction": "EU",
      "section": "Article 6",
      "excerpt": "Personal data must be processed lawfully, fairly, and transparently. Consent or another lawful basis is required before using customer data."
    }
  ],
  "risk_flags": ["personal data misuse"],
  "recommendations": ["Clarify the strategy details and confirm the applicable compliance requirements."]
}
(Response shortened here for readability — the real payload includes full citation metadata, confidence scores, and a timestamp.)

Configuration

Set these as environment variables, or edit config.py directly:

Variable	Default	What it controls
EMBEDDING_MODEL	sentence-transformers/all-MiniLM-L6-v2	Which model generates the semantic vectors
EMBEDDING_DIM	384	Dimensionality of those vectors
RERANKING_ENABLED	true	Turn the cross-encoder reranking step on or off
RERANKER_MODEL	Xenova/ms-marco-MiniLM-L-6-v2	Which model does the reranking
RERANK_CANDIDATE_COUNT	20	How many candidates get passed to the reranker
MAX_CHUNK_SIZE	1200	Max characters per chunk
CHUNK_OVERLAP	150	Overlap between consecutive chunks
Known Limitations (Because Honesty Matters)

Jurisdiction metadata: Defaults to "unknown" for PDFs that don't have a structured header — most real-world PDFs don't, so this is common right now, not an edge case.
Rule-based compliance decisions: The decision layer relies on keyword/pattern matching rather than an LLM — it's fast and explainable, but it won't catch anything outside its known risk patterns.
In-memory store: The default vector store is in-memory for development. There is a Postgres + pgvector path for production shapes, but it requires its own setup.
Triage tool only: This is not legal advice — treat every decision as a starting point for human review, not a final word.
Contributing

Found a bug, have an idea, or just want to argue about retrieval architecture? Open an issue or send a pull request. If you're adding a feature, a quick heads-up first is appreciated so we're not duplicating work, but honestly, don't overthink it — just open the PR and we'll figure it out together.
