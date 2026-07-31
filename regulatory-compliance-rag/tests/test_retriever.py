from modules.agent import ComplianceAgent
from modules.document_processor import process_document
from modules.retriever import Retriever
from modules.vector_store import VectorStore


def build_store() -> VectorStore:
    store = VectorStore()
    store.add_chunks(
        process_document(
            source_id="gdpr-001",
            source_name="gdpr.txt",
            text="Personal data must only be processed with a lawful basis and valid consent.",
        )
    )
    store.add_chunks(
        process_document(
            source_id="sec-001",
            source_name="sec.txt",
            text="Market abuse and insider trading are prohibited.",
        )
    )
    return store


def test_retriever_returns_relevant_rules() -> None:
    retriever = Retriever(build_store())
    results = retriever.retrieve("Use customer data for targeted EU trading signals")
    assert results
    assert results[0].source_id == "gdpr-001"


def test_agent_blocks_high_risk_strategy() -> None:
    store = build_store()
    retriever = Retriever(store)
    agent = ComplianceAgent()
    retrieved = retriever.retrieve("Use insider tips for trading ahead of announcements")
    response = agent.evaluate("Use insider tips for trading ahead of announcements", retrieved)
    assert response.decision == "BLOCKED"


def test_agent_can_return_needs_review_for_ambiguous_strategy() -> None:
	retriever = Retriever(build_store())
	agent = ComplianceAgent()
	response = agent.evaluate("Maybe use customer data for a borderline strategy", retriever.retrieve("borderline customer data strategy"))
	assert response.decision == "NEEDS_REVIEW"