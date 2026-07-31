from modules.agent import ComplianceAgent
from modules.models import Citation
from modules.retriever import RetrievedRegulation


def test_agent_approves_low_risk_strategy() -> None:
    agent = ComplianceAgent()
    response = agent.evaluate(
        "Rebalance a diversified ETF basket based on public market indicators.",
        [
            RetrievedRegulation(
                source_id="sec-001",
                source_name="sec.txt",
                regulation_title="SEC Rule",
                jurisdiction="SEC",
                effective_date="2024-01-01",
                section="general",
                chunk_id="sec-001-chunk-1",
                text="Market abuse and insider trading are prohibited.",
                confidence=0.82,
            )
        ],
    )
    assert response.decision == "APPROVED"
    assert response.citations
    assert response.strategy_id


def test_citation_shape() -> None:
    citation = Citation(
        source_id="gdrp-001",
        source_name="gdpr.txt",
        regulation_title="GDPR Core",
        jurisdiction="EU",
        effective_date="2024-01-01",
        section="article-32",
        chunk_id="gdrp-001-chunk-1",
        confidence=0.91,
        excerpt="Personal data must only be processed with consent.",
    )
    assert citation.source_name == "gdpr.txt"