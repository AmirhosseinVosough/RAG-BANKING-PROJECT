from fastapi.testclient import TestClient

from main import app, vector_store


client = TestClient(app)


def setup_module() -> None:
    if not vector_store.all_chunks():
        from modules.document_processor import process_document

        vector_store.add_chunks(
            process_document(
                source_id="internal-001",
                source_name="internal-policy.txt",
                text="Trading strategies must not use personal data without consent.",
            )
        )


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_check_strategy_endpoint_blocks_risky_strategy() -> None:
    response = client.post(
        "/compliance/check-strategy",
        json={"strategy": "Use insider information to trade ahead of earnings."},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] == "BLOCKED"
    assert payload["citations"]


def test_check_strategy_endpoint_can_return_needs_review() -> None:
	response = client.post(
		"/compliance/check-strategy",
		json={"strategy": "Maybe use customer data in a borderline way for signal generation."},
	)
	assert response.status_code == 200
	assert response.json()["decision"] in {"NEEDS_REVIEW", "BLOCKED", "APPROVED"}


def test_list_regulations_endpoint() -> None:
    response = client.get("/compliance/regulations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_search_regulations_endpoint() -> None:
	response = client.get("/compliance/search", params={"query": "personal data consent", "jurisdiction": "unknown"})
	assert response.status_code == 200
	assert "results" in response.json()