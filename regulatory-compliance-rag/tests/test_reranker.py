from config import settings
from modules.reranker import Reranker
from modules.retriever import Retriever


class FakeCrossEncoder:
	def rerank(self, query: str, documents: list[str]) -> list[float]:
		assert query == "insider trading controls"
		assert documents == ["general policy", "insider trading prohibition"]
		return [0.1, 0.9]


class FakeVectorStore:
	def __init__(self) -> None:
		self.requested_top_k: int | None = None

	def search(self, query: str, top_k: int, jurisdiction: str | None = None, source_id: str | None = None) -> list[dict[str, object]]:
		self.requested_top_k = top_k
		return [
			_result("general", "general policy"),
			_result("insider", "insider trading prohibition"),
		]


def _result(chunk_id: str, text: str) -> dict[str, object]:
	return {
		"source_id": "sec-001",
		"source_name": "sec.txt",
		"regulation_title": "SEC Rule",
		"jurisdiction": "US",
		"effective_date": "2024-01-01",
		"section": "general",
		"chunk_id": chunk_id,
		"text": text,
		"confidence": 0.5,
	}


def test_reranker_reorders_candidates_by_cross_encoder_score() -> None:
	reranker = Reranker(enabled=True, model_factory=FakeCrossEncoder)
	results = reranker.rerank(
		"insider trading controls",
		[_result("general", "general policy"), _result("insider", "insider trading prohibition")],
		top_k=1,
	)
	assert [result["chunk_id"] for result in results] == ["insider"]


def test_retriever_fetches_a_larger_candidate_pool_before_reranking() -> None:
	store = FakeVectorStore()
	reranker = Reranker(enabled=True, model_factory=FakeCrossEncoder)
	results = Retriever(store, reranker=reranker).retrieve("insider trading controls", top_k=1)
	assert store.requested_top_k == settings.rerank_candidate_count
	assert [result.chunk_id for result in results] == ["insider"]


def test_reranker_falls_back_to_first_stage_order_when_disabled() -> None:
	reranker = Reranker(enabled=False)
	results = reranker.rerank("anything", [_result("first", "first"), _result("second", "second")], top_k=1)
	assert [result["chunk_id"] for result in results] == ["first"]


def test_reranker_falls_back_to_first_stage_order_when_model_is_unavailable() -> None:
	def unavailable_model() -> FakeCrossEncoder:
		raise RuntimeError("model download failed")

	reranker = Reranker(enabled=True, model_factory=unavailable_model)
	results = reranker.rerank("anything", [_result("first", "first"), _result("second", "second")], top_k=1)
	assert [result["chunk_id"] for result in results] == ["first"]
