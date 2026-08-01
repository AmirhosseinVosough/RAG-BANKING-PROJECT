from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Protocol

from config import settings


logger = logging.getLogger(__name__)


class CrossEncoder(Protocol):
	def rerank(self, query: str, documents: list[str]) -> Iterable[float]: ...


class Reranker:
	"""Rerank a small set of retrieved chunks with a cross-encoder."""

	def __init__(
		self,
		enabled: bool | None = None,
		model_name: str | None = None,
		model_factory: Callable[[], CrossEncoder] | None = None,
	) -> None:
		self.enabled = settings.reranking_enabled if enabled is None else enabled
		self.model_name = model_name or settings.reranker_model_name
		self._model_factory = model_factory
		self._model: CrossEncoder | None = None
		self._unavailable = False

	def rerank(self, query: str, candidates: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
		"""Return the highest cross-encoder-scored candidates, preserving ties' input order."""
		if not self.enabled or self._unavailable or not candidates:
			return candidates[:top_k]

		try:
			scores = list(self._get_model().rerank(query, [str(candidate["text"]) for candidate in candidates]))
			if len(scores) != len(candidates):
				raise RuntimeError("Reranker returned a score count that does not match the candidate count.")
		except Exception as exc:  # pragma: no cover - model download/runtime failures depend on the host
			self._unavailable = True
			logger.warning("Reranker unavailable; returning first-stage retrieval order: %s", exc)
			return candidates[:top_k]

		ranked = sorted(
			enumerate(zip(candidates, scores)),
			key=lambda item: (-float(item[1][1]), item[0]),
		)
		return [candidate for _, (candidate, _) in ranked[:top_k]]

	def _get_model(self) -> CrossEncoder:
		if self._model is None:
			if self._model_factory is not None:
				self._model = self._model_factory()
			else:
				try:
					from fastembed.rerank.cross_encoder import TextCrossEncoder
				except ImportError as exc:
					raise RuntimeError(
						"Cross-encoder reranking requires fastembed. Install dependencies with `pip install -r requirements.txt`."
					) from exc
				self._model = TextCrossEncoder(model_name=self.model_name)
		return self._model

