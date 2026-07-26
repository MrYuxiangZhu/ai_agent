"""Hybrid retrieval combining vector and keyword search."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Mapping, Optional

from konwledge.core.config import RetrievalConfig
from konwledge.core.interfaces import EmbeddingModel, KeywordIndex, Reranker, VectorStore
from konwledge.core.models import SearchResult


class HybridRetriever:
    """Retrieve with dense vectors, BM25 keywords, metadata filters and rerank."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._reranker = reranker
        self._config = config or RetrievalConfig()

    def retrieve(self, query: str, metadata_filter: Optional[Mapping[str, object]] = None) -> List[SearchResult]:
        query_vector = self._embedding_model.embed_text(query)
        vector_results = self._vector_store.search(query_vector, self._config.vector_top_k, metadata_filter)
        keyword_results = self._keyword_index.search(query, self._config.keyword_top_k, metadata_filter)
        merged = self._merge(vector_results, keyword_results)
        candidates = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        if self._config.use_rerank and self._reranker:
            return self._reranker.rerank(query, candidates, self._config.final_top_k)
        return candidates[: self._config.final_top_k]

    def _merge(self, vector_results: List[SearchResult], keyword_results: List[SearchResult]) -> Dict[str, SearchResult]:
        vector_scores = self._normalize({item.chunk.id: item.vector_score for item in vector_results})
        keyword_scores = self._normalize({item.chunk.id: item.keyword_score for item in keyword_results})
        by_id: Dict[str, SearchResult] = {}
        for item in vector_results + keyword_results:
            chunk_id = item.chunk.id
            vector_score = vector_scores.get(chunk_id, 0.0)
            keyword_score = keyword_scores.get(chunk_id, 0.0)
            score = self._config.vector_weight * vector_score + self._config.keyword_weight * keyword_score
            reason_parts = []
            if vector_score:
                reason_parts.append("vector")
            if keyword_score:
                reason_parts.append("bm25")
            current = by_id.get(chunk_id)
            candidate = replace(
                item,
                score=score,
                vector_score=vector_score,
                keyword_score=keyword_score,
                reason="+".join(reason_parts),
            )
            if current is None or candidate.score > current.score:
                by_id[chunk_id] = candidate
        return by_id

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        lower = min(values)
        upper = max(values)
        if upper == lower:
            return {key: 1.0 for key in scores}
        return {key: (value - lower) / (upper - lower) for key, value in scores.items()}
