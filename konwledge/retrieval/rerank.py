"""Rerankers used after first-stage retrieval."""

from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence

from konwledge.core.models import SearchResult
from konwledge.processing.text import tokenize


class LexicalReranker:
    """Lightweight reranker based on query term coverage and original scores."""

    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> List[SearchResult]:
        query_terms = set(tokenize(query))
        reranked: List[SearchResult] = []
        for result in results:
            chunk_terms = set(tokenize(result.chunk.text))
            coverage = len(query_terms & chunk_terms) / max(1, len(query_terms))
            title_bonus = 0.1 if query_terms & set(tokenize(result.chunk.title)) else 0.0
            rerank_score = 0.7 * result.score + 0.3 * coverage + title_bonus
            reranked.append(
                replace(
                    result,
                    score=rerank_score,
                    rerank_score=rerank_score,
                    reason=f"{result.reason}+lexical_rerank",
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]
