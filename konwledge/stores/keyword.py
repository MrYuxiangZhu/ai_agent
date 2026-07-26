"""BM25 keyword index."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Optional, Sequence

from konwledge.core.models import Chunk, SearchResult
from konwledge.processing.text import tokenize
from konwledge.stores.vector import _metadata_matches


class BM25KeywordIndex:
    """Dependency-free BM25 index for exact and sparse retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: Dict[str, Chunk] = {}
        self._term_freqs: Dict[str, Counter] = {}
        self._doc_freqs: Dict[str, int] = defaultdict(int)
        self._doc_lengths: Dict[str, int] = {}

    def add(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            if chunk.id in self._chunks:
                continue
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            self._chunks[chunk.id] = chunk
            self._term_freqs[chunk.id] = counts
            self._doc_lengths[chunk.id] = len(tokens)
            for term in counts:
                self._doc_freqs[term] += 1

    def search(
        self,
        query: str,
        top_k: int,
        metadata_filter: Optional[Mapping[str, object]] = None,
    ) -> List[SearchResult]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        average_length = self._average_doc_length()
        results: List[SearchResult] = []
        for chunk_id, chunk in self._chunks.items():
            if not _metadata_matches(chunk.metadata, metadata_filter):
                continue
            score = self._score(query_terms, chunk_id, average_length)
            if score > 0:
                results.append(SearchResult(chunk=chunk, score=score, keyword_score=score, reason="bm25"))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def _score(self, query_terms: List[str], chunk_id: str, average_length: float) -> float:
        total_docs = max(1, len(self._chunks))
        length = self._doc_lengths.get(chunk_id, 0)
        counts = self._term_freqs.get(chunk_id, Counter())
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            df = self._doc_freqs.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            denominator = tf + self._k1 * (1 - self._b + self._b * length / max(1.0, average_length))
            score += idf * (tf * (self._k1 + 1)) / denominator
        return score

    def _average_doc_length(self) -> float:
        if not self._doc_lengths:
            return 1.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)
