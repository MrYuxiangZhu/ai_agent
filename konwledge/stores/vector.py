"""Vector store implementations."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from konwledge.core.models import Chunk, SearchResult
from konwledge.processing.text import cosine_similarity


def _metadata_matches(metadata: Mapping[str, object], metadata_filter: Optional[Mapping[str, object]]) -> bool:
    if not metadata_filter:
        return True
    for key, expected in metadata_filter.items():
        if metadata.get(key) != expected:
            return False
    return True


class InMemoryVectorStore:
    """A small persistent vector store suitable for local demos and tests."""

    def __init__(self) -> None:
        self._chunks: Dict[str, Chunk] = {}
        self._vectors: Dict[str, List[float]] = {}

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, vectors):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = [float(value) for value in vector]

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        metadata_filter: Optional[Mapping[str, object]] = None,
    ) -> List[SearchResult]:
        results: List[SearchResult] = []
        for chunk_id, vector in self._vectors.items():
            chunk = self._chunks[chunk_id]
            if not _metadata_matches(chunk.metadata, metadata_filter):
                continue
            score = cosine_similarity(query_vector, vector)
            results.append(SearchResult(chunk=chunk, score=score, vector_score=score, reason="vector"))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def persist(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": {chunk_id: asdict(chunk) for chunk_id, chunk in self._chunks.items()},
            "vectors": self._vectors,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self._chunks = {chunk_id: Chunk(**chunk_data) for chunk_id, chunk_data in payload.get("chunks", {}).items()}
        self._vectors = {
            chunk_id: [float(value) for value in vector]
            for chunk_id, vector in payload.get("vectors", {}).items()
        }

    def all_chunks(self) -> List[Chunk]:
        return list(self._chunks.values())


class ExternalVectorStoreAdapter:
    """Base adapter for FAISS, Milvus, Qdrant, Chroma and pgvector.

    Production integrations should subclass this adapter and implement the same
    ``add/search/persist/load`` contract. Keeping this interface stable allows
    the retrieval pipeline to stay unchanged.
    """

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        metadata_filter: Optional[Mapping[str, object]] = None,
    ) -> List[SearchResult]:
        raise NotImplementedError

    def persist(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError
